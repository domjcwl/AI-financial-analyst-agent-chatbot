r"""Deep-research stock-picking agent built on LangGraph.

Graph (multi-item dispatch):

  START
    -> user_query
    -> query_planner          (produces ResearchPlan: list of QueryItems)
       |
       +-- dispatch_research (LangGraph Send):
       |     for each item in plan.items:
       |       if item.query_type == "stock_analysis":
       |           Send -> fundamentals_agent / sentiment_agent /
       |                   technical_agent     / macro_agent      (target=ticker)
       |       elif item.query_type == "industry_analysis":
       |           Send -> sentiment_agent / macro_agent          (target=industry)
       |           (fundamentals + technical are SKIPPED)
       v
    -> aggregator
    -> critic ---(gather_more)--> query_planner   # evidence loop
              \--(sufficient)--> risk
                                 -> synthesis
                                    -> human_review ---(revise)--> synthesis
                                                  \--(approve)--> final_report
                                                                  -> END

Per-dimension state slots are Dict[target, Finding] (target = ticker for stock
items, industry name for industry items) with a merge reducer so parallel Send
invocations don't clobber one another. Each slot is a Pydantic model defined in
schemas.py.

The CLI / Rich rendering layer lives in cli.py; this module only defines the
graph and exposes `app` and `build_graph()` for import.
"""

# Silence third-party noise (yfinance HTTP 404s, urllib3, etc.) BEFORE those
# modules are imported, so their loggers inherit the muted level.
import logging
import warnings
for _noisy in ("yfinance", "peewee", "urllib3", "urllib3.connectionpool",
               "requests", "httpx", "httpcore"):
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from typing import Annotated, Any, Dict, List, Optional, TypedDict
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, Send, interrupt
import httpx
import os

from schemas import (
    CriticDecision,
    FinalReport,
    FundamentalsFindings,
    Holding,
    HumanReview,
    IndustryOutlook,
    InvestmentThesis,
    KeyMetric,
    MacroFindings,
    QueryItem,
    ResearchPlan,
    SentimentFindings,
    StockOutlook,
    TechnicalFindings,
)
from tools import (
    FUNDAMENTAL_TOOLS,
    MACRO_TOOLS,
    SENTIMENT_TOOLS,
    TECHNICAL_TOOLS,
)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# ---------- Tunables ----------

MAX_RESEARCH_ITERATIONS = 2   # caps the critic -> planner loop
MAX_REVISION_ITERATIONS = 2   # caps the human -> synthesis loop
SUBAGENT_RECURSION_LIMIT = 40 # how many ReAct steps a sub-agent may take

# Model tiers. Reasoning nodes (planner, critic, risk, synthesis, final_report)
# get the larger model because their decisions gate the rest of the workflow.
# Sub-agents stay on the cheaper model — they're tool-heavy, not reasoning-heavy.
REASONING_MODEL = "gpt-4o"
SUBAGENT_MODEL = "gpt-4o-mini"
EXTRACTOR_MODEL = "gpt-4o-mini"

# Minimum self-rated confidence per dimension below which the critic should
# treat a finding as insufficient. Mirrored in the critic prompt.
MIN_DIMENSION_CONFIDENCE = 0.6

# Shared httpx client to bypass corporate proxy/firewall (matches sample pattern)
_http = httpx.Client(verify=False)


def _llm(model: str = SUBAGENT_MODEL, temperature: float = 0.1) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        http_client=_http,
        api_key=api_key,
    )


# ---------- State ----------

def _merge_findings(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Reducer for per-dimension finding dicts. Later writes overwrite earlier
    entries for the same target; new targets are added."""
    if not a:
        return b or {}
    if not b:
        return a
    return {**a, **b}


class State(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    research_plan: Optional[ResearchPlan]
    fundamentals: Annotated[Dict[str, FundamentalsFindings], _merge_findings]
    sentiment:    Annotated[Dict[str, SentimentFindings],    _merge_findings]
    technical:    Annotated[Dict[str, TechnicalFindings],    _merge_findings]
    macro:        Annotated[Dict[str, MacroFindings],        _merge_findings]
    critic: Optional[CriticDecision]
    research_iterations: int
    thesis: Optional[InvestmentThesis]
    human_review: Optional[HumanReview]
    revision_iterations: int
    final_report: Optional[FinalReport]


# ---------- Sub-agent helper ----------

# Universal research discipline injected into every sub-agent's system message.
# Forces the multi-tool-call, multi-source, citation-required behaviour that
# distinguishes a real research pass from a one-shot LLM guess.
_RESEARCH_DISCIPLINE = (
    "RESEARCH DISCIPLINE — non-negotiable:\n"
    "1. Call each of your tools AT LEAST ONCE before drafting any answer. Make "
    "   MULTIPLE calls with different query phrasings when using web/news search.\n"
    "2. Address EVERY plan question explicitly. If a question cannot be answered "
    "   from available data, add it to `unanswered_questions` in the finding.\n"
    "3. Cite SPECIFIC numbers (revenue %, margin %, $ values, RSI, multiples) — "
    "   not vague qualifiers like 'strong' or 'good'. Pull numbers from tool "
    "   results, not from memory.\n"
    "4. Capture every source URL you relied on; the structured-output step will "
    "   populate `sources`. Deduplicate.\n"
    "5. Calibrate confidence honestly: 0.8+ requires 3+ sources and all "
    "   questions answered; 0.5-0.7 for partial coverage; <0.5 when data is "
    "   sparse, stale, or contradictory.\n"
    "6. Look for DISCONFIRMING evidence as well as supporting; do not cherry-pick.\n"
)


def _run_subagent(
    role_system: str,
    task_prompt: str,
    tools: list,
    schema,
    questions: Optional[List[str]] = None,
    fallback_factory=None,
    model: str = SUBAGENT_MODEL,
):
    """Spin up a ReAct sub-agent with a curated tool set, then extract a
    schema-validated finding from its final answer.

    On failure (tool blow-up, parser error, schema-validation error), returns
    the result of `fallback_factory()` so downstream nodes still receive a
    valid Finding instead of crashing the graph. The fallback carries
    confidence=0 so the critic will route to gather_more.
    """
    system_with_discipline = f"{role_system}\n\n{_RESEARCH_DISCIPLINE}"
    agent = create_agent(_llm(model=model), tools=tools)
    try:
        result = agent.invoke(
            {"messages": [
                SystemMessage(content=system_with_discipline),
                HumanMessage(content=task_prompt),
            ]},
            config={"recursion_limit": SUBAGENT_RECURSION_LIMIT},
        )
        # Prefer the last AI message (the agent's final synthesis); fall back to
        # whatever the last message is so we never lose what the agent produced.
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        raw_text = ai_msgs[-1].content if ai_msgs else result["messages"][-1].content

        extractor_system = (
            "Extract a structured finding from the analyst's research notes. "
            "Preserve specific numbers, multiples, and source URLs verbatim. "
            "Populate `unanswered_questions` with any plan question that the "
            "notes did not address. Do not invent data — leave fields empty "
            "or use a low confidence value when evidence is missing."
        )
        if questions:
            extractor_system += (
                "\n\nORIGINAL PLAN QUESTIONS — every one must either be answered "
                "in the finding or appear in `unanswered_questions`:\n- "
                + "\n- ".join(questions)
            )

        extractor = _llm(model=EXTRACTOR_MODEL, temperature=0).with_structured_output(schema)
        return extractor.invoke([
            SystemMessage(content=extractor_system),
            HumanMessage(content=raw_text),
        ])
    except Exception as exc:
        if fallback_factory is None:
            raise
        logging.getLogger(__name__).warning(
            "Sub-agent for %s failed: %s — emitting fallback finding.",
            schema.__name__, exc,
        )
        return fallback_factory(str(exc))


# ---------- Nodes ----------

def user_query_node(state: State) -> State:
    """Entry node: surface the user's question into the message stream."""
    return {"messages": [HumanMessage(content=state["user_query"])]}


_ALL_DIMS = ("fundamentals", "sentiment", "technical", "macro")
_STOCK_DIMS = {"fundamentals", "sentiment", "technical", "macro"}
_INDUSTRY_DIMS = {"sentiment", "macro"}

_DIM_TO_QUESTION_FIELD = {
    "fundamentals": "fundamentals_questions",
    "sentiment":    "sentiment_questions",
    "technical":    "technical_questions",
    "macro":        "macro_questions",
}


def _valid_dims_for(item: QueryItem) -> set:
    return _STOCK_DIMS if item.query_type == "stock_analysis" else _INDUSTRY_DIMS


def query_planner_node(state: State) -> State:
    """Decompose the user query into a structured ResearchPlan.

    The plan is a LIST of QueryItems. Each item is either:
      - stock_analysis    (target = ticker, all 4 agents run)
      - industry_analysis (target = industry label, only sentiment + macro run)

    On re-runs after an insufficient critic, this node:
      - locks the items list (query_type + target per item)
      - asks the LLM to refine questions ONLY for the (target, dimension) pairs
        the critic flagged
      - preserves the prior plan's questions for all other (target, dimension)
        pairs verbatim
    """
    previous_plan: Optional[ResearchPlan] = state.get("research_plan")
    crit = state.get("critic")
    is_rerun = (
        crit is not None
        and not crit.sufficient
        and previous_plan is not None
    )

    base_system = (
        "You are a senior buy-side research lead. Decompose the user's query into a "
        "ResearchPlan containing one or more QueryItems.\n\n"
        "There are exactly TWO query types:\n"
        "  - stock_analysis    : a specific stock / ETF / company is named "
        "(target = ticker). Runs fundamentals + sentiment + technical + macro.\n"
        "  - industry_analysis : a sector / industry is referenced without a specific "
        "company (target = a short canonical industry label). Runs ONLY sentiment + macro.\n\n"
        "Decomposition rules:\n"
        "1. ONE item per named stock. Resolve company names to tickers "
        "   (e.g. 'Google' -> 'GOOGL', 'Lockheed' -> 'LMT'). ETFs are also "
        "   stock_analysis (e.g. 'XLE', 'VOO', 'SPY').\n"
        "2. ONE item per named industry / sector. Use a short canonical label "
        "   (e.g. 'technology', 'healthcare', 'energy', 'financials', "
        "   'consumer discretionary', 'industrials', 'utilities').\n"
        "3. A single query MAY contain a mix of both types — produce as many items "
        "   as the user implied. Examples:\n"
        "     - 'Analyse NVDA, META, JPM, XLE' -> 4 stock_analysis items.\n"
        "     - 'Analyse the tech and healthcare industries' -> 2 industry_analysis items.\n"
        "     - 'Analyse the tech industry and tell me if Google is a buy' -> "
        "       1 industry_analysis (technology) + 1 stock_analysis (GOOGL).\n"
        "4. holdings: populate with quantity ONLY if the user states positions "
        "   (e.g. '3 VOO, 4 GOOGL'). Otherwise empty.\n\n"
        "Per-item questions (3-5 each):\n"
        "  - stock_analysis items: populate fundamentals_questions, "
        "    sentiment_questions, technical_questions, AND macro_questions.\n"
        "  - industry_analysis items: populate ONLY sentiment_questions and "
        "    macro_questions. Leave fundamentals_questions and "
        "    technical_questions EMPTY — those agents will be skipped.\n"
        "  - Each question must be specific enough that an analyst could answer "
        "    it with a number or a named source. Avoid open-ended 'how is X doing?'.\n"
    )

    if is_rerun:
        gap_dims = set(crit.gap_dimensions) or set(_ALL_DIMS)
        gap_targets = set(crit.gap_targets) or set(previous_plan.all_targets())
        item_lines = [
            f"    - [{i.query_type}] {i.target}" for i in previous_plan.items
        ]
        gap_dims_str = ", ".join(sorted(gap_dims))
        gap_targets_str = ", ".join(sorted(gap_targets))
        extra = (
            "\n\nRE-PLAN — narrow scope.\n"
            "A prior research round was judged insufficient. The plan ITEMS are "
            "FIXED — do NOT add, remove, or change query_type / target for any item:\n"
            + "\n".join(item_lines) + "\n\n"
            f"Refine questions ONLY for items with target in: {gap_targets_str}\n"
            f"And ONLY for these dimensions: {gap_dims_str}\n\n"
            "Leave question lists EMPTY for any (item, dimension) combination not "
            "in the refine set — they will be preserved from the prior plan.\n\n"
            "Gap details to address:\n- " + "\n- ".join(crit.gaps)
        )
        system = base_system + extra
    else:
        system = base_system

    planner = _llm(model=REASONING_MODEL, temperature=0).with_structured_output(ResearchPlan)
    plan = planner.invoke([
        SystemMessage(content=system),
        HumanMessage(content=state["user_query"]),
    ])

    if is_rerun:
        # Lock items + selectively merge questions per (target, dim). For each
        # previous item, if it's in gap_targets, take the NEW questions for any
        # dim in gap_dims, else keep the previous questions.
        gap_dims = set(crit.gap_dimensions) or set(_ALL_DIMS)
        gap_targets = set(crit.gap_targets) or set(previous_plan.all_targets())
        new_by_target = {i.target: i for i in plan.items}

        merged_items: List[QueryItem] = []
        for prev in previous_plan.items:
            if prev.target not in gap_targets:
                merged_items.append(prev)
                continue
            new = new_by_target.get(prev.target)
            if new is None:
                merged_items.append(prev)
                continue
            updates = {}
            for dim, field in _DIM_TO_QUESTION_FIELD.items():
                if dim in gap_dims and dim in _valid_dims_for(prev):
                    updates[field] = getattr(new, field)
                else:
                    updates[field] = getattr(prev, field)
            merged_items.append(prev.model_copy(update=updates))

        plan = previous_plan.model_copy(update={"items": merged_items})

    return {"research_plan": plan}


# ---- Dispatcher: fan out research work using Send ----

def dispatch_research(state: State):
    """Conditional edge from query_planner. Returns a list of Send objects so
    LangGraph fans out per-item × per-dimension in parallel.

    Gap-aware: when re-running after an insufficient critic, only dispatch to
    the (item, dimension) pairs the critic flagged. Findings from earlier
    rounds remain in state untouched (merge reducer overwrites only the keys
    that this round produces).
    """
    plan: ResearchPlan = state["research_plan"]
    crit: Optional[CriticDecision] = state.get("critic")
    is_rerun = crit is not None and not crit.sufficient

    wanted_dims = set(crit.gap_dimensions) if (is_rerun and crit.gap_dimensions) else None
    wanted_targets = set(crit.gap_targets) if (is_rerun and crit.gap_targets) else None

    sends: List[Send] = []
    iteration = state.get("research_iterations", 0)

    def _send(node: str, payload: dict):
        payload["iteration"] = iteration
        payload["is_rerun"] = is_rerun
        sends.append(Send(node, payload))

    for item in plan.items:
        if wanted_targets is not None and item.target not in wanted_targets:
            continue
        item_valid = _valid_dims_for(item)
        item_dims = (wanted_dims & item_valid) if wanted_dims else item_valid
        if not item_dims:
            # Gap dims didn't intersect this item's valid set — fall back to
            # the item's full valid set so we never emit zero work for it.
            item_dims = item_valid

        common = {
            "target": item.target,
            "query_type": item.query_type,
        }
        if "fundamentals" in item_dims:
            _send("fundamentals_agent", {**common, "questions": item.fundamentals_questions})
        if "sentiment" in item_dims:
            _send("sentiment_agent", {**common, "questions": item.sentiment_questions})
        if "technical" in item_dims:
            _send("technical_agent", {**common, "questions": item.technical_questions})
        if "macro" in item_dims:
            _send("macro_agent", {**common, "questions": item.macro_questions})

    return sends


# ---- Research agents: receive a Task dict (from Send), return dict-keyed slot update ----

def _rerun_note(task: dict) -> str:
    if task.get("is_rerun"):
        return (
            f"\n\nNOTE: This is research ROUND {task.get('iteration', 1) + 1}. A prior "
            "round was judged insufficient for this dimension — go DEEPER than usual: "
            "more tool calls, more query variations, more disconfirming evidence.\n"
        )
    return ""


def fundamentals_node(task: dict) -> State:
    ticker = task["target"]
    questions = task.get("questions") or [f"General fundamental health of {ticker}"]
    role = (
        f"You are a senior fundamentals analyst covering {ticker}. Your job is to "
        "produce a quantitatively specific assessment of revenue trajectory, "
        "profitability, valuation, balance-sheet health, and forward guidance."
    )
    prompt = (
        f"Research the fundamentals of {ticker}.\n\n"
        f"Workflow:\n"
        f"1. Call get_stock_fundamentals({ticker}) for the latest snapshot — "
        f"   capture P/E, EV/EBITDA, margins, ROE, debt/equity, FCF, beta.\n"
        f"2. Call get_earnings_history({ticker}) for the quarterly EPS surprise track.\n"
        f"3. Call tavily_web_search 2-3 times with DIFFERENT query phrasings — for "
        f"   example: latest 10-K/10-Q commentary, segment trends, management changes, "
        f"   competitive position, recent guidance updates. Verify the numbers against "
        f"   the qualitative narrative.\n\n"
        f"Address each question:\n- " + "\n- ".join(questions)
        + f"\n\nReturn a finding for ticker '{ticker}' that includes specific numbers "
          "(revenue growth %, margins %, multiples, $ values), the most recent guidance, "
          "and source URLs.\n"
        + _rerun_note(task)
    )

    def _fallback(err: str) -> FundamentalsFindings:
        return FundamentalsFindings(
            ticker=ticker,
            revenue_trend="unknown",
            profitability="unknown",
            valuation="unknown",
            balance_sheet="unknown",
            guidance="unknown",
            summary=f"Fundamentals research failed: {err}",
            confidence=0.0,
            unanswered_questions=list(questions),
        )

    finding = _run_subagent(
        role_system=role,
        task_prompt=prompt,
        tools=FUNDAMENTAL_TOOLS,
        schema=FundamentalsFindings,
        questions=questions,
        fallback_factory=_fallback,
    )
    return {"fundamentals": {ticker: finding}}


def sentiment_node(task: dict) -> State:
    target = task["target"]
    query_type = task["query_type"]
    questions = task.get("questions") or []

    if query_type == "stock_analysis":
        role = (
            f"You are a news & sentiment analyst covering {target}. Cover headlines, "
            "analyst consensus, social sentiment, and concrete near-term catalysts."
        )
        prompt = (
            f"Research news & sentiment for {target}.\n\n"
            f"Workflow:\n"
            f"1. Call tavily_news_search('{target} stock', days=14) for the most recent "
            f"   news cycle. Also try 'days=7' for very recent events.\n"
            f"2. Run a second tavily_news_search with a different angle (earnings, "
            f"   analyst price target, downgrade/upgrade, regulatory).\n"
            f"3. Use tavily_web_search to find analyst consensus and any aggregated "
            f"   price target (e.g. CNN Money, Yahoo Finance analyst page, Zacks).\n"
            f"4. Probe for DISCONFIRMING news (lawsuits, downgrades, customer losses) "
            f"   so the assessment isn't one-sided.\n\n"
            f"Address each question:\n- " + "\n- ".join(questions)
            + f"\n\nReturn a finding with subject='{target}': headline summary, analyst "
              "consensus, average price target (if available), social sentiment label, "
              "specific named catalysts with dates if possible, and source URLs.\n"
            + _rerun_note(task)
        )
    else:  # industry_analysis
        role = (
            f"You are a news & sentiment analyst covering the {target} industry. "
            "Surface specific company names, dates, and concrete catalysts shaping the sector."
        )
        prompt = (
            f"Research news & sentiment for the '{target}' industry.\n\n"
            f"Workflow:\n"
            f"1. tavily_news_search('{target} industry', days=14) — capture recent "
            f"   developments across the sector.\n"
            f"2. tavily_news_search with a second angle (regulation, earnings cycle, "
            f"   M&A, leading company catalysts).\n"
            f"3. tavily_web_search for analyst / industry write-ups; pull names of "
            f"   specific companies, tickers, and dated events that the research surfaces.\n\n"
            f"Address each question:\n- " + "\n- ".join(questions)
            + f"\n\nReturn a finding with subject='{target}'. Populate "
              "notable_catalysts with SPECIFIC named events / companies / dates from "
              "the research, not generic phrases. Sources required.\n"
            + _rerun_note(task)
        )

    def _fallback(err: str) -> SentimentFindings:
        return SentimentFindings(
            subject=target,
            headline_summary=f"Sentiment research failed: {err}",
            analyst_consensus="unknown",
            social_sentiment="neutral",
            summary=f"Sentiment research failed: {err}",
            confidence=0.0,
            unanswered_questions=list(questions),
        )

    finding = _run_subagent(
        role_system=role,
        task_prompt=prompt,
        tools=SENTIMENT_TOOLS,
        schema=SentimentFindings,
        questions=questions,
        fallback_factory=_fallback,
    )
    return {"sentiment": {target: finding}}


def technical_node(task: dict) -> State:
    ticker = task["target"]
    questions = task.get("questions") or [f"Trend, momentum, and key levels for {ticker}"]
    role = (
        f"You are a technical analyst covering {ticker}. Be precise with numbers — "
        "RSI, SMA values, price vs MA, support / resistance levels in $."
    )
    prompt = (
        f"Run a technical read on {ticker}.\n\n"
        f"Workflow:\n"
        f"1. Call get_price_history({ticker}, period='6mo') — capture last close, "
        f"   SMA20/50/200, RSI14, period high/low, volume averages.\n"
        f"2. Call get_price_history({ticker}, period='1y') for a longer-term trend read.\n"
        f"3. Call get_stock_quote({ticker}) to confirm current price vs the historical close.\n"
        f"4. From the numbers, identify support and resistance levels (recent swing lows/highs, "
        f"   round numbers near the price, the 50/200 SMA).\n\n"
        f"Address each question:\n- " + "\n- ".join(questions)
        + f"\n\nReturn a finding for ticker '{ticker}' with: current_price (numeric), "
          "trend label, moving_averages commentary citing specific SMA values, RSI14 "
          "(numeric), volume_profile vs the 30d average, explicit support/resistance "
          "levels in $, and a momentum_signal label.\n"
        + _rerun_note(task)
    )

    def _fallback(err: str) -> TechnicalFindings:
        return TechnicalFindings(
            ticker=ticker,
            trend="sideways",
            moving_averages="unknown",
            volume_profile="unknown",
            support_resistance="unknown",
            momentum_signal="neutral",
            summary=f"Technical research failed: {err}",
            confidence=0.0,
            unanswered_questions=list(questions),
        )

    finding = _run_subagent(
        role_system=role,
        task_prompt=prompt,
        tools=TECHNICAL_TOOLS,
        schema=TechnicalFindings,
        questions=questions,
        fallback_factory=_fallback,
    )
    return {"technical": {ticker: finding}}


def macro_node(task: dict) -> State:
    target = task["target"]
    query_type = task["query_type"]
    questions = task.get("questions") or []

    if query_type == "stock_analysis":
        role = (
            f"You are a macro & sector analyst covering {target}. Place the company "
            "within its sector and surface the cyclical / structural drivers that "
            "matter for forward returns."
        )
        prompt = (
            f"Research the macro & sector context for {target}.\n\n"
            f"Workflow:\n"
            f"1. Call get_peer_comparison({target}) — capture named peers and any "
            f"   relative-value commentary.\n"
            f"2. tavily_web_search for the sector outlook (e.g. '{target} sector "
            f"   outlook 2025', industry growth drivers, regulatory headwinds).\n"
            f"3. tavily_web_search for macro drivers relevant to this sector (rates, "
            f"   FX, commodity, regulation, demand cycle).\n\n"
            f"Address each question:\n- " + "\n- ".join(questions)
            + f"\n\nReturn a finding with subject='{target}': sector, industry_trends "
              "with specific named drivers, competitor_comparison naming 2+ peers, "
              "macro_drivers, source URLs.\n"
            + _rerun_note(task)
        )
    else:  # industry_analysis
        role = (
            f"You are a macro & sector analyst covering the {target} industry. "
            "Identify the structural drivers, leading companies, and macro context."
        )
        prompt = (
            f"Research the macro & sector context for the '{target}' industry.\n\n"
            f"Workflow:\n"
            f"1. tavily_web_search('{target} sector outlook') — capture the most "
            f"   relevant structural drivers and growth/headwind narrative.\n"
            f"2. tavily_web_search for the competitive landscape — name specific "
            f"   companies / tickers that lead this industry.\n"
            f"3. tavily_web_search for the macro context (rates, regulation, "
            f"   supply / demand, geopolitical).\n\n"
            f"Address each question:\n- " + "\n- ".join(questions)
            + f"\n\nReturn a finding with subject='{target}': sector, industry trends "
              "with named drivers, competitor_comparison naming 2+ leading companies, "
              "macro drivers, source URLs.\n"
            + _rerun_note(task)
        )

    def _fallback(err: str) -> MacroFindings:
        return MacroFindings(
            subject=target,
            sector="unknown",
            industry_trends="unknown",
            competitor_comparison="unknown",
            macro_drivers="unknown",
            summary=f"Macro research failed: {err}",
            confidence=0.0,
            unanswered_questions=list(questions),
        )

    finding = _run_subagent(
        role_system=role,
        task_prompt=prompt,
        tools=MACRO_TOOLS,
        schema=MacroFindings,
        questions=questions,
        fallback_factory=_fallback,
    )
    return {"macro": {target: finding}}


def aggregator_node(state: State) -> State:
    """Synchronisation point after parallel research. Bumps the loop counter."""
    return {"research_iterations": state.get("research_iterations", 0) + 1}


# ---- Helpers to assemble multi-item payloads ----

def _findings_block(label: str, findings: Dict[str, Any]) -> str:
    if not findings:
        return f"{label}: (none)\n"
    parts = [f"=== {label} ==="]
    for key, f in findings.items():
        parts.append(f"[{key}]\n{f.model_dump_json(indent=2)}")
    return "\n".join(parts) + "\n"


def _all_findings_payload(state: State) -> str:
    return (
        _findings_block("FUNDAMENTALS", state.get("fundamentals") or {})
        + _findings_block("SENTIMENT",   state.get("sentiment")    or {})
        + _findings_block("TECHNICAL",   state.get("technical")    or {})
        + _findings_block("MACRO",       state.get("macro")        or {})
    )


def _plan_summary(plan: ResearchPlan) -> str:
    """Compact textual summary of the plan items for LLM context."""
    lines = ["PLAN ITEMS:"]
    for item in plan.items:
        lines.append(f"  - [{item.query_type}] {item.target}")
    return "\n".join(lines) + "\n"


def critic_node(state: State) -> State:
    """Investment-committee chair: is the evidence decision-ready?

    Concrete tests the critic applies:
      - Every plan question answered (no entries in `unanswered_questions`).
      - Confidence per applicable (target, dimension) is >= MIN_DIMENSION_CONFIDENCE.
      - Sources are present for sentiment / fundamentals / macro findings.
      - No internal contradictions between dimensions.

    When sufficient=False, populate `gap_dimensions` and `gap_targets` precisely
    so the dispatcher only re-runs what's actually missing.
    """
    plan: ResearchPlan = state["research_plan"]
    system = (
        "You are a skeptical investment-committee chair. Decide if the assembled "
        "research is decision-ready.\n\n"
        "Recall: stock_analysis items should have findings across all 4 dimensions; "
        "industry_analysis items should have findings only for sentiment + macro.\n\n"
        "Mark sufficient=false if ANY of these are true:\n"
        f"  - Any applicable (target, dimension) finding has confidence < {MIN_DIMENSION_CONFIDENCE}.\n"
        "  - Any finding has non-empty `unanswered_questions`.\n"
        "  - Fundamentals / sentiment / macro findings are missing source URLs.\n"
        "  - Findings contradict each other (e.g. sentiment bullish but technical screams "
        "    breakdown without explanation).\n"
        "  - Numbers are absent where they were required (vague qualifiers only).\n\n"
        "When sufficient=false:\n"
        "  - `gaps` must be specific: cite target + dimension + what's missing.\n"
        "  - `gap_dimensions` must list ONLY the dimensions that need rework "
        "    (any subset of: fundamentals, sentiment, technical, macro).\n"
        "  - `gap_targets` must list ONLY the targets that need rework (empty = all).\n"
        "  - Be surgical — listing every dimension wastes a research round.\n"
    )
    payload = (
        f"USER QUERY: {state['user_query']}\n"
        f"RESEARCH ROUND: {state.get('research_iterations', 0)}\n\n"
        + _plan_summary(plan)
        + "\n"
        + _all_findings_payload(state)
    )
    judge = _llm(model=REASONING_MODEL, temperature=0).with_structured_output(CriticDecision)
    return {"critic": judge.invoke([SystemMessage(content=system), HumanMessage(content=payload)])}


def synthesis_node(state: State) -> State:
    """Portfolio-manager voice: integrate the research into per-item outlooks
    (each carrying its own future outlook, risk view, and recommendation)."""
    plan: ResearchPlan = state["research_plan"]
    has_stocks = plan.has_stock_items()
    has_industries = plan.has_industry_items()

    type_guidance_parts: List[str] = []
    if has_stocks:
        type_guidance_parts.append(
            "STOCK_OUTLOOKS — produce ONE entry per stock_analysis target. For each stock:\n"
            "  - future_outlook: where the stock is headed and why, grounded in the findings.\n"
            "  - bull_case: 2-4 specific bullish points (numbers, catalysts, analyst datapoints).\n"
            "  - bear_case: 2-4 specific bearish points (disconfirming evidence from findings).\n"
            "  - risk_rating: low / moderate / high / very_high.\n"
            "  - risk_assessment: volatility (RSI, % off highs, beta), drawdown exposure, "
            "    leverage (debt/equity, totalDebt), business concentration, idiosyncratic risks.\n"
            "  - recommendation: Strong Buy / Buy / Hold / Sell / Strong Sell — must follow the "
            "    weight of bull vs bear evidence, not be retrofitted to it.\n"
            "  - conviction + rationale (1-2 sentences)."
        )
    if has_industries:
        type_guidance_parts.append(
            "INDUSTRY_OUTLOOKS — produce ONE entry per industry_analysis target. For each industry:\n"
            "  - future_outlook: the sector trajectory, drivers, and headwinds — specific named "
            "    catalysts and players where available.\n"
            "  - risk_rating: low / moderate / high / very_high.\n"
            "  - risk_assessment: macro drivers (rates, FX, demand cycle), regulatory exposure, "
            "    cyclicality, competitive intensity.\n"
            "  - recommendation: Overweight / Neutral / Underweight — sector positioning call.\n"
            "  - conviction + rationale (1-2 sentences)."
        )
    if has_stocks and has_industries:
        type_guidance_parts.append(
            "MIXED plan — thesis_summary must connect the dots: explain how the industry "
            "backdrop shapes each stock recommendation."
        )
    if not has_stocks:
        type_guidance_parts.append(
            "INDUSTRY-ONLY plan — leave stock_outlooks EMPTY. thesis_summary should directly "
            "answer the user's question using the industry findings."
        )
    if not has_industries:
        type_guidance_parts.append(
            "STOCK-ONLY plan — leave industry_outlooks EMPTY."
        )

    system = (
        "You are a portfolio manager writing an investment thesis. Integrate the "
        "research into a structured InvestmentThesis with per-item outlooks.\n\n"
        "Discipline:\n"
        "  - Every bull / bear case point and every risk claim must be backed by a "
        "    SPECIFIC fact from the findings (a number, a named catalyst, an analyst "
        "    datapoint). No generic 'strong fundamentals' filler.\n"
        "  - Recommendations must follow the evidence, not the other way around. If "
        "    the bear case outweighs the bull case, the recommendation reflects that.\n"
        "  - Acknowledge contradictions explicitly — don't paper over them.\n"
        "  - overall_conviction reflects the FULL set of items, not the single loudest one.\n\n"
        + "\n\n".join(type_guidance_parts)
    )

    revision_note = ""
    if state.get("human_review") and state["human_review"].decision == "revise":
        revision_note = (
            f"\n\nREVISION FEEDBACK FROM INVESTOR (address explicitly):\n"
            f"{state['human_review'].feedback}"
        )

    holdings_str = ""
    if plan.holdings:
        holdings_str = "\nHOLDINGS (quantities):\n" + "\n".join(
            f"  {h.ticker}: {h.quantity}" for h in plan.holdings
        )

    payload = (
        f"USER QUERY: {state['user_query']}\n"
        + _plan_summary(plan)
        + holdings_str + "\n\n"
        + _all_findings_payload(state)
        + revision_note
    )
    pm = _llm(model=REASONING_MODEL, temperature=0.2).with_structured_output(InvestmentThesis)
    return {"thesis": pm.invoke([SystemMessage(content=system), HumanMessage(content=payload)])}


def human_review_node(state: State) -> State:
    """Pause the graph for investor approval (LangGraph interrupt)."""
    thesis: InvestmentThesis = state["thesis"]
    plan: ResearchPlan = state["research_plan"]
    is_informational = not plan.has_stock_items()
    payload = {
        "items": [
            {"query_type": i.query_type, "target": i.target} for i in plan.items
        ],
        "stock_targets": plan.stock_targets(),
        "industry_targets": plan.industry_targets(),
        "overall_conviction": thesis.overall_conviction,
        "stock_outlooks": [o.model_dump() for o in thesis.stock_outlooks],
        "industry_outlooks": [o.model_dump() for o in thesis.industry_outlooks],
        "thesis_summary": thesis.thesis_summary,
        "is_informational": is_informational,
        "instruction": (
            "Resume with {'decision': 'approve'} to finalise, or "
            "{'decision': 'revise', 'feedback': '<what to change>'} to iterate."
        ),
    }
    response = interrupt(payload)
    review = HumanReview.model_validate(response)
    return {
        "human_review": review,
        "revision_iterations": state.get("revision_iterations", 0)
        + (1 if review.decision == "revise" else 0),
    }


def final_report_node(state: State) -> State:
    plan: ResearchPlan = state["research_plan"]
    has_stocks = plan.has_stock_items()
    has_industries = plan.has_industry_items()

    sections: List[str] = ["  1. Headline", "  2. Executive summary"]
    next_no = 3
    if has_industries:
        sections.append(
            f"  {next_no}. Industry breakdown — ONE subsection per industry, each containing: "
            "future outlook, risk assessment, recommendation (Overweight/Neutral/Underweight). "
            "Cite specific drivers, named companies, and sources."
        )
        next_no += 1
    if has_stocks:
        sections.append(
            f"  {next_no}. Per-stock breakdown — ONE subsection per ticker, each containing: "
            "future outlook, bull case, bear case, risk assessment, recommendation "
            "(Strong Buy/Buy/Hold/Sell/Strong Sell). Cite specific numbers and sources."
        )
        next_no += 1
    sections.append(
        f"  {next_no}. Evidence by dimension (Fundamentals, Sentiment, Technical, Macro)"
    )

    if has_stocks and has_industries:
        mix_note = (
            "MIXED plan — connect the two: explain how each industry's outlook shapes the "
            "stock recommendations that sit inside it.\n\n"
        )
    elif has_industries and not has_stocks:
        mix_note = (
            "INDUSTRY-ONLY plan — the report is informational, not a per-stock recommendation. "
            "Set primary_recommendation to a brief sector summary (e.g. 'Overweight Technology, "
            "Neutral Healthcare').\n\n"
        )
    else:
        mix_note = ""

    system = (
        "You are a senior equity-research editor. Produce a polished, investor-ready "
        "markdown report from the InvestmentThesis and research findings.\n\n"
        f"{mix_note}"
        "Sections (in order):\n"
        + "\n".join(sections) + "\n\n"
        "Discipline: cite specific numbers and named sources from the research. "
        "No filler. Use markdown tables where they aid comparison.\n\n"
        "Populate tickers_covered with the stock_outlook tickers and "
        "industries_covered with the industry_outlook labels. primary_recommendation "
        "should be a concise headline phrase summarising the calls."
    )

    payload = (
        f"USER QUERY: {state['user_query']}\n"
        + _plan_summary(plan)
        + f"\nSTOCK TARGETS:    {plan.stock_targets()}\n"
        + f"INDUSTRY TARGETS: {plan.industry_targets()}\n\n"
        f"THESIS:\n{state['thesis'].model_dump_json(indent=2)}\n\n"
        + _all_findings_payload(state)
    )
    editor = _llm(model=REASONING_MODEL, temperature=0.2).with_structured_output(FinalReport)
    report = editor.invoke([SystemMessage(content=system), HumanMessage(content=payload)])
    return {
        "final_report": report,
        "messages": [SystemMessage(content=report.full_markdown)],
    }


# ---------- Conditional routers ----------

def critic_router(state: State) -> str:
    if state["critic"].sufficient:
        return "sufficient"
    if state.get("research_iterations", 0) >= MAX_RESEARCH_ITERATIONS:
        return "sufficient"  # bail out rather than spin forever
    return "gather_more"


def human_router(state: State) -> str:
    review = state.get("human_review")
    if review is None or review.decision == "approve":
        return "approve"
    if state.get("revision_iterations", 0) >= MAX_REVISION_ITERATIONS:
        return "approve"
    return "revise"


# ---------- Build the graph ----------

RESEARCH_NODES = ["fundamentals_agent", "sentiment_agent", "technical_agent", "macro_agent"]


def build_graph():
    graph = StateGraph(State)

    graph.add_node("user_query", user_query_node)
    graph.add_node("query_planner", query_planner_node)
    graph.add_node("fundamentals_agent", fundamentals_node)
    graph.add_node("sentiment_agent", sentiment_node)
    graph.add_node("technical_agent", technical_node)
    graph.add_node("macro_agent", macro_node)
    graph.add_node("aggregator", aggregator_node)
    graph.add_node("critic", critic_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("final_report", final_report_node)

    graph.add_edge(START, "user_query")
    graph.add_edge("user_query", "query_planner")

    # Dynamic fan-out via Send (per-item × per-dimension).
    graph.add_conditional_edges("query_planner", dispatch_research, RESEARCH_NODES)
    for node in RESEARCH_NODES:
        graph.add_edge(node, "aggregator")

    graph.add_edge("aggregator", "critic")
    graph.add_conditional_edges(
        "critic",
        critic_router,
        {"sufficient": "synthesis", "gather_more": "query_planner"},
    )

    graph.add_edge("synthesis", "human_review")
    graph.add_conditional_edges(
        "human_review",
        human_router,
        {"approve": "final_report", "revise": "synthesis"},
    )
    graph.add_edge("final_report", END)

    # Register all Pydantic schemas explicitly with the checkpoint serializer
    # to silence "unregistered type" warnings and stay forward-compatible.
    serde = JsonPlusSerializer(allowed_msgpack_modules=[
        ResearchPlan,
        QueryItem,
        Holding,
        FundamentalsFindings,
        KeyMetric,
        SentimentFindings,
        TechnicalFindings,
        MacroFindings,
        CriticDecision,
        InvestmentThesis,
        StockOutlook,
        IndustryOutlook,
        HumanReview,
        FinalReport,
    ])
    return graph.compile(checkpointer=MemorySaver(serde=serde))


app = build_graph()


if __name__ == "__main__":
    # The interactive CLI lives in cli.py — delegate to it so `python agent.py`
    # still works for users who used to launch it that way.
    from cli import main as _cli_main
    _cli_main()
