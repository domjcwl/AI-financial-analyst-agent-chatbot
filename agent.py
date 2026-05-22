r"""Deep-research stock-picking agent built on LangGraph.

Graph (variant of stock_research_langgraph.svg with dynamic dispatch):

  START
    -> user_query
    -> query_planner          (produces ResearchPlan: tickers + query_type)
       |
       +-- dispatch_research (LangGraph Send):
       |
       |   if query_type in {single_ticker, portfolio}:
       |       for each ticker:
       |           Send -> fundamentals_agent (ticker)
       |           Send -> sentiment_agent    (ticker)
       |           Send -> technical_agent    (ticker)
       |           Send -> macro_agent        (ticker)
       |   else (general — no specific stock):
       |       Send -> sentiment_agent  (topic, no ticker)
       |       Send -> macro_agent      (topic, no ticker)
       |       (fundamentals + technical are SKIPPED)
       v
    -> aggregator
    -> critic ---(gather_more)--> query_planner   # evidence loop
              \--(sufficient)--> risk
                                 -> synthesis
                                    -> human_review ---(revise)--> synthesis
                                                  \--(approve)--> final_report
                                                                  -> END

Per-dimension state slots are Dict[ticker, Finding] (or {"MARKET": Finding} for
general queries) with a merge reducer so parallel Send invocations don't clobber
one another. Each slot is a Pydantic model defined in schemas.py.
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
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, Send, interrupt
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
import httpx
import os

from schemas import (
    MARKET_KEY,
    CriticDecision,
    FinalReport,
    FundamentalsFindings,
    Holding,
    HumanReview,
    InvestmentThesis,
    KeyMetric,
    MacroFindings,
    ResearchPlan,
    RiskAssessment,
    SentimentFindings,
    TechnicalFindings,
    TickerRecommendation,
    TickerRisk,
)
from tools import (
    FUNDAMENTAL_TOOLS,
    MACRO_TOOLS,
    SENTIMENT_TOOLS,
    TECHNICAL_TOOLS,
)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

MAX_RESEARCH_ITERATIONS = 2   # caps the critic -> planner loop
MAX_REVISION_ITERATIONS = 2   # caps the human -> synthesis loop

# Shared httpx client to bypass corporate proxy/firewall (matches sample pattern)
_http = httpx.Client(verify=False)


def _llm(model: str = "gpt-4o-mini", temperature: float = 0.1) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        http_client=_http,
        api_key=api_key,
    )


# ---------- State ----------

def _merge_findings(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Reducer for per-dimension finding dicts. Later writes overwrite earlier
    entries for the same ticker; new tickers are added."""
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
    risk: Optional[RiskAssessment]
    thesis: Optional[InvestmentThesis]
    human_review: Optional[HumanReview]
    revision_iterations: int
    final_report: Optional[FinalReport]


# ---------- Sub-agent helper ----------

def _run_subagent(prompt: str, tools: list, schema, model: str = "gpt-4o-mini"):
    """Spin up a ReAct sub-agent with a curated tool set, then extract a
    schema-validated finding from its final answer."""
    agent = create_agent(_llm(model=model), tools=tools)
    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    raw_text = result["messages"][-1].content
    extractor = _llm(model=model, temperature=0).with_structured_output(schema)
    return extractor.invoke([
        SystemMessage(content="Extract a structured finding from the research notes."),
        HumanMessage(content=raw_text),
    ])


# ---------- Nodes ----------

def user_query_node(state: State) -> State:
    """Entry node: surface the user's question into the message stream."""
    return {"messages": [HumanMessage(content=state["user_query"])]}


def query_planner_node(state: State) -> State:
    """Decompose the user query into a structured ResearchPlan.

    Critical instruction to the LLM: only populate `tickers` when a specific
    stock, ETF, or company is named. Open-ended market queries get query_type
    'general' with an empty tickers list — the dispatcher will skip the
    fundamentals & technical agents in that case.
    """
    system = (
        "You are a senior buy-side research lead. Decompose the user's query into a "
        "structured ResearchPlan.\n\n"
        "Classify the query as one of:\n"
        "  - single_ticker : exactly one stock / ETF / company is named\n"
        "  - portfolio     : multiple specific tickers are named (with or without quantities)\n"
        "  - general       : NO specific company is named (e.g. 'what stocks will IPO this "
        "year', 'how is the AI sector doing?')\n\n"
        "Rules:\n"
        "1. tickers: list every named symbol exactly. Resolve company names to tickers "
        "   (e.g. 'Google' -> 'GOOGL', 'Lockheed' -> 'LMT'). For general queries, leave EMPTY.\n"
        "2. holdings: populate with quantity ONLY if the user states positions "
        "   (e.g. '3 VOO, 4 GOOGL'). Otherwise empty.\n"
        "3. topic: REQUIRED for general queries — a short phrase describing the subject "
        "   (e.g. 'upcoming IPOs in 2025', 'AI sector outlook'). Null otherwise.\n"
        "4. Questions: 3-5 sharp, decision-relevant questions per applicable dimension.\n"
        "   For general queries: leave fundamentals_questions and technical_questions EMPTY "
        "   (those agents will be skipped), but DO populate sentiment_questions and "
        "   macro_questions about the broader topic.\n"
    )
    extra = ""
    if state.get("critic") and not state["critic"].sufficient:
        extra = (
            "\n\nIMPORTANT: a prior research round was judged insufficient. "
            "Re-plan so the new round closes these specific gaps:\n- "
            + "\n- ".join(state["critic"].gaps)
        )

    planner = _llm(temperature=0).with_structured_output(ResearchPlan)
    plan = planner.invoke([
        SystemMessage(content=system + extra),
        HumanMessage(content=state["user_query"]),
    ])
    return {"research_plan": plan}


# ---- Dispatcher: fan out research work using Send ----

def dispatch_research(state: State):
    """Conditional edge from query_planner. Returns a list of Send objects so
    LangGraph fans out per-ticker × per-dimension in parallel."""
    plan: ResearchPlan = state["research_plan"]
    sends: List[Send] = []

    if plan.tickers:
        # Per-ticker fan-out across all 4 dimensions
        for ticker in plan.tickers:
            sends.append(Send("fundamentals_agent", {
                "ticker": ticker,
                "questions": plan.fundamentals_questions,
            }))
            sends.append(Send("sentiment_agent", {
                "ticker": ticker,
                "questions": plan.sentiment_questions,
            }))
            sends.append(Send("technical_agent", {
                "ticker": ticker,
                "questions": plan.technical_questions,
            }))
            sends.append(Send("macro_agent", {
                "ticker": ticker,
                "questions": plan.macro_questions,
            }))
    else:
        # General / topic query — fundamentals & technical SKIPPED.
        topic = plan.topic or state["user_query"]
        sends.append(Send("sentiment_agent", {
            "ticker": None,
            "topic": topic,
            "questions": plan.sentiment_questions,
        }))
        sends.append(Send("macro_agent", {
            "ticker": None,
            "topic": topic,
            "questions": plan.macro_questions,
        }))

    return sends


# ---- Research agents: receive a Task dict (from Send), return dict-keyed slot update ----

def fundamentals_node(task: dict) -> State:
    ticker = task["ticker"]
    questions = task.get("questions") or [f"General fundamental health of {ticker}"]
    prompt = (
        f"You are a fundamentals analyst covering {ticker}.\n"
        f"Call get_stock_fundamentals and get_earnings_history first, then "
        f"tavily_web_search for qualitative context (10-K/10-Q commentary, "
        f"management changes, segment trends). Address each question:\n- "
        + "\n- ".join(questions)
        + f"\n\nReturn a finding for ticker '{ticker}': revenue trend, profitability, "
          "valuation, balance sheet, guidance, key_metrics, summary, confidence (0-1), "
          "and source URLs."
    )
    finding = _run_subagent(prompt, FUNDAMENTAL_TOOLS, FundamentalsFindings)
    return {"fundamentals": {ticker: finding}}


def sentiment_node(task: dict) -> State:
    ticker = task.get("ticker")
    questions = task.get("questions") or []
    if ticker:
        subject = ticker
        prompt = (
            f"You are a news & sentiment analyst covering {ticker}.\n"
            f"Use tavily_news_search and tavily_web_search to address:\n- "
            + "\n- ".join(questions)
            + f"\n\nReturn a finding for ticker '{ticker}': headline summary, analyst "
              "consensus, average price target if available, social sentiment label, "
              "notable catalysts, summary, confidence (0-1), source URLs."
        )
        key = ticker
    else:
        topic = task.get("topic") or "broad market"
        subject = topic
        prompt = (
            f"You are a news & sentiment analyst. The user is asking about: '{topic}'.\n"
            f"No single ticker is in scope — research the topic broadly using "
            f"tavily_news_search and tavily_web_search. Address:\n- "
            + "\n- ".join(questions)
            + f"\n\nReturn a finding using ticker='{MARKET_KEY}': headline summary, "
              "analyst consensus on the theme, average price target (null is fine), "
              "social sentiment, notable catalysts (specific names if discovered), "
              "summary, confidence (0-1), source URLs."
        )
        key = MARKET_KEY
    finding = _run_subagent(prompt, SENTIMENT_TOOLS, SentimentFindings)
    return {"sentiment": {key: finding}}


def technical_node(task: dict) -> State:
    ticker = task["ticker"]
    questions = task.get("questions") or [f"Trend, momentum, and key levels for {ticker}"]
    prompt = (
        f"You are a technical analyst covering {ticker}.\n"
        f"Call get_price_history (period='6mo') and get_stock_quote. Address:\n- "
        + "\n- ".join(questions)
        + f"\n\nReturn a finding for ticker '{ticker}': current price, trend, moving "
          "averages commentary, RSI, volume profile, support/resistance, momentum "
          "signal, summary, confidence (0-1)."
    )
    finding = _run_subagent(prompt, TECHNICAL_TOOLS, TechnicalFindings)
    return {"technical": {ticker: finding}}


def macro_node(task: dict) -> State:
    ticker = task.get("ticker")
    questions = task.get("questions") or []
    if ticker:
        prompt = (
            f"You are a macro & sector analyst covering {ticker}.\n"
            f"Use get_peer_comparison and tavily_web_search to address:\n- "
            + "\n- ".join(questions)
            + f"\n\nReturn a finding for ticker '{ticker}': sector, industry trends, "
              "competitor comparison, macro drivers, summary, confidence (0-1), source URLs."
        )
        key = ticker
    else:
        topic = task.get("topic") or "broad market"
        prompt = (
            f"You are a macro & sector analyst. The user is asking about: '{topic}'.\n"
            f"Research the topic broadly using tavily_web_search. Address:\n- "
            + "\n- ".join(questions)
            + f"\n\nReturn a finding using ticker='{MARKET_KEY}': the most relevant "
              "sector, industry trends, competitor / peer landscape, macro drivers, "
              "summary, confidence (0-1), source URLs."
        )
        key = MARKET_KEY
    finding = _run_subagent(prompt, MACRO_TOOLS, MacroFindings)
    return {"macro": {key: finding}}


def aggregator_node(state: State) -> State:
    """Synchronisation point after parallel research. Bumps the loop counter."""
    return {"research_iterations": state.get("research_iterations", 0) + 1}


# ---- Helpers to assemble multi-ticker payloads ----

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


def critic_node(state: State) -> State:
    """Investment-committee chair: is the evidence decision-ready?"""
    plan: ResearchPlan = state["research_plan"]
    system = (
        "You are a skeptical investment-committee chair. Evaluate the research findings "
        "(possibly across multiple tickers) for completeness, internal consistency, and "
        "decision-readiness. Mark sufficient=false if any dimension for any ticker has "
        "confidence < 0.6, unanswered plan questions, missing data, or contradictions. "
        "List specific gaps when not sufficient (cite ticker + dimension)."
    )
    payload = (
        f"USER QUERY: {state['user_query']}\n"
        f"QUERY TYPE: {plan.query_type}\n"
        f"TICKERS: {plan.tickers or '(none — general query)'}\n\n"
        + _all_findings_payload(state)
    )
    judge = _llm(temperature=0).with_structured_output(CriticDecision)
    return {"critic": judge.invoke([SystemMessage(content=system), HumanMessage(content=payload)])}


def risk_node(state: State) -> State:
    plan: ResearchPlan = state["research_plan"]
    system = (
        "You are a risk officer. Build a risk assessment from the assembled research. "
        "Cover volatility, drawdown, financial leverage, business concentration, and "
        "macro risk. For multi-ticker analyses, also populate per_ticker_risks with one "
        "entry per ticker. Give an overall risk rating that reflects the full set."
    )
    payload = (
        f"QUERY TYPE: {plan.query_type}\nTICKERS: {plan.tickers}\n\n"
        + _all_findings_payload(state)
    )
    risk_llm = _llm(temperature=0.1).with_structured_output(RiskAssessment)
    return {"risk": risk_llm.invoke([SystemMessage(content=system), HumanMessage(content=payload)])}


def synthesis_node(state: State) -> State:
    """Portfolio-manager voice: integrate research + risk into a thesis."""
    plan: ResearchPlan = state["research_plan"]
    is_general = plan.query_type == "general"
    system = (
        "You are a portfolio manager writing an investment thesis. Integrate the research "
        "and risk assessment into a structured InvestmentThesis.\n\n"
        "Rules:\n"
        "- For single_ticker queries: produce a clear primary_recommendation and one "
        "  per_ticker_recommendations entry. Populate bull_case, bear_case, key_risks, "
        "  and catalysts.\n"
        "- For portfolio queries: produce per_ticker_recommendations for EVERY ticker. "
        "  primary_recommendation summarises the portfolio (use 'Mixed' if recs diverge). "
        "  Populate bull_case, bear_case, key_risks, and catalysts.\n"
        "- For general (ticker-less) queries: this is an informational question, NOT an "
        "  investment recommendation. Set primary_recommendation='N/A', leave "
        "  per_ticker_recommendations empty (unless specific tickers were surfaced worth "
        "  flagging), and leave bull_case AND bear_case as EMPTY lists. Use thesis_summary "
        "  to directly answer the user's question using the research findings. Populate "
        "  key_risks and catalysts only if genuinely relevant to the topic; otherwise "
        "  leave them empty.\n"
        "- Be specific, avoid hedge-everywhere prose."
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
        f"QUERY TYPE: {plan.query_type}\nTICKERS: {plan.tickers}{holdings_str}\n\n"
        + _all_findings_payload(state)
        + f"\nRISK:\n{state['risk'].model_dump_json(indent=2)}{revision_note}"
    )
    pm = _llm(model="gpt-4o", temperature=0.2).with_structured_output(InvestmentThesis)
    return {"thesis": pm.invoke([SystemMessage(content=system), HumanMessage(content=payload)])}


def human_review_node(state: State) -> State:
    """Pause the graph for investor approval (LangGraph interrupt)."""
    thesis: InvestmentThesis = state["thesis"]
    risk: RiskAssessment = state["risk"]
    plan: ResearchPlan = state["research_plan"]
    payload = {
        "query_type": plan.query_type,
        "tickers": plan.tickers,
        "primary_recommendation": thesis.primary_recommendation,
        "overall_conviction": thesis.overall_conviction,
        "per_ticker_recommendations": [
            r.model_dump() for r in thesis.per_ticker_recommendations
        ],
        "thesis_summary": thesis.thesis_summary,
        "bull_case": thesis.bull_case,
        "bear_case": thesis.bear_case,
        "key_risks": thesis.key_risks,
        "overall_risk_rating": risk.overall_risk_rating,
        "is_general": plan.query_type == "general",
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
    if plan.query_type == "general":
        system = (
            "You are a senior research editor. The user asked an informational question, "
            "NOT for an investment recommendation. Produce a polished markdown report that "
            "DIRECTLY answers the question using the research findings.\n\n"
            "Do NOT include bull case, bear case, buy/sell recommendations, or per-ticker "
            "calls unless the research specifically surfaced names worth flagging.\n\n"
            "Sections: headline, executive summary (the answer), supporting details with "
            "specific names / dates / sources from the research, and (only if relevant) a "
            "brief note on risks or caveats. Set primary_recommendation to 'N/A'."
        )
    else:
        system = (
            "You are a senior equity-research editor. Produce a polished, investor-ready "
            "markdown report. Sections: headline, executive summary, detailed thesis, "
            "per-ticker breakdown (one subsection per ticker if multiple), evidence by "
            "dimension, risk disclosures, and the final recommendation(s). Be specific."
        )
    payload = (
        f"USER QUERY: {state['user_query']}\nQUERY TYPE: {plan.query_type}\n"
        f"TICKERS: {plan.tickers}\n\n"
        f"THESIS:\n{state['thesis'].model_dump_json(indent=2)}\n\n"
        f"RISK:\n{state['risk'].model_dump_json(indent=2)}\n\n"
        + _all_findings_payload(state)
    )
    editor = _llm(model="gpt-4o", temperature=0.2).with_structured_output(FinalReport)
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
    graph.add_node("risk", risk_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("final_report", final_report_node)

    graph.add_edge(START, "user_query")
    graph.add_edge("user_query", "query_planner")

    # Dynamic fan-out via Send (per-ticker × per-dimension, or topic-only for general)
    graph.add_conditional_edges("query_planner", dispatch_research, RESEARCH_NODES)
    for node in RESEARCH_NODES:
        graph.add_edge(node, "aggregator")

    graph.add_edge("aggregator", "critic")
    graph.add_conditional_edges(
        "critic",
        critic_router,
        {"sufficient": "risk", "gather_more": "query_planner"},
    )

    graph.add_edge("risk", "synthesis")
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
        Holding,
        FundamentalsFindings,
        KeyMetric,
        SentimentFindings,
        TechnicalFindings,
        MacroFindings,
        CriticDecision,
        RiskAssessment,
        TickerRisk,
        InvestmentThesis,
        TickerRecommendation,
        HumanReview,
        FinalReport,
    ])
    return graph.compile(checkpointer=MemorySaver(serde=serde))


app = build_graph()


# ---------- Pretty terminal rendering (rich) ----------

console = Console()

NODE_STYLE = {
    "query_planner":      ("Query Planner",                "blue"),
    "fundamentals_agent": ("Fundamentals Agent",           "yellow"),
    "sentiment_agent":    ("News & Sentiment Agent",       "magenta"),
    "technical_agent":    ("Technical Analysis Agent",     "green"),
    "macro_agent":        ("Macro & Sector Agent",         "cyan"),
    "aggregator":         ("Aggregator",                   "cyan"),
    "critic":             ("Critic",                       "yellow"),
    "risk":               ("Risk Assessment",              "red"),
    "synthesis":          ("Synthesis & Thesis",           "magenta"),
    "final_report":       ("Final Report",                 "green"),
}


def _format_payload(node: str, payload: dict):
    if node == "query_planner":
        plan: Optional[ResearchPlan] = payload.get("research_plan")
        if not plan:
            return None
        t = Table(show_header=False, box=None, padding=(0, 1))
        t.add_column(style="bold")
        t.add_column()
        t.add_row("Query type", plan.query_type)
        t.add_row("Tickers", ", ".join(plan.tickers) if plan.tickers else "(none — general query)")
        if plan.holdings:
            t.add_row("Holdings", ", ".join(f"{h.quantity:g} {h.ticker}" for h in plan.holdings))
        if plan.topic:
            t.add_row("Topic", plan.topic)
        t.add_row("Horizon", plan.investor_horizon)
        t.add_row("Questions",
                  f"fund ({len(plan.fundamentals_questions)})  "
                  f"sent ({len(plan.sentiment_questions)})  "
                  f"tech ({len(plan.technical_questions)})  "
                  f"macro ({len(plan.macro_questions)})")
        return t

    if node == "fundamentals_agent":
        d = payload.get("fundamentals") or {}
        if not d:
            return None
        ticker, f = next(iter(d.items()))
        return (f"[bold]Ticker:[/] {ticker}\n"
                f"[bold]Summary:[/] {f.summary}\n"
                f"[bold]Valuation:[/] {f.valuation}\n"
                f"[bold]Confidence:[/] {f.confidence:.2f}")

    if node == "sentiment_agent":
        d = payload.get("sentiment") or {}
        if not d:
            return None
        key, s = next(iter(d.items()))
        return (f"[bold]Subject:[/] {key}\n"
                f"[bold]Social sentiment:[/] {s.social_sentiment}\n"
                f"[bold]Analyst consensus:[/] {s.analyst_consensus}\n"
                f"[bold]Headlines:[/] {s.headline_summary}\n"
                f"[bold]Confidence:[/] {s.confidence:.2f}")

    if node == "technical_agent":
        d = payload.get("technical") or {}
        if not d:
            return None
        ticker, t = next(iter(d.items()))
        price = f"${t.current_price:.2f}" if t.current_price else "n/a"
        rsi = f"{t.rsi:.1f}" if t.rsi else "n/a"
        return (f"[bold]Ticker:[/] {ticker}\n"
                f"[bold]Trend:[/] {t.trend}   [bold]Momentum:[/] {t.momentum_signal}\n"
                f"[bold]Price:[/] {price}   [bold]RSI14:[/] {rsi}\n"
                f"[bold]Summary:[/] {t.summary}\n"
                f"[bold]Confidence:[/] {t.confidence:.2f}")

    if node == "macro_agent":
        d = payload.get("macro") or {}
        if not d:
            return None
        key, m = next(iter(d.items()))
        return (f"[bold]Subject:[/] {key}\n"
                f"[bold]Sector:[/] {m.sector}\n"
                f"[bold]Industry:[/] {m.industry_trends}\n"
                f"[bold]Summary:[/] {m.summary}\n"
                f"[bold]Confidence:[/] {m.confidence:.2f}")

    if node == "aggregator":
        return f"Research round [bold]{payload.get('research_iterations')}[/] complete."

    if node == "critic":
        c = payload.get("critic")
        if not c:
            return None
        verdict = "[bold green]SUFFICIENT[/]" if c.sufficient else "[bold yellow]NEEDS MORE[/]"
        body = f"[bold]Verdict:[/] {verdict}\n[bold]Rationale:[/] {c.rationale}"
        if c.gaps:
            body += "\n[bold]Gaps:[/]\n  - " + "\n  - ".join(c.gaps)
        return body

    if node == "risk":
        r = payload.get("risk")
        if not r:
            return None
        body = (f"[bold]Overall risk:[/] [bold red]{r.overall_risk_rating}[/]\n"
                f"[bold]Summary:[/] {r.summary}")
        if r.per_ticker_risks:
            body += "\n[bold]Per ticker:[/]\n" + "\n".join(
                f"  - {pr.ticker}: [red]{pr.risk_rating}[/]"
                for pr in r.per_ticker_risks
            )
        return body

    if node == "synthesis":
        th = payload.get("thesis")
        if not th:
            return None
        is_general = th.primary_recommendation == "N/A"
        if is_general:
            body = f"[bold]Answer:[/] {th.thesis_summary}"
        else:
            body = (f"[bold]Recommendation:[/] [bold]{th.primary_recommendation}[/]   "
                    f"[bold]Conviction:[/] {th.overall_conviction}\n"
                    f"[bold]Thesis:[/] {th.thesis_summary}")
        if th.per_ticker_recommendations:
            body += "\n[bold]Per ticker:[/]\n" + "\n".join(
                f"  - {r.ticker}: [bold]{r.recommendation}[/] ({r.conviction}) — {r.rationale}"
                for r in th.per_ticker_recommendations
            )
        return body

    if node == "final_report":
        fr = payload.get("final_report")
        if not fr:
            return None
        return f"[bold]Headline:[/] {fr.headline}\n[dim]Full report rendered below.[/]"

    return None


def render_node_update(node: str, payload):
    if node == "__interrupt__" or node == "user_query":
        return
    if not isinstance(payload, dict):
        return
    spec = NODE_STYLE.get(node)
    if not spec:
        return
    title, color = spec
    body = _format_payload(node, payload)
    if body is None:
        return
    # For per-ticker agents, append the ticker to the panel title.
    suffix = ""
    for slot in ("fundamentals", "sentiment", "technical", "macro"):
        d = payload.get(slot)
        if d:
            keys = list(d.keys())
            if len(keys) == 1:
                suffix = f"  -  {keys[0]}"
            break
    console.print(Panel(body, title=f"[bold {color}]{title}[/]{suffix}",
                        border_style=color, expand=True))


def prompt_human(payload: dict) -> dict:
    """Interactive human-in-the-loop review prompt."""
    console.print(Rule(title="[bold yellow]Investor Review Required[/]", style="yellow"))

    summary = Table(show_header=False, box=None, padding=(0, 1))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("Query type", str(payload.get("query_type")))
    tickers = payload.get("tickers") or []
    summary.add_row("Tickers", ", ".join(tickers) if tickers else "(general query)")
    summary.add_row("Recommendation", f"[bold]{payload.get('primary_recommendation')}[/]")
    summary.add_row("Conviction", str(payload.get("overall_conviction")))
    summary.add_row("Risk rating", str(payload.get("overall_risk_rating")))
    console.print(Panel(summary, border_style="yellow"))

    per_ticker = payload.get("per_ticker_recommendations") or []
    if per_ticker:
        tbl = Table(title="Per-ticker recommendations", title_style="bold yellow")
        tbl.add_column("Ticker", style="bold")
        tbl.add_column("Recommendation", style="bold")
        tbl.add_column("Conviction")
        tbl.add_column("Rationale", overflow="fold")
        for r in per_ticker:
            tbl.add_row(str(r.get("ticker")), str(r.get("recommendation")),
                        str(r.get("conviction")), str(r.get("rationale")))
        console.print(tbl)

    console.print(f"\n[bold]Answer:[/] {payload.get('thesis_summary')}\n"
                  if payload.get("is_general")
                  else f"\n[bold]Thesis:[/] {payload.get('thesis_summary')}\n")

    def _bullets(items, style):
        if not items:
            return f"[{style}]  (none)[/]"
        return "\n".join(f"[{style}]  - {it}[/]" for it in items)

    # Skip bull/bear case sections for general informational queries.
    if not payload.get("is_general"):
        console.print("[bold green]Bull case[/]")
        console.print(_bullets(payload.get("bull_case", []), "green"))
        console.print("\n[bold red]Bear case[/]")
        console.print(_bullets(payload.get("bear_case", []), "red"))

    if payload.get("key_risks"):
        console.print("\n[bold yellow]Key risks[/]")
        console.print(_bullets(payload.get("key_risks", []), "yellow"))
    console.print()

    decision = Prompt.ask(
        "[bold]Approve the thesis or request a revision?[/]",
        choices=["approve", "revise"],
        default="approve",
    )
    if decision == "revise":
        feedback = Prompt.ask("[yellow]What would you like changed?[/]")
        return {"decision": "revise", "feedback": feedback}
    return {"decision": "approve"}


# ---------- Interactive runner ----------

def run_research(query: str, config: dict) -> None:
    """Stream the graph for one user query, handling interrupts interactively."""
    inputs = {
        "user_query": query,
        "research_iterations": 0,
        "revision_iterations": 0,
        "fundamentals": {},
        "sentiment": {},
        "technical": {},
        "macro": {},
    }

    # Text shown on the spinner WHILE the next step is being generated.
    # Indexed by the node that just completed (so we describe what's coming).
    next_step_message = {
        "user_query":         "Planning research...",
        "query_planner":      "Dispatching research agents...",
        "fundamentals_agent": "Gathering research findings...",
        "sentiment_agent":    "Gathering research findings...",
        "technical_agent":    "Gathering research findings...",
        "macro_agent":        "Gathering research findings...",
        "aggregator":         "Critiquing evidence...",
        "critic":             "Assessing risk...",
        "risk":               "Synthesising thesis...",
        "synthesis":          "Awaiting investor review...",
        "human_review":       "Generating final report...",
        "final_report":       "Wrapping up...",
    }

    resume_value = None
    first = True

    while True:
        if first:
            stream_input = inputs
            current_status = "Planning research..."
            first = False
        else:
            stream_input = Command(resume=resume_value)
            current_status = (
                "Revising thesis..."
                if resume_value.get("decision") == "revise"
                else "Generating final report..."
            )

        with console.status(f"[dim]{current_status}[/]", spinner="dots") as status:
            for event in app.stream(stream_input, config=config, stream_mode="updates"):
                for node, payload in event.items():
                    render_node_update(node, payload)
                    msg = next_step_message.get(node)
                    if msg:
                        status.update(f"[dim]{msg}[/]")

        snapshot = app.get_state(config)
        pending = [t for t in snapshot.tasks if t.interrupts]
        if not pending:
            break
        resume_value = prompt_human(pending[0].interrupts[0].value)

    final = app.get_state(config).values.get("final_report")
    if final:
        console.print(Rule(title="[bold green]Final Recommendation Report[/]", style="green"))
        console.print(Markdown(final.full_markdown))
        console.print(Rule(style="green"))


def main() -> None:
    console.print(Rule(title="[bold blue]Deep Research Stock Picker[/]", style="blue"))
    console.print(
        "[dim]Ask any investment question. Examples:\n"
        "  - 'Is TSLA a buy for the next year?' (single ticker)\n"
        "  - 'Analyse my positions: 3 VOO, 4 GOOGL, 7 NVDA, 2 JPM' (portfolio)\n"
        "  - 'What stocks are going to IPO this year?' (general — no ticker)\n"
        "Type 'exit' to quit.[/]\n"
    )

    session = 0
    while True:
        session += 1
        try:
            query = Prompt.ask("[bold cyan]Question[/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/]")
            return
        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye.[/]")
            return

        config = {"configurable": {"thread_id": f"session-{session}"}}
        try:
            run_research(query, config)
        except Exception:
            console.print_exception()
        console.print()


if __name__ == "__main__":
    main()
