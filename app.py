"""Streamlit frontend for the deep-research stock-picking agent.

Mirrors the same information as cli.py — query plan, per-agent findings,
critic verdict, thesis synthesis, human-in-the-loop review, and final report —
inside a ChatGPT-style chat UI.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import uuid
from typing import Optional

import streamlit as st
from langgraph.types import Command

from agent import app
from schemas import QueryItem, ResearchPlan


# ---------- Page setup ----------

st.set_page_config(
    page_title="Deep Research Stock Picker",
    page_icon="•",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 820px;
            padding-top: 2.5rem;
            padding-bottom: 6rem;
        }
        [data-testid="stChatMessage"] {
            padding: 0.6rem 0.9rem;
            border-radius: 14px;
        }
        h1, h2, h3, h4, h5 { letter-spacing: -0.01em; }
        /* Tighten section headings inside the expanders */
        [data-testid="stExpander"] h5 {
            margin-top: 0.9rem;
            margin-bottom: 0.25rem;
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            opacity: 0.78;
        }
        [data-testid="stExpander"] h5:first-child {
            margin-top: 0.1rem;
        }
        .agent-divider {
            border: none;
            border-top: 1px solid rgba(125, 125, 125, 0.18);
            margin: 0.6rem 0 0.6rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Node display config ----------

# (Pretty title, Streamlit colour name) per node.
NODE_INFO: dict[str, tuple[str, str]] = {
    "query_planner":      ("Query Planner",            "blue"),
    "fundamentals_agent": ("Fundamentals Agent",       "orange"),
    "sentiment_agent":    ("News & Sentiment Agent",   "violet"),
    "technical_agent":    ("Technical Analysis Agent", "green"),
    "macro_agent":        ("Macro & Sector Agent",     "blue"),
    "aggregator":         ("Aggregator",               "blue"),
    "critic":             ("Critic",                   "orange"),
    "synthesis":          ("Synthesis & Thesis",       "violet"),
    "final_report":       ("Final Report",             "green"),
}

_DIM_META = [
    ("fundamentals", "fundamentals_agent", "Fundamentals"),
    ("sentiment",    "sentiment_agent",    "Sentiment"),
    ("technical",    "technical_agent",    "Technical"),
    ("macro",        "macro_agent",        "Macro"),
]

_STOCK_DIMS = {"fundamentals", "sentiment", "technical", "macro"}
_INDUSTRY_DIMS = {"sentiment", "macro"}


def _valid_dims(item: QueryItem) -> set[str]:
    return _STOCK_DIMS if item.query_type == "stock_analysis" else _INDUSTRY_DIMS


def _active_dims_for_item(
    item: QueryItem,
    gap_dims: Optional[set[str]],
    gap_targets: Optional[set[str]],
) -> set[str]:
    valid = _valid_dims(item)
    if gap_dims is None:
        return valid
    if gap_targets is not None and item.target not in gap_targets:
        return set()
    scoped = gap_dims & valid
    return scoped or valid


def _predicted_routes(
    plan: ResearchPlan,
    gap_dims: Optional[set[str]],
    gap_targets: Optional[set[str]],
) -> list[str]:
    counts: dict[str, int] = {dim: 0 for dim, *_ in _DIM_META}
    target_lists: dict[str, list[str]] = {dim: [] for dim, *_ in _DIM_META}
    for item in plan.items:
        active = _active_dims_for_item(item, gap_dims, gap_targets)
        for dim in active:
            counts[dim] += 1
            target_lists[dim].append(item.target)

    routes: list[str] = []
    for dim, node, _label in _DIM_META:
        n = counts[dim]
        if n == 0:
            continue
        if n == 1:
            routes.append(f"{node} ({target_lists[dim][0]})")
        else:
            routes.append(f"{node} × {n}")
    return routes


# ---------- Small helpers ----------

def _e(value) -> str:
    """Escape `$` so st.markdown doesn't treat dollar amounts as LaTeX delimiters."""
    if value is None:
        return ""
    return str(value).replace("$", r"\$")


def _section(title: str) -> None:
    st.markdown(f"##### {title}")


def _kv_block(rows: list[tuple[str, str]]) -> None:
    """Render a vertical list of bold-label / value rows."""
    st.markdown("\n".join(f"**{k}:** {v}  " for k, v in rows))


def _unanswered_section(questions: list[str]) -> None:
    if not questions:
        return
    _section(f"Unanswered questions ({len(questions)})")
    for q in questions:
        st.markdown(f"- _{_e(q)}_")


def _expander_label(title: str, color: str, suffix: str = "") -> str:
    """Coloured pill label used as the expander header."""
    return f":{color}-background[ **{title}** ]{suffix}"


def _payload_suffix(payload: dict) -> str:
    """Append the target to the panel title (e.g. ' — NVDA') when single-target."""
    for slot in ("fundamentals", "sentiment", "technical", "macro"):
        d = payload.get(slot)
        if d:
            keys = list(d.keys())
            if len(keys) == 1:
                return f" — `{keys[0]}`"
            break
    return ""


# ---------- Per-agent panel renderers ----------

def _render_query_planner(
    plan: ResearchPlan,
    gap_dims: Optional[set[str]],
    gap_targets: Optional[set[str]],
) -> None:
    # --- Summary block ---
    n_stock = len(plan.stock_targets())
    n_industry = len(plan.industry_targets())
    type_parts: list[str] = []
    if n_stock:
        type_parts.append(f"{n_stock} × stock_analysis")
    if n_industry:
        type_parts.append(f"{n_industry} × industry_analysis")

    _section("Query type")
    st.markdown(", ".join(type_parts) or "(none)")

    if plan.stock_targets():
        _section("Stocks")
        st.markdown(_e(", ".join(plan.stock_targets())))
    if plan.industry_targets():
        _section("Industries")
        st.markdown(_e(", ".join(plan.industry_targets())))
    if plan.holdings:
        _section("Holdings")
        st.markdown(
            _e(", ".join(f"{h.quantity:g} {h.ticker}" for h in plan.holdings))
        )

    _section("Horizon")
    st.markdown(plan.investor_horizon)

    _section("Routing")
    routes = _predicted_routes(plan, gap_dims, gap_targets)
    if routes:
        for r in routes:
            st.markdown(f"- {r}")
    else:
        st.markdown("_(nothing — re-plan empty)_")

    # --- Per-item generated questions, grouped by research dimension ---
    _section("Per-item questions")
    _render_query_planner_items(plan, gap_dims, gap_targets)


def _render_query_planner_items(
    plan: ResearchPlan,
    gap_dims: Optional[set[str]],
    gap_targets: Optional[set[str]],
) -> None:
    for item in plan.items:
        active = _active_dims_for_item(item, gap_dims, gap_targets)
        type_color = "orange" if item.query_type == "stock_analysis" else "blue"
        type_label = "stock" if item.query_type == "stock_analysis" else "industry"

        skipped_note = (
            " <i style='opacity:0.6'>(skipped this round)</i>" if not active else ""
        )
        st.markdown(
            f"<p style='margin-top:0.8rem;margin-bottom:0.3rem;font-size:1.02em'>"
            f"<span style='opacity:0.75'>:{type_color}[`[{type_label}]`]</span> "
            f"<b>{_e(item.target)}</b>{skipped_note}</p>",
            unsafe_allow_html=True,
        )
        if not active:
            continue

        item_fields = {
            "fundamentals": item.fundamentals_questions,
            "sentiment":    item.sentiment_questions,
            "technical":    item.technical_questions,
            "macro":        item.macro_questions,
        }

        # One bordered card per research dimension, laid out 2-per-row, so each
        # dimension's questions are visually partitioned.
        dim_blocks = [
            (node_name, label, item_fields[dim])
            for dim, node_name, label in _DIM_META
            if dim in active
        ]
        for i in range(0, len(dim_blocks), 2):
            cols = st.columns(2)
            for col, (node_name, label, qs) in zip(cols, dim_blocks[i:i + 2]):
                _, dim_color = NODE_INFO[node_name]
                with col:
                    with st.container(border=True):
                        count_chip = (
                            f"_{len(qs)} question{'s' if len(qs) != 1 else ''}_"
                        )
                        st.markdown(
                            f":{dim_color}-background[ **{label}** ] &nbsp; {count_chip}"
                        )
                        if not qs:
                            st.caption("0 — agent will run with default workflow")
                        else:
                            for q in qs:
                                st.markdown(f"- {_e(q)}")


def _render_technical_agent(payload: dict) -> None:
    d = payload.get("technical") or {}
    if not d:
        return
    ticker, t = next(iter(d.items()))

    _section("Ticker")
    st.markdown(f"`{_e(ticker)}`")

    _section("Momentum")
    _kv_block([
        ("Trend",    _e(t.trend)),
        ("Signal",   _e(t.momentum_signal)),
    ])

    _section("Technical indicators")
    price = rf"\${t.current_price:.2f}" if t.current_price else "n/a"
    rsi = f"{t.rsi:.1f}" if t.rsi else "n/a"
    _kv_block([
        ("Current price",        price),
        ("RSI (14)",             rsi),
        ("Moving averages",      _e(t.moving_averages)),
        ("Volume profile",       _e(t.volume_profile)),
        ("Support / resistance", _e(t.support_resistance)),
    ])

    _section("Summary")
    st.markdown(_e(t.summary))
    st.caption(f"Confidence: {t.confidence:.2f}")

    _unanswered_section(t.unanswered_questions)


def _render_sentiment_agent(payload: dict) -> None:
    d = payload.get("sentiment") or {}
    if not d:
        return
    key, s = next(iter(d.items()))

    _section("Subject")
    st.markdown(f"`{_e(key)}`")

    _section("Social sentiment")
    st.markdown(_e(s.social_sentiment))

    _section("Analyst consensus")
    rows: list[tuple[str, str]] = [("Consensus", _e(s.analyst_consensus))]
    if s.average_price_target is not None:
        rows.append(("Average price target", rf"\${s.average_price_target:.2f}"))
    _kv_block(rows)

    _section("Recent news")
    st.markdown(_e(s.headline_summary))
    if s.notable_catalysts:
        st.markdown("**Notable catalysts:**")
        for c in s.notable_catalysts:
            st.markdown(f"- {_e(c)}")

    _section("Overall summary")
    st.markdown(_e(s.summary))
    st.caption(f"Confidence: {s.confidence:.2f}")

    _unanswered_section(s.unanswered_questions)

    if s.sources:
        _section("Sources")
        for src in s.sources:
            st.markdown(f"- {_e(src)}")


def _render_fundamentals_agent(payload: dict) -> None:
    d = payload.get("fundamentals") or {}
    if not d:
        return
    ticker, f = next(iter(d.items()))

    _section("Ticker")
    st.markdown(f"`{_e(ticker)}`")

    _section("Key metrics")
    if f.key_metrics:
        for m in f.key_metrics:
            st.markdown(f"- `{_e(m.name)}`: {_e(m.value)}")
    else:
        st.caption("_(none reported)_")

    _section("Revenue trend")
    st.markdown(_e(f.revenue_trend))

    _section("Profitability")
    _kv_block([
        ("Margins & earnings", _e(f.profitability)),
        ("Valuation",          _e(f.valuation)),
    ])

    _section("Balance sheet")
    _kv_block([
        ("Position", _e(f.balance_sheet)),
        ("Guidance", _e(f.guidance)),
    ])

    _section("Overall summary")
    st.markdown(_e(f.summary))
    st.caption(f"Confidence: {f.confidence:.2f}")

    _unanswered_section(f.unanswered_questions)

    if f.sources:
        _section("Sources")
        for src in f.sources:
            st.markdown(f"- {_e(src)}")


def _render_macro_agent(payload: dict) -> None:
    d = payload.get("macro") or {}
    if not d:
        return
    key, m = next(iter(d.items()))

    _section("Subject")
    st.markdown(f"`{_e(key)}`")

    _section("Sector")
    st.markdown(_e(m.sector))

    _section("Industry news")
    _kv_block([
        ("Industry trends",      _e(m.industry_trends)),
        ("Competitor landscape", _e(m.competitor_comparison)),
        ("Macro drivers",        _e(m.macro_drivers)),
    ])

    _section("Overall summary")
    st.markdown(_e(m.summary))
    st.caption(f"Confidence: {m.confidence:.2f}")

    _unanswered_section(m.unanswered_questions)

    if m.sources:
        _section("Sources")
        for src in m.sources:
            st.markdown(f"- {_e(src)}")


def _render_aggregator(payload: dict) -> None:
    st.markdown(
        f"Research round **{payload.get('research_iterations')}** complete."
    )


def _render_critic(payload: dict) -> None:
    c = payload.get("critic")
    if not c:
        return

    _section("Verdict")
    if c.sufficient:
        st.markdown(":green-background[ **SUFFICIENT** ] — evidence clears the bar.")
    else:
        st.markdown(
            ":orange-background[ **NEEDS MORE** ] — another research round required."
        )

    _section("Rationale")
    st.markdown(_e(c.rationale))

    _section("Gaps")
    has_any_gap = bool(c.gap_dimensions or c.gap_targets or c.gaps)
    if not has_any_gap:
        st.caption("_(no gaps reported)_")
    else:
        if c.gap_dimensions:
            st.markdown(
                "**Dimensions to re-research:** "
                + ", ".join(f"`{d}`" for d in c.gap_dimensions)
            )
        if c.gap_targets:
            st.markdown(
                "**Targets to re-research:** "
                + ", ".join(f"`{_e(t)}`" for t in c.gap_targets)
            )
        if c.gaps:
            st.markdown("**Specific gaps:**")
            for g in c.gaps:
                st.markdown(f"- {_e(g)}")


def _render_synthesis(payload: dict) -> None:
    th = payload.get("thesis")
    if not th:
        return
    _render_thesis_body(
        thesis_summary=th.thesis_summary,
        overall_conviction=th.overall_conviction,
        stock_outlooks=[
            {
                "ticker":          o.ticker,
                "recommendation":  o.recommendation,
                "conviction":      o.conviction,
                "risk_rating":     o.risk_rating,
                "future_outlook":  o.future_outlook,
                "bull_case":       o.bull_case,
                "bear_case":       o.bear_case,
                "risk_assessment": o.risk_assessment,
                "rationale":       o.rationale,
            }
            for o in th.stock_outlooks
        ],
        industry_outlooks=[
            {
                "industry":        o.industry,
                "recommendation":  o.recommendation,
                "conviction":      o.conviction,
                "risk_rating":     o.risk_rating,
                "future_outlook":  o.future_outlook,
                "risk_assessment": o.risk_assessment,
                "rationale":       o.rationale,
            }
            for o in th.industry_outlooks
        ],
    )


# ---------- Synthesis / thesis body ----------

_RISK_COLOR = {
    "low":       "green",
    "moderate":  "blue",
    "high":      "orange",
    "very_high": "red",
}
_CONVICTION_COLOR = {"low": "gray", "medium": "blue", "high": "green"}
_REC_COLOR_STOCK = {
    "Strong Buy":  "green",
    "Buy":         "green",
    "Hold":        "blue",
    "Sell":        "orange",
    "Strong Sell": "red",
}
_REC_COLOR_INDUSTRY = {
    "Overweight":  "green",
    "Neutral":     "blue",
    "Underweight": "orange",
}


def _badge(label: str, value: str, color: str) -> str:
    return f"**{label}:** :{color}-background[ {value} ]"


def _render_thesis_body(
    *,
    thesis_summary: Optional[str],
    overall_conviction: Optional[str],
    stock_outlooks: list[dict],
    industry_outlooks: list[dict],
) -> None:
    # --- Top-level thesis ---
    _section("Thesis")
    st.markdown(_e(thesis_summary) or "_(no thesis emitted)_")

    if overall_conviction:
        conv_color = _CONVICTION_COLOR.get(overall_conviction, "gray")
        st.markdown(_badge("Overall conviction", overall_conviction, conv_color))

    # --- Stock outlooks ---
    if stock_outlooks:
        _section(f"Stock outlooks ({len(stock_outlooks)})")
        for o in stock_outlooks:
            _render_stock_outlook_card(o)

    # --- Industry outlooks ---
    if industry_outlooks:
        _section(f"Industry outlooks ({len(industry_outlooks)})")
        for o in industry_outlooks:
            _render_industry_outlook_card(o)


def _render_stock_outlook_card(o: dict) -> None:
    rec = o.get("recommendation") or "—"
    conv = o.get("conviction") or "—"
    risk = o.get("risk_rating") or "—"
    rec_color  = _REC_COLOR_STOCK.get(rec, "gray")
    conv_color = _CONVICTION_COLOR.get(conv, "gray")
    risk_color = _RISK_COLOR.get(risk, "gray")

    with st.container(border=True):
        st.markdown(
            f"#### :violet[Stock] &nbsp; `{_e(o.get('ticker'))}`"
        )
        st.markdown(
            f"{_badge('Recommendation', rec, rec_color)} &nbsp; "
            f"{_badge('Conviction', conv, conv_color)} &nbsp; "
            f"{_badge('Risk', risk, risk_color)}"
        )

        st.markdown("**Future outlook**")
        st.markdown(_e(o.get("future_outlook")))

        bull = o.get("bull_case") or []
        bear = o.get("bear_case") or []
        col_bull, col_bear = st.columns(2)
        with col_bull:
            st.markdown(":green-background[ **Bull case** ]")
            if bull:
                for it in bull:
                    st.markdown(f"- {_e(it)}")
            else:
                st.caption("_(none)_")
        with col_bear:
            st.markdown(":red-background[ **Bear case** ]")
            if bear:
                for it in bear:
                    st.markdown(f"- {_e(it)}")
            else:
                st.caption("_(none)_")

        st.markdown("**Risk assessment**")
        st.markdown(_e(o.get("risk_assessment")))

        st.markdown("**Rationale**")
        st.markdown(_e(o.get("rationale")))


def _render_industry_outlook_card(o: dict) -> None:
    rec = o.get("recommendation") or "—"
    conv = o.get("conviction") or "—"
    risk = o.get("risk_rating") or "—"
    rec_color  = _REC_COLOR_INDUSTRY.get(rec, "gray")
    conv_color = _CONVICTION_COLOR.get(conv, "gray")
    risk_color = _RISK_COLOR.get(risk, "gray")

    with st.container(border=True):
        st.markdown(
            f"#### :violet[Industry] &nbsp; `{_e(o.get('industry'))}`"
        )
        st.markdown(
            f"{_badge('Recommendation', rec, rec_color)} &nbsp; "
            f"{_badge('Conviction', conv, conv_color)} &nbsp; "
            f"{_badge('Risk', risk, risk_color)}"
        )

        st.markdown("**Future outlook**")
        st.markdown(_e(o.get("future_outlook")))

        st.markdown("**Risk assessment**")
        st.markdown(_e(o.get("risk_assessment")))

        st.markdown("**Rationale**")
        st.markdown(_e(o.get("rationale")))


# ---------- Top-level block renderer ----------

# Map node-name -> render function (one per agent panel).
_PANEL_RENDERERS = {
    "fundamentals_agent": _render_fundamentals_agent,
    "sentiment_agent":    _render_sentiment_agent,
    "technical_agent":    _render_technical_agent,
    "macro_agent":        _render_macro_agent,
    "aggregator":         _render_aggregator,
    "critic":             _render_critic,
    "synthesis":          _render_synthesis,
}


def render_block(block: dict) -> None:
    btype = block["type"]

    if btype == "user":
        with st.chat_message("user"):
            st.markdown(block["text"])

    elif btype == "node":
        node = block["node"]
        payload = block["payload"]
        title, color = NODE_INFO.get(node, (node, "gray"))
        label = _expander_label(title, color, _payload_suffix(payload))

        with st.chat_message("assistant"):
            with st.expander(label, expanded=False):
                if node == "query_planner":
                    plan = payload.get("research_plan")
                    if plan:
                        _render_query_planner(
                            plan,
                            block.get("gap_dims"),
                            block.get("gap_targets"),
                        )
                else:
                    renderer = _PANEL_RENDERERS.get(node)
                    if renderer:
                        renderer(payload)

    elif btype == "final_report":
        label = _expander_label("Final Recommendation Report", "green")
        with st.chat_message("assistant"):
            with st.expander(label, expanded=False):
                if block.get("headline"):
                    st.markdown(f"### {_e(block['headline'])}")
                if block.get("primary_recommendation"):
                    _kv_block([(
                        "Recommendation",
                        _e(block["primary_recommendation"]),
                    )])
                covered_chips: list[str] = []
                for t in block.get("tickers_covered") or []:
                    covered_chips.append(f"`{_e(t)}`")
                for ind in block.get("industries_covered") or []:
                    covered_chips.append(f"`{_e(ind)}`")
                if covered_chips:
                    st.markdown("**Covered:** " + " ".join(covered_chips))

                st.markdown("<hr class='agent-divider'/>", unsafe_allow_html=True)
                st.markdown(_e(block["markdown"]))

    elif btype == "hitl_response":
        with st.chat_message("user"):
            if block["decision"] == "approve":
                st.markdown("**Approved.**")
            else:
                st.markdown(
                    f"**Requested revision:** {block.get('feedback') or '_(no feedback)_'}"
                )


# ---------- Human-in-the-loop form ----------

def render_hitl_form(payload: dict) -> None:
    """Investor review — uses the same thesis layout as the synthesis panel."""
    with st.chat_message("assistant"):
        with st.container(border=True):
            st.markdown(
                ":orange-background[ **Investor Review Required** ]"
            )
            st.caption(
                "Approve the thesis or request a revision before continuing."
            )
            st.markdown("<hr class='agent-divider'/>", unsafe_allow_html=True)

            _render_thesis_body(
                thesis_summary=payload.get("thesis_summary"),
                overall_conviction=payload.get("overall_conviction"),
                stock_outlooks=payload.get("stock_outlooks") or [],
                industry_outlooks=payload.get("industry_outlooks") or [],
            )

            st.markdown("<hr class='agent-divider'/>", unsafe_allow_html=True)

            with st.form("hitl_form", clear_on_submit=False):
                decision = st.radio(
                    "Approve the thesis or request a revision?",
                    options=["approve", "revise"],
                    horizontal=True,
                )
                feedback = st.text_area(
                    "Revision feedback",
                    placeholder=(
                        "What would you like changed? (Required for revise.)"
                    ),
                )
                submitted = st.form_submit_button("Submit", type="primary")

            if submitted:
                if decision == "revise" and not feedback.strip():
                    st.error("Please provide feedback to request a revision.")
                    return
                st.session_state.blocks.append({
                    "type": "hitl_response",
                    "decision": decision,
                    "feedback": feedback if decision == "revise" else None,
                })
                st.session_state.pending_resume = (
                    {"decision": "revise", "feedback": feedback}
                    if decision == "revise"
                    else {"decision": "approve"}
                )
                st.session_state.action = "resume"
                st.session_state.interrupt_payload = None
                st.rerun()


# ---------- Streaming driver ----------

def _stream_graph(stream_input) -> None:
    """Stream the graph, render each node update inline, append to history."""
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    next_step_message = {
        "user_query":         "Planning research...",
        "query_planner":      "Dispatching research agents...",
        "fundamentals_agent": "Gathering research findings...",
        "sentiment_agent":    "Gathering research findings...",
        "technical_agent":    "Gathering research findings...",
        "macro_agent":        "Gathering research findings...",
        "aggregator":         "Critiquing evidence...",
        "critic":             "Synthesising thesis...",
        "synthesis":          "Awaiting investor review...",
        "human_review":       "Generating final report...",
        "final_report":       "Wrapping up...",
    }
    initial_label = (
        "Revising thesis..."
        if isinstance(stream_input, Command)
        and (stream_input.resume or {}).get("decision") == "revise"
        else (
            "Generating final report..."
            if isinstance(stream_input, Command)
            else "Planning research..."
        )
    )

    stream_area = st.container()
    status_placeholder = st.empty()
    status = status_placeholder.status(initial_label, expanded=False)

    gap_dims = st.session_state.get("gap_dims")
    gap_targets = st.session_state.get("gap_targets")

    for event in app.stream(stream_input, config=config, stream_mode="updates"):
        for node, payload in event.items():
            if node in ("__interrupt__", "user_query"):
                continue
            if not isinstance(payload, dict):
                continue

            # Track critic gaps so the NEXT query_planner panel reflects what
            # will actually be researched.
            if node == "critic":
                c = payload.get("critic")
                if c and not c.sufficient:
                    gap_dims = set(c.gap_dimensions) or {
                        "fundamentals", "sentiment", "technical", "macro"
                    }
                    gap_targets = set(c.gap_targets) if c.gap_targets else None
                else:
                    gap_dims = None
                    gap_targets = None

            if node not in NODE_INFO:
                continue

            if node == "final_report":
                fr = payload.get("final_report")
                if fr is not None:
                    block = {
                        "type": "final_report",
                        "headline": fr.headline,
                        "primary_recommendation": fr.primary_recommendation,
                        "tickers_covered": list(fr.tickers_covered or []),
                        "industries_covered": list(fr.industries_covered or []),
                        "markdown": fr.full_markdown,
                    }
                    st.session_state.blocks.append(block)
                    with stream_area:
                        render_block(block)
                msg = next_step_message.get(node)
                if msg:
                    status.update(label=msg)
                continue

            block = {
                "type": "node",
                "node": node,
                "payload": payload,
                "gap_dims": gap_dims,
                "gap_targets": gap_targets,
            }
            st.session_state.blocks.append(block)
            with stream_area:
                render_block(block)

            msg = next_step_message.get(node)
            if msg:
                status.update(label=msg)

    status_placeholder.empty()

    st.session_state.gap_dims = gap_dims
    st.session_state.gap_targets = gap_targets


def _handle_post_stream() -> None:
    """After a stream pass, check if the graph paused for HITL."""
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    snapshot = app.get_state(config)
    pending = [t for t in snapshot.tasks if t.interrupts]
    if pending:
        st.session_state.interrupt_payload = pending[0].interrupts[0].value
    else:
        st.session_state.interrupt_payload = None


# ---------- Session state ----------

def _init_state() -> None:
    defaults = {
        "blocks": [],
        "thread_id": None,
        "action": None,            # None | 'run' | 'resume'
        "pending_query": None,
        "pending_resume": None,
        "interrupt_payload": None,
        "gap_dims": None,
        "gap_targets": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def _new_conversation() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    _init_state()


# ---------- App ----------

def main() -> None:
    _init_state()

    with st.sidebar:
        st.markdown("### Deep Research Stock Picker")
        st.markdown(
            "A multi-agent research workflow that plans, gathers, critiques, "
            "and synthesises an investment thesis — with an investor review step."
        )
        st.button(
            "New conversation",
            on_click=_new_conversation,
            use_container_width=True,
        )

    # Empty-state hero
    if not st.session_state.blocks and st.session_state.action is None:
        st.markdown("## Deep Research Stock Picker")
        st.caption("Ask any investment question. Examples:")
        st.markdown(
            "- *Is TSLA a buy for the next year?*  *(one stock_analysis)*\n"
            "- *Analyse my positions: 3 VOO, 4 GOOGL, 7 NVDA, 2 JPM*  *(multi stock_analysis)*\n"
            "- *How are the tech and healthcare industries doing?*  *(multi industry_analysis)*\n"
            "- *Analyse the tech industry and tell me if GOOGL is a buy*  *(mixed)*"
        )

    # Replay all blocks accumulated so far.
    for block in st.session_state.blocks:
        render_block(block)

    # Run any pending action: a new query or a resume after HITL.
    if st.session_state.action == "run":
        query = st.session_state.pending_query
        st.session_state.action = None
        st.session_state.pending_query = None
        st.session_state.thread_id = f"session-{uuid.uuid4()}"
        st.session_state.gap_dims = None
        st.session_state.gap_targets = None

        stream_input = {
            "user_query": query,
            "research_iterations": 0,
            "revision_iterations": 0,
            "fundamentals": {},
            "sentiment": {},
            "technical": {},
            "macro": {},
        }
        try:
            _stream_graph(stream_input)
            _handle_post_stream()
        except Exception as exc:
            st.error(f"Run failed: {exc}")
            st.exception(exc)

    elif st.session_state.action == "resume":
        resume_value = st.session_state.pending_resume
        st.session_state.action = None
        st.session_state.pending_resume = None

        try:
            _stream_graph(Command(resume=resume_value))
            _handle_post_stream()
        except Exception as exc:
            st.error(f"Resume failed: {exc}")
            st.exception(exc)

    # If the graph is paused for HITL, show the review form.
    if st.session_state.interrupt_payload:
        render_hitl_form(st.session_state.interrupt_payload)

    # Chat input — disabled while a HITL form is pending so the user resolves it first.
    disabled = st.session_state.interrupt_payload is not None
    placeholder = (
        "Resolve the investor review above to continue…"
        if disabled
        else "Ask any investment question…"
    )
    if query := st.chat_input(placeholder, disabled=disabled):
        st.session_state.blocks.append({"type": "user", "text": query})
        st.session_state.pending_query = query
        st.session_state.action = "run"
        st.rerun()


if __name__ == "__main__":
    main()
