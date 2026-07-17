"""Interactive Rich-powered CLI for the deep-research stock-picking agent.

The graph itself lives in agent.py; this module only handles:
  - Per-node panel rendering as the graph streams updates
  - The human-in-the-loop approval prompt
  - The session loop (read query -> stream -> render final report)

Run with:
    python cli.py            # preferred entrypoint
    python agent.py          # also works (delegates here)
"""

from __future__ import annotations

from typing import Optional

from langgraph.types import Command
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from agent import app
from schemas import InvestmentThesis, QueryItem, ResearchPlan
from tools import validate_query


console = Console()


# ---------- Per-node panel rendering ----------

NODE_STYLE = {
    "query_planner":      ("Query Planner",                "blue"),
    "fundamentals_agent": ("Fundamentals Agent",           "yellow"),
    "sentiment_agent":    ("News & Sentiment Agent",       "magenta"),
    "technical_agent":    ("Technical Analysis Agent",     "green"),
    "macro_agent":        ("Macro & Sector Agent",         "cyan"),
    "aggregator":         ("Aggregator",                   "cyan"),
    "critic":             ("Critic",                       "yellow"),
    "synthesis":          ("Synthesis & Thesis",           "magenta"),
    "final_report":       ("Final Report",                 "green"),
}

# Maps each research dimension to its (agent node name, panel style colour).
# Iteration order determines the visual order of question sections in the
# Query Planner panel.
_DIM_META = [
    ("fundamentals", "fundamentals_agent", "Fundamentals", "yellow"),
    ("sentiment",    "sentiment_agent",    "Sentiment",    "magenta"),
    ("technical",    "technical_agent",    "Technical",    "green"),
    ("macro",        "macro_agent",        "Macro",        "cyan"),
]

_STOCK_DIMS = {"fundamentals", "sentiment", "technical", "macro"}


# Module-level state used to align the Query Planner panel with what the
# graph will actually research. Updated when the critic flags gaps and
# reset at the start of each new query.
_last_gap_dims: Optional[set[str]] = None
_last_gap_targets: Optional[set[str]] = None


def _reset_render_state() -> None:
    """Clear cross-node CLI state at the start of a new query session."""
    global _last_gap_dims, _last_gap_targets
    _last_gap_dims = None
    _last_gap_targets = None


def _valid_dims(item: QueryItem) -> set[str]:
    return _STOCK_DIMS


def _unanswered_block(questions: list[str]) -> str:
    """Markup block listing the agent's unanswered questions (or empty string)."""
    if not questions:
        return ""
    body = f"\n[bold dim]Unanswered questions ({len(questions)}):[/]\n"
    body += "\n".join(f"[dim]  - {q}[/]" for q in questions)
    return body


def _active_dims_for_item(item: QueryItem) -> set[str]:
    """The set of dimensions that will actually be researched for this item
    after the planner step — mirrors dispatch_research's gating.

      - First-pass: the item's full valid set.
      - Re-run after critic gaps: intersection with gap_dimensions, fall back
        to the full set if the intersection is empty.
      - On a re-run, items NOT in gap_targets are skipped entirely.
    """
    valid = _valid_dims(item)
    if _last_gap_dims is None:
        return valid
    if _last_gap_targets is not None and item.target not in _last_gap_targets:
        return set()  # this item is skipped on the re-run
    scoped = _last_gap_dims & valid
    return scoped or valid


def _predicted_routes(plan: ResearchPlan) -> list[str]:
    """Mirror dispatch_research so the CLI's Routing row reflects what will
    actually fan out."""
    counts: dict[str, int] = {dim: 0 for dim, *_ in _DIM_META}
    target_lists: dict[str, list[str]] = {dim: [] for dim, *_ in _DIM_META}
    for item in plan.items:
        active = _active_dims_for_item(item)
        for dim in active:
            counts[dim] += 1
            target_lists[dim].append(item.target)

    routes: list[str] = []
    for dim, node, _label, _color in _DIM_META:
        n = counts[dim]
        if n == 0:
            continue
        if n == 1:
            routes.append(f"{node} ({target_lists[dim][0]})")
        else:
            routes.append(f"{node} x {n}")
    return routes


def _item_block(item: QueryItem, active_dims: set[str]) -> Text:
    """Render one QueryItem as a labelled, indented block with its questions."""
    type_color = "yellow"
    type_label = "stock"

    out = Text()
    out.append(f"\n[{type_label}] ", style=f"bold {type_color}")
    out.append(f"{item.target}", style="bold")
    if not active_dims:
        out.append("  (skipped this round)\n", style="dim")
        return out
    out.append("\n")

    item_fields = {
        "fundamentals": item.fundamentals_questions,
        "sentiment":    item.sentiment_questions,
        "technical":    item.technical_questions,
        "macro":        item.macro_questions,
    }
    for dim, _node, label, color in _DIM_META:
        if dim not in active_dims:
            continue
        qs = item_fields[dim]
        if not qs:
            out.append(
                f"  {label} questions (0 — agent will run with default workflow)\n",
                style=f"dim {color}",
            )
            continue
        out.append(f"  {label} questions ({len(qs)}):\n", style=f"bold {color}")
        for q in qs:
            out.append(f"    - {q}\n", style=color)
    return out


def _format_payload(node: str, payload: dict):
    if node == "query_planner":
        plan: Optional[ResearchPlan] = payload.get("research_plan")
        if not plan:
            return None
        t = Table(show_header=False, box=None, padding=(0, 1))
        t.add_column(style="bold")
        t.add_column()
        t.add_row("Query type", "stock_analysis")
        if plan.stock_targets():
            t.add_row("Stock", ", ".join(plan.stock_targets()))
        t.add_row("Horizon", plan.investor_horizon)
        t.add_row("Routing", ", ".join(_predicted_routes(plan)) or "(nothing — re-plan empty)")

        sections: list = [t]
        for item in plan.items:
            sections.append(_item_block(item, _active_dims_for_item(item)))
        return Group(*sections)

    if node == "fundamentals_agent":
        d = payload.get("fundamentals") or {}
        if not d:
            return None
        ticker, f = next(iter(d.items()))
        body = (f"[bold]Ticker:[/] {ticker}\n"
                f"[bold]Summary:[/] {f.summary}\n"
                f"[bold]Valuation:[/] {f.valuation}\n"
                f"[bold]Confidence:[/] {f.confidence:.2f}")
        body += _unanswered_block(f.unanswered_questions)
        return body

    if node == "sentiment_agent":
        d = payload.get("sentiment") or {}
        if not d:
            return None
        key, s = next(iter(d.items()))
        body = (f"[bold]Subject:[/] {key}\n"
                f"[bold]Social sentiment:[/] {s.social_sentiment}\n"
                f"[bold]Analyst consensus:[/] {s.analyst_consensus}\n"
                f"[bold]Headlines:[/] {s.headline_summary}\n"
                f"[bold]Confidence:[/] {s.confidence:.2f}")
        body += _unanswered_block(s.unanswered_questions)
        return body

    if node == "technical_agent":
        d = payload.get("technical") or {}
        if not d:
            return None
        ticker, t = next(iter(d.items()))
        price = f"${t.current_price:.2f}" if t.current_price else "n/a"
        rsi = f"{t.rsi:.1f}" if t.rsi else "n/a"
        body = (f"[bold]Ticker:[/] {ticker}\n"
                f"[bold]Trend:[/] {t.trend}   [bold]Momentum:[/] {t.momentum_signal}\n"
                f"[bold]Price:[/] {price}   [bold]RSI14:[/] {rsi}\n"
                f"[bold]Summary:[/] {t.summary}\n"
                f"[bold]Confidence:[/] {t.confidence:.2f}")
        body += _unanswered_block(t.unanswered_questions)
        return body

    if node == "macro_agent":
        d = payload.get("macro") or {}
        if not d:
            return None
        key, m = next(iter(d.items()))
        body = (f"[bold]Subject:[/] {key}\n"
                f"[bold]Sector:[/] {m.sector}\n"
                f"[bold]Industry:[/] {m.industry_trends}\n"
                f"[bold]Summary:[/] {m.summary}\n"
                f"[bold]Confidence:[/] {m.confidence:.2f}")
        body += _unanswered_block(m.unanswered_questions)
        return body

    if node == "aggregator":
        return f"Research round [bold]{payload.get('research_iterations')}[/] complete."

    if node == "critic":
        c = payload.get("critic")
        if not c:
            return None
        verdict = "[bold green]SUFFICIENT[/]" if c.sufficient else "[bold yellow]NEEDS MORE[/]"
        body = f"[bold]Verdict:[/] {verdict}\n[bold]Rationale:[/] {c.rationale}"
        if c.gap_dimensions:
            body += f"\n[bold]Gap dimensions:[/] {', '.join(c.gap_dimensions)}"
        if c.gap_targets:
            body += f"\n[bold]Gap targets:[/] {', '.join(c.gap_targets)}"
        if c.gaps:
            body += "\n[bold]Gaps:[/]\n  - " + "\n  - ".join(c.gaps)
        return body

    if node == "synthesis":
        th: Optional[InvestmentThesis] = payload.get("thesis")
        if not th:
            return None
        body = (f"[bold]Thesis:[/] {th.thesis_summary}\n"
                f"[bold]Overall conviction:[/] {th.overall_conviction}")
        if th.stock_outlooks:
            body += "\n[bold]Stock outlooks:[/]\n" + "\n".join(
                f"  - {o.ticker}: [bold]{o.recommendation}[/] "
                f"({o.conviction}) — risk: [red]{o.risk_rating}[/]"
                for o in th.stock_outlooks
            )
        return body

    if node == "final_report":
        fr = payload.get("final_report")
        if not fr:
            return None
        return f"[bold]Headline:[/] {fr.headline}\n[dim]Full report rendered below.[/]"

    return None


def render_node_update(node: str, payload):
    global _last_gap_dims, _last_gap_targets
    if node == "__interrupt__" or node == "user_query":
        return
    if not isinstance(payload, dict):
        return
    # Track the critic's gap dims/targets so the NEXT query_planner panel only
    # shows questions for (item, dimension) pairs that will actually be researched.
    if node == "critic":
        c = payload.get("critic")
        if c and not c.sufficient:
            _last_gap_dims = set(c.gap_dimensions) or {
                "fundamentals", "sentiment", "technical", "macro"
            }
            _last_gap_targets = set(c.gap_targets) if c.gap_targets else None
        else:
            _last_gap_dims = None
            _last_gap_targets = None
    spec = NODE_STYLE.get(node)
    if not spec:
        return
    title, color = spec
    body = _format_payload(node, payload)
    if body is None:
        return
    # For per-item agents, append the target to the panel title.
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


# ---------- Human-in-the-loop ----------

def prompt_human(payload: dict) -> dict:
    """Interactive human-in-the-loop review prompt."""
    console.print(Rule(title="[bold yellow]Investor Review Required[/]", style="yellow"))

    summary = Table(show_header=False, box=None, padding=(0, 1))
    summary.add_column(style="bold")
    summary.add_column()
    stock_targets = payload.get("stock_targets") or []
    if stock_targets:
        summary.add_row("Stock", ", ".join(stock_targets))
    summary.add_row("Overall conviction", str(payload.get("overall_conviction")))
    console.print(Panel(summary, border_style="yellow"))

    console.print(f"\n[bold]Thesis:[/] {payload.get('thesis_summary')}\n")

    def _bullets(items, style):
        if not items:
            return f"[{style}]  (none)[/]"
        return "\n".join(f"[{style}]  - {it}[/]" for it in items)

    stock_outlooks = payload.get("stock_outlooks") or []
    for o in stock_outlooks:
        console.print(Rule(title=f"[bold yellow]Stock: {o.get('ticker')}[/]", style="yellow"))
        console.print(
            f"[bold]Recommendation:[/] [bold]{o.get('recommendation')}[/]   "
            f"[bold]Conviction:[/] {o.get('conviction')}   "
            f"[bold]Risk:[/] [red]{o.get('risk_rating')}[/]"
        )
        console.print(f"[bold]Future outlook:[/] {o.get('future_outlook')}")
        console.print("[bold green]Bull case[/]")
        console.print(_bullets(o.get('bull_case', []), "green"))
        console.print("[bold red]Bear case[/]")
        console.print(_bullets(o.get('bear_case', []), "red"))
        console.print(f"[bold yellow]Risk assessment:[/] {o.get('risk_assessment')}")
        console.print(f"[bold]Rationale:[/] {o.get('rationale')}")
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
    _reset_render_state()
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
        "critic":             "Synthesising thesis...",
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
                    # Stop the spinner so the panel prints cleanly above it,
                    # then restart with the next-step message. Without this,
                    # rapid parallel completions (4 sub-agents finishing near
                    # the same time) collide with Rich's Live redraw and the
                    # panels can be lost on Windows terminals.
                    status.stop()
                    render_node_update(node, payload)
                    msg = next_step_message.get(node)
                    if msg:
                        status.update(f"[dim]{msg}[/]")
                    status.start()

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
        "[dim]Enter a single stock ticker symbol to analyse. Examples:\n"
        "  - 'NVDA'\n"
        "  - 'GOOGL'\n"
        "  - 'JPM'\n"
        "Sentences, questions, and multiple symbols are not accepted.\n"
        "Type 'exit' to quit.[/]\n"
    )

    session = 0
    while True:
        session += 1
        try:
            query = Prompt.ask("[bold cyan]Ticker[/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/]")
            return
        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye.[/]")
            return

        ticker, error = validate_query(query)
        if error:
            console.print(f"[bold red]Invalid input:[/] {error}")
            console.print()
            continue
        query = ticker

        config = {"configurable": {"thread_id": f"session-{session}"}}
        try:
            run_research(query, config)
        except Exception:
            console.print_exception()
        console.print()


if __name__ == "__main__":
    main()