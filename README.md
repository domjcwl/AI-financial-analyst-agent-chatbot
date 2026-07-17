# Deep Research Stock Picker

An AI financial analyst agent built on [LangGraph](https://github.com/langchain-ai/langgraph). It turns a single stock ticker into an investor-ready equity research report: a query planner decomposes the request, four specialist sub-agents research it in parallel, a critic decides whether the evidence is decision-ready, a portfolio-manager node synthesises an investment thesis, and a human-in-the-loop step gives an investor final sign-off before the report is generated.

## How it works

![LangGraph flow](docs/langgraph_flow.png)

1. **`user_query`** — the validated ticker enters the graph.
2. **`query_planner`** — an LLM decomposes the ticker into a `ResearchPlan` with 3-5 targeted questions per research dimension.
3. **Parallel research (`Send` fan-out)** — four ReAct sub-agents run concurrently, each with its own curated tool set:
   - `fundamentals_agent` — valuation, margins, balance sheet, guidance (yfinance + web search)
   - `sentiment_agent` — news, analyst consensus, catalysts (Tavily news search)
   - `technical_agent` — price trend, moving averages, RSI, support/resistance (yfinance price history)
   - `macro_agent` — sector positioning, peers, macro drivers (Tavily web search)
4. **`aggregator`** — synchronisation point after the parallel fan-out.
5. **`critic`** — a skeptical investment-committee chair that checks confidence scores, missing sources, unanswered questions, and contradictions. If the evidence isn't decision-ready, it routes back to `query_planner` with a surgical list of which (ticker, dimension) pairs need more research — capped at `MAX_RESEARCH_ITERATIONS`.
6. **`synthesis`** — a portfolio-manager node integrates all findings into an `InvestmentThesis`: future outlook, bull/bear case, risk rating, and a Buy/Hold/Sell-style recommendation.
7. **`human_review`** — the graph pauses (LangGraph `interrupt`) for investor approval. The investor can approve or request a revision with feedback, looping back to `synthesis` — capped at `MAX_REVISION_ITERATIONS`.
8. **`final_report`** — an editor node produces the polished markdown report.

Every research and reasoning node is grounded to only use tool results or upstream findings — the prompts explicitly forbid falling back on the model's own training knowledge, so every number and claim in the final report traces back to a tool call made during that run.

## Project structure

| File | Purpose |
|---|---|
| [agent.py](agent.py) | LangGraph graph definition — state, nodes, routers, `build_graph()` |
| [tools.py](tools.py) | `@tool`-decorated functions (Tavily search, yfinance market data) and ticker input validation |
| [schemas.py](schemas.py) | Pydantic schemas for the plan, per-dimension findings, critic decision, thesis, and final report |
| [cli.py](cli.py) | Interactive Rich-powered terminal UI |
| [app.py](app.py) | Streamlit chat UI |

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with:

```
OPENAI_API_KEY=your-openai-key
TAVILY_API_KEY=your-tavily-key
```

## Usage

**CLI:**

```bash
python cli.py
```

**Streamlit app:**

```bash
streamlit run app.py
```

Enter a single stock ticker (e.g. `NVDA`, `GOOGL`, `JPM`). Sentences, questions, and multiple symbols are rejected by input validation before the graph runs.

## Tunables

Defined at the top of [agent.py](agent.py):

- `MAX_RESEARCH_ITERATIONS` — caps the critic → planner evidence-gathering loop (default 2)
- `MAX_REVISION_ITERATIONS` — caps the human → synthesis revision loop (default 2)
- `SUBAGENT_RECURSION_LIMIT` — max ReAct steps per sub-agent (default 40)
- `REASONING_MODEL` / `SUBAGENT_MODEL` / `EXTRACTOR_MODEL` — model tiers (reasoning nodes use a larger model than tool-heavy sub-agents)
