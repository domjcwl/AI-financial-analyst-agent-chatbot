from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import httpx
import os
import requests
from datetime import datetime, timedelta

load_dotenv()
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
FMP_API_KEY     = os.getenv("FMP_API_KEY")       # https://financialmodelingprep.com/
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")   # https://finnhub.io/  (free: 60 req/min)

# New FMP stable base URL (replaces deprecated /api/v3/)
FMP_BASE = "https://financialmodelingprep.com/stable"


# ── Agent State ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ── Tools ────────────────────────────────────────────────────────────────────

@tool
def smart_news_search(ticker: str) -> str:
    """
    Find the latest news articles related to a stock ticker using Finnhub.
    Provide the ticker symbol (e.g. 'AAPL').
    Returns up to 5 recent news articles with title, source, date, summary and URL.
    """
    try:
        today     = datetime.today().strftime("%Y-%m-%d")
        from_date = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")

        url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": ticker.upper(),
            "from":   from_date,
            "to":     today,
            "token":  FINNHUB_API_KEY,
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        articles = response.json()

        if not articles:
            return f"No recent news found for '{ticker}' in the past 30 days."

        # Sort by newest first and take top 5
        articles = sorted(articles, key=lambda x: x.get("datetime", 0), reverse=True)[:5]

        lines = [f"📰 Latest news for {ticker.upper()} (past 30 days):\n"]
        for i, art in enumerate(articles, 1):
            title    = art.get("headline", "N/A")
            source   = art.get("source", "Unknown")
            pub_ts   = art.get("datetime", 0)
            pub_date = datetime.fromtimestamp(pub_ts).strftime("%Y-%m-%d") if pub_ts else "N/A"
            summary  = art.get("summary", "")[:200]
            url_link = art.get("url", "")
            lines.append(
                f"{i}. [{source}] {title} ({pub_date})\n"
                f"   {summary}\n"
                f"   🔗 {url_link}\n"
            )
        return "\n".join(lines)

    except Exception as e:
        return f"❌ News search failed for '{ticker}': {str(e)}"


@tool
def get_financial_statements(
    ticker: str,
    period: str = "annual",
    limit: int = 1,
) -> str:
    """
    Fetch income statement, balance sheet, and cash flow statement for a stock ticker.

    Args:
        ticker : Stock ticker symbol, e.g. 'AAPL', 'MSFT'.
        period : 'annual' (default) or 'quarter'. Use 'quarter' when the user asks
                 for quarterly data or a specific quarter (e.g. Q1 2023).
        limit  : How many periods to return (default 1 = latest only).
                 Set higher to retrieve historical data, e.g.:
                   - "last 3 years"   -> limit=3,  period='annual'
                   - "last 4 quarters"-> limit=4,  period='quarter'
                   - "since 2020"     -> limit=5,  period='annual'  (approx years since 2020)
                 Maximum useful value is 10 for annual, 16 for quarterly.

    Examples the LLM should handle:
        "Show AAPL financials"                 -> ticker='AAPL', period='annual', limit=1
        "Show AAPL last 5 years of financials" -> ticker='AAPL', period='annual', limit=5
        "MSFT quarterly financials"            -> ticker='MSFT', period='quarter', limit=4
        "NVDA income statements since 2019"    -> ticker='NVDA', period='annual', limit=6
    """
    try:
        sym        = ticker.upper()
        period     = period.lower().strip()
        # FMP accepts 'annual' or 'quarter'
        fmp_period = "quarter" if period in ("quarter", "quarterly", "q") else "annual"
        limit      = max(1, min(limit, 20))   # clamp to sensible range

        def fetch(endpoint: str) -> list:
            url = (
                f"{FMP_BASE}/{endpoint}"
                f"?symbol={sym}&period={fmp_period}&limit={limit}&apikey={FMP_API_KEY}"
            )
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            return data if isinstance(data, list) else []

        incomes   = fetch("income-statement")
        balances  = fetch("balance-sheet-statement")
        cashflows = fetch("cash-flow-statement")

        if not incomes:
            return f"❌ No financial statements found for '{sym}'. Check the ticker symbol."

        # Build a lookup by date for balance sheet and cash flow
        balance_by_date  = {b.get("date"): b for b in balances}
        cashflow_by_date = {c.get("date"): c for c in cashflows}

        def fmt(val):
            if val is None:
                return "N/A"
            try:
                return f"${float(val):,.0f}"
            except (ValueError, TypeError):
                return str(val)

        period_label = "Quarterly" if fmp_period == "quarter" else "Annual"
        sections = [
            f"📊 {period_label} Financial Statements for {sym} "
            f"({len(incomes)} period(s) returned)\n"
        ]

        for income in incomes:
            date     = income.get("date", "N/A")
            balance  = balance_by_date.get(date, {})
            cashflow = cashflow_by_date.get(date, {})

            sections.append(f"{'='*52}")
            sections.append(f"  Period ending: {date}")
            sections.append(f"{'='*52}")
            sections.append("── INCOME STATEMENT ──────────────────────────────")
            sections.append(f"  Revenue              : {fmt(income.get('revenue'))}")
            sections.append(f"  Gross Profit         : {fmt(income.get('grossProfit'))}")
            sections.append(f"  Operating Income     : {fmt(income.get('operatingIncome'))}")
            sections.append(f"  Net Income           : {fmt(income.get('netIncome'))}")
            sections.append(f"  EPS (diluted)        : {income.get('eps', 'N/A')}")
            sections.append(f"  EBITDA               : {fmt(income.get('ebitda'))}")
            sections.append("")
            sections.append("── BALANCE SHEET ─────────────────────────────────")
            sections.append(f"  Total Assets         : {fmt(balance.get('totalAssets'))}")
            sections.append(f"  Total Liabilities    : {fmt(balance.get('totalLiabilities'))}")
            sections.append(f"  Total Equity         : {fmt(balance.get('totalStockholdersEquity'))}")
            sections.append(f"  Cash & Equivalents   : {fmt(balance.get('cashAndCashEquivalents'))}")
            sections.append(f"  Total Debt           : {fmt(balance.get('totalDebt'))}")
            sections.append("")
            sections.append("── CASH FLOW STATEMENT ───────────────────────────")
            sections.append(f"  Operating Cash Flow  : {fmt(cashflow.get('operatingCashFlow'))}")
            sections.append(f"  Capital Expenditures : {fmt(cashflow.get('capitalExpenditure'))}")
            sections.append(f"  Free Cash Flow       : {fmt(cashflow.get('freeCashFlow'))}")
            sections.append(f"  Dividends Paid       : {fmt(cashflow.get('dividendsPaid'))}")
            sections.append("")

        return "\n".join(sections).strip()

    except Exception as e:
        return f"❌ Failed to fetch financial statements for '{ticker}': {str(e)}"


@tool
def get_analyst_data(ticker: str) -> str:
    """
    Fetch analyst sentiment data for a stock ticker.
    Uses FMP stable endpoints:
      - /stable/grades                  -> recent upgrade/downgrade actions (last 30)
      - /stable/price-target-consensus  -> high / low / median / consensus price targets
    """
    try:
        sym = ticker.upper()

        # 1. Recent analyst grade actions (upgrades / downgrades / initiations)
        grades_url = f"{FMP_BASE}/grades?symbol={sym}&limit=30&apikey={FMP_API_KEY}"
        grades_r   = requests.get(grades_url, timeout=10)
        grades_r.raise_for_status()
        raw = grades_r.json()
        grades_list = raw if isinstance(raw, list) else []

        # 2. Consensus price target (high / low / median / consensus)
        consensus_url = f"{FMP_BASE}/price-target-consensus?symbol={sym}&apikey={FMP_API_KEY}"
        consensus_r   = requests.get(consensus_url, timeout=10)
        consensus_r.raise_for_status()
        consensus_data = consensus_r.json()
        c = consensus_data[0] if isinstance(consensus_data, list) and consensus_data else {}

        if not grades_list and not c:
            return f"❌ No analyst data found for '{ticker}'."

        def fmt(val):
            try:
                return f"${float(val):,.2f}"
            except (ValueError, TypeError):
                return str(val) if val else "N/A"

        lines = [f"🎯 Analyst Data for {sym}\n"]

        # Price target consensus block
        if c:
            lines.append("── PRICE TARGET CONSENSUS ───────────────────────")
            lines.append(f"  Consensus Target : {fmt(c.get('priceTarget') or c.get('consensus'))}")
            lines.append(f"  High Target      : {fmt(c.get('priceTargetHigh') or c.get('high'))}")
            lines.append(f"  Low Target       : {fmt(c.get('priceTargetLow')  or c.get('low'))}")
            lines.append(f"  Median Target    : {fmt(c.get('priceTargetMedian') or c.get('median'))}")
            lines.append("")

        # Tally grade actions into sentiment buckets
        if grades_list:
            buy_kw  = {"buy", "strong buy", "outperform", "overweight",
                       "positive", "conviction buy", "add", "accumulate"}
            hold_kw = {"hold", "neutral", "market perform", "equal weight",
                       "peer perform", "in-line", "sector perform", "fair value"}
            sell_kw = {"sell", "strong sell", "underperform", "underweight",
                       "negative", "reduce", "avoid"}

            counts = {"Buy / Outperform": 0, "Hold / Neutral": 0, "Sell / Underperform": 0}
            recent_actions = []

            for g in grades_list:
                new_grade = (g.get("newGrade") or "").strip().lower()
                action    = (g.get("action")   or "").strip()
                firm      = g.get("gradingCompany") or g.get("company") or "Unknown"
                date      = (g.get("date") or "")[:10]
                prev      = g.get("previousGrade") or ""

                if any(k in new_grade for k in buy_kw):
                    counts["Buy / Outperform"] += 1
                elif any(k in new_grade for k in sell_kw):
                    counts["Sell / Underperform"] += 1
                elif any(k in new_grade for k in hold_kw):
                    counts["Hold / Neutral"] += 1

                if len(recent_actions) < 5:
                    grade_display = g.get("newGrade") or new_grade.title()
                    prev_display  = f" (from {prev})" if prev else ""
                    recent_actions.append(
                        f"  • {date}  {firm:<30} {action:<12} -> {grade_display}{prev_display}"
                    )

            total = sum(counts.values()) or 1
            lines.append("── GRADE SENTIMENT (last 30 actions) ────────────")
            for label, n in counts.items():
                pct = n / total * 100
                bar = "█" * int(pct / 5)
                lines.append(f"  {label:<22}: {n:>3}  {bar:<20} {pct:.1f}%")

            lines.append("")
            lines.append("── MOST RECENT ACTIONS ──────────────────────────")
            lines.extend(recent_actions)

        return "\n".join(lines)

    except Exception as e:
        return f"❌ Failed to fetch analyst data for '{ticker}': {str(e)}"


@tool
def get_key_metrics(ticker: str) -> str:
    """
    Fetch key financial metrics for a stock ticker, including:
      - Price, market cap, 52-week range, beta  (FMP /stable/profile)
      - Valuation ratios: P/E, P/B, P/S, EV/EBITDA, PEG  (FMP /stable/ratios TTM)
      - Profitability: ROE, ROA, net/gross margin
      - Liquidity & leverage: current ratio, debt/equity, interest coverage
      - Dividends: yield and per-share amount
    Use this for a snapshot of a company's fundamental health and valuation.
    """
    try:
        sym = ticker.upper()

        # 1. Company profile — price, market cap, 52-week range, beta, sector
        profile_url = f"{FMP_BASE}/profile?symbol={sym}&apikey={FMP_API_KEY}"
        profile_r   = requests.get(profile_url, timeout=10)
        profile_r.raise_for_status()
        profile_list = profile_r.json()
        p = profile_list[0] if isinstance(profile_list, list) and profile_list else {}

        # 2. TTM financial ratios — P/E, ROE, margins, etc.
        # /stable/ratios returns the most reliable field names for the stable API
        ratios_url = f"{FMP_BASE}/ratios?symbol={sym}&period=TTM&limit=1&apikey={FMP_API_KEY}"
        ratios_r   = requests.get(ratios_url, timeout=10)
        ratios_r.raise_for_status()
        ratios_list = ratios_r.json()
        r = ratios_list[0] if isinstance(ratios_list, list) and ratios_list else {}

        # 3. TTM key metrics — EV/EBITDA, PEG, free cash flow yield, etc.
        km_url  = f"{FMP_BASE}/key-metrics?symbol={sym}&period=TTM&limit=1&apikey={FMP_API_KEY}"
        km_r    = requests.get(km_url, timeout=10)
        km_r.raise_for_status()
        km_list = km_r.json()
        m = km_list[0] if isinstance(km_list, list) and km_list else {}

        if not p and not r and not m:
            return f"❌ No key metrics found for '{sym}'. Check the ticker symbol."

        def f2(val, suffix="", prefix=""):
            try:
                return f"{prefix}{float(val):.2f}{suffix}"
            except (ValueError, TypeError):
                return "N/A"

        def pct(val):
            """Render a ratio that may already be a decimal (0.25) or percent (25.0)."""
            try:
                v = float(val)
                # FMP returns most margin/yield ratios as decimals (0.xx)
                return f"{v * 100:.2f}%" if abs(v) < 10 else f"{v:.2f}%"
            except (ValueError, TypeError):
                return "N/A"

        mktcap = p.get("mktCap") or p.get("marketCap") or 0
        try:
            mktcap_fmt = f"${float(mktcap):,.0f}"
        except (ValueError, TypeError):
            mktcap_fmt = "N/A"

        # 52-week range comes as "low-high" string in profile
        range_str = str(p.get("range", ""))
        if "-" in range_str:
            parts    = range_str.split("-")
            wk52_low  = f2(parts[0].strip(), prefix="$")
            wk52_high = f2(parts[-1].strip(), prefix="$")
        else:
            wk52_low = wk52_high = "N/A"

        report = f"""
📈 Key Metrics for {sym} (TTM)

── PRICE & MARKET ────────────────────────────────
  Current Price        : {f2(p.get('price'), prefix='$')}
  Market Cap           : {mktcap_fmt}
  52-Week High         : {wk52_high}
  52-Week Low          : {wk52_low}
  Beta                 : {f2(p.get('beta'))}
  Sector / Industry    : {p.get('sector', 'N/A')} / {p.get('industry', 'N/A')}

── VALUATION ─────────────────────────────────────
  P/E Ratio (TTM)      : {f2(r.get('priceEarningsRatio') or r.get('peRatio') or m.get('peRatio'))}
  P/B Ratio            : {f2(r.get('priceToBookRatio') or r.get('pbRatio') or m.get('pbRatio'))}
  P/S Ratio            : {f2(r.get('priceToSalesRatio') or m.get('priceToSalesRatio'))}
  EV / EBITDA          : {f2(m.get('evToEbitda') or m.get('enterpriseValueOverEBITDA'))}
  PEG Ratio            : {f2(r.get('priceEarningsToGrowthRatio') or m.get('pegRatio'))}
  Price / FCF          : {f2(r.get('priceToFreeCashFlowsRatio') or m.get('pfcfRatio'))}

── PROFITABILITY ─────────────────────────────────
  Gross Profit Margin  : {pct(r.get('grossProfitMargin'))}
  Operating Margin     : {pct(r.get('operatingProfitMargin'))}
  Net Profit Margin    : {pct(r.get('netProfitMargin'))}
  Return on Equity     : {pct(r.get('returnOnEquity') or r.get('roe'))}
  Return on Assets     : {pct(r.get('returnOnAssets') or r.get('roa'))}
  Return on Inv. Cap   : {pct(r.get('returnOnCapitalEmployed'))}

── LIQUIDITY & LEVERAGE ──────────────────────────
  Current Ratio        : {f2(r.get('currentRatio'))}
  Quick Ratio          : {f2(r.get('quickRatio'))}
  Debt to Equity       : {f2(r.get('debtEquityRatio') or r.get('debtToEquity'))}
  Interest Coverage    : {f2(r.get('interestCoverage'))}

── DIVIDENDS ─────────────────────────────────────
  Dividend Yield       : {pct(r.get('dividendYield'))}
  Dividend Per Share   : {f2(m.get('dividendPerShare'), prefix='$')}
  Payout Ratio         : {pct(r.get('payoutRatio'))}
""".strip()

        return report

    except Exception as e:
        return f"❌ Failed to fetch key metrics for '{ticker}': {str(e)}"


@tool
def get_technical_indicators(ticker: str) -> str:
    """
    Fetch key technical indicators for a stock ticker using FMP stable endpoints.
    Returns the latest values for:
      - RSI(14)       — momentum / overbought-oversold (>70 overbought, <30 oversold)
      - SMA(50)       — 50-day simple moving average
      - SMA(200)      — 200-day simple moving average (trend baseline)
      - EMA(12)       — 12-day exponential moving average
      - EMA(26)       — 26-day exponential moving average (EMA12 vs EMA26 = MACD signal)
      - ADX(14)       — trend strength (>25 = strong trend)
      - Williams %R   — overbought/oversold oscillator
    All indicators use daily (1day) timeframe.
    Use this whenever the user asks about RSI, moving averages, momentum, or technical analysis.
    """
    try:
        sym  = ticker.upper()
        BASE = f"{FMP_BASE}/technical-indicators"

        def fetch_indicator(name: str, period: int) -> dict:
            """Fetch a single indicator and return the most recent data point."""
            url = (
                f"{BASE}/{name}"
                f"?symbol={sym}&periodLength={period}&timeframe=1day"
                f"&limit=1&apikey={FMP_API_KEY}"
            )
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            return data[0] if isinstance(data, list) and data else {}

        rsi       = fetch_indicator("rsi",      14)
        sma50     = fetch_indicator("sma",      50)
        sma200    = fetch_indicator("sma",     200)
        ema12     = fetch_indicator("ema",      12)
        ema26     = fetch_indicator("ema",      26)
        adx       = fetch_indicator("adx",      14)
        williams  = fetch_indicator("williams", 14)

        if not any([rsi, sma50, sma200, ema12, ema26, adx, williams]):
            return f"❌ No technical indicator data found for '{sym}'."

        def f2(d, key):
            """Extract and format a float value from a dict."""
            val = d.get(key) if d else None
            try:
                return f"{float(val):.2f}"
            except (ValueError, TypeError):
                return "N/A"

        def f2price(d, key):
            try:
                return f"${float(d.get(key)):.2f}"
            except (ValueError, TypeError):
                return "N/A"

        # Derive MACD signal from EMA crossover
        try:
            macd_val = float(ema12.get("ema", 0)) - float(ema26.get("ema", 0))
            macd_str = f"{macd_val:+.2f} ({'bullish cross' if macd_val > 0 else 'bearish cross'})"
        except (ValueError, TypeError):
            macd_str = "N/A"

        # RSI interpretation
        try:
            rsi_val = float(rsi.get("rsi", 50))
            if rsi_val >= 70:
                rsi_signal = "⚠️  Overbought"
            elif rsi_val <= 30:
                rsi_signal = "⚠️  Oversold"
            else:
                rsi_signal = "✅  Neutral"
        except (ValueError, TypeError):
            rsi_signal = ""

        # ADX interpretation
        try:
            adx_val = float(adx.get("adx", 0))
            adx_signal = "Strong trend" if adx_val >= 25 else "Weak / no trend"
        except (ValueError, TypeError):
            adx_signal = ""

        # SMA trend signal
        try:
            price  = float(sma50.get("close", 0))
            s50    = float(sma50.get("sma", 0))
            s200   = float(sma200.get("sma", 0))
            if price > s50 > s200:
                trend_signal = "📈 Bullish (price > SMA50 > SMA200)"
            elif price < s50 < s200:
                trend_signal = "📉 Bearish (price < SMA50 < SMA200)"
            else:
                trend_signal = "↔️  Mixed"
        except (ValueError, TypeError):
            trend_signal = "N/A"

        # Reference date from RSI (most recent)
        ref_date = (rsi or sma50 or {}).get("date", "N/A")

        report = f"""
📊 Technical Indicators for {sym}  (as of {ref_date}, daily)

── MOMENTUM ──────────────────────────────────────
  RSI (14)             : {f2(rsi, 'rsi')}  {rsi_signal}
  Williams %R (14)     : {f2(williams, 'williams')}

── MOVING AVERAGES ───────────────────────────────
  SMA (50-day)         : {f2price(sma50, 'sma')}
  SMA (200-day)        : {f2price(sma200, 'sma')}
  EMA (12-day)         : {f2price(ema12, 'ema')}
  EMA (26-day)         : {f2price(ema26, 'ema')}
  Close Price          : {f2price(sma50, 'close')}

── MACD (EMA12 − EMA26) ──────────────────────────
  MACD Value           : {macd_str}

── TREND STRENGTH ────────────────────────────────
  ADX (14)             : {f2(adx, 'adx')}  ({adx_signal})
  Trend Signal         : {trend_signal}
""".strip()

        return report

    except Exception as e:
        return f"❌ Failed to fetch technical indicators for '{ticker}': {str(e)}"


# ── Model & Tool Node ────────────────────────────────────────────────────────

tools     = [smart_news_search, get_financial_statements, get_analyst_data, get_key_metrics, get_technical_indicators]
tool_node = ToolNode(tools)

model = ChatOpenAI(
    model="gpt-4o",
    http_client=httpx.Client(verify=False),   # bypass corporate proxy/firewall if needed
    api_key=OPENAI_API_KEY,
    temperature=0,
).bind_tools(tools)


# ── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an expert financial analyst assistant with deep knowledge of equity markets,
fundamental analysis, and investment research.

You have access to four tools:
  • smart_news_search        – latest company news via Finnhub (past 30 days)
  • get_financial_statements – income statement, balance sheet, cash flow (FMP stable API).
                               Supports period=('annual'|'quarter') and limit=N for historical data.
                               Infer limit from phrases like "last 3 years" (limit=3, period=annual),
                               "since 2020" (limit=years since 2020), "last 4 quarters" (limit=4, period=quarter).
  • get_analyst_data         – analyst grades summary + price target consensus (FMP stable API)
  • get_key_metrics          – valuation (P/E, P/B, P/S, EV/EBITDA, PEG), profitability (ROE, margins),
                               liquidity, dividends. Use for fundamental health snapshot.
  • get_technical_indicators – RSI(14), SMA(50/200), EMA(12/26), MACD signal, ADX(14), Williams%%R.
                               Use whenever the user asks about RSI, moving averages, momentum,
                               overbought/oversold signals, or any technical analysis.

GUIDELINES:
- When the user mentions a ticker or company, proactively call the relevant tools to
  ground your answer in real data.
- For broad market research questions, combine news + key metrics + analyst data.
- For deep-dive analysis requests, use all four tools and synthesise the results.
- Always note that figures are as of the latest available period — not real-time intraday prices.
- Be concise but thorough. Use bullet points and structured sections for readability.
- If the user provides a company name without a ticker, infer the most likely ticker
  before calling tools.
- Never fabricate financial figures. If a tool returns an error, tell the user and
  suggest they verify the ticker symbol.
- If the user wants to END the conversation (e.g. "bye", "exit", "quit", "stop"),
  respond ONLY with the exact string: TERMINATE
"""


# ── Agent Node ───────────────────────────────────────────────────────────────

def my_agent(state: AgentState) -> AgentState:
    # ── Greeting on first entry ───────────────────────────────────────────────
    if not state["messages"]:
        greeting = AIMessage(content=(
            "👋 Hello! I'm your AI Financial Analyst.\n\n"
            "I can help you with:\n"
            "  📰 Latest news & sentiment for any stock (Finnhub)\n"
            "  📊 Financial statements — income, balance sheet, cash flow (FMP)\n"
            "  🎯 Analyst ratings & price targets (FMP)\n"
            "  📈 Key metrics — valuation, profitability, dividends (FMP)\n"
            "  📉 Technical indicators — RSI, SMA, EMA, MACD, ADX (FMP)\n\n"
            "Just tell me a ticker symbol or company name to get started.\n"
            "Type 'exit' or 'bye' to end the session."
        ))
        print(f"\n🤖 AI: {greeting.content}")
        return {"messages": [greeting]}

    system_msg = SystemMessage(content=SYSTEM_PROMPT)

    # ── After tool execution: invoke LLM with full context ────────────────────
    # The LLM may respond with plain text OR trigger another tool call.
    # should_continue() handles both cases — no printing here so we don't
    # print intermediate tool-chaining responses that have no content.
    if isinstance(state["messages"][-1], ToolMessage):
        response = model.invoke([system_msg] + list(state["messages"]))
        # Only print if the model produced a visible reply (not a silent tool call)
        if response.content and not (hasattr(response, "tool_calls") and response.tool_calls):
            print(f"\n🤖 AI: {response.content}")
        return {"messages": [response]}

    # ── Normal user turn: collect input then invoke ───────────────────────────
    user_input   = input("\n👤 You: ").strip()
    user_message = HumanMessage(content=user_input)

    all_messages = [system_msg] + list(state["messages"]) + [user_message]
    response     = model.invoke(all_messages)

    # Only print if this is a direct reply, not a tool-dispatch message
    if response.content and not (hasattr(response, "tool_calls") and response.tool_calls):
        print(f"\n🤖 AI: {response.content}")

    return {"messages": [user_message, response]}


# ── Routing ──────────────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]

    if isinstance(last, AIMessage):
        # Tool call present → dispatch to tools regardless of where we came from
        if hasattr(last, "tool_calls") and last.tool_calls:
            names = [tc["name"] for tc in last.tool_calls]
            print(f"\n🔧 Calling tools: {names}")
            return "tools"

        # Termination signal
        if "TERMINATE" in last.content:
            print("\n👋 Goodbye! Happy investing.")
            return "end"

    # Otherwise wait for next user input
    return "agent"


# ── Graph ────────────────────────────────────────────────────────────────────

workflow = StateGraph(AgentState)

workflow.add_node("agent", my_agent)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "agent": "agent",
        "end":   END,
    },
)

workflow.add_edge("tools", "agent")

graph = workflow.compile()


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🚀 Financial Analyst Agent started  (type 'exit' or 'bye' to quit)\n")
    graph.invoke({"messages": []})