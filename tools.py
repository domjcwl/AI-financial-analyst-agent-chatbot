"""Tool layer for the deep-research stock agent.

All @tool decorated functions live here so the graph code in agent.py stays focused
on orchestration. Each research agent receives a curated subset of these tools.

External services used:
  - Tavily   : general web search + recent news (TAVILY_API_KEY required)
  - yfinance : free price/fundamentals snapshot (no key required)

Corporate-proxy note: if your environment intercepts TLS, set
REQUESTS_CA_BUNDLE / SSL_CERT_FILE in .env so that Tavily (requests-based) and
yfinance can reach the public web. The httpx client used by ChatOpenAI in
agent.py already uses verify=False to match the sample's pattern.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
from typing import Optional

# Mute noisy library loggers before importing yfinance (it logs HTTP 404s etc.).
for _noisy in ("yfinance", "peewee", "urllib3", "urllib3.connectionpool",
               "requests", "httpx", "httpcore"):
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)

import yfinance as yf
from dotenv import load_dotenv
from langchain_core.tools import tool
from tavily import TavilyClient

load_dotenv()


@contextlib.contextmanager
def _silence_io():
    """Swallow stdout/stderr — used to suppress yfinance's raw `print()` output
    on 404s and similar errors that bypass the logging module."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield


# ---------- Tavily client (lazy) ----------

_tavily_client: Optional[TavilyClient] = None


def _tavily() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError(
                "TAVILY_API_KEY is not set. Add it to your .env file."
            )
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


# ---------- Web research tools ----------

@tool
def tavily_web_search(query: str, max_results: int = 5) -> str:
    """General web research: competitor info, qualitative context, industry write-ups.
    Returns a digest of top results with title, URL, and snippet."""
    client = _tavily()
    res = client.search(query=query, max_results=max_results, search_depth="advanced")
    items = res.get("results", [])
    if not items:
        return "No results."
    out = []
    for i, it in enumerate(items, 1):
        out.append(
            f"[{i}] {it.get('title')}\n"
            f"    URL: {it.get('url')}\n"
            f"    {(it.get('content') or '')[:600]}"
        )
    return "\n\n".join(out)


@tool
def tavily_news_search(query: str, days: int = 14, max_results: int = 6) -> str:
    """Recent news headlines (default last 14 days). Use for sentiment, catalysts, events."""
    client = _tavily()
    res = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
        topic="news",
        days=days,
    )
    items = res.get("results", [])
    if not items:
        return "No news."
    out = []
    for i, it in enumerate(items, 1):
        out.append(
            f"[{i}] {it.get('title')} ({it.get('published_date', '')})\n"
            f"    URL: {it.get('url')}\n"
            f"    {(it.get('content') or '')[:600]}"
        )
    return "\n\n".join(out)


# ---------- Market-data tools ----------

@tool
def get_stock_quote(ticker: str) -> dict:
    """Latest price snapshot for a ticker via yfinance fast_info."""
    with _silence_io():
        t = yf.Ticker(ticker)
        info = t.fast_info or {}
    return {
        "ticker": ticker,
        "last_price": _safe_float(info.get("last_price")),
        "previous_close": _safe_float(info.get("previous_close")),
        "currency": info.get("currency"),
        "market_cap": info.get("market_cap"),
        "fifty_two_week_high": info.get("year_high"),
        "fifty_two_week_low": info.get("year_low"),
    }


@tool
def get_stock_fundamentals(ticker: str) -> dict:
    """Fundamental metrics from yfinance: valuation, profitability, growth,
    balance sheet, cash flow, dividends, beta. Returns a flat dict."""
    with _silence_io():
        t = yf.Ticker(ticker)
        info = t.info or {}
    keys = [
        "longName", "sector", "industry", "marketCap",
        "trailingPE", "forwardPE", "priceToBook", "priceToSalesTrailing12Months",
        "enterpriseValue", "enterpriseToEbitda",
        "profitMargins", "operatingMargins", "grossMargins",
        "returnOnEquity", "returnOnAssets",
        "revenueGrowth", "earningsGrowth",
        "totalRevenue", "grossProfits", "ebitda",
        "totalCash", "totalDebt", "debtToEquity",
        "freeCashflow", "operatingCashflow",
        "dividendYield", "payoutRatio",
        "beta", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
    ]
    return {k: info.get(k) for k in keys}


@tool
def get_earnings_history(ticker: str) -> str:
    """Recent quarterly earnings: EPS estimate vs actual and surprise %."""
    with _silence_io():
        t = yf.Ticker(ticker)
        try:
            df = t.earnings_history
        except Exception as e:
            return f"Earnings history unavailable: {e}"
    if df is None or len(df) == 0:
        return "No earnings history available."
    return df.to_string()


@tool
def get_price_history(ticker: str, period: str = "6mo") -> dict:
    """OHLCV history with computed indicators (SMA20/50/200, RSI14, period range,
    volume profile). period: '1mo', '3mo', '6mo', '1y', '2y', '5y'."""
    with _silence_io():
        t = yf.Ticker(ticker)
        hist = t.history(period=period, auto_adjust=True)
    if hist.empty:
        return {"error": f"No price data for {ticker} over {period}."}

    closes = hist["Close"]
    vol = hist["Volume"]

    sma20 = closes.rolling(20).mean().iloc[-1] if len(closes) >= 20 else None
    sma50 = closes.rolling(50).mean().iloc[-1] if len(closes) >= 50 else None
    sma200 = closes.rolling(200).mean().iloc[-1] if len(closes) >= 200 else None

    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    rsi14 = (100 - (100 / (1 + rs))).iloc[-1]

    last_close = float(closes.iloc[-1])
    period_high = float(closes.max())
    period_low = float(closes.min())

    return {
        "ticker": ticker,
        "period": period,
        "last_close": last_close,
        "sma20": _safe_float(sma20),
        "sma50": _safe_float(sma50),
        "sma200": _safe_float(sma200),
        "rsi14": _safe_float(rsi14),
        "period_high": period_high,
        "period_low": period_low,
        "avg_volume_30d": float(vol.tail(30).mean()),
        "latest_volume": float(vol.iloc[-1]),
        "pct_off_period_high": (last_close / period_high - 1) * 100,
        "pct_above_period_low": (last_close / period_low - 1) * 100,
    }


@tool
def get_peer_comparison(ticker: str) -> str:
    """Use Tavily to identify sector peers and surface competitor commentary."""
    client = _tavily()
    q = (
        f"top competitors and peer companies for {ticker} stock with comparable "
        f"market cap, valuation, and growth metrics"
    )
    res = client.search(query=q, max_results=5, search_depth="advanced")
    items = res.get("results", [])
    if not items:
        return f"No peer info found for {ticker}."
    out = [f"Peer research for {ticker}:"]
    for i, it in enumerate(items, 1):
        out.append(
            f"[{i}] {it.get('title')}\n"
            f"    URL: {it.get('url')}\n"
            f"    {(it.get('content') or '')[:600]}"
        )
    return "\n".join(out)


# ---------- helpers ----------

def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        if f != f:  # NaN check
            return None
        return f
    except (TypeError, ValueError):
        return None


# ---------- Curated tool buckets per research agent ----------

FUNDAMENTAL_TOOLS = [get_stock_fundamentals, get_earnings_history, tavily_web_search]
SENTIMENT_TOOLS = [tavily_news_search, tavily_web_search]
TECHNICAL_TOOLS = [get_price_history, get_stock_quote]
MACRO_TOOLS = [get_peer_comparison, tavily_web_search]
