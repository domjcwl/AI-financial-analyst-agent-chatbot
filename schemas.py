from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# Sentinel used as the "ticker" key for findings that are not tied to a single
# stock (e.g. "what stocks will IPO this year?"). Kept as a plain string so it
# round-trips cleanly through OpenAI structured output and the checkpointer.
MARKET_KEY = "MARKET"


# ---------- Planning ----------

class Holding(BaseModel):
    """A single portfolio holding (used when the user gives quantities)."""
    ticker: str
    quantity: float = Field(description="Number of shares or units")


class ResearchPlan(BaseModel):
    """Decomposed research plan produced by the Query Planner.

    `query_type` controls which research agents the dispatcher fans out to:
      - "single_ticker" : one ticker, all four dimensions
      - "portfolio"     : multiple tickers, all four dimensions for each
      - "general"       : no specific ticker — only sentiment + macro run
    """
    query_type: Literal["single_ticker", "portfolio", "general"]
    tickers: List[str] = Field(
        default_factory=list,
        description=(
            "Stock tickers to research. Populate for single_ticker / portfolio "
            "queries. Leave EMPTY for general queries with no specific company."
        ),
    )
    holdings: List[Holding] = Field(
        default_factory=list,
        description=(
            "Portfolio holdings with quantities, only when the user explicitly "
            "states positions. Empty otherwise."
        ),
    )
    topic: Optional[str] = Field(
        default=None,
        description=(
            "Subject of a general (ticker-less) query, e.g. 'upcoming IPOs', "
            "'AI sector outlook'. Required when query_type == 'general'."
        ),
    )
    fundamentals_questions: List[str] = Field(
        default_factory=list,
        description="3-5 fundamentals questions. EMPTY when query_type == 'general'.",
    )
    sentiment_questions: List[str] = Field(
        default_factory=list,
        description="3-5 news & sentiment questions.",
    )
    technical_questions: List[str] = Field(
        default_factory=list,
        description="3-5 technical analysis questions. EMPTY when query_type == 'general'.",
    )
    macro_questions: List[str] = Field(
        default_factory=list,
        description="3-5 macro / sector questions.",
    )
    investor_horizon: Literal["short_term", "medium_term", "long_term"] = Field(
        default="medium_term",
        description="Time horizon implied by the user's query",
    )


# ---------- Per-dimension findings ----------

class KeyMetric(BaseModel):
    """A single fundamental metric. Typed name/value pair so that OpenAI's strict
    JSON-schema structured output accepts it (open-ended dicts are rejected)."""
    name: str = Field(description="Metric name, e.g. 'trailingPE', 'revenueGrowth'")
    value: str = Field(description="Metric value as a string (preserves units)")


class FundamentalsFindings(BaseModel):
    ticker: str
    revenue_trend: str = Field(description="Recent revenue growth trajectory")
    profitability: str = Field(description="Margin profile and earnings quality")
    valuation: str = Field(description="P/E, EV/EBITDA, FCF yield observations")
    balance_sheet: str = Field(description="Debt, cash, liquidity assessment")
    guidance: str = Field(description="Forward guidance / earnings outlook")
    key_metrics: List[KeyMetric] = Field(
        default_factory=list,
        description="Raw fundamental metrics as typed name/value pairs",
    )
    summary: str
    confidence: float = Field(ge=0, le=1, description="Self-rated confidence 0-1")
    sources: List[str] = Field(default_factory=list)


class SentimentFindings(BaseModel):
    ticker: str = Field(description="Ticker, or 'MARKET' for general / ticker-less queries")
    headline_summary: str
    analyst_consensus: str
    average_price_target: Optional[float] = None
    social_sentiment: Literal["bullish", "neutral", "bearish", "mixed"]
    notable_catalysts: List[str] = Field(default_factory=list)
    summary: str
    confidence: float = Field(ge=0, le=1)
    sources: List[str] = Field(default_factory=list)


class TechnicalFindings(BaseModel):
    ticker: str
    current_price: Optional[float] = None
    trend: Literal["uptrend", "downtrend", "sideways"]
    moving_averages: str = Field(description="Commentary on SMA20/50/200")
    rsi: Optional[float] = None
    volume_profile: str
    support_resistance: str
    momentum_signal: Literal["bullish", "bearish", "neutral"]
    summary: str
    confidence: float = Field(ge=0, le=1)


class MacroFindings(BaseModel):
    ticker: str = Field(description="Ticker, or 'MARKET' for general / ticker-less queries")
    sector: str
    industry_trends: str
    competitor_comparison: str
    macro_drivers: str = Field(description="Rates, FX, regulation, macro context")
    summary: str
    confidence: float = Field(ge=0, le=1)
    sources: List[str] = Field(default_factory=list)


# ---------- Critic / risk / thesis ----------

class CriticDecision(BaseModel):
    """Did the assembled evidence clear the bar to move to thesis-building?"""
    sufficient: bool
    rationale: str
    gaps: List[str] = Field(
        default_factory=list,
        description="Specific missing data or unanswered questions; empty if sufficient",
    )


class TickerRisk(BaseModel):
    """Per-position risk view, used inside RiskAssessment for portfolios."""
    ticker: str
    risk_rating: Literal["low", "moderate", "high", "very_high"]
    primary_concerns: List[str] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    overall_risk_rating: Literal["low", "moderate", "high", "very_high"]
    volatility_assessment: str
    drawdown_risk: str
    financial_leverage_risk: str
    business_concentration_risk: str
    macro_risk: str
    per_ticker_risks: List[TickerRisk] = Field(
        default_factory=list,
        description=(
            "One entry per ticker for portfolio / multi-ticker analyses. "
            "Empty for general (ticker-less) queries."
        ),
    )
    summary: str


class TickerRecommendation(BaseModel):
    """Per-ticker action inside an InvestmentThesis."""
    ticker: str
    recommendation: Literal["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
    conviction: Literal["low", "medium", "high"]
    rationale: str = Field(description="1-2 sentence justification")


class InvestmentThesis(BaseModel):
    primary_recommendation: Literal[
        "Strong Buy", "Buy", "Hold", "Sell", "Strong Sell", "Mixed", "N/A"
    ] = Field(
        description=(
            "Headline call. Use 'Mixed' for portfolios with diverging per-ticker "
            "recommendations; use 'N/A' for general (ticker-less) queries."
        ),
    )
    target_horizon: str
    per_ticker_recommendations: List[TickerRecommendation] = Field(
        default_factory=list,
        description=(
            "One entry per ticker analysed. Empty for general (ticker-less) queries."
        ),
    )
    bull_case: List[str]
    bear_case: List[str]
    key_risks: List[str]
    catalysts: List[str]
    overall_conviction: Literal["low", "medium", "high"]
    thesis_summary: str


# ---------- Human-in-the-loop ----------

class HumanReview(BaseModel):
    decision: Literal["approve", "revise"]
    feedback: Optional[str] = Field(
        default=None,
        description="Required when decision == 'revise'",
    )


# ---------- Final deliverable ----------

class FinalReport(BaseModel):
    headline: str
    primary_recommendation: str = Field(
        description="Headline recommendation; mirrors InvestmentThesis.primary_recommendation"
    )
    tickers_covered: List[str] = Field(default_factory=list)
    executive_summary: str
    detailed_thesis: str
    risk_disclosures: List[str]
    full_markdown: str = Field(
        description="Polished, investor-ready markdown report (the deliverable)"
    )
