from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ---------- Planning ----------

class QueryItem(BaseModel):
    """One analysis unit inside a ResearchPlan.

    The agent supports a SINGLE query type — stock_analysis — which runs the
    fundamentals + sentiment + technical + macro agents on one ticker.
    """
    query_type: Literal["stock_analysis"] = "stock_analysis"
    target: str = Field(
        description="The ticker symbol to research (e.g. 'NVDA', 'GOOGL', 'JPM').",
    )
    fundamentals_questions: List[str] = Field(
        default_factory=list,
        description="3-5 fundamentals questions.",
    )
    sentiment_questions: List[str] = Field(
        default_factory=list,
        description="3-5 news & sentiment questions.",
    )
    technical_questions: List[str] = Field(
        default_factory=list,
        description="3-5 technical analysis questions.",
    )
    macro_questions: List[str] = Field(
        default_factory=list,
        description="3-5 macro / sector questions.",
    )


class ResearchPlan(BaseModel):
    """Research plan produced by the Query Planner.

    The agent analyses exactly ONE stock per query, so the plan always holds a
    single stock_analysis item, e.g. 'NVDA' -> 1 stock_analysis item.
    """
    items: List[QueryItem] = Field(
        min_length=1,
        max_length=1,
        description="Exactly one stock_analysis item.",
    )
    investor_horizon: Literal["short_term", "medium_term", "long_term"] = Field(
        default="medium_term",
        description="Time horizon implied by the user's query",
    )

    # --- convenience accessors ---

    def stock_targets(self) -> List[str]:
        return [i.target for i in self.items if i.query_type == "stock_analysis"]

    def all_targets(self) -> List[str]:
        return [i.target for i in self.items]

    def has_stock_items(self) -> bool:
        return any(i.query_type == "stock_analysis" for i in self.items)


# ---------- Per-dimension findings ----------

class KeyMetric(BaseModel):
    """A single fundamental metric. Typed name/value pair so that OpenAI's strict
    JSON-schema structured output accepts it (open-ended dicts are rejected)."""
    name: str = Field(description="Metric name, e.g. 'trailingPE', 'revenueGrowth'")
    value: str = Field(description="Metric value as a string (preserves units)")


class FundamentalsFindings(BaseModel):
    ticker: str
    revenue_trend: str = Field(description="Recent revenue growth trajectory with specific YoY/QoQ %")
    profitability: str = Field(description="Margin profile and earnings quality with specific margin %")
    valuation: str = Field(description="P/E, EV/EBITDA, FCF yield with concrete multiples")
    balance_sheet: str = Field(description="Debt, cash, liquidity with specific $ figures")
    guidance: str = Field(description="Forward guidance / earnings outlook")
    key_metrics: List[KeyMetric] = Field(
        default_factory=list,
        description="Raw fundamental metrics as typed name/value pairs",
    )
    summary: str
    confidence: float = Field(
        ge=0, le=1,
        description=(
            "Calibrated confidence 0-1. Use >=0.8 ONLY when 3+ sources, all plan "
            "questions addressed, and numerical specifics present. Use <0.5 when "
            "data is sparse, contradictory, or stale."
        ),
    )
    sources: List[str] = Field(default_factory=list, description="Source URLs, deduplicated")
    unanswered_questions: List[str] = Field(
        default_factory=list,
        description="Plan questions you could not address; empty if all answered.",
    )


class SentimentFindings(BaseModel):
    subject: str = Field(description="Ticker symbol under analysis.")
    headline_summary: str
    analyst_consensus: str
    average_price_target: Optional[float] = None
    social_sentiment: Literal["bullish", "neutral", "bearish", "mixed"]
    notable_catalysts: List[str] = Field(default_factory=list)
    summary: str
    confidence: float = Field(
        ge=0, le=1,
        description=(
            "Calibrated 0-1. >=0.8 needs 3+ recent sources (within ~14 days) and "
            "all plan questions addressed; <0.5 when news is sparse or stale."
        ),
    )
    sources: List[str] = Field(default_factory=list, description="Source URLs, deduplicated")
    unanswered_questions: List[str] = Field(
        default_factory=list,
        description="Plan questions you could not address; empty if all answered.",
    )


class TechnicalFindings(BaseModel):
    ticker: str
    current_price: Optional[float] = None
    trend: Literal["uptrend", "downtrend", "sideways"]
    moving_averages: str = Field(description="Commentary on SMA20/50/200 with explicit values")
    rsi: Optional[float] = None
    volume_profile: str = Field(description="Volume vs 30d avg, with explicit ratios")
    support_resistance: str = Field(description="Specific $ levels for support and resistance")
    momentum_signal: Literal["bullish", "bearish", "neutral"]
    summary: str
    confidence: float = Field(
        ge=0, le=1,
        description="Calibrated 0-1. >=0.8 only when price + indicators all retrieved.",
    )
    unanswered_questions: List[str] = Field(
        default_factory=list,
        description="Plan questions you could not address; empty if all answered.",
    )


class MacroFindings(BaseModel):
    subject: str = Field(description="Ticker symbol under analysis.")
    sector: str
    industry_trends: str
    competitor_comparison: str
    macro_drivers: str = Field(description="Rates, FX, regulation, macro context")
    summary: str
    confidence: float = Field(
        ge=0, le=1,
        description="Calibrated 0-1. >=0.8 needs sector data + 2+ named peers + macro context.",
    )
    sources: List[str] = Field(default_factory=list, description="Source URLs, deduplicated")
    unanswered_questions: List[str] = Field(
        default_factory=list,
        description="Plan questions you could not address; empty if all answered.",
    )


# ---------- Critic / thesis ----------

class CriticDecision(BaseModel):
    """Did the assembled evidence clear the bar to move to thesis-building?"""
    sufficient: bool
    rationale: str
    gaps: List[str] = Field(
        default_factory=list,
        description="Specific missing data or unanswered questions; empty if sufficient",
    )
    gap_dimensions: List[Literal["fundamentals", "sentiment", "technical", "macro"]] = Field(
        default_factory=list,
        description=(
            "Dimensions needing more research on the next round. Drives gap-aware "
            "re-dispatch — only listed dimensions are re-run. Empty means all "
            "dimensions if sufficient=False (fallback)."
        ),
    )
    gap_targets: List[str] = Field(
        default_factory=list,
        description=(
            "Specific tickers needing more research. Empty = re-research all."
        ),
    )


class StockOutlook(BaseModel):
    """Per-stock synthesis: future outlook, bull/bear case, risk, recommendation."""
    ticker: str
    future_outlook: str = Field(
        description="Forward-looking narrative — where this stock is headed and why."
    )
    bull_case: List[str] = Field(
        description="2-4 specific bullish points, each backed by a fact from the findings."
    )
    bear_case: List[str] = Field(
        description="2-4 specific bearish points, each backed by a fact from the findings."
    )
    risk_rating: Literal["low", "moderate", "high", "very_high"]
    risk_assessment: str = Field(
        description=(
            "Narrative risk view — volatility, drawdown exposure, leverage, "
            "business concentration, idiosyncratic risks."
        ),
    )
    recommendation: Literal["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
    conviction: Literal["low", "medium", "high"]
    rationale: str = Field(description="1-2 sentence justification for the recommendation.")


class InvestmentThesis(BaseModel):
    """Integrated thesis: the stock outlook plus a headline summary."""
    thesis_summary: str = Field(
        description="Top-level narrative integrating all dimensions of the analysis.",
    )
    overall_conviction: Literal["low", "medium", "high"]
    stock_outlooks: List[StockOutlook] = Field(
        default_factory=list,
        description="One entry for the analysed stock.",
    )


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
        description=(
            "Concise headline phrase summarising the call "
            "(e.g. 'Buy NVDA'). Free-form."
        ),
    )
    tickers_covered: List[str] = Field(
        default_factory=list,
        description="The analysed ticker.",
    )
    executive_summary: str
    detailed_thesis: str
    risk_disclosures: List[str]
    full_markdown: str = Field(
        description="Polished, investor-ready markdown report (the deliverable)"
    )
