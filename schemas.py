from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ---------- Planning ----------

class Holding(BaseModel):
    """A single portfolio holding (used when the user gives quantities)."""
    ticker: str
    quantity: float = Field(description="Number of shares or units")


class QueryItem(BaseModel):
    """One analysis unit inside a ResearchPlan.

    A single user query can decompose into multiple items. Routing per item:
      - stock_analysis    : runs fundamentals + sentiment + technical + macro
                            agents on the named ticker.
      - industry_analysis : runs ONLY sentiment + macro agents on the named
                            sector / industry (fundamentals and technical are
                            not meaningful at the industry level).
    """
    query_type: Literal["stock_analysis", "industry_analysis"]
    target: str = Field(
        description=(
            "What to research. For stock_analysis: the ticker symbol (e.g. "
            "'NVDA', 'GOOGL', 'XLE'). For industry_analysis: a short canonical "
            "industry/sector label (e.g. 'technology', 'healthcare', 'energy')."
        ),
    )
    fundamentals_questions: List[str] = Field(
        default_factory=list,
        description="3-5 fundamentals questions. EMPTY for industry_analysis items.",
    )
    sentiment_questions: List[str] = Field(
        default_factory=list,
        description="3-5 news & sentiment questions.",
    )
    technical_questions: List[str] = Field(
        default_factory=list,
        description="3-5 technical analysis questions. EMPTY for industry_analysis items.",
    )
    macro_questions: List[str] = Field(
        default_factory=list,
        description="3-5 macro / sector questions.",
    )


class ResearchPlan(BaseModel):
    """Decomposed research plan produced by the Query Planner.

    A query can decompose into MULTIPLE items. Examples:
      - 'Analyse my portfolio of NVDA, META, JPM, XLE'
            -> 4 stock_analysis items
      - 'Analyse the tech and healthcare industries'
            -> 2 industry_analysis items
      - 'Analyse the tech industry and tell me if Google is a good buy'
            -> 1 industry_analysis (technology) + 1 stock_analysis (GOOGL)
    """
    items: List[QueryItem] = Field(
        min_length=1,
        description="One or more analysis items, each routed independently.",
    )
    holdings: List[Holding] = Field(
        default_factory=list,
        description=(
            "Portfolio holdings with quantities, only when the user explicitly "
            "states positions (e.g. '3 VOO, 4 GOOGL'). Empty otherwise."
        ),
    )
    investor_horizon: Literal["short_term", "medium_term", "long_term"] = Field(
        default="medium_term",
        description="Time horizon implied by the user's query",
    )

    # --- convenience accessors ---

    def stock_targets(self) -> List[str]:
        return [i.target for i in self.items if i.query_type == "stock_analysis"]

    def industry_targets(self) -> List[str]:
        return [i.target for i in self.items if i.query_type == "industry_analysis"]

    def all_targets(self) -> List[str]:
        return [i.target for i in self.items]

    def has_stock_items(self) -> bool:
        return any(i.query_type == "stock_analysis" for i in self.items)

    def has_industry_items(self) -> bool:
        return any(i.query_type == "industry_analysis" for i in self.items)


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
    subject: str = Field(
        description="Ticker symbol (stock_analysis) or industry name (industry_analysis)."
    )
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
    subject: str = Field(
        description="Ticker symbol (stock_analysis) or industry name (industry_analysis)."
    )
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
            "Specific item targets (tickers or industry names) needing more "
            "research. Empty = re-research all items."
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


class IndustryOutlook(BaseModel):
    """Per-industry synthesis: future outlook, risk, recommendation."""
    industry: str
    future_outlook: str = Field(
        description="Forward-looking narrative for the industry — drivers, trajectory."
    )
    risk_rating: Literal["low", "moderate", "high", "very_high"]
    risk_assessment: str = Field(
        description="Narrative risk view — macro, regulatory, cyclical, competitive risks."
    )
    recommendation: Literal["Overweight", "Neutral", "Underweight"]
    conviction: Literal["low", "medium", "high"]
    rationale: str = Field(description="1-2 sentence justification for the recommendation.")


class InvestmentThesis(BaseModel):
    """Integrated thesis: per-stock and per-industry outlooks plus a headline summary."""
    thesis_summary: str = Field(
        description=(
            "Top-level narrative integrating all analyses. For mixed plans, "
            "explain how the industry backdrop shapes each stock call."
        ),
    )
    overall_conviction: Literal["low", "medium", "high"]
    stock_outlooks: List[StockOutlook] = Field(
        default_factory=list,
        description="One entry per stock_analysis item. Empty for industry-only plans.",
    )
    industry_outlooks: List[IndustryOutlook] = Field(
        default_factory=list,
        description="One entry per industry_analysis item. Empty for stock-only plans.",
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
            "Concise headline phrase summarising the calls across all outlooks "
            "(e.g. 'Buy NVDA, Hold META, Overweight Technology'). Free-form."
        ),
    )
    tickers_covered: List[str] = Field(
        default_factory=list,
        description="Tickers from stock_analysis items.",
    )
    industries_covered: List[str] = Field(
        default_factory=list,
        description="Industries from industry_analysis items.",
    )
    executive_summary: str
    detailed_thesis: str
    risk_disclosures: List[str]
    full_markdown: str = Field(
        description="Polished, investor-ready markdown report (the deliverable)"
    )
