from dataclasses import dataclass


VALID_QUALITY_MODES = {"strict", "balanced", "lenient"}
MIN_STRIKE_PCT = 0.50
MAX_STRIKE_PCT = 1.50
MIN_DTE = 0
MAX_DTE = 365
MIN_MAX_TRADE_AGE_HOURS = 1.0
MAX_MAX_TRADE_AGE_HOURS = 24.0 * 14.0
DEFAULT_WEB_DTE_MIN = 7


@dataclass(frozen=True)
class SurfaceRequest:
    ticker: str
    strike_min_pct: float = 0.93
    strike_max_pct: float = 1.07
    dte_min: int = 1
    dte_max: int = 60
    smooth: bool = True
    risk_free_rate: float = 0.02
    dividend_yield: float = 0.0
    quality_mode: str = "lenient"
    max_trade_age_hours: float = 120.0
    include_low_confidence: bool = False
