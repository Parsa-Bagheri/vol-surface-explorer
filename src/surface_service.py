from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Dict

import pandas as pd

from src.data_cleaner import (
    build_diagnostics_report,
    prepare_options_data,
)
from src.data_fetch import get_current_price, get_options_data
from src.time_utils import normalize_as_of_utc
from src.visualizer import create_vol_surface


TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,11}$")
MIN_STRIKE_PCT = 0.50
MAX_STRIKE_PCT = 1.50
MIN_DTE = 0
MAX_DTE = 365
MIN_MAX_TRADE_AGE_HOURS = 1.0
MAX_MAX_TRADE_AGE_HOURS = 24.0 * 14.0
DEFAULT_STRIKE_MIN_PCT = 0.90
DEFAULT_STRIKE_MAX_PCT = 1.10
DEFAULT_DTE_MIN = 0
DEFAULT_DTE_MAX = 60
DEFAULT_RISK_FREE_RATE = 0.02
DEFAULT_DIVIDEND_YIELD = 0.0
DEFAULT_MAX_TRADE_AGE_HOURS = 120.0


@dataclass(frozen=True)
class SurfaceRequest:
    ticker: str
    strike_min_pct: float = DEFAULT_STRIKE_MIN_PCT
    strike_max_pct: float = DEFAULT_STRIKE_MAX_PCT
    dte_min: int = DEFAULT_DTE_MIN
    dte_max: int = DEFAULT_DTE_MAX
    smooth: bool = True
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
    dividend_yield: float = DEFAULT_DIVIDEND_YIELD
    max_trade_age_hours: float = DEFAULT_MAX_TRADE_AGE_HOURS
    include_low_confidence: bool = False


@dataclass
class SurfaceBuildResult:
    current_price: float
    cleaned_options_df: pd.DataFrame
    diagnostics: Dict[str, Any]
    figure: Any


def validate_ticker_symbol(ticker: str) -> str:
    normalized = (ticker or "").strip().upper()
    if not normalized:
        raise ValueError("A ticker symbol is required.")
    if not TICKER_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Ticker symbols may only contain letters, numbers, dots, and hyphens, "
            "and must be 12 characters or fewer."
        )
    return normalized


def _finite_float(name: str, value: float) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number.")
    return converted


def _validated_request(request: SurfaceRequest) -> SurfaceRequest:
    ticker = validate_ticker_symbol(request.ticker)

    strike_min_pct = _finite_float("strike_min_pct", request.strike_min_pct)
    strike_max_pct = _finite_float("strike_max_pct", request.strike_max_pct)
    if strike_min_pct <= 0 or strike_max_pct <= 0:
        raise ValueError("Strike percentage bounds must be positive.")
    if strike_min_pct > strike_max_pct:
        strike_min_pct, strike_max_pct = strike_max_pct, strike_min_pct
    if strike_min_pct < MIN_STRIKE_PCT or strike_max_pct > MAX_STRIKE_PCT:
        raise ValueError(
            f"Strike percentage bounds must stay between {MIN_STRIKE_PCT:.0%} "
            f"and {MAX_STRIKE_PCT:.0%} of spot."
        )

    dte_min = int(request.dte_min)
    dte_max = int(request.dte_max)
    if dte_min > dte_max:
        dte_min, dte_max = dte_max, dte_min
    if dte_min < MIN_DTE or dte_max > MAX_DTE:
        raise ValueError(f"DTE bounds must stay between {MIN_DTE} and {MAX_DTE} days.")

    risk_free_rate = _finite_float("risk_free_rate", request.risk_free_rate)
    dividend_yield = _finite_float("dividend_yield", request.dividend_yield)
    if not -1.0 <= risk_free_rate <= 1.0:
        raise ValueError("risk_free_rate must stay between -1.0 and 1.0.")
    if not -1.0 <= dividend_yield <= 1.0:
        raise ValueError("dividend_yield must stay between -1.0 and 1.0.")

    max_trade_age_hours = _finite_float(
        "max_trade_age_hours", request.max_trade_age_hours
    )
    if not MIN_MAX_TRADE_AGE_HOURS <= max_trade_age_hours <= MAX_MAX_TRADE_AGE_HOURS:
        raise ValueError(
            "max_trade_age_hours must stay between "
            f"{MIN_MAX_TRADE_AGE_HOURS:.0f} and {MAX_MAX_TRADE_AGE_HOURS:.0f}."
        )

    return SurfaceRequest(
        ticker=ticker,
        strike_min_pct=strike_min_pct,
        strike_max_pct=strike_max_pct,
        dte_min=dte_min,
        dte_max=dte_max,
        smooth=bool(request.smooth),
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        max_trade_age_hours=max_trade_age_hours,
        include_low_confidence=bool(request.include_low_confidence),
    )


def build_surface_bundle(request: SurfaceRequest) -> SurfaceBuildResult:
    validated_request = _validated_request(request)
    valuation_time_utc = normalize_as_of_utc()

    current_price = get_current_price(validated_request.ticker)
    if current_price is None:
        raise ValueError(
            f"Could not fetch the current price for {validated_request.ticker}."
        )

    min_strike_abs = current_price * validated_request.strike_min_pct
    max_strike_abs = current_price * validated_request.strike_max_pct

    raw_options_df = get_options_data(
        validated_request.ticker,
        min_dte=validated_request.dte_min,
        max_dte=validated_request.dte_max,
        as_of_utc=valuation_time_utc,
    )
    if raw_options_df.empty:
        raise ValueError(
            f"No options data was returned for {validated_request.ticker}."
        )

    cleaned_options_df = prepare_options_data(
        raw_options_df,
        min_strike=min_strike_abs,
        max_strike=max_strike_abs,
        min_dte=validated_request.dte_min,
        max_dte=validated_request.dte_max,
        underlying_price=current_price,
        risk_free_rate=validated_request.risk_free_rate,
        dividend_yield=validated_request.dividend_yield,
        max_trade_age_hours=validated_request.max_trade_age_hours,
        as_of_utc=valuation_time_utc,
    )

    if cleaned_options_df.empty:
        raise ValueError(
            "No suitable options remained after filtering. Adjust the strike or DTE range."
        )

    diagnostics = build_diagnostics_report(cleaned_options_df)
    diagnostics["request"] = asdict(validated_request)
    diagnostics["request"]["valuation_time_utc"] = valuation_time_utc.isoformat()
    figure = create_vol_surface(
        cleaned_options_df,
        smooth=validated_request.smooth,
        include_low_confidence=validated_request.include_low_confidence,
        underlying_price=current_price,
        risk_free_rate=validated_request.risk_free_rate,
        strike_range=(min_strike_abs, max_strike_abs),
        dte_range=(validated_request.dte_min, validated_request.dte_max),
    )

    return SurfaceBuildResult(
        current_price=float(current_price),
        cleaned_options_df=cleaned_options_df,
        diagnostics=diagnostics,
        figure=figure,
    )
