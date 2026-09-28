from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict

import pandas as pd

from src.data_cleaner import (
    build_diagnostics_report,
    build_internal_validation_report,
    prepare_options_data,
)
from src.data_fetch import get_current_price, get_options_data
from src.time_utils import normalize_as_of_utc
from src.visualizer import create_vol_surface, select_surface_quotes
from src.ticker_validation import validate_ticker_symbol
from src.contract_outcomes import build_contract_outcomes
from src.surface_config import (
    MAX_DTE,
    MAX_MAX_TRADE_AGE_HOURS,
    MAX_STRIKE_PCT,
    MIN_DTE,
    MIN_MAX_TRADE_AGE_HOURS,
    MIN_STRIKE_PCT,
    VALID_QUALITY_MODES,
    SurfaceRequest,
)


@dataclass
class SurfaceBuildResult:
    request: SurfaceRequest
    current_price: float
    raw_options_df: pd.DataFrame
    cleaned_options_df: pd.DataFrame
    diagnostics: Dict[str, Any]
    figure: Any


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

    quality_mode = str(request.quality_mode or "").strip().lower()
    if quality_mode not in VALID_QUALITY_MODES:
        raise ValueError("Invalid quality mode.")

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
        quality_mode=quality_mode,
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

    raw_options_df = raw_options_df.copy()
    raw_options_df["contractId"] = range(len(raw_options_df))
    cleaned_options_df = prepare_options_data(
        raw_options_df,
        min_strike=min_strike_abs,
        max_strike=max_strike_abs,
        option_type_to_plot="both",
        min_dte=validated_request.dte_min,
        max_dte=validated_request.dte_max,
        underlying_price=current_price,
        risk_free_rate=validated_request.risk_free_rate,
        dividend_yield=validated_request.dividend_yield,
        quality_mode=validated_request.quality_mode,
        max_trade_age_hours=validated_request.max_trade_age_hours,
        as_of_utc=valuation_time_utc,
    )

    if cleaned_options_df.empty:
        raise ValueError(
            "No suitable options remained after filtering. Adjust the strike or DTE range."
        )

    surface_input = cleaned_options_df
    if not validated_request.include_low_confidence:
        surface_input = surface_input[
            surface_input["includeInSurface"].fillna(False).astype(bool)
        ]
    selected_quotes = select_surface_quotes(
        surface_input, current_price,
        validated_request.risk_free_rate, validated_request.dividend_yield,
    )

    diagnostics = build_diagnostics_report(
        cleaned_options_df,
        raw_row_count=len(raw_options_df),
    )
    diagnostics["request"] = {
        "ticker": validated_request.ticker,
        "strike_min_pct": validated_request.strike_min_pct,
        "strike_max_pct": validated_request.strike_max_pct,
        "dte_min": validated_request.dte_min,
        "dte_max": validated_request.dte_max,
        "quality_mode": validated_request.quality_mode,
        "max_trade_age_hours": validated_request.max_trade_age_hours,
        "smooth": bool(validated_request.smooth),
        "include_low_confidence": bool(validated_request.include_low_confidence),
        "valuation_time_utc": valuation_time_utc.isoformat(),
    }
    diagnostics["fetch"] = raw_options_df.attrs.get("fetchDiagnostics", {})
    diagnostics["contract_outcomes"] = build_contract_outcomes(
        raw_options_df, cleaned_options_df, validated_request, current_price,
        selected_quotes=selected_quotes,
    )
    diagnostics["internal_validation"] = build_internal_validation_report(
        cleaned_options_df,
        underlying_price=current_price,
        risk_free_rate=validated_request.risk_free_rate,
        dividend_yield=validated_request.dividend_yield,
        dte_range=(validated_request.dte_min, validated_request.dte_max),
    )

    figure = create_vol_surface(
        cleaned_options_df,
        validated_request.ticker,
        smooth=validated_request.smooth,
        include_low_confidence=validated_request.include_low_confidence,
        underlying_price=current_price,
        risk_free_rate=validated_request.risk_free_rate,
        dividend_yield=validated_request.dividend_yield,
        strike_range=(min_strike_abs, max_strike_abs),
        dte_range=(validated_request.dte_min, validated_request.dte_max),
        selected_quotes=selected_quotes,
    )

    return SurfaceBuildResult(
        request=validated_request,
        current_price=float(current_price),
        raw_options_df=raw_options_df,
        cleaned_options_df=cleaned_options_df,
        diagnostics=diagnostics,
        figure=figure,
    )
