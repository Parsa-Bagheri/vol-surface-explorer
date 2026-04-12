from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from src.data_cleaner import (
    build_diagnostics_report,
    build_internal_validation_report,
    prepare_options_data,
)
from src.data_fetch import get_current_price, get_options_data
from src.visualizer import create_vol_surface


@dataclass(frozen=True)
class SurfaceRequest:
    ticker: str
    strike_min_pct: float = 0.93
    strike_max_pct: float = 1.07
    dte_min: int = 1
    dte_max: int = 60
    smooth: bool = True
    iv_source: str = "auto"
    risk_free_rate: float = 0.02
    dividend_yield: float = 0.0
    quality_mode: str = "lenient"
    max_trade_age_hours: float = 72.0
    include_low_confidence: bool = False


@dataclass
class SurfaceBuildResult:
    request: SurfaceRequest
    current_price: float
    raw_options_df: pd.DataFrame
    cleaned_options_df: pd.DataFrame
    diagnostics: Dict[str, Any]
    figure: Any


def _validated_request(request: SurfaceRequest) -> SurfaceRequest:
    ticker = (request.ticker or "").strip().upper()
    if not ticker:
        raise ValueError("A ticker symbol is required.")

    strike_min_pct = float(request.strike_min_pct)
    strike_max_pct = float(request.strike_max_pct)
    if strike_min_pct <= 0 or strike_max_pct <= 0:
        raise ValueError("Strike percentage bounds must be positive.")
    if strike_min_pct > strike_max_pct:
        strike_min_pct, strike_max_pct = strike_max_pct, strike_min_pct

    dte_min = int(request.dte_min)
    dte_max = int(request.dte_max)
    if dte_min < 1 or dte_max < 1:
        raise ValueError("DTE bounds must be at least 1 day.")
    if dte_min > dte_max:
        dte_min, dte_max = dte_max, dte_min

    return SurfaceRequest(
        ticker=ticker,
        strike_min_pct=strike_min_pct,
        strike_max_pct=strike_max_pct,
        dte_min=dte_min,
        dte_max=dte_max,
        smooth=bool(request.smooth),
        iv_source=request.iv_source,
        risk_free_rate=float(request.risk_free_rate),
        dividend_yield=float(request.dividend_yield),
        quality_mode=request.quality_mode,
        max_trade_age_hours=float(request.max_trade_age_hours),
        include_low_confidence=bool(request.include_low_confidence),
    )


def build_surface_bundle(request: SurfaceRequest) -> SurfaceBuildResult:
    validated_request = _validated_request(request)

    current_price = get_current_price(validated_request.ticker)
    if current_price is None:
        raise ValueError(
            f"Could not fetch the current price for {validated_request.ticker}."
        )

    min_strike_abs = current_price * validated_request.strike_min_pct
    max_strike_abs = current_price * validated_request.strike_max_pct

    raw_options_df = get_options_data(validated_request.ticker)
    if raw_options_df.empty:
        raise ValueError(
            f"No options data was returned for {validated_request.ticker}."
        )

    cleaned_options_df = prepare_options_data(
        raw_options_df,
        min_strike=min_strike_abs,
        max_strike=max_strike_abs,
        option_type_to_plot="both",
        min_dte=validated_request.dte_min,
        max_dte=validated_request.dte_max,
        iv_source=validated_request.iv_source,
        underlying_price=current_price,
        risk_free_rate=validated_request.risk_free_rate,
        dividend_yield=validated_request.dividend_yield,
        quality_mode=validated_request.quality_mode,
        max_trade_age_hours=validated_request.max_trade_age_hours,
    )

    if cleaned_options_df.empty:
        raise ValueError(
            "No suitable options remained after filtering. Adjust the strike or DTE range."
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
        "iv_source": validated_request.iv_source,
        "quality_mode": validated_request.quality_mode,
        "max_trade_age_hours": validated_request.max_trade_age_hours,
        "smooth": bool(validated_request.smooth),
        "include_low_confidence": bool(validated_request.include_low_confidence),
    }
    diagnostics["fetch"] = raw_options_df.attrs.get("fetchDiagnostics", {})
    diagnostics["internal_validation"] = build_internal_validation_report(
        cleaned_options_df,
        underlying_price=current_price,
        risk_free_rate=validated_request.risk_free_rate,
        dividend_yield=validated_request.dividend_yield,
    )

    figure = create_vol_surface(
        cleaned_options_df,
        validated_request.ticker,
        smooth=validated_request.smooth,
        include_low_confidence=validated_request.include_low_confidence,
        underlying_price=current_price,
        risk_free_rate=validated_request.risk_free_rate,
        dividend_yield=validated_request.dividend_yield,
    )

    return SurfaceBuildResult(
        request=validated_request,
        current_price=float(current_price),
        raw_options_df=raw_options_df,
        cleaned_options_df=cleaned_options_df,
        diagnostics=diagnostics,
        figure=figure,
    )
