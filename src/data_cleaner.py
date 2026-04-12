from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


IV_PLACEHOLDER_THRESHOLD = 1e-4
MAX_REASONABLE_IV = 5.0
ARBITRAGE_EPSILON = 1e-8

QUALITY_MODE_CONFIG = {
    "strict": {
        "spread_ratio_max": 0.50,
        "allow_medium_surface": False,
        "min_volume": 10.0,
        "min_open_interest": 50.0,
    },
    "balanced": {
        "spread_ratio_max": 1.00,
        "allow_medium_surface": True,
        "min_volume": 5.0,
        "min_open_interest": 25.0,
    },
    "lenient": {
        "spread_ratio_max": 2.00,
        "allow_medium_surface": True,
        "min_volume": 1.0,
        "min_open_interest": 5.0,
    },
}


def _to_float(value) -> float:
    try:
        if value is None:
            return float("nan")
        converted = float(value)
        if np.isfinite(converted):
            return converted
        return float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _black_scholes_price(
    option_type: str,
    spot: float,
    strike: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Compute the Black-Scholes-Merton theoretical price for a European option."""
    if (
        time_to_expiration <= 0
        or spot <= 0
        or strike <= 0
        or option_type not in {"call", "put"}
    ):
        return float("nan")
    if volatility < 0:
        return float("nan")
    if volatility == 0:
        lower, _ = _no_arbitrage_bounds(
            option_type=option_type,
            spot=spot,
            strike=strike,
            time_to_expiration=time_to_expiration,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
        )
        return float(lower)

    sqrt_t = np.sqrt(time_to_expiration)
    d1 = (
        np.log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiration
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    discounted_spot = spot * np.exp(-dividend_yield * time_to_expiration)
    discounted_strike = strike * np.exp(-risk_free_rate * time_to_expiration)

    if option_type == "call":
        price = discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)
    else:
        price = discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1)

    return float(price)


def _no_arbitrage_bounds(
    option_type: str,
    spot: float,
    strike: float,
    time_to_expiration: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> Tuple[float, float]:
    discounted_spot = spot * np.exp(-dividend_yield * time_to_expiration)
    discounted_strike = strike * np.exp(-risk_free_rate * time_to_expiration)

    if option_type == "call":
        lower = max(0.0, discounted_spot - discounted_strike)
        upper = discounted_spot
    else:
        lower = max(0.0, discounted_strike - discounted_spot)
        upper = discounted_strike

    return float(lower), float(upper)


def _implied_volatility(
    option_type: str,
    spot: float,
    strike: float,
    time_to_expiration: float,
    risk_free_rate: float,
    market_price: float,
    dividend_yield: float = 0.0,
) -> float:
    """Solve for implied volatility using Black-Scholes-Merton and no-arbitrage bounds."""
    if any(val is None for val in (spot, strike, time_to_expiration, market_price)):
        return float("nan")
    if (
        option_type not in {"call", "put"}
        or time_to_expiration <= 0
        or market_price < 0
        or spot <= 0
        or strike <= 0
    ):
        return float("nan")

    lower, upper = _no_arbitrage_bounds(
        option_type, spot, strike, time_to_expiration, risk_free_rate, dividend_yield
    )
    if abs(market_price - lower) <= ARBITRAGE_EPSILON:
        return 0.0
    if market_price < lower - ARBITRAGE_EPSILON or market_price > upper + ARBITRAGE_EPSILON:
        return float("nan")

    def objective(vol: float) -> float:
        return (
            _black_scholes_price(
                option_type,
                spot,
                strike,
                time_to_expiration,
                risk_free_rate,
                vol,
                dividend_yield,
            )
            - market_price
        )

    try:
        return float(brentq(objective, 1e-6, MAX_REASONABLE_IV, maxiter=200, xtol=1e-8))
    except ValueError:
        return float("nan")


def _normalize_iv_source(iv_source: str) -> str:
    normalized = (iv_source or "auto").strip().lower()
    if normalized not in {"auto", "yfinance", "black-scholes"}:
        print(f"Warning: Invalid iv_source '{iv_source}'. Defaulting to 'auto'.")
        return "auto"
    return normalized


def _normalize_quality_mode(quality_mode: str) -> str:
    normalized = (quality_mode or "lenient").strip().lower()
    if normalized not in QUALITY_MODE_CONFIG:
        print(f"Warning: Invalid quality_mode '{quality_mode}'. Defaulting to 'lenient'.")
        return "lenient"
    return normalized


def _split_flags(flags: str) -> List[str]:
    if not flags or flags == "none":
        return []
    return [flag for flag in flags.split(";") if flag]


def _select_market_price(
    row: pd.Series,
    max_trade_age_hours: float,
    spread_ratio_max: float,
    now_utc: pd.Timestamp,
) -> Tuple[float, str, Set[str], float]:
    """Select market price source and attach quality flags."""
    flags: Set[str] = set()
    bid = _to_float(row.get("bid"))
    ask = _to_float(row.get("ask"))
    last_price = _to_float(row.get("lastPrice"))
    last_trade_date = row.get("lastTradeDate")

    has_two_sided_quote = (
        np.isfinite(bid)
        and np.isfinite(ask)
        and bid > 0
        and ask > 0
        and ask >= bid
    )

    spread_ratio = float("nan")
    if has_two_sided_quote:
        market_price = (bid + ask) / 2
        price_source = "mid"
        if market_price > 0:
            spread_ratio = (ask - bid) / market_price
            if np.isfinite(spread_ratio) and spread_ratio > spread_ratio_max:
                flags.add("wide_spread")
        return float(market_price), price_source, flags, spread_ratio

    flags.add("no_two_sided_quote")
    if np.isfinite(last_price) and last_price > 0:
        price_source = "lastPrice"
        market_price = last_price
        if pd.isna(last_trade_date):
            flags.add("stale_last_trade")
        else:
            age_hours = (now_utc - last_trade_date).total_seconds() / 3600.0
            if age_hours > max_trade_age_hours:
                flags.add("stale_last_trade")
        return float(market_price), price_source, flags, spread_ratio

    if np.isfinite(bid) and bid > 0:
        return float(bid), "bid", flags, spread_ratio
    if np.isfinite(ask) and ask > 0:
        return float(ask), "ask", flags, spread_ratio

    return float("nan"), "none", flags, spread_ratio


def _resolve_row_iv(
    row: pd.Series,
    iv_source_mode: str,
    quality_mode: str,
    underlying_price: float,
    risk_free_rate: float,
    dividend_yield: float,
    max_trade_age_hours: float,
    now_utc: pd.Timestamp,
) -> pd.Series:
    flags: Set[str] = set()
    quality_config = QUALITY_MODE_CONFIG[quality_mode]
    spread_ratio_max = quality_config["spread_ratio_max"]

    option_type = str(row.get("optionType", "")).lower()
    strike = _to_float(row.get("strike"))
    time_to_expiration_years = _to_float(row.get("time_to_expiration_years"))
    volume = _to_float(row.get("volume"))
    open_interest = _to_float(row.get("openInterest"))
    implied_vol_raw = _to_float(row.get("impliedVolatilityRaw"))

    if not np.isfinite(volume) or volume <= 0:
        flags.add("volume_zero_or_missing")
    elif volume < quality_config["min_volume"]:
        flags.add("low_volume")

    if not np.isfinite(open_interest) or open_interest <= 0:
        flags.add("oi_zero_or_missing")
    elif open_interest < quality_config["min_open_interest"]:
        flags.add("low_open_interest")

    provider_iv_valid = bool(
        np.isfinite(implied_vol_raw)
        and implied_vol_raw > IV_PLACEHOLDER_THRESHOLD
        and implied_vol_raw <= MAX_REASONABLE_IV
    )
    if np.isfinite(implied_vol_raw) and implied_vol_raw <= IV_PLACEHOLDER_THRESHOLD:
        flags.add("placeholder_iv")
    if np.isfinite(implied_vol_raw) and implied_vol_raw > MAX_REASONABLE_IV:
        flags.add("iv_outlier")

    market_price, price_source, price_flags, spread_ratio = _select_market_price(
        row,
        max_trade_age_hours=max_trade_age_hours,
        spread_ratio_max=spread_ratio_max,
        now_utc=now_utc,
    )
    flags.update(price_flags)

    bs_iv = float("nan")
    if (
        option_type in {"call", "put"}
        and np.isfinite(underlying_price)
        and underlying_price > 0
        and np.isfinite(strike)
        and strike > 0
        and np.isfinite(time_to_expiration_years)
        and time_to_expiration_years > 0
        and np.isfinite(market_price)
        and market_price > 0
    ):
        lower, upper = _no_arbitrage_bounds(
            option_type=option_type,
            spot=underlying_price,
            strike=strike,
            time_to_expiration=time_to_expiration_years,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
        )
        if market_price < lower - ARBITRAGE_EPSILON or market_price > upper + ARBITRAGE_EPSILON:
            flags.add("arbitrage_violation")
        else:
            bs_iv = _implied_volatility(
                option_type=option_type,
                spot=underlying_price,
                strike=strike,
                time_to_expiration=time_to_expiration_years,
                risk_free_rate=risk_free_rate,
                market_price=market_price,
                dividend_yield=dividend_yield,
            )
            if np.isfinite(bs_iv) and bs_iv > MAX_REASONABLE_IV:
                flags.add("iv_outlier")
                bs_iv = float("nan")

    quote_sanity_ok = (
        "no_two_sided_quote" not in flags
        and "wide_spread" not in flags
        and "arbitrage_violation" not in flags
    )

    implied_vol_final = float("nan")
    iv_source_used = "none"

    if iv_source_mode == "yfinance":
        if provider_iv_valid:
            implied_vol_final = implied_vol_raw
            iv_source_used = "yfinance"
    elif iv_source_mode == "black-scholes":
        if np.isfinite(bs_iv) and bs_iv > 0:
            implied_vol_final = bs_iv
            iv_source_used = "black-scholes"
    else:  # auto
        if provider_iv_valid and quote_sanity_ok:
            implied_vol_final = implied_vol_raw
            iv_source_used = "yfinance"
        elif np.isfinite(bs_iv) and bs_iv > 0:
            implied_vol_final = bs_iv
            iv_source_used = "black-scholes"
        elif provider_iv_valid:
            implied_vol_final = implied_vol_raw
            iv_source_used = "yfinance"
            flags.add("provider_used_without_quote_sanity")

    if not np.isfinite(implied_vol_final) or implied_vol_final <= 0:
        flags.add("iv_unavailable")
        implied_vol_final = float("nan")
        iv_source_used = "none"

    if np.isfinite(implied_vol_final) and implied_vol_final > MAX_REASONABLE_IV:
        flags.add("iv_outlier")
        flags.add("iv_unavailable")
        implied_vol_final = float("nan")
        iv_source_used = "none"

    confidence_level = "high"
    has_volume_issue = "volume_zero_or_missing" in flags or "low_volume" in flags
    has_open_interest_issue = "oi_zero_or_missing" in flags or "low_open_interest" in flags
    has_liquidity_issue = has_volume_issue or has_open_interest_issue
    if (
        "iv_unavailable" in flags
        or "arbitrage_violation" in flags
        or "stale_last_trade" in flags
        or (has_volume_issue and has_open_interest_issue)
        or ("no_two_sided_quote" in flags and has_liquidity_issue)
    ):
        confidence_level = "low"
    elif (
        iv_source_used == "black-scholes"
        or "no_two_sided_quote" in flags
        or "wide_spread" in flags
        or has_liquidity_issue
        or "provider_used_without_quote_sanity" in flags
    ):
        confidence_level = "medium"

    include_in_surface = bool(np.isfinite(implied_vol_final) and implied_vol_final > 0)
    if include_in_surface:
        if quality_mode == "strict":
            include_in_surface = confidence_level == "high"
        else:
            include_in_surface = confidence_level in {"high", "medium"}

    volume_score = 0.0
    if np.isfinite(volume) and volume > 0:
        volume_score = min(float(np.log1p(volume) / np.log1p(100.0)), 1.0)
    open_interest_score = 0.0
    if np.isfinite(open_interest) and open_interest > 0:
        open_interest_score = min(float(np.log1p(open_interest) / np.log1p(500.0)), 1.0)
    liquidity_score = 0.5 * volume_score + 0.5 * open_interest_score
    spread_penalty = 1.0
    if np.isfinite(spread_ratio) and spread_ratio > 0:
        spread_penalty = 1.0 / (1.0 + spread_ratio)
    confidence_multiplier = {"high": 1.0, "medium": 0.6, "low": 0.25}[confidence_level]
    surface_weight = max(0.05, confidence_multiplier * (0.5 + liquidity_score) * spread_penalty)

    return pd.Series(
        {
            "marketPrice": market_price,
            "priceSourceUsed": price_source,
            "spreadRatio": spread_ratio,
            "impliedVolatilityFinal": implied_vol_final,
            "ivSourceUsed": iv_source_used,
            "confidenceLevel": confidence_level,
            "includeInSurface": bool(include_in_surface),
            "surfaceWeight": float(surface_weight),
            "qualityFlags": ";".join(sorted(flags)) if flags else "none",
        }
    )


def prepare_options_data(
    df: pd.DataFrame,
    min_strike: float = None,
    max_strike: float = None,
    min_date: str = None,
    max_date: str = None,
    option_type_to_plot: str = "both",
    min_dte: int = None,
    max_dte: int = None,
    iv_source: str = "auto",
    underlying_price: float = None,
    risk_free_rate: float = 0.02,
    dividend_yield: float = 0.0,
    quality_mode: str = "lenient",
    max_trade_age_hours: float = 72.0,
) -> pd.DataFrame:
    """
    Clean and enrich options data for volatility-surface visualization.

    The returned frame retains low-confidence rows for diagnostics but marks
    surface-suitable rows via `includeInSurface`.
    """
    if df is None or df.empty:
        print("No data to clean")
        return pd.DataFrame()

    iv_source_mode = _normalize_iv_source(iv_source)
    quality_mode = _normalize_quality_mode(quality_mode)

    clean_df = df.copy()
    required_columns = [
        "strike",
        "expirationDate",
        "optionType",
        "impliedVolatility",
        "volume",
        "openInterest",
        "bid",
        "ask",
        "lastPrice",
        "lastTradeDate",
    ]
    for column in required_columns:
        if column not in clean_df.columns:
            clean_df[column] = np.nan

    if option_type_to_plot.lower() == "call":
        clean_df = clean_df[clean_df["optionType"] == "call"].copy()
        print(f"Filtering for CALL options. Rows remaining: {len(clean_df)}")
    elif option_type_to_plot.lower() == "put":
        clean_df = clean_df[clean_df["optionType"] == "put"].copy()
        print(f"Filtering for PUT options. Rows remaining: {len(clean_df)}")
    elif option_type_to_plot.lower() != "both":
        print(f"Warning: Invalid option_type_to_plot '{option_type_to_plot}'. Using 'both'.")

    if clean_df.empty:
        print(f"No data remaining after filtering for option type: {option_type_to_plot}")
        return pd.DataFrame()

    expiration_utc = pd.to_datetime(clean_df["expirationDate"], errors="coerce", utc=True)
    clean_df["expirationDate"] = expiration_utc.dt.tz_convert(None)
    clean_df["lastTradeDate"] = pd.to_datetime(
        clean_df["lastTradeDate"], utc=True, errors="coerce"
    )
    clean_df["strike"] = pd.to_numeric(clean_df["strike"], errors="coerce")
    clean_df["impliedVolatilityRaw"] = pd.to_numeric(
        clean_df["impliedVolatility"], errors="coerce"
    )
    clean_df["volume"] = pd.to_numeric(clean_df["volume"], errors="coerce")
    clean_df["openInterest"] = pd.to_numeric(clean_df["openInterest"], errors="coerce")
    clean_df["bid"] = pd.to_numeric(clean_df["bid"], errors="coerce")
    clean_df["ask"] = pd.to_numeric(clean_df["ask"], errors="coerce")
    clean_df["lastPrice"] = pd.to_numeric(clean_df["lastPrice"], errors="coerce")

    # DTE uses expiration close at ~4pm ET (21:00 UTC) for near-expiry stability.
    now_utc = pd.Timestamp.now(tz="UTC")
    expiration_close_utc = expiration_utc + pd.Timedelta(hours=21)
    remaining_days = (
        expiration_close_utc - now_utc
    ).dt.total_seconds() / (24.0 * 60.0 * 60.0)
    clean_df["time_to_expiration_years"] = remaining_days / 365.25
    clean_df["days_to_expiration"] = np.ceil(remaining_days).astype("Int64")

    clean_df = clean_df.replace([np.inf, -np.inf], np.nan)
    clean_df = clean_df.dropna(subset=["expirationDate", "strike", "days_to_expiration"])
    clean_df = clean_df[clean_df["strike"] > 0]
    clean_df = clean_df[clean_df["days_to_expiration"] > 0]
    clean_df = clean_df[clean_df["time_to_expiration_years"] > 0]

    if min_strike is not None:
        clean_df = clean_df[clean_df["strike"] >= min_strike]
    if max_strike is not None:
        clean_df = clean_df[clean_df["strike"] <= max_strike]

    if min_date is not None:
        clean_df = clean_df[clean_df["expirationDate"] >= pd.Timestamp(min_date)]
    if max_date is not None:
        clean_df = clean_df[clean_df["expirationDate"] <= pd.Timestamp(max_date)]

    if min_dte is not None:
        clean_df = clean_df[clean_df["days_to_expiration"] >= min_dte]
        print(f"Filtering for DTE >= {min_dte}. Rows remaining: {len(clean_df)}")

    if max_dte is not None:
        clean_df = clean_df[clean_df["days_to_expiration"] <= max_dte]
        print(f"Filtering for DTE <= {max_dte}. Rows remaining: {len(clean_df)}")

    if clean_df.empty:
        return pd.DataFrame()

    underlying_price = _to_float(underlying_price)
    if iv_source_mode in {"auto", "black-scholes"} and (
        not np.isfinite(underlying_price) or underlying_price <= 0
    ):
        if iv_source_mode == "black-scholes":
            raise ValueError(
                "underlying_price must be provided and positive when iv_source='black-scholes'."
            )
        print("Warning: Invalid underlying price for auto IV fallback. Black-Scholes fallback disabled.")

    resolved = clean_df.apply(
        lambda row: _resolve_row_iv(
            row=row,
            iv_source_mode=iv_source_mode,
            quality_mode=quality_mode,
            underlying_price=underlying_price,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            max_trade_age_hours=max_trade_age_hours,
            now_utc=now_utc,
        ),
        axis=1,
    )
    clean_df = pd.concat([clean_df, resolved], axis=1)
    clean_df["impliedVolatility"] = clean_df["impliedVolatilityFinal"]
    clean_df["ivComputationMethod"] = clean_df["ivSourceUsed"]

    output_columns = [
        "strike",
        "days_to_expiration",
        "time_to_expiration_years",
        "impliedVolatilityRaw",
        "impliedVolatilityFinal",
        "impliedVolatility",
        "expirationDate",
        "optionType",
        "volume",
        "openInterest",
        "bid",
        "ask",
        "lastPrice",
        "lastTradeDate",
        "marketPrice",
        "priceSourceUsed",
        "spreadRatio",
        "ivSourceUsed",
        "ivComputationMethod",
        "confidenceLevel",
        "qualityFlags",
        "includeInSurface",
        "surfaceWeight",
    ]

    for column in output_columns:
        if column not in clean_df.columns:
            clean_df[column] = np.nan

    final_df = clean_df[output_columns].copy()
    final_df["days_to_expiration"] = final_df["days_to_expiration"].astype(int)
    final_df = final_df.sort_values(
        ["days_to_expiration", "strike", "optionType"], ascending=[True, True, True]
    ).reset_index(drop=True)
    return final_df


def build_internal_validation_report(
    df: pd.DataFrame,
    underlying_price: float,
    risk_free_rate: float = 0.02,
    dividend_yield: float = 0.0,
) -> Dict[str, float]:
    """Build repricing diagnostics from final IV vs selected market price."""
    if df is None or df.empty:
        return {
            "rows_checked": 0,
            "rows_with_market_price": 0,
            "repricing_mae": None,
            "repricing_rmse": None,
            "repricing_p95_abs_error": None,
            "arbitrage_bound_violations": 0,
        }

    checks_df = df.copy()
    checks_df = checks_df.replace([np.inf, -np.inf], np.nan)
    checks_df = checks_df.dropna(
        subset=["impliedVolatilityFinal", "time_to_expiration_years", "strike", "optionType"]
    )
    checks_df = checks_df[
        (checks_df["impliedVolatilityFinal"] > 0)
        & (checks_df["time_to_expiration_years"] > 0)
        & (checks_df["strike"] > 0)
    ]

    if checks_df.empty:
        return {
            "rows_checked": 0,
            "rows_with_market_price": 0,
            "repricing_mae": None,
            "repricing_rmse": None,
            "repricing_p95_abs_error": None,
            "arbitrage_bound_violations": 0,
        }

    model_prices: List[float] = []
    market_prices: List[float] = []
    arbitrage_violations = 0
    for _, row in checks_df.iterrows():
        option_type = str(row["optionType"]).lower()
        strike = _to_float(row["strike"])
        t = _to_float(row["time_to_expiration_years"])
        market_price = _to_float(row.get("marketPrice"))
        iv = _to_float(row["impliedVolatilityFinal"])
        if option_type not in {"call", "put"}:
            continue

        if np.isfinite(market_price) and market_price > 0:
            lower, upper = _no_arbitrage_bounds(
                option_type,
                underlying_price,
                strike,
                t,
                risk_free_rate,
                dividend_yield,
            )
            if market_price < lower - ARBITRAGE_EPSILON or market_price > upper + ARBITRAGE_EPSILON:
                arbitrage_violations += 1

        model_price = _black_scholes_price(
            option_type=option_type,
            spot=underlying_price,
            strike=strike,
            time_to_expiration=t,
            risk_free_rate=risk_free_rate,
            volatility=iv,
            dividend_yield=dividend_yield,
        )
        if np.isfinite(market_price) and market_price > 0 and np.isfinite(model_price):
            market_prices.append(float(market_price))
            model_prices.append(float(model_price))

    if not market_prices:
        return {
            "rows_checked": int(len(checks_df)),
            "rows_with_market_price": 0,
            "repricing_mae": None,
            "repricing_rmse": None,
            "repricing_p95_abs_error": None,
            "arbitrage_bound_violations": int(arbitrage_violations),
        }

    error = np.array(model_prices) - np.array(market_prices)
    abs_error = np.abs(error)
    return {
        "rows_checked": int(len(checks_df)),
        "rows_with_market_price": int(len(market_prices)),
        "repricing_mae": float(abs_error.mean()),
        "repricing_rmse": float(np.sqrt(np.mean(error**2))),
        "repricing_p95_abs_error": float(np.quantile(abs_error, 0.95)),
        "arbitrage_bound_violations": int(arbitrage_violations),
    }


def build_diagnostics_report(
    df: pd.DataFrame,
    raw_row_count: int = None,
) -> Dict[str, object]:
    """Summarize quality, fallback usage, and exclusions."""
    if df is None or df.empty:
        return {
            "raw_row_count": int(raw_row_count) if raw_row_count is not None else None,
            "rows_retained": 0,
            "rows_surface_included": 0,
            "rows_surface_excluded": 0,
            "fallback_iv_fraction": 0.0,
            "iv_source_counts": {},
            "confidence_counts": {},
            "flag_counts": {},
            "dropped_rows_by_reason": {},
            "per_dte_quality_summary": [],
        }

    retained_df = df.copy()
    included_mask = retained_df["includeInSurface"].fillna(False).astype(bool)
    included_df = retained_df[included_mask]
    finite_iv_mask = retained_df["impliedVolatilityFinal"].notna()

    iv_source_counts = (
        retained_df["ivSourceUsed"].fillna("none").value_counts().sort_index().to_dict()
    )
    fallback_count = int((retained_df["ivSourceUsed"] == "black-scholes").sum())
    fallback_denom = max(int(finite_iv_mask.sum()), 1)
    fallback_iv_fraction = fallback_count / fallback_denom

    confidence_counts = (
        retained_df["confidenceLevel"].fillna("unknown").value_counts().sort_index().to_dict()
    )

    flag_counts: Dict[str, int] = {}
    excluded_reason_counts: Dict[str, int] = {}
    excluded_df = retained_df[~included_mask]
    for flag_blob in retained_df["qualityFlags"].fillna("none"):
        for flag in _split_flags(flag_blob):
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    for flag_blob in excluded_df["qualityFlags"].fillna("none"):
        for flag in _split_flags(flag_blob):
            excluded_reason_counts[flag] = excluded_reason_counts.get(flag, 0) + 1

    per_dte_summary = []
    grouped = retained_df.groupby("days_to_expiration", dropna=True)
    for dte, group in grouped:
        included_group = group[group["includeInSurface"].fillna(False).astype(bool)]
        per_dte_summary.append(
            {
                "days_to_expiration": int(dte),
                "rows": int(len(group)),
                "included_rows": int(len(included_group)),
                "high_confidence_rows": int((group["confidenceLevel"] == "high").sum()),
                "medium_confidence_rows": int((group["confidenceLevel"] == "medium").sum()),
                "low_confidence_rows": int((group["confidenceLevel"] == "low").sum()),
                "fallback_rows": int((group["ivSourceUsed"] == "black-scholes").sum()),
            }
        )
    per_dte_summary.sort(key=lambda item: item["days_to_expiration"])

    return {
        "raw_row_count": int(raw_row_count) if raw_row_count is not None else None,
        "rows_retained": int(len(retained_df)),
        "rows_surface_included": int(len(included_df)),
        "rows_surface_excluded": int(len(retained_df) - len(included_df)),
        "fallback_iv_fraction": float(fallback_iv_fraction),
        "iv_source_counts": {str(k): int(v) for k, v in iv_source_counts.items()},
        "confidence_counts": {str(k): int(v) for k, v in confidence_counts.items()},
        "flag_counts": {str(k): int(v) for k, v in sorted(flag_counts.items())},
        "dropped_rows_by_reason": {
            str(k): int(v) for k, v in sorted(excluded_reason_counts.items())
        },
        "per_dte_quality_summary": per_dte_summary,
    }
