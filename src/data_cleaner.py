from __future__ import annotations

import math
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr

from src.time_utils import expiration_close_utc as option_expiration_close_utc
from src.time_utils import normalize_as_of_utc


MAX_REASONABLE_IV = 5.0
ARBITRAGE_EPSILON = 1e-8
MIN_FORWARD_PAIR_COUNT = 3
MAX_FORWARD_ABS_LOG_MONEYNESS = 0.15
MIN_REASONABLE_IMPLIED_DIVIDEND_YIELD = -0.25
MAX_REASONABLE_IMPLIED_DIVIDEND_YIELD = 0.25

QUALITY_CONFIG = {
    "spread_ratio_max": 2.00,
    "recompute_spread_ratio_max": 0.50,
    "iv_bid_ask_width_max": 0.10,
    "iv_bid_ask_width_ratio_max": 0.75,
    "min_volume": 1.0,
    "min_open_interest": 5.0,
}

PRIMARY_EXCLUSION_REASON_PRIORITY = (
    "arbitrage_violation",
    "stale_last_trade",
    "last_trade_outside_quote",
    "wide_recompute_spread",
    "wide_iv_bid_ask",
    "iv_bid_ask_width_unavailable",
    "one_sided_quote_price",
    "no_two_sided_quote",
    "recomputed_iv_unreliable",
    "iv_unavailable",
)


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


def _black_scholes_price_from_terms(
    option_type: str,
    log_spot_over_strike: float,
    discounted_spot: float,
    discounted_strike: float,
    time_to_expiration: float,
    sqrt_t: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
) -> float:
    vol_sqrt_t = volatility * sqrt_t
    d1 = (
        log_spot_over_strike
        + (risk_free_rate - dividend_yield + 0.5 * volatility * volatility)
        * time_to_expiration
    ) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    if option_type == "call":
        return float(discounted_spot * ndtr(d1) - discounted_strike * ndtr(d2))
    return float(discounted_strike * ndtr(-d2) - discounted_spot * ndtr(-d1))


def _black_scholes_prices_array(
    option_types: np.ndarray | str,
    spot: float,
    strikes: np.ndarray,
    time_to_expiration: np.ndarray | float,
    risk_free_rate: float,
    volatility: np.ndarray | float,
    dividend_yield: np.ndarray | float,
) -> np.ndarray:
    strikes = np.asarray(strikes, dtype=float)
    prices = np.full(strikes.shape, np.nan, dtype=float)
    if spot <= 0:
        return prices

    try:
        option_types = np.broadcast_to(
            np.asarray(option_types, dtype=object), strikes.shape
        )
        time_to_expiration = np.broadcast_to(
            np.asarray(time_to_expiration, dtype=float), strikes.shape
        )
        volatility = np.broadcast_to(
            np.asarray(volatility, dtype=float), strikes.shape
        )
        dividend_yield = np.broadcast_to(
            np.asarray(dividend_yield, dtype=float), strikes.shape
        )
    except ValueError:
        return prices

    is_call = option_types == "call"
    is_put = option_types == "put"
    valid = (
        (is_call | is_put)
        & np.isfinite(strikes)
        & (strikes > 0)
        & np.isfinite(time_to_expiration)
        & (time_to_expiration > 0)
        & np.isfinite(volatility)
        & (volatility >= 0)
        & np.isfinite(dividend_yield)
    )
    if not np.any(valid):
        return prices

    discounted_spot = spot * np.exp(-dividend_yield * time_to_expiration)
    discounted_strike = strikes * np.exp(-risk_free_rate * time_to_expiration)

    zero_volatility = valid & (volatility == 0)
    if np.any(zero_volatility):
        call_zero = zero_volatility & is_call
        put_zero = zero_volatility & is_put
        prices[call_zero] = np.maximum(
            0.0,
            discounted_spot[call_zero] - discounted_strike[call_zero],
        )
        prices[put_zero] = np.maximum(
            0.0,
            discounted_strike[put_zero] - discounted_spot[put_zero],
        )

    positive_volatility = valid & (volatility > 0)
    if np.any(positive_volatility):
        sqrt_t = np.sqrt(time_to_expiration[positive_volatility])
        vol_sqrt_t = volatility[positive_volatility] * sqrt_t
        d1 = (
            np.log(spot / strikes[positive_volatility])
            + (
                risk_free_rate
                - dividend_yield[positive_volatility]
                + 0.5 * volatility[positive_volatility] ** 2
            )
            * time_to_expiration[positive_volatility]
        ) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
        positive_prices = np.empty(np.count_nonzero(positive_volatility), dtype=float)
        positive_is_call = is_call[positive_volatility]
        discounted_spot_positive = discounted_spot[positive_volatility]
        discounted_strike_positive = discounted_strike[positive_volatility]
        positive_prices[positive_is_call] = (
            discounted_spot_positive[positive_is_call] * ndtr(d1[positive_is_call])
            - discounted_strike_positive[positive_is_call] * ndtr(d2[positive_is_call])
        )
        positive_prices[~positive_is_call] = (
            discounted_strike_positive[~positive_is_call] * ndtr(-d2[~positive_is_call])
            - discounted_spot_positive[~positive_is_call] * ndtr(-d1[~positive_is_call])
        )
        prices[positive_volatility] = positive_prices

    return prices


def _no_arbitrage_bounds(
    option_type: str,
    spot: float,
    strike: float,
    time_to_expiration: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> Tuple[float, float]:
    discounted_spot = spot * math.exp(-dividend_yield * time_to_expiration)
    discounted_strike = strike * math.exp(-risk_free_rate * time_to_expiration)

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

    log_spot_over_strike = math.log(spot / strike)
    discounted_spot = spot * math.exp(-dividend_yield * time_to_expiration)
    discounted_strike = strike * math.exp(-risk_free_rate * time_to_expiration)
    sqrt_t = math.sqrt(time_to_expiration)

    def objective(vol: float) -> float:
        return (
            _black_scholes_price_from_terms(
                option_type=option_type,
                log_spot_over_strike=log_spot_over_strike,
                discounted_spot=discounted_spot,
                discounted_strike=discounted_strike,
                time_to_expiration=time_to_expiration,
                sqrt_t=sqrt_t,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                volatility=vol,
            )
            - market_price
        )

    try:
        return float(brentq(objective, 1e-6, MAX_REASONABLE_IV, maxiter=200, xtol=1e-8))
    except ValueError:
        return float("nan")


def _split_flags(flags: str) -> List[str]:
    if not flags or flags == "none":
        return []
    return [flag for flag in flags.split(";") if flag]


def _primary_exclusion_reason(row: pd.Series) -> str:
    flags = set(_split_flags(str(row["qualityFlags"])))
    for reason in PRIMARY_EXCLUSION_REASON_PRIORITY:
        if reason in flags:
            return reason

    has_volume_issue = bool(
        {"volume_zero_or_missing", "low_volume"}.intersection(flags)
    )
    has_open_interest_issue = bool(
        {"oi_zero_or_missing", "low_open_interest"}.intersection(flags)
    )
    if has_volume_issue and has_open_interest_issue:
        return "insufficient_liquidity"

    confidence_level = str(row["confidenceLevel"]).lower()
    if confidence_level in {"low", "medium"}:
        return f"{confidence_level}_confidence"
    return "quality_filter"


def _select_market_price(
    row: pd.Series,
    max_trade_age_hours: float,
    spread_ratio_max: float,
    recompute_spread_ratio_max: float,
    now_utc: pd.Timestamp,
) -> Tuple[float, str, Set[str], float]:
    """Select market price source and attach quality flags."""
    flags: Set[str] = set()
    bid = _to_float(row["bid"])
    ask = _to_float(row["ask"])
    last_price = _to_float(row["lastPrice"])
    last_trade_date = row["lastTradeDate"]
    last_trade_is_stale = pd.isna(last_trade_date) or (
        now_utc - last_trade_date
    ).total_seconds() / 3600.0 > max_trade_age_hours

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
        spread_ratio = (ask - bid) / market_price
        if spread_ratio > spread_ratio_max:
            flags.add("wide_spread")

        if (
            spread_ratio > recompute_spread_ratio_max
            and np.isfinite(last_price)
            and last_price > 0
        ):
            if last_trade_is_stale:
                flags.add("stale_last_trade")
            elif bid <= last_price <= ask:
                flags.add("recent_trade_inside_wide_quote")
            else:
                flags.add("last_trade_outside_quote")
        return float(market_price), "mid", flags, spread_ratio

    flags.add("no_two_sided_quote")
    if np.isfinite(last_price) and last_price > 0:
        if last_trade_is_stale:
            flags.add("stale_last_trade")
        return float(last_price), "lastPrice", flags, spread_ratio

    if np.isfinite(bid) and bid > 0:
        flags.add("one_sided_quote_price")
        return float(bid), "bid", flags, spread_ratio
    if np.isfinite(ask) and ask > 0:
        flags.add("one_sided_quote_price")
        return float(ask), "ask", flags, spread_ratio

    return float("nan"), "none", flags, spread_ratio


def _compute_bid_ask_iv_width(
    option_type: str,
    spot: float,
    strike: float,
    time_to_expiration: float,
    risk_free_rate: float,
    dividend_yield: float,
    bid: float,
    ask: float,
    midpoint_iv: float,
) -> Tuple[float, float]:
    """Convert a two-sided quote spread into an implied-volatility interval."""
    bid_iv = _implied_volatility(
        option_type=option_type,
        spot=spot,
        strike=strike,
        time_to_expiration=time_to_expiration,
        risk_free_rate=risk_free_rate,
        market_price=bid,
        dividend_yield=dividend_yield,
    )
    ask_iv = _implied_volatility(
        option_type=option_type,
        spot=spot,
        strike=strike,
        time_to_expiration=time_to_expiration,
        risk_free_rate=risk_free_rate,
        market_price=ask,
        dividend_yield=dividend_yield,
    )
    if not np.isfinite(bid_iv) or not np.isfinite(ask_iv):
        return float("nan"), float("nan")

    iv_width = max(float(ask_iv - bid_iv), 0.0)
    denominator = midpoint_iv
    if not np.isfinite(denominator) or denominator <= 0:
        denominator = 0.5 * (bid_iv + ask_iv)
    iv_width_ratio = (
        float(iv_width / denominator)
        if np.isfinite(denominator) and denominator > 0
        else float("nan")
    )
    return float(iv_width), iv_width_ratio


def _estimate_forward_terms(
    df: pd.DataFrame,
    underlying_price: float,
    risk_free_rate: float,
    fallback_dividend_yield: float,
) -> pd.DataFrame:
    """Estimate expiry-level forwards from put-call parity when quotes are usable."""
    forward_df = df.copy()
    forward_df["forwardPrice"] = underlying_price * np.exp(
        (risk_free_rate - fallback_dividend_yield)
        * forward_df["time_to_expiration_years"]
    )
    forward_df["dividendYieldUsed"] = float(fallback_dividend_yield)

    for expiration_date, group in forward_df.groupby("expirationDate", dropna=True):
        time_to_expiration = _to_float(group["time_to_expiration_years"].iloc[0])

        calls = group[group["optionType"] == "call"][["strike", "bid", "ask"]].copy()
        puts = group[group["optionType"] == "put"][["strike", "bid", "ask"]].copy()
        if calls.empty or puts.empty:
            continue

        paired = calls.merge(puts, on="strike", suffixes=("_call", "_put"))
        paired = paired[
            (paired["bid_call"] > 0)
            & (paired["ask_call"] > paired["bid_call"])
            & (paired["bid_put"] > 0)
            & (paired["ask_put"] > paired["bid_put"])
            & (paired["strike"] > 0)
        ].copy()
        paired["log_moneyness_abs"] = np.abs(np.log(paired["strike"] / underlying_price))
        paired = paired[paired["log_moneyness_abs"] <= MAX_FORWARD_ABS_LOG_MONEYNESS]
        if len(paired) < MIN_FORWARD_PAIR_COUNT:
            continue

        paired["call_mid"] = 0.5 * (paired["bid_call"] + paired["ask_call"])
        paired["put_mid"] = 0.5 * (paired["bid_put"] + paired["ask_put"])
        discount_factor = np.exp(-risk_free_rate * time_to_expiration)
        paired["forward_candidate"] = paired["strike"] + (
            paired["call_mid"] - paired["put_mid"]
        ) / discount_factor
        paired = paired[
            np.isfinite(paired["forward_candidate"]) & (paired["forward_candidate"] > 0)
        ].copy()
        if len(paired) < MIN_FORWARD_PAIR_COUNT:
            continue

        paired = paired.sort_values("log_moneyness_abs").head(7)
        forward_price = float(paired["forward_candidate"].median())
        implied_dividend_yield = float(
            risk_free_rate - np.log(forward_price / underlying_price) / time_to_expiration
        )
        if not (
            np.isfinite(implied_dividend_yield)
            and MIN_REASONABLE_IMPLIED_DIVIDEND_YIELD
            <= implied_dividend_yield
            <= MAX_REASONABLE_IMPLIED_DIVIDEND_YIELD
        ):
            continue

        mask = forward_df["expirationDate"] == expiration_date
        forward_df.loc[mask, "forwardPrice"] = forward_price
        forward_df.loc[mask, "dividendYieldUsed"] = implied_dividend_yield

    return forward_df


def _resolve_row_iv(
    row: pd.Series,
    underlying_price: float,
    risk_free_rate: float,
    max_trade_age_hours: float,
    now_utc: pd.Timestamp,
) -> pd.Series:
    flags: Set[str] = set()
    spread_ratio_max = QUALITY_CONFIG["spread_ratio_max"]

    option_type = str(row["optionType"]).lower()
    strike = _to_float(row["strike"])
    time_to_expiration_years = _to_float(row["time_to_expiration_years"])
    volume = _to_float(row["volume"])
    open_interest = _to_float(row["openInterest"])
    bid = _to_float(row["bid"])
    ask = _to_float(row["ask"])
    row_dividend_yield = _to_float(row["dividendYieldUsed"])

    if not np.isfinite(volume) or volume <= 0:
        flags.add("volume_zero_or_missing")
    elif volume < QUALITY_CONFIG["min_volume"]:
        flags.add("low_volume")

    if not np.isfinite(open_interest) or open_interest <= 0:
        flags.add("oi_zero_or_missing")
    elif open_interest < QUALITY_CONFIG["min_open_interest"]:
        flags.add("low_open_interest")

    market_price, price_source, price_flags, spread_ratio = _select_market_price(
        row,
        max_trade_age_hours=max_trade_age_hours,
        spread_ratio_max=spread_ratio_max,
        recompute_spread_ratio_max=QUALITY_CONFIG["recompute_spread_ratio_max"],
        now_utc=now_utc,
    )
    flags.update(price_flags)

    bs_iv = float("nan")
    iv_bid_ask_width = float("nan")
    iv_bid_ask_width_ratio = float("nan")
    if (
        option_type in {"call", "put"}
        and np.isfinite(market_price)
        and market_price > 0
    ):
        lower, upper = _no_arbitrage_bounds(
            option_type=option_type,
            spot=underlying_price,
            strike=strike,
            time_to_expiration=time_to_expiration_years,
            risk_free_rate=risk_free_rate,
            dividend_yield=row_dividend_yield,
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
                dividend_yield=row_dividend_yield,
            )
            if price_source == "mid":
                (
                    iv_bid_ask_width,
                    iv_bid_ask_width_ratio,
                ) = _compute_bid_ask_iv_width(
                    option_type=option_type,
                    spot=underlying_price,
                    strike=strike,
                    time_to_expiration=time_to_expiration_years,
                    risk_free_rate=risk_free_rate,
                    dividend_yield=row_dividend_yield,
                    bid=bid,
                    ask=ask,
                    midpoint_iv=bs_iv,
                )
                if not np.isfinite(iv_bid_ask_width):
                    flags.add("iv_bid_ask_width_unavailable")
                else:
                    if iv_bid_ask_width > QUALITY_CONFIG["iv_bid_ask_width_max"]:
                        flags.add("wide_iv_bid_ask")
                    if (
                        np.isfinite(iv_bid_ask_width_ratio)
                        and iv_bid_ask_width_ratio
                        > QUALITY_CONFIG["iv_bid_ask_width_ratio_max"]
                    ):
                        flags.add("wide_iv_bid_ask")

    if (
        price_source == "mid"
        and spread_ratio > QUALITY_CONFIG["recompute_spread_ratio_max"]
    ):
        flags.add("wide_recompute_spread")

    recomputed_iv_usable = bool(
        np.isfinite(bs_iv)
        and bs_iv > 0
        and "arbitrage_violation" not in flags
        and "stale_last_trade" not in flags
        and "one_sided_quote_price" not in flags
        and "wide_recompute_spread" not in flags
        and "wide_iv_bid_ask" not in flags
        and "iv_bid_ask_width_unavailable" not in flags
        and price_source in {"mid", "lastPrice"}
    )
    recomputed_iv_has_full_quote_support = bool(
        recomputed_iv_usable and "no_two_sided_quote" not in flags
    )

    implied_vol_final = float("nan")
    if recomputed_iv_usable:
        implied_vol_final = bs_iv
        if not recomputed_iv_has_full_quote_support:
            flags.add("recomputed_iv_low_quote_support")
    elif np.isfinite(bs_iv) and bs_iv > 0:
        flags.add("recomputed_iv_unreliable")

    if not np.isfinite(implied_vol_final) or implied_vol_final <= 0:
        flags.add("iv_unavailable")
        implied_vol_final = float("nan")

    confidence_level = "high"
    has_volume_issue = "volume_zero_or_missing" in flags or "low_volume" in flags
    has_open_interest_issue = "oi_zero_or_missing" in flags or "low_open_interest" in flags
    has_liquidity_issue = has_volume_issue or has_open_interest_issue
    if (
        "iv_unavailable" in flags
        or "arbitrage_violation" in flags
        or "recomputed_iv_unreliable" in flags
        or (has_volume_issue and has_open_interest_issue)
    ):
        confidence_level = "low"
    elif (
        "stale_last_trade" in flags
        or "no_two_sided_quote" in flags
        or "wide_spread" in flags
        or "wide_recompute_spread" in flags
        or "wide_iv_bid_ask" in flags
        or "iv_bid_ask_width_unavailable" in flags
        or has_liquidity_issue
    ):
        confidence_level = "medium"

    include_in_surface = bool(
        np.isfinite(implied_vol_final)
        and implied_vol_final > 0
        and confidence_level in {"high", "medium"}
    )

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
            "spreadRatio": spread_ratio,
            "impliedVolatilityFinal": implied_vol_final,
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
    min_dte: int = None,
    max_dte: int = None,
    underlying_price: float = None,
    risk_free_rate: float = 0.02,
    dividend_yield: float = 0.0,
    max_trade_age_hours: float = 120.0,
    as_of_utc: pd.Timestamp = None,
) -> pd.DataFrame:
    """
    Clean and enrich options data for volatility-surface visualization.

    The returned frame retains low-confidence rows for diagnostics but marks
    surface-suitable rows via `includeInSurface`.
    """
    if df is None or df.empty:
        print("No data to clean")
        return pd.DataFrame()

    required_columns = {
        "strike",
        "expirationDate",
        "optionType",
        "volume",
        "openInterest",
        "bid",
        "ask",
        "lastPrice",
        "lastTradeDate",
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Options data missing required columns: {', '.join(missing_columns)}"
        )
    clean_df = df.copy()

    as_of_utc = normalize_as_of_utc(as_of_utc)
    expiration_utc = pd.to_datetime(clean_df["expirationDate"], errors="coerce", utc=True)
    clean_df["expirationDate"] = expiration_utc.dt.tz_convert(None)
    expiration_close_times_utc = expiration_utc.map(option_expiration_close_utc)
    clean_df["lastTradeDate"] = pd.to_datetime(
        clean_df["lastTradeDate"], utc=True, errors="coerce"
    )
    clean_df["strike"] = pd.to_numeric(clean_df["strike"], errors="coerce")
    clean_df["volume"] = pd.to_numeric(clean_df["volume"], errors="coerce")
    clean_df["openInterest"] = pd.to_numeric(clean_df["openInterest"], errors="coerce")
    clean_df["bid"] = pd.to_numeric(clean_df["bid"], errors="coerce")
    clean_df["ask"] = pd.to_numeric(clean_df["ask"], errors="coerce")
    clean_df["lastPrice"] = pd.to_numeric(clean_df["lastPrice"], errors="coerce")

    remaining_days = (
        expiration_close_times_utc - as_of_utc
    ).dt.total_seconds() / (24.0 * 60.0 * 60.0)
    clean_df["time_to_expiration_years"] = remaining_days / 365.25
    clean_df["days_to_expiration"] = (
        clean_df["expirationDate"] - as_of_utc.tz_convert(None).normalize()
    ).dt.days.astype("Int64")

    clean_df = clean_df.replace([np.inf, -np.inf], np.nan)
    clean_df = clean_df.dropna(subset=["expirationDate", "strike", "days_to_expiration"])
    clean_df = clean_df[clean_df["strike"] > 0]
    clean_df = clean_df[clean_df["days_to_expiration"] >= 0]
    clean_df = clean_df[clean_df["time_to_expiration_years"] > 0]

    if min_strike is not None:
        clean_df = clean_df[clean_df["strike"] >= min_strike]
    if max_strike is not None:
        clean_df = clean_df[clean_df["strike"] <= max_strike]

    if min_dte is not None:
        clean_df = clean_df[clean_df["days_to_expiration"] >= min_dte]
        print(f"Filtering for DTE >= {min_dte}. Rows remaining: {len(clean_df)}")

    if max_dte is not None:
        clean_df = clean_df[clean_df["days_to_expiration"] <= max_dte]
        print(f"Filtering for DTE <= {max_dte}. Rows remaining: {len(clean_df)}")

    if clean_df.empty:
        return pd.DataFrame()

    underlying_price = _to_float(underlying_price)
    if not np.isfinite(underlying_price) or underlying_price <= 0:
        raise ValueError(
            "underlying_price must be provided and positive for Black-Scholes IV."
        )

    clean_df = _estimate_forward_terms(
        clean_df,
        underlying_price=underlying_price,
        risk_free_rate=risk_free_rate,
        fallback_dividend_yield=dividend_yield,
    )

    resolved = clean_df.apply(
        lambda row: _resolve_row_iv(
            row=row,
            underlying_price=underlying_price,
            risk_free_rate=risk_free_rate,
            max_trade_age_hours=max_trade_age_hours,
            now_utc=as_of_utc,
        ),
        axis=1,
    )
    clean_df = pd.concat([clean_df, resolved], axis=1)

    output_columns = [
        "strike",
        "days_to_expiration",
        "time_to_expiration_years",
        "impliedVolatilityFinal",
        "optionType",
        "volume",
        "openInterest",
        "spreadRatio",
        "forwardPrice",
        "dividendYieldUsed",
        "confidenceLevel",
        "qualityFlags",
        "includeInSurface",
        "surfaceWeight",
    ]

    clean_df = clean_df[output_columns].copy()
    clean_df["days_to_expiration"] = clean_df["days_to_expiration"].astype(int)
    return clean_df.sort_values(
        ["days_to_expiration", "strike", "optionType"], ascending=[True, True, True]
    ).reset_index(drop=True)


def build_diagnostics_report(df: pd.DataFrame) -> Dict[str, object]:
    """Summarize quality and exclusions."""
    if df is None or df.empty:
        return {
            "rows_retained": 0,
            "rows_surface_included": 0,
            "rows_surface_excluded": 0,
            "surface_dte_min": None,
            "surface_dte_max": None,
            "dropped_rows_by_reason": {},
        }

    included_mask = df["includeInSurface"].fillna(False).astype(bool)
    included_df = df[included_mask]
    surface_dte = pd.to_numeric(included_df["days_to_expiration"], errors="coerce").dropna()
    excluded_df = df[~included_mask]
    excluded_reason_counts = (
        excluded_df.apply(_primary_exclusion_reason, axis=1)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return {
        "rows_retained": int(len(df)),
        "rows_surface_included": int(len(included_df)),
        "rows_surface_excluded": int(len(df) - len(included_df)),
        "surface_dte_min": int(surface_dte.min()) if not surface_dte.empty else None,
        "surface_dte_max": int(surface_dte.max()) if not surface_dte.empty else None,
        "dropped_rows_by_reason": {
            str(k): int(v) for k, v in sorted(excluded_reason_counts.items())
        },
    }
