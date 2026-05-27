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
INV_SQRT_TWO = 1.0 / math.sqrt(2.0)

QUALITY_MODE_CONFIG = {
    "strict": {
        "spread_ratio_max": 0.50,
        "recompute_spread_ratio_max": 0.20,
        "iv_bid_ask_width_max": 0.06,
        "iv_bid_ask_width_ratio_max": 0.35,
        "allow_medium_surface": False,
        "min_volume": 10.0,
        "min_open_interest": 50.0,
    },
    "balanced": {
        "spread_ratio_max": 1.00,
        "recompute_spread_ratio_max": 0.35,
        "iv_bid_ask_width_max": 0.08,
        "iv_bid_ask_width_ratio_max": 0.50,
        "allow_medium_surface": True,
        "min_volume": 5.0,
        "min_open_interest": 25.0,
    },
    "lenient": {
        "spread_ratio_max": 2.00,
        "recompute_spread_ratio_max": 0.50,
        "iv_bid_ask_width_max": 0.10,
        "iv_bid_ask_width_ratio_max": 0.75,
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


def _normal_cdf(value: float) -> float:
    return 0.5 * math.erfc(-float(value) * INV_SQRT_TWO)


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
        return float(discounted_spot * _normal_cdf(d1) - discounted_strike * _normal_cdf(d2))
    return float(discounted_strike * _normal_cdf(-d2) - discounted_spot * _normal_cdf(-d1))


def _black_scholes_prices_array(
    option_types: np.ndarray,
    spot: float,
    strikes: np.ndarray,
    time_to_expiration: np.ndarray,
    risk_free_rate: float,
    volatility: np.ndarray,
    dividend_yield: np.ndarray,
) -> np.ndarray:
    option_types = np.asarray(option_types, dtype=object)
    strikes = np.asarray(strikes, dtype=float)
    time_to_expiration = np.asarray(time_to_expiration, dtype=float)
    volatility = np.asarray(volatility, dtype=float)
    dividend_yield = np.asarray(dividend_yield, dtype=float)
    prices = np.full(strikes.shape, np.nan, dtype=float)
    if (
        option_types.shape != strikes.shape
        or strikes.shape != time_to_expiration.shape
        or strikes.shape != volatility.shape
        or strikes.shape != dividend_yield.shape
        or spot <= 0
    ):
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

    return _black_scholes_price_from_terms(
        option_type=option_type,
        log_spot_over_strike=math.log(spot / strike),
        discounted_spot=spot * math.exp(-dividend_yield * time_to_expiration),
        discounted_strike=strike * math.exp(-risk_free_rate * time_to_expiration),
        time_to_expiration=time_to_expiration,
        sqrt_t=math.sqrt(time_to_expiration),
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
    )


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
    recompute_spread_ratio_max: float,
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

            if (
                np.isfinite(spread_ratio)
                and spread_ratio > recompute_spread_ratio_max
                and np.isfinite(last_price)
                and last_price > 0
            ):
                if pd.isna(last_trade_date):
                    flags.add("stale_last_trade")
                else:
                    age_hours = (now_utc - last_trade_date).total_seconds() / 3600.0
                    if age_hours > max_trade_age_hours:
                        flags.add("stale_last_trade")
                    elif bid <= last_price <= ask:
                        flags.add("recent_trade_inside_wide_quote")
                    else:
                        flags.add("last_trade_outside_quote")
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
) -> Tuple[float, float, float, float]:
    """Convert a two-sided quote spread into an implied-volatility interval."""
    if (
        option_type not in {"call", "put"}
        or not np.isfinite(spot)
        or spot <= 0
        or not np.isfinite(strike)
        or strike <= 0
        or not np.isfinite(time_to_expiration)
        or time_to_expiration <= 0
        or not np.isfinite(bid)
        or not np.isfinite(ask)
        or bid <= 0
        or ask <= 0
        or ask < bid
    ):
        return float("nan"), float("nan"), float("nan"), float("nan")

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
        return bid_iv, ask_iv, float("nan"), float("nan")

    iv_width = max(float(ask_iv - bid_iv), 0.0)
    denominator = midpoint_iv
    if not np.isfinite(denominator) or denominator <= 0:
        denominator = 0.5 * (bid_iv + ask_iv)
    iv_width_ratio = (
        float(iv_width / denominator)
        if np.isfinite(denominator) and denominator > 0
        else float("nan")
    )
    return float(bid_iv), float(ask_iv), float(iv_width), iv_width_ratio


def _fallback_forward_price(
    spot: float,
    time_to_expiration: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> float:
    if (
        not np.isfinite(spot)
        or spot <= 0
        or not np.isfinite(time_to_expiration)
        or time_to_expiration <= 0
    ):
        return float("nan")
    return float(spot * np.exp((risk_free_rate - dividend_yield) * time_to_expiration))


def _estimate_forward_terms(
    df: pd.DataFrame,
    underlying_price: float,
    risk_free_rate: float,
    fallback_dividend_yield: float,
) -> pd.DataFrame:
    """Estimate expiry-level forwards from put-call parity when quotes are usable."""
    forward_df = df.copy()
    forward_df["forwardPrice"] = forward_df["time_to_expiration_years"].apply(
        lambda t: _fallback_forward_price(
            spot=underlying_price,
            time_to_expiration=_to_float(t),
            risk_free_rate=risk_free_rate,
            dividend_yield=fallback_dividend_yield,
        )
    )
    forward_df["dividendYieldUsed"] = float(fallback_dividend_yield)
    forward_df["forwardEstimationMethod"] = "configured_dividend_yield"

    if not np.isfinite(underlying_price) or underlying_price <= 0:
        return forward_df

    for expiration_date, group in forward_df.groupby("expirationDate", dropna=True):
        time_to_expiration = _to_float(group["time_to_expiration_years"].iloc[0])
        if not np.isfinite(time_to_expiration) or time_to_expiration <= 0:
            continue

        calls = group[group["optionType"] == "call"][
            ["strike", "bid", "ask", "volume", "openInterest"]
        ].copy()
        puts = group[group["optionType"] == "put"][
            ["strike", "bid", "ask", "volume", "openInterest"]
        ].copy()
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
        if paired.empty:
            continue

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
        forward_df.loc[mask, "forwardEstimationMethod"] = "put_call_parity"

    return forward_df


def _resolve_row_iv(
    row: pd.Series,
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
    bid = _to_float(row.get("bid"))
    ask = _to_float(row.get("ask"))
    row_dividend_yield = _to_float(row.get("dividendYieldUsed"))
    if not np.isfinite(row_dividend_yield):
        row_dividend_yield = dividend_yield

    if not np.isfinite(volume) or volume <= 0:
        flags.add("volume_zero_or_missing")
    elif volume < quality_config["min_volume"]:
        flags.add("low_volume")

    if not np.isfinite(open_interest) or open_interest <= 0:
        flags.add("oi_zero_or_missing")
    elif open_interest < quality_config["min_open_interest"]:
        flags.add("low_open_interest")

    market_price, price_source, price_flags, spread_ratio = _select_market_price(
        row,
        max_trade_age_hours=max_trade_age_hours,
        spread_ratio_max=spread_ratio_max,
        recompute_spread_ratio_max=quality_config["recompute_spread_ratio_max"],
        now_utc=now_utc,
    )
    flags.update(price_flags)

    bs_iv = float("nan")
    bid_iv = float("nan")
    ask_iv = float("nan")
    iv_bid_ask_width = float("nan")
    iv_bid_ask_width_ratio = float("nan")
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
            if np.isfinite(bs_iv) and bs_iv > MAX_REASONABLE_IV:
                flags.add("iv_outlier")
                bs_iv = float("nan")

            if price_source == "mid":
                (
                    bid_iv,
                    ask_iv,
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
                    if iv_bid_ask_width > quality_config["iv_bid_ask_width_max"]:
                        flags.add("wide_iv_bid_ask")
                    if (
                        np.isfinite(iv_bid_ask_width_ratio)
                        and iv_bid_ask_width_ratio
                        > quality_config["iv_bid_ask_width_ratio_max"]
                    ):
                        flags.add("wide_iv_bid_ask")

    if (
        price_source == "mid"
        and np.isfinite(spread_ratio)
        and spread_ratio > quality_config["recompute_spread_ratio_max"]
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
        recomputed_iv_usable
        and "stale_last_trade" not in flags
        and "no_two_sided_quote" not in flags
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

    if np.isfinite(implied_vol_final) and implied_vol_final > MAX_REASONABLE_IV:
        flags.add("iv_outlier")
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
        "arbitrage_violation" in flags
        or "stale_last_trade" in flags
        or "no_two_sided_quote" in flags
        or "wide_spread" in flags
        or "wide_recompute_spread" in flags
        or "wide_iv_bid_ask" in flags
        or "iv_bid_ask_width_unavailable" in flags
        or has_liquidity_issue
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
            "blackScholesImpliedVolatility": bs_iv,
            "bidImpliedVolatility": bid_iv,
            "askImpliedVolatility": ask_iv,
            "ivBidAskWidth": iv_bid_ask_width,
            "ivBidAskWidthRatio": iv_bid_ask_width_ratio,
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
    min_date: str = None,
    max_date: str = None,
    option_type_to_plot: str = "both",
    min_dte: int = None,
    max_dte: int = None,
    underlying_price: float = None,
    risk_free_rate: float = 0.02,
    dividend_yield: float = 0.0,
    quality_mode: str = "lenient",
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

    quality_mode = _normalize_quality_mode(quality_mode)

    clean_df = df.copy()
    required_columns = [
        "strike",
        "expirationDate",
        "optionType",
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

    as_of_utc = normalize_as_of_utc(as_of_utc)
    expiration_utc = pd.to_datetime(clean_df["expirationDate"], errors="coerce", utc=True)
    clean_df["expirationDate"] = expiration_utc.dt.tz_convert(None)
    if "expirationCloseUtc" in clean_df.columns:
        expiration_close_times_utc = pd.to_datetime(
            clean_df["expirationCloseUtc"], errors="coerce", utc=True
        )
    else:
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
            quality_mode=quality_mode,
            underlying_price=underlying_price,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            max_trade_age_hours=max_trade_age_hours,
            now_utc=as_of_utc,
        ),
        axis=1,
    )
    clean_df = pd.concat([clean_df, resolved], axis=1)
    clean_df["impliedVolatility"] = clean_df["impliedVolatilityFinal"]

    output_columns = [
        "strike",
        "days_to_expiration",
        "time_to_expiration_years",
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
        "forwardPrice",
        "dividendYieldUsed",
        "forwardEstimationMethod",
        "blackScholesImpliedVolatility",
        "bidImpliedVolatility",
        "askImpliedVolatility",
        "ivBidAskWidth",
        "ivBidAskWidthRatio",
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
    dte_range: Tuple[int, int] | None = None,
) -> Dict[str, float]:
    """Build repricing diagnostics from final IV vs selected market price."""
    requested_dte_min = None
    requested_dte_max = None
    if dte_range is not None:
        requested_dte_min, requested_dte_max = int(dte_range[0]), int(dte_range[1])
        if requested_dte_min > requested_dte_max:
            requested_dte_min, requested_dte_max = requested_dte_max, requested_dte_min

    if df is None or df.empty:
        return {
            "rows_checked": 0,
            "rows_with_market_price": 0,
            "repricing_mae": None,
            "repricing_rmse": None,
            "repricing_p95_abs_error": None,
            "arbitrage_bound_violations": 0,
            "requested_dte_min": requested_dte_min,
            "requested_dte_max": requested_dte_max,
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
            "requested_dte_min": requested_dte_min,
            "requested_dte_max": requested_dte_max,
        }

    option_types = checks_df["optionType"].astype(str).str.lower().to_numpy(dtype=object)
    strikes = pd.to_numeric(checks_df["strike"], errors="coerce").to_numpy(dtype=float)
    times = pd.to_numeric(
        checks_df["time_to_expiration_years"], errors="coerce"
    ).to_numpy(dtype=float)
    if "marketPrice" in checks_df.columns:
        market_prices = pd.to_numeric(
            checks_df["marketPrice"], errors="coerce"
        ).to_numpy(dtype=float)
    else:
        market_prices = np.full(len(checks_df), np.nan, dtype=float)
    implied_volatility = pd.to_numeric(
        checks_df["impliedVolatilityFinal"], errors="coerce"
    ).to_numpy(dtype=float)
    if "dividendYieldUsed" in checks_df.columns:
        dividend_yields = pd.to_numeric(
            checks_df["dividendYieldUsed"], errors="coerce"
        ).fillna(dividend_yield).to_numpy(dtype=float)
    else:
        dividend_yields = np.full(len(checks_df), dividend_yield, dtype=float)

    discounted_spot = underlying_price * np.exp(-dividend_yields * times)
    discounted_strikes = strikes * np.exp(-risk_free_rate * times)
    call_mask = option_types == "call"
    put_mask = option_types == "put"
    lower_bounds = np.full(len(checks_df), np.nan, dtype=float)
    upper_bounds = np.full(len(checks_df), np.nan, dtype=float)
    lower_bounds[call_mask] = np.maximum(
        0.0, discounted_spot[call_mask] - discounted_strikes[call_mask]
    )
    upper_bounds[call_mask] = discounted_spot[call_mask]
    lower_bounds[put_mask] = np.maximum(
        0.0, discounted_strikes[put_mask] - discounted_spot[put_mask]
    )
    upper_bounds[put_mask] = discounted_strikes[put_mask]
    market_price_mask = np.isfinite(market_prices) & (market_prices > 0)
    arbitrage_violations = int(
        np.count_nonzero(
            market_price_mask
            & np.isfinite(lower_bounds)
            & np.isfinite(upper_bounds)
            & (
                (market_prices < lower_bounds - ARBITRAGE_EPSILON)
                | (market_prices > upper_bounds + ARBITRAGE_EPSILON)
            )
        )
    )

    model_prices = _black_scholes_prices_array(
        option_types=option_types,
        spot=underlying_price,
        strikes=strikes,
        time_to_expiration=times,
        risk_free_rate=risk_free_rate,
        volatility=implied_volatility,
        dividend_yield=dividend_yields,
    )
    comparable_mask = market_price_mask & np.isfinite(model_prices)

    if not np.any(comparable_mask):
        return {
            "rows_checked": int(len(checks_df)),
            "rows_with_market_price": 0,
            "repricing_mae": None,
            "repricing_rmse": None,
            "repricing_p95_abs_error": None,
            "arbitrage_bound_violations": int(arbitrage_violations),
            "requested_dte_min": requested_dte_min,
            "requested_dte_max": requested_dte_max,
        }

    error = model_prices[comparable_mask] - market_prices[comparable_mask]
    abs_error = np.abs(error)
    return {
        "rows_checked": int(len(checks_df)),
        "rows_with_market_price": int(np.count_nonzero(comparable_mask)),
        "repricing_mae": float(abs_error.mean()),
        "repricing_rmse": float(np.sqrt(np.mean(error**2))),
        "repricing_p95_abs_error": float(np.quantile(abs_error, 0.95)),
        "arbitrage_bound_violations": int(arbitrage_violations),
        "requested_dte_min": requested_dte_min,
        "requested_dte_max": requested_dte_max,
    }


def build_diagnostics_report(
    df: pd.DataFrame,
    raw_row_count: int = None,
) -> Dict[str, object]:
    """Summarize quality and exclusions."""
    if df is None or df.empty:
        return {
            "raw_row_count": int(raw_row_count) if raw_row_count is not None else None,
            "rows_retained": 0,
            "rows_surface_included": 0,
            "rows_surface_excluded": 0,
            "observed_dte_min": None,
            "observed_dte_max": None,
            "surface_dte_min": None,
            "surface_dte_max": None,
            "confidence_counts": {},
            "flag_counts": {},
            "dropped_rows_by_reason": {},
            "per_dte_quality_summary": [],
        }

    retained_df = df.copy()
    included_mask = retained_df["includeInSurface"].fillna(False).astype(bool)
    included_df = retained_df[included_mask]
    observed_dte = pd.to_numeric(retained_df["days_to_expiration"], errors="coerce").dropna()
    surface_dte = pd.to_numeric(included_df["days_to_expiration"], errors="coerce").dropna()

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
            }
        )
    per_dte_summary.sort(key=lambda item: item["days_to_expiration"])

    return {
        "raw_row_count": int(raw_row_count) if raw_row_count is not None else None,
        "rows_retained": int(len(retained_df)),
        "rows_surface_included": int(len(included_df)),
        "rows_surface_excluded": int(len(retained_df) - len(included_df)),
        "observed_dte_min": int(observed_dte.min()) if not observed_dte.empty else None,
        "observed_dte_max": int(observed_dte.max()) if not observed_dte.empty else None,
        "surface_dte_min": int(surface_dte.min()) if not surface_dte.empty else None,
        "surface_dte_max": int(surface_dte.max()) if not surface_dte.empty else None,
        "confidence_counts": {str(k): int(v) for k, v in confidence_counts.items()},
        "flag_counts": {str(k): int(v) for k, v in sorted(flag_counts.items())},
        "dropped_rows_by_reason": {
            str(k): int(v) for k, v in sorted(excluded_reason_counts.items())
        },
        "per_dte_quality_summary": per_dte_summary,
    }
