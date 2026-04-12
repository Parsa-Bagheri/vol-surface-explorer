from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.data_cleaner import (
    _black_scholes_price,
    _implied_volatility,
    _no_arbitrage_bounds,
    build_diagnostics_report,
    prepare_options_data,
)


def _black_scholes_call_price(
    spot: float,
    strike: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
) -> float:
    if time_to_expiration <= 0 or volatility <= 0:
        raise ValueError("Time to expiration and volatility must be positive for pricing.")

    sqrt_t = np.sqrt(time_to_expiration)
    d1 = (
        np.log(spot / strike)
        + (risk_free_rate + 0.5 * volatility**2) * time_to_expiration
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    return spot * norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiration) * norm.cdf(d2)


def _base_row(**overrides):
    now = datetime.now(timezone.utc)
    row = {
        "strike": 100.0,
        "expirationDate": (now + timedelta(days=30)).date(),
        "optionType": "call",
        "impliedVolatility": 0.3,
        "volume": 50.0,
        "openInterest": 100.0,
        "bid": 4.5,
        "ask": 5.5,
        "lastPrice": 5.0,
        "lastTradeDate": now.isoformat(),
    }
    row.update(overrides)
    return row


def test_prepare_options_data_black_scholes_iv():
    spot_price = 100.0
    strike_price = 100.0
    days_to_expiration = 30
    risk_free_rate = 0.02
    true_volatility = 0.25
    time_to_expiration_years = days_to_expiration / 365.25

    theoretical_price = _black_scholes_call_price(
        spot=spot_price,
        strike=strike_price,
        time_to_expiration=time_to_expiration_years,
        risk_free_rate=risk_free_rate,
        volatility=true_volatility,
    )

    raw_df = pd.DataFrame(
        [
            _base_row(
                strike=strike_price,
                expirationDate=(datetime.now(timezone.utc) + timedelta(days=days_to_expiration)).date(),
                impliedVolatility=0.0,
                bid=theoretical_price * 0.99,
                ask=theoretical_price * 1.01,
                lastPrice=theoretical_price,
            )
        ]
    )

    cleaned_df = prepare_options_data(
        raw_df,
        option_type_to_plot="call",
        iv_source="black-scholes",
        underlying_price=spot_price,
        risk_free_rate=risk_free_rate,
    )

    assert not cleaned_df.empty
    computed_iv = cleaned_df["impliedVolatilityFinal"].iloc[0]
    assert abs(computed_iv - true_volatility) < 5e-3
    assert cleaned_df["ivComputationMethod"].iloc[0] == "black-scholes"
    assert bool(cleaned_df["includeInSurface"].iloc[0]) is True


def test_black_scholes_price_matches_dividend_adjusted_formula():
    spot_price = 100.0
    strike_price = 105.0
    risk_free_rate = 0.03
    dividend_yield = 0.01
    time_to_expiration_years = 45.0 / 365.25
    volatility = 0.24

    sqrt_t = np.sqrt(time_to_expiration_years)
    d1 = (
        np.log(spot_price / strike_price)
        + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiration_years
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    expected_call = spot_price * np.exp(-dividend_yield * time_to_expiration_years) * norm.cdf(d1) - strike_price * np.exp(-risk_free_rate * time_to_expiration_years) * norm.cdf(d2)
    expected_put = strike_price * np.exp(-risk_free_rate * time_to_expiration_years) * norm.cdf(-d2) - spot_price * np.exp(-dividend_yield * time_to_expiration_years) * norm.cdf(-d1)

    actual_call = _black_scholes_price(
        option_type="call",
        spot=spot_price,
        strike=strike_price,
        time_to_expiration=time_to_expiration_years,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )
    actual_put = _black_scholes_price(
        option_type="put",
        spot=spot_price,
        strike=strike_price,
        time_to_expiration=time_to_expiration_years,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )

    assert abs(actual_call - expected_call) < 1e-10
    assert abs(actual_put - expected_put) < 1e-10


def test_zero_volatility_limit_returns_lower_bound_and_inverts_to_zero():
    spot_price = 100.0
    strike_price = 90.0
    time_to_expiration_years = 30.0 / 365.25
    risk_free_rate = 0.02
    dividend_yield = 0.01
    lower_bound, _ = _no_arbitrage_bounds(
        option_type="call",
        spot=spot_price,
        strike=strike_price,
        time_to_expiration=time_to_expiration_years,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )

    price_at_zero_vol = _black_scholes_price(
        option_type="call",
        spot=spot_price,
        strike=strike_price,
        time_to_expiration=time_to_expiration_years,
        risk_free_rate=risk_free_rate,
        volatility=0.0,
        dividend_yield=dividend_yield,
    )
    implied_volatility = _implied_volatility(
        option_type="call",
        spot=spot_price,
        strike=strike_price,
        time_to_expiration=time_to_expiration_years,
        risk_free_rate=risk_free_rate,
        market_price=lower_bound,
        dividend_yield=dividend_yield,
    )

    assert abs(price_at_zero_vol - lower_bound) < 1e-12
    assert implied_volatility == 0.0


def test_auto_falls_back_from_placeholder_provider_iv():
    spot_price = 100.0
    true_volatility = 0.22
    days_to_expiration = 30
    t = days_to_expiration / 365.25
    market_price = _black_scholes_call_price(spot_price, 100.0, t, 0.02, true_volatility)

    raw_df = pd.DataFrame(
        [
            _base_row(
                impliedVolatility=1e-5,
                bid=market_price * 0.99,
                ask=market_price * 1.01,
                lastPrice=market_price,
            )
        ]
    )
    cleaned_df = prepare_options_data(
        raw_df,
        option_type_to_plot="call",
        iv_source="auto",
        underlying_price=spot_price,
        quality_mode="lenient",
    )

    assert len(cleaned_df) == 1
    row = cleaned_df.iloc[0]
    assert row["ivSourceUsed"] == "black-scholes"
    assert abs(row["impliedVolatilityFinal"] - true_volatility) < 5e-3
    assert "placeholder_iv" in row["qualityFlags"]
    assert bool(row["includeInSurface"]) is True


def test_arbitrage_violation_flagged_and_excluded():
    spot_price = 100.0
    raw_df = pd.DataFrame(
        [
            _base_row(
                optionType="call",
                impliedVolatility=1e-5,
                bid=149.0,
                ask=151.0,
                lastPrice=150.0,
            )
        ]
    )

    cleaned_df = prepare_options_data(
        raw_df,
        option_type_to_plot="call",
        iv_source="auto",
        underlying_price=spot_price,
        quality_mode="lenient",
    )

    assert len(cleaned_df) == 1
    row = cleaned_df.iloc[0]
    assert "arbitrage_violation" in row["qualityFlags"]
    assert bool(row["includeInSurface"]) is False
    assert pd.isna(row["impliedVolatilityFinal"])


def test_lenient_mode_retains_low_confidence_rows_but_excludes_from_surface():
    spot_price = 100.0
    stale_trade = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    raw_df = pd.DataFrame(
        [
            _base_row(
                impliedVolatility=0.28,
                bid=0.0,
                ask=0.0,
                lastPrice=5.0,
                lastTradeDate=stale_trade,
            )
        ]
    )

    cleaned_df = prepare_options_data(
        raw_df,
        option_type_to_plot="call",
        iv_source="auto",
        underlying_price=spot_price,
        quality_mode="lenient",
        max_trade_age_hours=72.0,
    )

    assert len(cleaned_df) == 1
    row = cleaned_df.iloc[0]
    assert row["confidenceLevel"] == "low"
    assert bool(row["includeInSurface"]) is False
    assert "stale_last_trade" in row["qualityFlags"]


def test_nan_open_interest_is_flagged_deterministically():
    spot_price = 100.0
    raw_df = pd.DataFrame(
        [_base_row(openInterest=np.nan, volume=np.nan, impliedVolatility=0.31)]
    )

    cleaned_df = prepare_options_data(
        raw_df,
        option_type_to_plot="call",
        iv_source="auto",
        underlying_price=spot_price,
        quality_mode="lenient",
    )

    assert len(cleaned_df) == 1
    row = cleaned_df.iloc[0]
    assert "oi_zero_or_missing" in row["qualityFlags"]
    assert row["confidenceLevel"] in {"medium", "low"}


def test_zero_volume_and_zero_open_interest_are_excluded_from_surface():
    spot_price = 100.0
    raw_df = pd.DataFrame(
        [_base_row(openInterest=0.0, volume=0.0, impliedVolatility=0.31)]
    )

    cleaned_df = prepare_options_data(
        raw_df,
        option_type_to_plot="call",
        iv_source="auto",
        underlying_price=spot_price,
        quality_mode="lenient",
    )

    assert len(cleaned_df) == 1
    row = cleaned_df.iloc[0]
    assert "oi_zero_or_missing" in row["qualityFlags"]
    assert "volume_zero_or_missing" in row["qualityFlags"]
    assert row["confidenceLevel"] == "low"
    assert bool(row["includeInSurface"]) is False
    assert row["surfaceWeight"] < 0.5


def test_min_dte_filter_removes_short_dated_contracts():
    spot_price = 100.0
    now = datetime.now(timezone.utc)
    raw_df = pd.DataFrame(
        [
            _base_row(expirationDate=(now + timedelta(days=10)).date(), strike=100.0),
            _base_row(expirationDate=(now + timedelta(days=45)).date(), strike=105.0),
        ]
    )

    cleaned_df = prepare_options_data(
        raw_df,
        option_type_to_plot="call",
        iv_source="auto",
        underlying_price=spot_price,
        min_dte=20,
        max_dte=60,
        quality_mode="lenient",
    )

    assert len(cleaned_df) == 1
    assert cleaned_df["days_to_expiration"].iloc[0] >= 20


def test_diagnostics_report_counts_flags_and_exclusions():
    spot_price = 100.0
    now = datetime.now(timezone.utc)
    raw_df = pd.DataFrame(
        [
            _base_row(impliedVolatility=0.25),
            _base_row(
                strike=105.0,
                impliedVolatility=1e-5,
                bid=0.0,
                ask=0.0,
                lastPrice=4.0,
                lastTradeDate=(now - timedelta(days=7)).isoformat(),
            ),
        ]
    )

    cleaned_df = prepare_options_data(
        raw_df,
        option_type_to_plot="call",
        iv_source="auto",
        underlying_price=spot_price,
        quality_mode="lenient",
    )
    report = build_diagnostics_report(cleaned_df, raw_row_count=len(raw_df))

    assert report["rows_retained"] == 2
    assert report["rows_surface_excluded"] >= 1
    assert "placeholder_iv" in report["flag_counts"]
    assert report["raw_row_count"] == 2
