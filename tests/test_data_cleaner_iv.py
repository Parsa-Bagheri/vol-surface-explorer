from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.data_cleaner import _black_scholes_prices_array, _implied_volatility, _no_arbitrage_bounds
from src.data_cleaner import build_diagnostics_report, prepare_options_data
from src.time_utils import expiration_dte


OUTPUT_COLUMNS = """
strike days_to_expiration time_to_expiration_years impliedVolatilityFinal optionType
volume openInterest spreadRatio forwardPrice dividendYieldUsed confidenceLevel qualityFlags
includeInSurface surfaceWeight
""".split()


def _black_scholes_call_price(spot, strike, time_to_expiration, risk_free_rate, volatility):
    sqrt_t = np.sqrt(time_to_expiration)
    d1 = (
        np.log(spot / strike)
        + (risk_free_rate + 0.5 * volatility**2) * time_to_expiration
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    return spot * norm.cdf(d1) - strike * np.exp(
        -risk_free_rate * time_to_expiration
    ) * norm.cdf(d2)


def _base_row(**overrides):
    now = datetime.now(timezone.utc)
    return {
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
        **overrides,
    }


def _clean_rows(*rows, **kwargs):
    spot = kwargs.pop("underlying_price", 100.0)
    return prepare_options_data(pd.DataFrame(rows), underlying_price=spot, **kwargs)


def _clean_row(overrides=None, **kwargs):
    return _clean_rows(_base_row(**(overrides or {})), **kwargs).iloc[0]


def _quote_price(volatility=0.24, dte=30):
    return _black_scholes_call_price(100.0, 100.0, dte / 365.25, 0.02, volatility)


def _trade_time(**delta):
    return (datetime.now(timezone.utc) - timedelta(**delta)).isoformat()


def test_prepare_options_data_black_scholes_iv():
    true_volatility = 0.25
    market_price = _quote_price(true_volatility)
    cleaned = _clean_rows(
        _base_row(
            impliedVolatility=0.0,
            bid=market_price * 0.99,
            ask=market_price * 1.01,
            lastPrice=market_price,
        ),
        risk_free_rate=0.02,
    )

    assert cleaned.columns.tolist() == OUTPUT_COLUMNS
    assert abs(cleaned.iloc[0]["impliedVolatilityFinal"] - true_volatility) < 5e-3
    assert bool(cleaned.iloc[0]["includeInSurface"]) is True


def test_prepare_options_data_rejects_incomplete_schema():
    with pytest.raises(ValueError, match="openInterest"):
        prepare_options_data(pd.DataFrame([_base_row()]).drop(columns=["openInterest"]), underlying_price=100.0)


def test_black_scholes_price_matches_dividend_adjusted_formula():
    spot, strike, rate, dividend, t, volatility = 100.0, 105.0, 0.03, 0.01, 45 / 365.25, 0.24
    sqrt_t = np.sqrt(t)
    d1 = (np.log(spot / strike) + (rate - dividend + 0.5 * volatility**2) * t) / (
        volatility * sqrt_t
    )
    d2 = d1 - volatility * sqrt_t
    discounted_spot = spot * np.exp(-dividend * t)
    discounted_strike = strike * np.exp(-rate * t)
    expected = [
        discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2),
        discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1),
    ]

    actual = _black_scholes_prices_array(
        np.array(["call", "put"]), spot, np.array([strike, strike]), t, rate, volatility, dividend
    )

    assert np.allclose(actual, expected, atol=1e-10)


def test_zero_volatility_limit_returns_lower_bound_and_inverts_to_zero():
    spot, strike, t, rate, dividend = 100.0, 90.0, 30 / 365.25, 0.02, 0.01
    lower, _ = _no_arbitrage_bounds("call", spot, strike, t, rate, dividend)
    zero_price = _black_scholes_prices_array(
        "call", spot, np.array([strike]), t, rate, 0.0, dividend
    )[0]

    assert abs(zero_price - lower) < 1e-12
    assert _implied_volatility("call", spot, strike, t, rate, lower, dividend) == 0.0


def test_raw_iv_field_is_ignored_when_quote_iv_is_available():
    market_price = _quote_price(0.22)
    row = _clean_row(
        {
            "impliedVolatility": 1e-5,
            "bid": market_price * 0.99,
            "ask": market_price * 1.01,
            "lastPrice": market_price,
        }
    )

    assert abs(row["impliedVolatilityFinal"] - 0.22) < 5e-3
    assert bool(row["includeInSurface"]) is True


@pytest.mark.parametrize(
    ("overrides", "extra_flags"),
    [
        ({"lastTradeDate": _trade_time(hours=12)}, set()),
        ({"lastPrice": 2.5}, {"recent_trade_inside_wide_quote"}),
        ({"lastTradeDate": _trade_time(days=10)}, {"stale_last_trade"}),
    ],
    ids=["wide-midpoint", "recent-inside-quote", "stale-wide-quote"],
)
def test_unreliable_wide_quotes_are_excluded(overrides, extra_flags):
    row = _clean_row(
        {
            "impliedVolatility": 0.35,
            "bid": 0.25,
            "ask": 8.0,
            "lastPrice": 4.0,
            "volume": 100.0,
            "openInterest": 500.0,
            **overrides,
        }
    )

    flags = set(row["qualityFlags"].split(";"))
    assert {
        "wide_recompute_spread",
        "wide_iv_bid_ask",
        "recomputed_iv_unreliable",
        *extra_flags,
    } <= flags
    assert pd.isna(row["impliedVolatilityFinal"])
    assert row["confidenceLevel"] == "low"
    assert bool(row["includeInSurface"]) is False


@pytest.mark.parametrize(
    ("raw_iv", "open_interest"),
    [(1e-5, 0.0), (0.28, 100.0)],
    ids=["low-open-interest", "normal-open-interest"],
)
def test_recent_last_price_iv_without_quotes_is_surface_eligible(raw_iv, open_interest):
    row = _clean_row(
        {
            "impliedVolatility": raw_iv,
            "bid": 0.0,
            "ask": 0.0,
            "lastPrice": _quote_price(),
            "lastTradeDate": _trade_time(hours=12),
            "volume": 100.0,
            "openInterest": open_interest,
        },
        max_trade_age_hours=72.0,
    )

    assert abs(row["impliedVolatilityFinal"] - 0.24) < 5e-3
    assert row["confidenceLevel"] == "medium"
    assert bool(row["includeInSurface"]) is True
    assert "no_two_sided_quote" in row["qualityFlags"]
    assert "recomputed_iv_low_quote_support" in row["qualityFlags"]
    assert "stale_last_trade" not in row["qualityFlags"]


def test_black_scholes_excludes_stale_last_price_when_quote_support_is_missing():
    row = _clean_row(
        {
            "bid": 0.0,
            "ask": 0.0,
            "lastPrice": _quote_price(),
            "lastTradeDate": _trade_time(days=10),
            "volume": 100.0,
            "openInterest": 500.0,
        },
        max_trade_age_hours=72.0,
    )

    assert pd.isna(row["impliedVolatilityFinal"])
    assert bool(row["includeInSurface"]) is False
    assert {"stale_last_trade", "recomputed_iv_unreliable"} <= set(
        row["qualityFlags"].split(";")
    )


def test_arbitrage_violation_flagged_and_excluded():
    row = _clean_row(
        {"impliedVolatility": 1e-5, "bid": 149.0, "ask": 151.0, "lastPrice": 150.0}
    )

    assert "arbitrage_violation" in row["qualityFlags"]
    assert bool(row["includeInSurface"]) is False
    assert pd.isna(row["impliedVolatilityFinal"])


@pytest.mark.parametrize(
    ("volume", "open_interest", "expected_flags", "confidence", "included"),
    [
        (np.nan, np.nan, {"oi_zero_or_missing"}, {"medium", "low"}, None),
        (0.0, 0.0, {"oi_zero_or_missing", "volume_zero_or_missing"}, {"low"}, False),
    ],
    ids=["nan-liquidity", "zero-liquidity"],
)
def test_missing_liquidity_is_flagged(volume, open_interest, expected_flags, confidence, included):
    row = _clean_row(
        {"volume": volume, "openInterest": open_interest, "impliedVolatility": 0.31}
    )

    assert expected_flags <= set(row["qualityFlags"].split(";"))
    assert row["confidenceLevel"] in confidence
    if included is not None:
        assert bool(row["includeInSurface"]) is included
        assert row["surfaceWeight"] < 0.5


def test_min_dte_filter_removes_short_dated_contracts():
    now = datetime.now(timezone.utc)
    cleaned = _clean_rows(
        _base_row(expirationDate=(now + timedelta(days=10)).date(), strike=100.0),
        _base_row(expirationDate=(now + timedelta(days=45)).date(), strike=105.0),
        min_dte=20,
        max_dte=60,
    )

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["days_to_expiration"] >= 20


@pytest.mark.parametrize(
    ("as_of", "expected"),
    [("2026-05-15 14:00:00", 0), ("2026-05-15 20:30:00", -1)],
    ids=["before-close", "after-close"],
)
def test_expiration_dte_respects_new_york_close(as_of, expected):
    assert expiration_dte("2026-05-15", pd.Timestamp(as_of, tz="UTC")) == expected


def test_prepare_options_data_retains_zero_dte_before_close():
    market_price = _black_scholes_call_price(100.0, 100.0, 6 / (24 * 365.25), 0.02, 0.30)
    cleaned = _clean_rows(
        _base_row(
            expirationDate=pd.Timestamp("2026-05-15").date(),
            bid=market_price * 0.99,
            ask=market_price * 1.01,
            lastPrice=market_price,
            lastTradeDate="2026-05-15T13:45:00Z",
        ),
        min_dte=0,
        max_dte=0,
        as_of_utc=pd.Timestamp("2026-05-15 14:00:00", tz="UTC"),
    )

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["days_to_expiration"] == 0
    assert cleaned.iloc[0]["time_to_expiration_years"] > 0


def test_diagnostics_report_counts_contracts_and_exclusions():
    report = build_diagnostics_report(
        _clean_rows(
            _base_row(impliedVolatility=0.25),
            _base_row(
                strike=105.0,
                impliedVolatility=1e-5,
                bid=0.0,
                ask=0.0,
                lastPrice=0.0,
                lastTradeDate=_trade_time(days=7),
            ),
        )
    )

    assert report["rows_retained"] == 2
    assert report["rows_surface_excluded"] >= 1
    assert sum(report["dropped_rows_by_reason"].values()) == report["rows_surface_excluded"]
    assert set(report) == {
        "rows_retained",
        "rows_surface_included",
        "rows_surface_excluded",
        "surface_dte_min",
        "surface_dte_max",
        "dropped_rows_by_reason",
    }


def test_diagnostics_exclusion_reasons_assign_one_primary_reason_per_contract():
    report = build_diagnostics_report(
        pd.DataFrame(
            [
                (10, True, "high", "none"),
                (10, False, "low", "iv_unavailable;recomputed_iv_unreliable;wide_recompute_spread"),
                (20, False, "low", "low_open_interest;volume_zero_or_missing"),
            ],
            columns="days_to_expiration includeInSurface confidenceLevel qualityFlags".split(),
        )
    )

    assert report["rows_surface_excluded"] == 2
    assert report["dropped_rows_by_reason"] == {
        "insufficient_liquidity": 1,
        "wide_recompute_spread": 1,
    }
    assert sum(report["dropped_rows_by_reason"].values()) == report["rows_surface_excluded"]
