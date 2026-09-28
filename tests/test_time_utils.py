import numpy as np
import pandas as pd
import pytest

from src.data_cleaner import _black_scholes_price, prepare_options_data
from src.time_utils import expiration_close_utc, expiration_dte, expiration_dtes


@pytest.mark.parametrize(
    "as_of,expiry,expected",
    [
        ("2026-09-28T19:00:00Z", "2026-09-28", 0),
        ("2026-09-29T01:00:00Z", "2026-09-29", 1),
        ("2026-12-15T01:00:00Z", "2026-12-15", 1),
        ("2026-09-29T04:01:00Z", "2026-09-29", 0),
        ("2026-12-15T05:01:00Z", "2026-12-15", 0),
        ("2026-03-07T18:00:00Z", "2026-03-09", 2),
        ("2026-10-31T18:00:00Z", "2026-11-02", 2),
    ],
)
def test_calendar_dte_is_consistent_for_scalar_and_frame(as_of, expiry, expected):
    as_of = pd.Timestamp(as_of)
    expiries = pd.Series([expiry, None], index=[5, 9])
    assert expiration_dte(expiry, as_of) == expected
    result = expiration_dtes(expiries, as_of)
    assert result.index.tolist() == [5, 9]
    assert result.loc[5] == expected
    assert pd.isna(result.loc[9])
    assert expiration_dte("not a date", as_of) is None


def _quote(expiry, as_of):
    years = (expiration_close_utc(expiry) - as_of).total_seconds() / (365.25 * 86400)
    price = _black_scholes_price("call", 100.0, 100.0, years, 0.02, 0.30)
    return {
        "contractId": 0, "strike": 100.0, "expirationDate": expiry,
        "optionType": "call", "bid": price * 0.995, "ask": price * 1.005,
        "lastPrice": price, "lastTradeDate": as_of,
        "volume": 100, "openInterest": 500,
    }


@pytest.mark.parametrize("as_of,expiry", [
    ("2026-09-29T01:00:00Z", "2026-09-29"),
    ("2026-12-15T01:00:00Z", "2026-12-15"),
])
def test_cleaning_keeps_tomorrow_after_utc_midnight(as_of, expiry):
    as_of = pd.Timestamp(as_of)
    cleaned = prepare_options_data(
        pd.DataFrame([_quote(expiry, as_of)]), underlying_price=100.0,
        min_dte=1, max_dte=1, as_of_utc=as_of,
    )
    assert len(cleaned) == 1
    assert cleaned.iloc[0].days_to_expiration == expiration_dte(expiry, as_of) == 1
    assert cleaned.iloc[0].impliedVolatilityFinal == pytest.approx(0.30, abs=1e-7)


@pytest.mark.parametrize("expiry,as_of", [
    ("2026-09-28", "2026-09-28T19:00:00Z"),
    ("2026-12-14", "2026-12-14T20:00:00Z"),
])
def test_zero_day_contract_uses_positive_fractional_time_until_close(expiry, as_of):
    as_of = pd.Timestamp(as_of)
    raw = pd.DataFrame([_quote(expiry, as_of)])
    cleaned = prepare_options_data(
        raw, underlying_price=100.0, min_dte=0, max_dte=0, as_of_utc=as_of,
    )
    assert len(cleaned) == 1
    assert cleaned.iloc[0].days_to_expiration == 0
    assert cleaned.iloc[0].includeInSurface
    assert cleaned.iloc[0].time_to_expiration_years == pytest.approx(1 / (365.25 * 24))
    assert cleaned.iloc[0].impliedVolatilityFinal == pytest.approx(0.30, abs=1e-7)
    for delay in [pd.Timedelta(hours=1), pd.Timedelta(hours=2)]:
        expired = prepare_options_data(
            raw, underlying_price=100.0, min_dte=0, max_dte=0, as_of_utc=as_of + delay,
        )
        assert expired.empty


@pytest.mark.parametrize("spot", [0.0, float("nan"), float("inf")])
def test_shared_array_pricing_rejects_invalid_spot(spot):
    from src.data_cleaner import _black_scholes_prices_array

    prices = _black_scholes_prices_array("call", spot, np.array([100.0]), 0.1, 0.02, 0.3, 0.0)
    assert np.isnan(prices).all()


def test_shared_array_pricing_broadcasts_and_matches_scalar_prices():
    from src.data_cleaner import _black_scholes_prices_array

    strikes = np.array([90.0, 100.0, 110.0])
    types = np.array(["call", "put", "call"])
    ivs = np.array([0.0, 0.2, 0.4])
    prices = _black_scholes_prices_array(types, 100.0, strikes, 0.1, 0.02, ivs, 0.01)
    expected = [
        _black_scholes_price(kind, 100.0, strike, 0.1, 0.02, iv, 0.01)
        for kind, strike, iv in zip(types, strikes, ivs)
    ]
    np.testing.assert_allclose(prices, expected, atol=1e-12)
    invalid = _black_scholes_prices_array(types, 100.0, strikes, [0.1, 0.2], 0.02, ivs, 0.01)
    assert np.isnan(invalid).all()
