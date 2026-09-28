from types import SimpleNamespace

import pandas as pd
import pytest

import src.data_fetch as data_fetch


def test_get_current_price_falls_back_when_one_day_history_is_empty(monkeypatch):
    class FakeTicker:
        fast_info = {}
        info = {}

        def history(self, period):
            if period == "1d":
                return pd.DataFrame(columns=["Close"])
            return pd.DataFrame({"Close": [99.0, 101.25]})

    monkeypatch.setattr(data_fetch.yf, "Ticker", lambda ticker: FakeTicker())

    assert data_fetch.get_current_price("SPY") == 101.25


def test_get_current_price_uses_info_when_history_is_unavailable(monkeypatch):
    class FakeTicker:
        fast_info = {}
        info = {"regularMarketPrice": 432.10}

        def history(self, period):
            raise RuntimeError("history unavailable")

    monkeypatch.setattr(data_fetch.yf, "Ticker", lambda ticker: FakeTicker())

    assert data_fetch.get_current_price("SPY") == 432.10


def test_get_current_price_returns_none_when_all_sources_are_unavailable(monkeypatch):
    class FakeTicker:
        fast_info = None
        info = None

        def history(self, period):
            return pd.DataFrame(columns=["Close"])

    monkeypatch.setattr(data_fetch.yf, "Ticker", lambda ticker: FakeTicker())

    assert data_fetch.get_current_price("SPY") is None


def _fake_chain(expiration):
    calls = pd.DataFrame(
        [
            {
                "contractSymbol": f"XLY{expiration}C00100000",
                "strike": 100.0,
                "bid": 1.0,
                "ask": 1.2,
                "impliedVolatility": 0.20,
                "openInterest": 100,
            }
        ]
    )
    puts = pd.DataFrame(
        [
            {
                "contractSymbol": f"XLY{expiration}P00100000",
                "strike": 100.0,
                "bid": 1.1,
                "ask": 1.3,
                "impliedVolatility": 0.22,
                "openInterest": 100,
            }
        ]
    )
    return SimpleNamespace(calls=calls, puts=puts)


def test_get_options_data_records_available_expirations_outside_requested_dte_range(monkeypatch):
    class FakeTicker:
        options = (
            "2026-05-15",
            "2026-05-22",
            "2026-05-29",
            "2026-06-05",
            "2026-06-12",
            "2026-06-18",
            "2026-07-17",
        )

        def option_chain(self, expiration):
            return _fake_chain(expiration)

    monkeypatch.setattr(data_fetch.yf, "Ticker", lambda ticker: FakeTicker())

    result = data_fetch.get_options_data(
        "XLY",
        retry_on_poor_quality=False,
        min_dte=7,
        max_dte=60,
        as_of_utc=pd.Timestamp("2026-05-11T03:50:46Z"),
    )

    diagnostics = result.attrs["fetchDiagnostics"]
    assert [
        item["days_to_expiration"] for item in diagnostics["selected_expirations"]
    ] == [12, 19, 26, 33, 39]
    assert [
        item["days_to_expiration"] for item in diagnostics["available_expirations"]
    ] == [5, 12, 19, 26, 33, 39, 68]
    assert sorted(result["expirationDteAtFetch"].unique().tolist()) == [
        12,
        19,
        26,
        33,
        39,
    ]


@pytest.mark.parametrize("as_of,expected_fetches", [
    ("2026-09-28T19:00:00Z", ["2026-09-28"]),
    ("2026-09-28T20:00:00Z", []),
    ("2026-09-28T21:00:00Z", []),
    ("2026-12-14T20:00:00Z", ["2026-12-14"]),
    ("2026-12-14T21:00:00Z", []),
])
def test_zero_day_fetch_only_includes_unexpired_chains(monkeypatch, as_of, expected_fetches):
    fetched = []
    expiry = as_of[:10]

    class FakeTicker:
        options = (expiry,)

        def option_chain(self, expiration):
            fetched.append(expiration)
            return _fake_chain(expiration)

    monkeypatch.setattr(data_fetch.yf, "Ticker", lambda ticker: FakeTicker())
    result = data_fetch.get_options_data(
        "SPY", retry_on_poor_quality=False, min_dte=0, max_dte=0,
        as_of_utc=pd.Timestamp(as_of),
    )
    assert fetched == expected_fetches
    if expected_fetches:
        assert result["expirationDteAtFetch"].unique().tolist() == [0]
    else:
        assert result.empty
