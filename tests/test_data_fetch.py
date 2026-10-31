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


@pytest.mark.parametrize("all_fail", [False, True])
def test_expiration_failures_preserve_partial_and_total_failure_metadata(monkeypatch, all_fail):
    class FakeTicker:
        options = ("2026-10-16", "2026-10-23")

        def option_chain(self, expiration):
            if all_fail or expiration == "2026-10-23":
                raise RuntimeError("Simulated expiration request failure")
            return _fake_chain(expiration)

    monkeypatch.setattr(data_fetch.yf, "Ticker", lambda ticker: FakeTicker())
    result = data_fetch.get_options_data("XLY", retry_on_poor_quality=False,
                                         as_of_utc=pd.Timestamp("2026-10-13T18:00:00Z"))
    fetch = result.attrs["fetchDiagnostics"]
    assert fetch["expirations_requested"] == 2
    assert fetch["expirations_fetched"] == (0 if all_fail else 1)
    assert fetch["outcomes"][-1]["status"] == "failed"
    assert "Simulated" in fetch["outcomes"][-1]["error"]
    assert result.empty == all_fail


def test_price_observation_records_history_fallback_and_source_time(monkeypatch):
    class FakeTicker:
        def history(self, period):
            if period == "1d":
                return pd.DataFrame()
            return pd.DataFrame({"Close": [101.25]}, index=pd.DatetimeIndex(["2026-10-12T20:00:00Z"]))

    monkeypatch.setattr(data_fetch.yf, "Ticker", lambda ticker: FakeTicker())
    result = data_fetch.get_price_observation("XLY")
    assert result["value"] == 101.25
    assert result["method"] == "history:5d:Close"
    assert result["source_timestamp_utc"] == "2026-10-12T20:00:00+00:00"


def test_workers_reuse_discovery_and_keep_independent_expiration_maps(monkeypatch):
    instances = []
    discovery_calls = []
    dates = ("2026-10-16", "2026-10-23")

    class FakeTicker:
        def __init__(self, ticker):
            self._expirations = {}
            instances.append(self)

        @property
        def options(self):
            discovery_calls.append(True)
            self._expirations = {date: index for index, date in enumerate(dates)}
            return dates

        def option_chain(self, expiration):
            if not self._expirations:
                self.options
            assert expiration in self._expirations
            return _fake_chain(expiration)

    monkeypatch.setattr(data_fetch.yf, "Ticker", FakeTicker)
    result = data_fetch.get_options_data("SPY", retry_on_poor_quality=False,
                                         as_of_utc=pd.Timestamp("2026-10-13T18:00:00Z"))
    assert len(discovery_calls) == 1
    assert result.attrs["fetchDiagnostics"]["expirations_fetched"] == 2
    assert set(result["expirationDate"].dt.strftime("%Y-%m-%d")) == set(dates)
    assert len(instances) == 3
    assert len({id(instance._expirations) for instance in instances}) == 3
    instances[1]._expirations["worker-only"] = 123
    assert all("worker-only" not in instance._expirations for instance in (instances[0], instances[2]))
