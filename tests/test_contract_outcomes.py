import pandas as pd
import pytest

import src.contract_outcomes as contract_outcomes
from src.contract_outcomes import build_contract_outcomes, exclusion_reason
from src.surface_service import SurfaceRequest


def test_outcomes_reconcile_and_follow_fitter_selection():
    base = {"days_to_expiration": 30, "time_to_expiration_years": 30 / 365.25,
            "impliedVolatilityFinal": 0.25, "dividendYieldUsed": 0.0,
            "forwardPrice": 100.0, "includeInSurface": True, "marketPrice": 2.0,
            "priceSourceUsed": "mid", "qualityFlags": "none"}
    cleaned = pd.DataFrame([
        dict(base, contractId=0, strike=110., optionType="call"),
        dict(base, contractId=1, strike=110., optionType="put"),
        dict(base, contractId=2, strike=100., optionType="call", includeInSurface=False,
             qualityFlags="wide_iv_bid_ask;iv_unavailable;low_volume"),
    ])
    raw = pd.DataFrame([
        {"contractId": i, "strike": strike, "expirationDate": "2026-12-18"}
        for i, strike in enumerate([110., 110., 100., 200.])
    ])
    result = build_contract_outcomes(raw, cleaned, SurfaceRequest(ticker="SPY"), 100.)
    assert result["fetched"] == 4
    assert result["used"] == 1
    assert result["excluded"] == 3
    assert {r["label"]: r["count"] for r in result["reasons"]} == {
        "Outside selected range": 1, "Alternative quote used": 1, "Quote too uncertain": 1
    }


def test_price_bounds_take_precedence_over_uncertainty_and_liquidity():
    assert exclusion_reason({"marketPrice": 2., "qualityFlags":
        "arbitrage_violation;wide_iv_bid_ask;low_volume"}) == "Price outside model bounds"


@pytest.mark.parametrize("changes,expected", [
    ({"marketPrice": None}, "Missing usable price"),
    ({"qualityFlags": "stale_last_trade;arbitrage_violation"}, "Missing usable price"),
    ({"qualityFlags": "iv_outlier;low_volume"}, "Quote too uncertain"),
    ({"qualityFlags": "low_volume", "impliedVolatilityFinal": None}, "Calculation failed"),
    ({"qualityFlags": "low_volume"}, "Insufficient liquidity"),
    ({"qualityFlags": "not_low_volume"}, "Calculation failed"),
])
def test_vectorized_reasons_preserve_priority_and_match_exact_flags(changes, expected):
    row = {"marketPrice": 2.0, "impliedVolatilityFinal": 0.2, "qualityFlags": "none"}
    row.update(changes)
    assert exclusion_reason(row) == expected


def test_preselected_quotes_are_reused_and_each_exclusion_counted_once(monkeypatch):
    def unexpected_selection(*args):
        pytest.fail("Accounting must reuse the fitter's selected quotes")

    monkeypatch.setattr(contract_outcomes, "select_surface_quotes", unexpected_selection)
    raw = pd.DataFrame([
        {"contractId": 0, "strike": 100.0, "expirationDate": "2026-12-18"},
        {"contractId": 1, "strike": 100.0, "expirationDate": "2026-12-18"},
        {"contractId": 2, "strike": 200.0, "expirationDate": "2026-12-18"},
        {"contractId": 3, "strike": "bad", "expirationDate": "bad"},
    ])
    cleaned = pd.DataFrame([
        {"contractId": 0, "includeInSurface": True, "priceSourceUsed": "lastPrice"},
        {"contractId": 1, "includeInSurface": True, "priceSourceUsed": "mid"},
    ])
    selected = cleaned.iloc[:1]
    result = build_contract_outcomes(raw, cleaned, SurfaceRequest(ticker="SPY"), 100.0, selected)
    assert result["used"] == 1
    assert result["accepted_trade_count"] == 1
    assert result["fetched"] == result["used"] + sum(r["count"] for r in result["reasons"])
    assert {r["label"]: r["count"] for r in result["reasons"]} == {
        "Outside selected range": 1, "Missing contract fields": 1, "Alternative quote used": 1,
    }
