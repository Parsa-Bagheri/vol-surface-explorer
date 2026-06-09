import pandas as pd
import plotly.graph_objects as go
import pytest

import src.surface_service as surface_service
from src.surface_service import SurfaceRequest, _validated_request


def test_build_surface_bundle_passes_requested_dte_range_to_fetch_clean_and_plot(monkeypatch):
    captured = {}
    raw_df = pd.DataFrame([{"strike": 100.0}])
    cleaned_df = pd.DataFrame(
        [
            {
                "strike": 100.0,
                "days_to_expiration": 12,
                "time_to_expiration_years": 12 / 365.25,
                "impliedVolatilityFinal": 0.22,
                "optionType": "call",
                "includeInSurface": True,
                "dividendYieldUsed": 0.0,
            }
        ]
    )

    monkeypatch.setattr(surface_service, "get_current_price", lambda ticker: 100.0)

    def fake_get_options_data(ticker, min_dte, max_dte, as_of_utc):
        captured["fetch_dte_range"] = (min_dte, max_dte)
        captured["fetch_as_of_utc"] = as_of_utc
        return raw_df.copy()

    def fake_prepare_options_data(_options_df, min_dte, max_dte, as_of_utc, **kwargs):
        captured["clean_dte_range"] = (min_dte, max_dte)
        captured["clean_as_of_utc"] = as_of_utc
        return cleaned_df.copy()

    def fake_create_vol_surface(df, dte_range, **kwargs):
        captured["plot_dte_range"] = dte_range
        captured["plot_strike_range"] = kwargs["strike_range"]
        return go.Figure()

    monkeypatch.setattr(surface_service, "get_options_data", fake_get_options_data)
    monkeypatch.setattr(surface_service, "prepare_options_data", fake_prepare_options_data)
    monkeypatch.setattr(
        surface_service,
        "build_diagnostics_report",
        lambda cleaned: {
            "rows_surface_included": len(cleaned),
            "surface_dte_min": int(cleaned["days_to_expiration"].min()),
            "surface_dte_max": int(cleaned["days_to_expiration"].max()),
        },
    )
    monkeypatch.setattr(surface_service, "create_vol_surface", fake_create_vol_surface)

    surface_service.build_surface_bundle(SurfaceRequest(ticker="XLY", dte_min=7, dte_max=60))

    assert captured["fetch_dte_range"] == (7, 60)
    assert captured["clean_dte_range"] == (7, 60)
    assert captured["plot_dte_range"] == (7, 60)
    assert captured["plot_strike_range"] == pytest.approx((90.0, 110.0))
    assert captured["fetch_as_of_utc"] == captured["clean_as_of_utc"]


def test_surface_request_accepts_zero_dte():
    request = _validated_request(SurfaceRequest(ticker="SPY", dte_min=0, dte_max=0))

    assert request.dte_min == 0
    assert request.dte_max == 0


def test_surface_request_defaults_allow_zero_dte():
    request = SurfaceRequest(ticker="SPY")

    assert request.strike_min_pct == 0.90
    assert request.strike_max_pct == 1.10
    assert request.dte_min == 0
    assert request.dte_max == 60
