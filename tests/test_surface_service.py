import pandas as pd
import plotly.graph_objects as go

import src.surface_service as surface_service
from src.surface_service import SurfaceRequest


def test_build_surface_bundle_passes_requested_dte_range_to_fetch_clean_and_plot(monkeypatch):
    captured = {}
    raw_df = pd.DataFrame([{"strike": 100.0}])
    raw_df.attrs["fetchDiagnostics"] = {
        "selected_expirations": [
            {"expiration": "2026-05-21", "days_to_expiration": 7},
            {"expiration": "2026-06-30", "days_to_expiration": 47},
        ],
        "available_expirations": [
            {"expiration": "2026-05-21", "days_to_expiration": 7},
            {"expiration": "2026-06-30", "days_to_expiration": 47},
            {"expiration": "2026-07-17", "days_to_expiration": 64},
        ],
    }
    cleaned_df = pd.DataFrame(
        [
            {
                "strike": 100.0,
                "days_to_expiration": 12,
                "time_to_expiration_years": 12 / 365.25,
                "impliedVolatilityFinal": 0.22,
                "optionType": "call",
                "includeInSurface": True,
                "marketPrice": 1.0,
                "dividendYieldUsed": 0.0,
            }
        ]
    )

    monkeypatch.setattr(surface_service, "get_current_price", lambda ticker: 100.0)

    def fake_get_options_data(ticker, min_dte, max_dte, as_of_utc):
        captured["fetch_dte_range"] = (min_dte, max_dte)
        captured["fetch_as_of_utc"] = as_of_utc
        return raw_df.copy()

    def fake_prepare_options_data(options_df, min_dte, max_dte, as_of_utc, **kwargs):
        captured["clean_dte_range"] = (min_dte, max_dte)
        captured["clean_as_of_utc"] = as_of_utc
        return cleaned_df.copy()

    def fake_create_vol_surface(df, ticker, dte_range, **kwargs):
        captured["plot_dte_range"] = dte_range
        captured["plot_strike_range"] = kwargs["strike_range"]
        return go.Figure()

    monkeypatch.setattr(surface_service, "get_options_data", fake_get_options_data)
    monkeypatch.setattr(surface_service, "prepare_options_data", fake_prepare_options_data)
    monkeypatch.setattr(
        surface_service,
        "build_diagnostics_report",
        lambda cleaned, raw_row_count: {
            "rows_surface_included": len(cleaned),
            "surface_dte_min": int(cleaned["days_to_expiration"].min()),
            "surface_dte_max": int(cleaned["days_to_expiration"].max()),
        },
    )
    monkeypatch.setattr(surface_service, "create_vol_surface", fake_create_vol_surface)

    result = surface_service.build_surface_bundle(
        SurfaceRequest(ticker="XLY", dte_min=7, dte_max=60, iv_source="yfinance")
    )

    assert captured["fetch_dte_range"] == (7, 60)
    assert captured["clean_dte_range"] == (7, 60)
    assert captured["plot_dte_range"] == (7, 60)
    assert captured["plot_strike_range"] == (93.0, 107.0)
    assert captured["fetch_as_of_utc"] == captured["clean_as_of_utc"]
    assert result.diagnostics["internal_validation"]["requested_dte_min"] == 7
    assert result.diagnostics["internal_validation"]["requested_dte_max"] == 60
    assert result.diagnostics["static_arbitrage"]["certification_scope"] in {
        "nodes_only",
        "grid",
        "not_certified",
    }
    assert result.diagnostics["dte_coverage"]["selected_max"] == 47
    assert result.diagnostics["dte_coverage"]["next_available_after_requested_max"] == 64
    assert "No listed expiration exists" in result.diagnostics["dte_coverage"]["notes"][0]
