import pandas as pd
import plotly.graph_objects as go

from src.surface_service import SurfaceBuildResult, SurfaceRequest
from web_app import create_app


def _build_result(request: SurfaceRequest) -> SurfaceBuildResult:
    figure = go.Figure()
    figure.add_scatter(x=[1, 2], y=[1, 2])
    diagnostics = {
        "request": {
            "dte_min": request.dte_min,
            "dte_max": request.dte_max,
        },
        "rows_retained": 12,
        "rows_surface_included": 8,
        "rows_surface_excluded": 4,
        "surface_dte_min": 14,
        "surface_dte_max": 68,
        "fallback_iv_fraction": 0.25,
        "flag_counts": {"low_volume": 2},
        "internal_validation": {"repricing_mae": 0.1234},
    }
    return SurfaceBuildResult(
        request=request,
        current_price=123.45,
        raw_options_df=pd.DataFrame([{"a": 1}] * 12),
        cleaned_options_df=pd.DataFrame([{"a": 1}] * 8),
        diagnostics=diagnostics,
        figure=figure,
    )


def test_index_renders_empty_state():
    app = create_app(surface_builder=_build_result)
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b"Volatility Surface Explorer" in response.data
    assert b"Lightweight web UI for visualizing and exploring implied volatility surfaces of equity options" in response.data
    assert b"Unified Arbitrage-Free Surface" not in response.data
    assert b"Ready when you are" in response.data
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]


def test_index_builds_surface_from_query_params():
    captured = {}

    def fake_builder(request: SurfaceRequest) -> SurfaceBuildResult:
        captured["request"] = request
        return _build_result(request)

    app = create_app(surface_builder=fake_builder)
    client = app.test_client()

    response = client.get(
        "/?ticker=tsla&strike_min_pct=85&strike_max_pct=120&dte_min=14&dte_max=90&iv_source=yfinance&smooth=1"
    )

    assert response.status_code == 200
    assert b"TSLA surface" in response.data
    assert b"plotly-graph-div" in response.data
    assert b"Requested DTE" in response.data
    assert b"Included DTE" in response.data
    assert captured["request"].ticker == "TSLA"
    assert captured["request"].strike_min_pct == 0.85
    assert captured["request"].strike_max_pct == 1.2
    assert captured["request"].dte_min == 14
    assert captured["request"].dte_max == 90
    assert captured["request"].iv_source == "yfinance"
    assert captured["request"].smooth is True


def test_index_shows_error_state_when_surface_build_fails():
    def failing_builder(_: SurfaceRequest) -> SurfaceBuildResult:
        raise ValueError("No suitable options remained after filtering.")

    app = create_app(surface_builder=failing_builder)
    client = app.test_client()

    response = client.get("/?ticker=BAD")

    assert response.status_code == 200
    assert b"build the surface" in response.data
    assert b"No suitable options remained after filtering." in response.data


def test_index_rejects_invalid_ticker_before_fetching():
    app = create_app()
    client = app.test_client()

    response = client.get("/?ticker=%3Cscript%3E")

    assert response.status_code == 200
    assert b"Ticker symbols may only contain" in response.data
