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
        "black_scholes_iv_fraction": 0.25,
        "provider_iv_fraction": 0.75,
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
    assert b"Lightweight web UI" not in response.data
    assert b"Unified Arbitrage-Free Surface" not in response.data
    assert b"Enter a ticker to build a surface" in response.data
    assert b"loading-surface-card" in response.data
    assert b"loading-spinner" in response.data
    assert b"range-control" in response.data
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
    assert b"Available DTE" in response.data
    assert b"Advanced info" in response.data
    assert b"Raw Contracts" in response.data
    assert b"plot-loading-overlay" in response.data
    assert b"Building surface" in response.data
    assert b"Interactive Plot" not in response.data
    assert captured["request"].ticker == "TSLA"
    assert captured["request"].strike_min_pct == 0.85
    assert captured["request"].strike_max_pct == 1.2
    assert captured["request"].dte_min == 14
    assert captured["request"].dte_max == 90
    assert captured["request"].iv_source == "yfinance"
    assert captured["request"].smooth is True


def test_index_uses_auto_iv_source_by_default():
    captured = {}

    def fake_builder(request: SurfaceRequest) -> SurfaceBuildResult:
        captured["request"] = request
        return _build_result(request)

    app = create_app(surface_builder=fake_builder)
    client = app.test_client()

    response = client.get("/?ticker=spy")

    assert response.status_code == 200
    assert captured["request"].ticker == "SPY"
    assert captured["request"].iv_source == "auto"
    assert b"value=\"auto\" selected" in response.data


def test_index_accepts_editable_range_values_across_full_bounds():
    captured = {}

    def fake_builder(request: SurfaceRequest) -> SurfaceBuildResult:
        captured["request"] = request
        return _build_result(request)

    app = create_app(surface_builder=fake_builder)
    client = app.test_client()

    response = client.get(
        "/?ticker=spy&strike_min_pct=55&strike_max_pct=145&dte_min=4&dte_max=240"
    )

    assert response.status_code == 200
    assert captured["request"].strike_min_pct == 0.55
    assert captured["request"].strike_max_pct == 1.45
    assert captured["request"].dte_min == 4
    assert captured["request"].dte_max == 240


def test_index_allows_smoothed_mode_to_be_disabled_from_form_submission():
    captured = {}

    def fake_builder(request: SurfaceRequest) -> SurfaceBuildResult:
        captured["request"] = request
        return _build_result(request)

    app = create_app(surface_builder=fake_builder)
    client = app.test_client()

    response = client.get(
        "/?ticker=spy&strike_min_pct=93&strike_max_pct=107&dte_min=7&dte_max=60"
        "&iv_source=yfinance&smooth=0"
    )

    assert response.status_code == 200
    assert captured["request"].smooth is False
    assert b'name="smooth" value="1" checked' not in response.data


def test_index_keeps_checked_smoothed_mode_when_hidden_fallback_is_submitted():
    captured = {}

    def fake_builder(request: SurfaceRequest) -> SurfaceBuildResult:
        captured["request"] = request
        return _build_result(request)

    app = create_app(surface_builder=fake_builder)
    client = app.test_client()

    response = client.get(
        "/?ticker=spy&strike_min_pct=93&strike_max_pct=107&dte_min=7&dte_max=60"
        "&iv_source=yfinance&smooth=0&smooth=1"
    )

    assert response.status_code == 200
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
