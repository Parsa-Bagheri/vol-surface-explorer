import pandas as pd
import plotly.graph_objects as go

from src.surface_service import SurfaceBuildResult, SurfaceRequest
from web_app import create_app


def _build_result(request: SurfaceRequest) -> SurfaceBuildResult:
    figure = go.Figure()
    figure.add_scatter(x=[1, 2], y=[1, 2])
    diagnostics = {
        "rows_retained": 12,
        "rows_surface_included": 8,
        "rows_surface_excluded": 4,
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
    assert b"Ready when you are" in response.data


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
