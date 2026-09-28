import pandas as pd
import pytest
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
        "flag_counts": {"low_volume": 2},
        "internal_validation": {"repricing_mae": 0.1234},
        "contract_outcomes": {"used": 8, "excluded": 4, "reasons": [{"label": "Insufficient liquidity", "count": 4}], "accepted_trade_count": 0},
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
    assert response.headers["Cache-Control"] == "public, max-age=0"
    assert "s-maxage=3600" in response.headers["Vercel-CDN-Cache-Control"]
    assert b"Include .TO for TSX stocks." in response.data
    assert b'/static/app.css?v=' in response.data
    assert b'/static/app.js?v=' in response.data


def test_index_builds_surface_from_query_params():
    captured = {}

    def fake_builder(request: SurfaceRequest) -> SurfaceBuildResult:
        captured["request"] = request
        return _build_result(request)

    app = create_app(surface_builder=fake_builder)
    client = app.test_client()

    response = client.get(
        "/?ticker=tsla&strike_min_pct=85&strike_max_pct=120&dte_min=14&dte_max=90&smooth=1"
    )

    assert response.status_code == 200
    assert b"TSLA surface" in response.data
    assert b"plotly-graph-div" in response.data
    assert b"Requested DTE" in response.data
    assert b"Available DTE" in response.data
    assert b"Advanced info" in response.data
    assert b"Contracts fetched" in response.data
    assert b"Contracts used" in response.data
    assert b"Repricing MAE" not in response.data
    assert b"Quality Summary" not in response.data
    assert b"Insufficient liquidity: 4" in response.data
    assert b"plot-loading-overlay" in response.data
    assert b"Building surface" in response.data
    assert b"Interactive Plot" not in response.data
    assert captured["request"].ticker == "TSLA"
    assert captured["request"].strike_min_pct == 0.85
    assert captured["request"].strike_max_pct == 1.2
    assert captured["request"].dte_min == 14
    assert captured["request"].dte_max == 90
    assert captured["request"].smooth is True
    assert response.headers["Cache-Control"] == "no-store"
    assert "Vercel-CDN-Cache-Control" not in response.headers


def test_index_has_no_iv_source_selector():
    captured = {}

    def fake_builder(request: SurfaceRequest) -> SurfaceBuildResult:
        captured["request"] = request
        return _build_result(request)

    app = create_app(surface_builder=fake_builder)
    client = app.test_client()

    response = client.get("/?ticker=spy")

    assert response.status_code == 200
    assert captured["request"].ticker == "SPY"
    assert b'name="iv_source"' not in response.data
    assert b"IV Source" not in response.data


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
        "&smooth=0"
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
        "&smooth=0&smooth=1"
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


@pytest.mark.parametrize("ticker", ["VIX", "^VIX", "vix"])
def test_vix_explains_unsupported_model_before_building(ticker):
    def unexpected_builder(request):
        pytest.fail("Unsupported VIX must not invoke the equity surface builder")

    response = create_app(surface_builder=unexpected_builder).test_client().get(
        "/", query_string={"ticker": ticker}
    )
    assert b"VIX options exist" in response.data
    assert b"expiry-specific VIX forward prices" in response.data
    with pytest.raises(ValueError, match="VIX options exist"):
        from src.surface_service import validate_ticker_symbol
        validate_ticker_symbol(ticker)


def test_index_accepts_a_zero_day_only_range():
    captured = {}

    def builder(request):
        captured["request"] = request
        return _build_result(request)

    response = create_app(surface_builder=builder).test_client().get(
        "/?ticker=spy&dte_min=0&dte_max=0"
    )
    assert response.status_code == 200
    assert captured["request"].dte_min == captured["request"].dte_max == 0
    assert b'data-min-gap="0"' in response.data
    assert b'name="dte_min" type="number" min="0"' in response.data


def test_empty_homepage_does_not_import_the_financial_stack():
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import sys; import web_app; "
        "assert web_app.app.test_client().get('/').status_code == 200; "
        "assert not {'numpy', 'pandas', 'scipy', 'plotly', 'yfinance'} & set(sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
