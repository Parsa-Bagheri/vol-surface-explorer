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
    assert response.headers["Cache-Control"] == "public, max-age=0, must-revalidate"
    assert response.headers["Vercel-CDN-Cache-Control"] == "public, s-maxage=30"


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


def test_preparation_is_explicit_and_does_not_fetch_or_build_a_surface():
    calls = []

    def unexpected_builder(request):
        pytest.fail("Preparing runtime must not fetch/build a surface")

    app = create_app(surface_builder=unexpected_builder,
                     runtime_preparer=lambda: calls.append("prepared"))
    client = app.test_client()
    assert client.get("/").status_code == 200
    assert calls == []
    assert client.get("/prepare-surface").status_code == 405
    response = client.post("/prepare-surface")
    assert response.status_code == 204
    assert response.headers["Cache-Control"] == "no-store"
    assert calls == ["prepared"]


def test_runtime_preparation_is_once_per_worker_and_never_fetches(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import web_app
    import src.data_fetch as data_fetch
    import src.numerical_engine as numerical_engine

    calls = []
    monkeypatch.setattr(web_app, "_runtime_prepared", False)
    monkeypatch.setattr(numerical_engine, "resolve_backend", lambda _: calls.append("backend") or "python")
    def forbidden(*args, **kwargs):
        pytest.fail("Runtime readiness must not access market data")
    monkeypatch.setattr(data_fetch, "get_current_price", forbidden)
    monkeypatch.setattr(data_fetch, "get_options_data", forbidden)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: web_app._prepare_surface_runtime(), range(8)))
    assert calls == ["backend"]
    assert web_app._runtime_prepared


def test_vercel_entrypoint_prepares_before_serving(monkeypatch):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path
    env = dict(os.environ, VOL_NUMERICAL_BACKEND="python", VOL_REQUEST_TIMING="1",
               PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c",
        "import api.index; import web_app; assert web_app._runtime_prepared"],
        cwd=Path(__file__).resolve().parents[1], env=env,
        capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout.split("VOLEXPLORER_TIMING ", 1)[1])
    assert record["event"] == "runtime_ready"
    assert record["preparation_ms"] > 0


def test_surface_cache_has_freshness_and_explicit_bypass():
    client = create_app(surface_builder=_build_result).test_client()
    response = client.get("/?ticker=SPY&dte_min=2&dte_max=60")
    assert response.headers["Vercel-CDN-Cache-Control"] == "public, s-maxage=30"
    assert response.headers["X-Surface-Built-At"].encode() in response.data
    assert b"Fetch latest" in response.data and b"fresh=1" in response.data
    assert response.headers["X-Surface-Timing-Scope"] == "original-build"
    fresh = client.get("/?ticker=SPY&dte_min=2&dte_max=60&fresh=1")
    assert fresh.headers["Cache-Control"] == "no-store"
    assert fresh.headers["Vercel-CDN-Cache-Control"] == "no-store"


@pytest.mark.parametrize("ticker", ["<script>", "SPY"])
def test_failed_or_empty_surfaces_are_never_cached(ticker):
    def empty(request):
        result = _build_result(request)
        result.figure = go.Figure()
        return result
    response = create_app(surface_builder=empty).test_client().get("/", query_string={"ticker": ticker})
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["Vercel-CDN-Cache-Control"] == "no-store"
    assert b'class="message-card error-card"' in response.data
    assert b'class="surface-freshness"' not in response.data


def test_cacheable_responses_preserve_distinct_query_controls():
    client = create_app(surface_builder=_build_result).test_client()
    for ticker, dte in [("SPY", 2), ("TSLA", 7)]:
        response = client.get("/", query_string={"ticker": ticker, "dte_min": dte, "dte_max": 60, "smooth": 0})
        assert (ticker + " surface").encode() in response.data
        assert f'dte_min={dte}'.encode() in response.data
        assert b"smooth=False" in response.data


def test_homepage_preloads_exactly_the_chart_script():
    import re
    from src.plotly_assets import plotly_script_url
    client = create_app(surface_builder=_build_result).test_client()
    homepage = client.get("/").get_data(as_text=True)
    chart = client.get("/?ticker=SPY").get_data(as_text=True)
    url = plotly_script_url()
    assert f'<link rel="preload" href="{url}" as="script">' in homepage
    scripts = re.findall(r'<script[^>]+src="([^"]+)"', chart)
    assert [script for script in scripts if "cdn.plot.ly" in script] == [url]


def test_build_hook_updates_manifest_to_installed_plotly(tmp_path, monkeypatch):
    import json
    import vercel_install
    from plotly.offline import get_plotlyjs_version
    monkeypatch.setattr(vercel_install, "__file__", str(tmp_path / "vercel_install.py"))
    vercel_install.write_plotly_manifest()
    data = json.loads((tmp_path / "static/plotly-version.json").read_text())
    assert data["version"] == get_plotlyjs_version()
