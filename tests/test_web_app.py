import pandas as pd
import plotly.graph_objects as go
import pytest

from src.surface_service import SurfaceBuildResult
from web_app import create_app


def _build_result(request):
    diagnostics = {
        "request": {"dte_min": request.dte_min, "dte_max": request.dte_max},
        "rows_retained": 12,
        "rows_surface_included": 8,
        "rows_surface_excluded": 4,
        "surface_dte_min": 14,
        "surface_dte_max": 68,
        "dropped_rows_by_reason": {"iv_unavailable": 3, "wide_spread": 1},
    }
    return SurfaceBuildResult(
        current_price=123.45,
        cleaned_options_df=pd.DataFrame([{"a": 1}] * 8),
        diagnostics=diagnostics,
        figure=go.Figure(go.Scatter(x=[1, 2], y=[1, 2])),
    )


def _get(path="/", builder=_build_result):
    return create_app(surface_builder=builder).test_client().get(path)


def _get_captured(path):
    captured = {}
    def builder(request):
        captured["request"] = request
        return _build_result(request)
    return _get(path, builder), captured["request"]


def _assert_contains(response, *values):
    for value in values:
        assert value in response.data


def test_index_renders_empty_state():
    response = _get()

    assert response.status_code == 200
    _assert_contains(
        response,
        b"Volatility Surface Explorer",
        b"Enter a ticker to build a surface",
        b"loading-surface-card",
        b"loading-spinner",
        b"range-control",
        b'id="strike-min-value" name="strike_min_pct" type="number" min="50" max="150" step="1" value="90"',
        b'id="strike-max-value" name="strike_max_pct" type="number" min="50" max="150" step="1" value="110"',
        b'id="dte-min-value" name="dte_min" type="number" min="0" max="365" step="1" value="0"',
        b'id="dte-max-value" name="dte_max" type="number" min="0" max="365" step="1" value="60"',
    )
    assert b"Lightweight web UI" not in response.data
    assert b"Unified Arbitrage-Free Surface" not in response.data
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]


def test_index_builds_surface_from_query_params():
    response, request = _get_captured(
        "/?ticker=tsla&strike_min_pct=85&strike_max_pct=120&dte_min=14&dte_max=90&smooth=1"
    )

    assert response.status_code == 200
    _assert_contains(
        response,
        b"TSLA surface",
        b"plotly-graph-div",
        b"Requested DTE",
        b"Available DTE",
        b"Advanced info",
        b"Raw Contracts",
        b"Filtered Contracts",
        b"Excluded Contracts",
        b"Reasons for Exclusion",
        b"IV Unavailable",
        b"Wide Spread",
        b"data-spot-line-toggle",
        b"plot-loading-overlay",
        b"Building surface",
    )
    for removed_text in (
        b"Surface Quotes",
        b"Retained Rows",
        b"Excluded Rows",
        b"Repricing MAE",
        b"Quality Summary",
        b"Flag Counts",
        b"Interactive Plot",
    ):
        assert removed_text not in response.data
    assert sum(_build_result(request).diagnostics["dropped_rows_by_reason"].values()) == 4
    assert (request.ticker, request.strike_min_pct, request.strike_max_pct) == (
        "TSLA",
        0.85,
        1.2,
    )
    assert (request.dte_min, request.dte_max, request.smooth) == (14, 90, True)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("ticker=spy&dte_min=0&dte_max=0", {"dte_min": 0, "dte_max": 0}),
        (
            "ticker=spy&strike_min_pct=55&strike_max_pct=145&dte_min=4&dte_max=240",
            {"strike_min_pct": 0.55, "strike_max_pct": 1.45, "dte_min": 4, "dte_max": 240},
        ),
        ("ticker=spy&smooth=0", {"smooth": False}),
        ("ticker=spy&smooth=0&smooth=1", {"smooth": True}),
    ],
    ids=["zero-dte", "full-range", "smooth-disabled", "smooth-checked"],
)
def test_index_parses_form_values(query, expected):
    response, request = _get_captured(f"/?{query}")

    assert response.status_code == 200
    for field, value in expected.items():
        assert getattr(request, field) == value
    if expected == {"dte_min": 0, "dte_max": 0}:
        _assert_contains(
            response,
            b'id="dte-min-value" name="dte_min" type="number" min="0"',
            b'data-range-control data-min-gap="0"',
        )
    if expected == {"smooth": False}:
        assert b'name="smooth" value="1" checked' not in response.data


def test_index_has_no_iv_source_selector():
    response, request = _get_captured("/?ticker=spy")

    assert response.status_code == 200
    assert request.ticker == "SPY"
    assert b'name="iv_source"' not in response.data
    assert b"IV Source" not in response.data


def test_index_shows_error_state_when_surface_build_fails():
    def failing_builder(_):
        raise ValueError("No suitable options remained after filtering.")

    response = _get("/?ticker=BAD", failing_builder)

    assert response.status_code == 200
    _assert_contains(response, b"build the surface", b"No suitable options remained after filtering.")


def test_index_rejects_invalid_ticker_before_fetching():
    response = create_app().test_client().get("/?ticker=%3Cscript%3E")

    assert response.status_code == 200
    assert b"Ticker symbols may only contain" in response.data
