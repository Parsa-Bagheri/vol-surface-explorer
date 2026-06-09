import numpy as np
import pandas as pd
import pytest

from src.data_cleaner import _black_scholes_prices_array
from src.visualizer import _build_arbitrage_free_surface, create_vol_surface


def _build_surface_input():
    return pd.DataFrame(
        [
            {
                "strike": strike,
                "days_to_expiration": dte,
                "time_to_expiration_years": dte / 365.25,
                "impliedVolatilityFinal": iv_base + (strike - 100.0) * 0.001,
                "optionType": option_type,
                "forwardPrice": 100.0 * np.exp(0.02 * dte / 365.25),
                "dividendYieldUsed": 0.0,
                "surfaceWeight": 1.0,
                "volume": 100.0,
                "openInterest": 200.0,
                "spreadRatio": 0.1,
                "confidenceLevel": "high",
                "includeInSurface": True,
            }
            for option_type, iv_base in [("call", 0.20), ("put", 0.24)]
            for strike in [95.0, 100.0, 105.0]
            for dte in [10, 20, 30]
        ]
    )


def test_create_vol_surface_smoothed_creates_one_unified_surface():
    df = _build_surface_input()
    fig = create_vol_surface(df, smooth=True, underlying_price=100.0)

    assert fig is not None
    trace_types = {trace.type for trace in fig.data}
    assert "surface" in trace_types
    assert "mesh3d" not in trace_types
    assert [trace.name for trace in fig.data].count("Current Spot") == 1


def test_current_spot_line_is_dotted_white_and_follows_the_surface():
    df = _build_surface_input()
    fig = create_vol_surface(df, smooth=True, underlying_price=100.0)

    spot_trace = next(trace for trace in fig.data if trace.name == "Current Spot")
    assert spot_trace.type == "scatter3d"
    assert spot_trace.mode == "lines"
    assert spot_trace.line.color == "#ffffff"
    assert spot_trace.line.dash == "dot"
    assert spot_trace.opacity < 1.0
    assert np.allclose(np.asarray(spot_trace.x, dtype=float), 100.0)


def test_unified_surface_collapses_call_and_put_quotes_into_one_node_per_strike_and_dte():
    df = _build_surface_input()
    _, _, _, surface_nodes = _build_arbitrage_free_surface(
        df=df,
        underlying_price=100.0,
        risk_free_rate=0.02,
        dte_step=1,
    )

    assert len(surface_nodes) == 9
    preferred_side_by_strike = dict(
        surface_nodes[["strike", "preferredSurfaceSide"]].drop_duplicates().itertuples(index=False, name=None)
    )
    assert preferred_side_by_strike[95.0] == "put"
    assert preferred_side_by_strike[105.0] == "call"


def test_arbitrage_free_surface_projection_enforces_discrete_static_arbitrage():
    rows = []
    for dte, ivs in [
        (10, {90.0: 0.20, 100.0: 0.36, 110.0: 0.16}),
        (20, {90.0: 0.15, 100.0: 0.12, 110.0: 0.10}),
        (30, {90.0: 0.16, 100.0: 0.14, 110.0: 0.12}),
    ]:
        for strike, implied_volatility in ivs.items():
            rows.append(
                {
                    "strike": strike,
                    "days_to_expiration": dte,
                    "time_to_expiration_years": dte / 365.25,
                    "impliedVolatilityFinal": implied_volatility,
                    "optionType": "call",
                    "forwardPrice": 100.0 * np.exp(0.02 * dte / 365.25),
                    "dividendYieldUsed": 0.0,
                    "surfaceWeight": 1.0,
                    "volume": 25.0 if strike == 100.0 else 1.0,
                    "openInterest": 150.0 if strike == 100.0 else 5.0,
                    "spreadRatio": 0.05 if strike == 100.0 else 0.40,
                    "confidenceLevel": "high" if strike == 100.0 else "medium",
                    "includeInSurface": True,
                }
            )
    df = pd.DataFrame(rows)

    grid_strike, grid_dte, iv_grid, _ = _build_arbitrage_free_surface(
        df=df,
        underlying_price=100.0,
        risk_free_rate=0.02,
        dte_step=1,
    )

    assert grid_strike.shape == grid_dte.shape == iv_grid.shape
    assert np.all(np.isfinite(iv_grid))
    assert np.all(iv_grid >= 0.0)

    for row_index in range(iv_grid.shape[0]):
        time_to_expiration = grid_dte[row_index, 0] / 365.25
        prices = _black_scholes_prices_array(
            option_types="call",
            spot=100.0,
            strikes=grid_strike[row_index],
            time_to_expiration=float(time_to_expiration),
            risk_free_rate=0.02,
            volatility=iv_grid[row_index],
            dividend_yield=0.0,
        )
        first_difference = np.diff(prices)
        strike_steps = np.diff(grid_strike[row_index])
        slopes = first_difference / strike_steps
        assert np.all(first_difference <= 1e-8)
        assert np.all(np.diff(slopes) >= -1e-6)

    total_variance = iv_grid**2 * (grid_dte / 365.25)
    assert np.all(np.diff(total_variance, axis=0) >= -1e-10)


def _build_selected_iv_surface_input(selected_iv: float):
    return pd.DataFrame(
        [
            {
                "strike": strike,
                "days_to_expiration": dte,
                "time_to_expiration_years": dte / 365.25,
                "impliedVolatilityFinal": selected_iv,
                "optionType": "call",
                "forwardPrice": 100.0 * np.exp(0.02 * dte / 365.25),
                "dividendYieldUsed": 0.0,
                "surfaceWeight": 1.0,
                "volume": 100.0,
                "openInterest": 250.0,
                "spreadRatio": 0.05,
                "confidenceLevel": "high",
                "includeInSurface": True,
            }
            for dte in [20, 45]
            for strike in [90.0, 95.0, 100.0, 105.0, 110.0]
        ]
    )


def test_surface_construction_uses_selected_iv():
    low_iv_df = _build_selected_iv_surface_input(selected_iv=0.20)
    high_iv_df = _build_selected_iv_surface_input(selected_iv=0.35)

    _, _, _, low_nodes = _build_arbitrage_free_surface(
        df=low_iv_df,
        underlying_price=100.0,
        risk_free_rate=0.02,
        dte_step=1,
    )
    _, _, _, high_nodes = _build_arbitrage_free_surface(
        df=high_iv_df,
        underlying_price=100.0,
        risk_free_rate=0.02,
        dte_step=1,
    )

    assert float(high_nodes["surfaceImpliedVolatility"].mean()) > float(
        low_nodes["surfaceImpliedVolatility"].mean()
    ) + 0.10


def test_include_low_confidence_allows_valid_rows_marked_excluded():
    df = _build_surface_input()
    df["includeInSurface"] = False
    df["confidenceLevel"] = "low"

    fig = create_vol_surface(
        df,
        smooth=True,
        include_low_confidence=True,
        underlying_price=100.0,
    )

    assert len(fig.data) >= 1


def test_create_vol_surface_supports_zero_dte_with_positive_time_remaining():
    df = _build_surface_input()
    zero_dte = df["days_to_expiration"] == 10
    df.loc[zero_dte, "days_to_expiration"] = 0
    df.loc[zero_dte, "time_to_expiration_years"] = 6.0 / (24.0 * 365.25)

    fig = create_vol_surface(
        df,
        smooth=False,
        underlying_price=100.0,
        dte_range=(0, 30),
    )

    spot_trace = next(trace for trace in fig.data if trace.name == "Current Spot")
    assert 0.0 in list(spot_trace.y)
    assert list(fig.layout.scene.yaxis.range) == [0, 30]


@pytest.mark.parametrize(
    ("range_arg", "axis", "expected"),
    [
        ({"dte_range": (7, 60)}, "yaxis", [7, 60]),
        ({"strike_range": (93.0, 107.0)}, "xaxis", [93.0, 107.0]),
    ],
    ids=["dte", "strike"],
)
def test_create_vol_surface_honors_requested_axis_range(range_arg, axis, expected):
    fig = create_vol_surface(
        _build_surface_input(), smooth=False, underlying_price=100.0, **range_arg
    )

    assert list(getattr(fig.layout.scene, axis).range) == expected


def test_create_vol_surface_leaves_title_to_web_shell():
    df = _build_surface_input()

    fig = create_vol_surface(df, smooth=True, underlying_price=100.0)

    assert fig.layout.title.text is None
    assert fig.layout.paper_bgcolor == "#101a2a"
    assert fig.layout.margin.t == 18
