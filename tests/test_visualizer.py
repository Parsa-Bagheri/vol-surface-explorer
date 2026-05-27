from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.data_cleaner import _black_scholes_price
from src.visualizer import _build_arbitrage_free_surface, create_vol_surface


def _build_surface_input():
    now = datetime.now(timezone.utc)
    base_rows = []
    for option_type, iv_base in [("call", 0.20), ("put", 0.24)]:
        for strike in [95.0, 100.0, 105.0]:
            for dte in [10, 20, 30]:
                base_rows.append(
                    {
                        "strike": strike,
                        "days_to_expiration": dte,
                        "time_to_expiration_years": dte / 365.25,
                        "impliedVolatilityRaw": iv_base,
                        "impliedVolatilityFinal": iv_base + (strike - 100.0) * 0.001,
                        "impliedVolatility": iv_base + (strike - 100.0) * 0.001,
                        "expirationDate": (now + timedelta(days=dte)).date(),
                        "optionType": option_type,
                        "volume": 100.0,
                        "openInterest": 200.0,
                        "bid": 1.0,
                        "ask": 1.2,
                        "lastPrice": 1.1,
                        "lastTradeDate": now.isoformat(),
                        "marketPrice": 1.1,
                        "priceSourceUsed": "mid",
                        "spreadRatio": 0.1,
                        "ivSourceUsed": "yfinance",
                        "ivComputationMethod": "yfinance",
                        "confidenceLevel": "high",
                        "qualityFlags": "none",
                        "includeInSurface": True,
                    }
                )
    return pd.DataFrame(base_rows)


def test_create_vol_surface_smoothed_creates_one_unified_surface():
    df = _build_surface_input()
    fig = create_vol_surface(df, ticker="TEST", smooth=True, underlying_price=100.0)

    assert fig is not None
    assert len(fig.data) == 1
    trace_types = {trace.type for trace in fig.data}
    assert "surface" in trace_types
    assert "mesh3d" not in trace_types


def test_unified_surface_collapses_call_and_put_quotes_into_one_node_per_strike_and_dte():
    df = _build_surface_input()
    _, _, _, _, surface_nodes = _build_arbitrage_free_surface(
        df=df,
        underlying_price=100.0,
        risk_free_rate=0.02,
        dividend_yield=0.0,
        dte_step=1,
    )

    assert len(surface_nodes) == 9
    preferred_side_by_strike = dict(
        surface_nodes[["strike", "preferredSurfaceSide"]].drop_duplicates().itertuples(index=False, name=None)
    )
    assert preferred_side_by_strike[95.0] == "put"
    assert preferred_side_by_strike[105.0] == "call"


def test_arbitrage_free_surface_projection_enforces_discrete_static_arbitrage():
    now = datetime.now(timezone.utc)
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
                    "expirationDate": (now + timedelta(days=dte)).date(),
                    "optionType": "call",
                    "volume": 25.0 if strike == 100.0 else 1.0,
                    "openInterest": 150.0 if strike == 100.0 else 5.0,
                    "spreadRatio": 0.05 if strike == 100.0 else 0.40,
                    "confidenceLevel": "high" if strike == 100.0 else "medium",
                    "qualityFlags": "none",
                    "includeInSurface": True,
                }
            )
    df = pd.DataFrame(rows)

    _, grid_strike, grid_dte, iv_grid, _ = _build_arbitrage_free_surface(
        df=df,
        underlying_price=100.0,
        risk_free_rate=0.02,
        dividend_yield=0.0,
        dte_step=1,
    )

    assert grid_strike.shape == grid_dte.shape == iv_grid.shape
    assert np.all(np.isfinite(iv_grid))
    assert np.all(iv_grid >= 0.0)

    for row_index in range(iv_grid.shape[0]):
        time_to_expiration = grid_dte[row_index, 0] / 365.25
        prices = np.array(
            [
                _black_scholes_price(
                    option_type="call",
                    spot=100.0,
                    strike=float(strike),
                    time_to_expiration=float(time_to_expiration),
                    risk_free_rate=0.02,
                    volatility=float(volatility),
                    dividend_yield=0.0,
                )
                for strike, volatility in zip(grid_strike[row_index], iv_grid[row_index])
            ]
        )
        first_difference = np.diff(prices)
        strike_steps = np.diff(grid_strike[row_index])
        slopes = first_difference / strike_steps
        assert np.all(first_difference <= 1e-8)
        assert np.all(np.diff(slopes) >= -1e-6)

    total_variance = iv_grid**2 * (grid_dte / 365.25)
    assert np.all(np.diff(total_variance, axis=0) >= -1e-10)


def _build_iv_source_surface_input(selected_iv: float, quoted_iv: float = 0.20):
    now = datetime.now(timezone.utc)
    rows = []
    for dte in [20, 45]:
        t = dte / 365.25
        for strike in [90.0, 95.0, 100.0, 105.0, 110.0]:
            quoted_market_price = _black_scholes_price(
                option_type="call",
                spot=100.0,
                strike=strike,
                time_to_expiration=t,
                risk_free_rate=0.02,
                volatility=quoted_iv,
                dividend_yield=0.0,
            )
            rows.append(
                {
                    "strike": strike,
                    "days_to_expiration": dte,
                    "time_to_expiration_years": t,
                    "impliedVolatilityFinal": selected_iv,
                    "expirationDate": (now + timedelta(days=dte)).date(),
                    "optionType": "call",
                    "volume": 100.0,
                    "openInterest": 250.0,
                    "marketPrice": quoted_market_price,
                    "spreadRatio": 0.05,
                    "confidenceLevel": "high",
                    "qualityFlags": "none",
                    "includeInSurface": True,
                }
            )
    return pd.DataFrame(rows)


def test_surface_construction_uses_selected_iv_not_market_price_fallback():
    low_iv_df = _build_iv_source_surface_input(selected_iv=0.20, quoted_iv=0.20)
    high_provider_iv_df = _build_iv_source_surface_input(selected_iv=0.35, quoted_iv=0.20)

    _, _, _, _, low_nodes = _build_arbitrage_free_surface(
        df=low_iv_df,
        underlying_price=100.0,
        risk_free_rate=0.02,
        dividend_yield=0.0,
        dte_step=1,
    )
    _, _, _, _, high_nodes = _build_arbitrage_free_surface(
        df=high_provider_iv_df,
        underlying_price=100.0,
        risk_free_rate=0.02,
        dividend_yield=0.0,
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
        ticker="TEST",
        smooth=True,
        include_low_confidence=True,
        underlying_price=100.0,
    )

    assert len(fig.data) >= 1


def test_create_vol_surface_honors_requested_dte_axis_range():
    df = _build_surface_input()

    fig = create_vol_surface(
        df,
        ticker="TEST",
        smooth=False,
        underlying_price=100.0,
        dte_range=(7, 60),
    )

    assert list(fig.layout.scene.yaxis.range) == [7, 60]


def test_create_vol_surface_leaves_title_to_web_shell():
    df = _build_surface_input()

    fig = create_vol_surface(df, ticker="TEST", smooth=True, underlying_price=100.0)

    assert fig.layout.title.text is None
    assert fig.layout.paper_bgcolor == "#101a2a"
    assert fig.layout.margin.t == 18
