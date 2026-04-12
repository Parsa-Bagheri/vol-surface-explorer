from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.data_cleaner import _black_scholes_price
from src.visualizer import _build_arbitrage_free_surface, create_vol_surface


def _build_surface_input():
    now = datetime.now(timezone.utc)
    base_rows = []
    for option_type, iv_base in [("call", 0.20), ("put", 0.24)]:
        for strike in [95.0, 105.0]:
            for dte in [10, 20]:
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

    assert len(surface_nodes) == 4
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
