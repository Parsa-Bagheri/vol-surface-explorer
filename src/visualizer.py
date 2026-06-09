from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import Bounds, LinearConstraint, minimize

from src.data_cleaner import (
    _black_scholes_prices_array,
    _implied_volatility,
)


ARBITRAGE_K_GRID_SIZE = 81
ARBITRAGE_PROJECTION_ITERATIONS = 3
SURFACE_ATM_LOG_MONEYNESS_BAND = 0.02
MIN_SMOOTH_SLICE_NODE_COUNT = 3
MIN_SMOOTH_MATURITY_SLICE_COUNT = 3
MIN_SMOOTH_SLICE_LOG_MONEYNESS_WIDTH = 0.01
MAX_SMOOTH_DTE_GRID_POINTS = 80
ARBITRAGE_FEASIBILITY_TOLERANCE = 1e-12


def _call_price_slice_is_feasible(
    strikes: np.ndarray,
    call_prices: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    discount_factor: float,
) -> bool:
    tolerance = ARBITRAGE_FEASIBILITY_TOLERANCE
    if (
        not np.all(np.isfinite(call_prices))
        or np.any(call_prices < lower_bounds - tolerance)
        or np.any(call_prices > upper_bounds + tolerance)
    ):
        return False

    strike_steps = np.diff(strikes)
    if np.any(strike_steps <= 0):
        return False

    price_decrease = call_prices[:-1] - call_prices[1:]
    if np.any(price_decrease < -tolerance):
        return False
    if np.any(price_decrease > strike_steps * discount_factor + tolerance):
        return False

    if len(strikes) >= 3:
        slopes = np.diff(call_prices) / strike_steps
        if np.any(np.diff(slopes) < -tolerance):
            return False

    return True


def _project_call_price_slice(
    strikes: np.ndarray,
    call_prices: np.ndarray,
    weights: np.ndarray,
    spot: float,
    time_to_expiration: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> np.ndarray:
    if len(strikes) < 2:
        return call_prices.astype(float)

    strikes = strikes.astype(float)
    call_prices = call_prices.astype(float)
    weights = np.clip(weights.astype(float), 0.05, None)
    discounted_spot = spot * np.exp(-dividend_yield * time_to_expiration)
    discount_factor = np.exp(-risk_free_rate * time_to_expiration)
    lower_bounds = np.maximum(0.0, discounted_spot - strikes * discount_factor)
    upper_bounds = np.full(len(strikes), discounted_spot)
    initial = np.clip(call_prices, lower_bounds, upper_bounds)

    if _call_price_slice_is_feasible(
        strikes=strikes,
        call_prices=call_prices,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        discount_factor=discount_factor,
    ):
        return call_prices.astype(float)

    constraints: List[LinearConstraint] = []
    monotone_matrix = []
    monotone_upper = []
    for index in range(len(strikes) - 1):
        row = np.zeros(len(strikes))
        row[index] = 1.0
        row[index + 1] = -1.0
        monotone_matrix.append(row)
        monotone_upper.append((strikes[index + 1] - strikes[index]) * discount_factor)
    constraints.append(
        LinearConstraint(
            np.array(monotone_matrix),
            np.zeros(len(monotone_matrix)),
            np.array(monotone_upper),
        )
    )

    convex_matrix = []
    for index in range(len(strikes) - 2):
        left_step = strikes[index + 1] - strikes[index]
        right_step = strikes[index + 2] - strikes[index + 1]
        if left_step <= 0 or right_step <= 0:
            continue
        row = np.zeros(len(strikes))
        row[index] = 1.0 / left_step
        row[index + 1] = -(1.0 / left_step) - (1.0 / right_step)
        row[index + 2] = 1.0 / right_step
        convex_matrix.append(row)
    if convex_matrix:
        constraints.append(
            LinearConstraint(np.array(convex_matrix), np.zeros(len(convex_matrix)), np.full(len(convex_matrix), np.inf))
        )

    def objective(candidate: np.ndarray) -> float:
        residual = candidate - call_prices
        return float(np.sum(weights * residual * residual))

    def objective_jacobian(candidate: np.ndarray) -> np.ndarray:
        return 2.0 * weights * (candidate - call_prices)

    result = minimize(
        objective,
        x0=initial,
        jac=objective_jacobian,
        method="SLSQP",
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if result.success and np.all(np.isfinite(result.x)):
        return result.x.astype(float)
    return initial


def _project_total_variance_grid(
    strike_grid: np.ndarray,
    time_to_expiration: np.ndarray,
    dividend_yields: np.ndarray,
    weights: np.ndarray,
    total_variance: np.ndarray,
    underlying_price: float,
    risk_free_rate: float,
) -> np.ndarray:
    for _ in range(ARBITRAGE_PROJECTION_ITERATIONS):
        for row_index, row_strikes in enumerate(strike_grid):
            row_time = float(time_to_expiration[row_index])
            row_dividend_yield = float(dividend_yields[row_index])
            row_volatility = np.sqrt(
                np.maximum(total_variance[row_index], 0.0) / row_time
            )
            call_prices = _black_scholes_prices_array(
                option_types="call",
                spot=underlying_price,
                strikes=row_strikes,
                time_to_expiration=row_time,
                risk_free_rate=risk_free_rate,
                volatility=row_volatility,
                dividend_yield=row_dividend_yield,
            )
            projected_prices = _project_call_price_slice(
                strikes=row_strikes,
                call_prices=call_prices,
                weights=weights[row_index],
                spot=underlying_price,
                time_to_expiration=row_time,
                risk_free_rate=risk_free_rate,
                dividend_yield=row_dividend_yield,
            )
            total_variance[row_index] = np.array(
                [
                    _implied_volatility(
                        option_type="call",
                        spot=underlying_price,
                        strike=float(strike),
                        time_to_expiration=row_time,
                        risk_free_rate=risk_free_rate,
                        market_price=float(projected_price),
                        dividend_yield=row_dividend_yield,
                    )
                    ** 2
                    * row_time
                    for strike, projected_price in zip(row_strikes, projected_prices)
                ],
                dtype=float,
            )

        total_variance = np.maximum.accumulate(total_variance, axis=0)

    return total_variance


def _build_surface_nodes(
    df: pd.DataFrame,
    underlying_price: float,
    risk_free_rate: float,
) -> pd.DataFrame:
    working_df = df.copy()
    working_df = working_df.replace([np.inf, -np.inf], np.nan)
    working_df = working_df.dropna(
        subset=[
            "strike",
            "days_to_expiration",
            "time_to_expiration_years",
            "impliedVolatilityFinal",
            "optionType",
            "forwardPrice",
            "dividendYieldUsed",
            "surfaceWeight",
        ]
    )
    working_df = working_df[
        (working_df["strike"] > 0)
        & (working_df["days_to_expiration"] >= 0)
        & (working_df["time_to_expiration_years"] > 0)
    ].copy()
    if working_df.empty:
        return pd.DataFrame()

    option_types = working_df["optionType"].astype(str).str.lower()
    strikes = pd.to_numeric(working_df["strike"], errors="coerce").to_numpy(dtype=float)
    time_to_expiration = pd.to_numeric(
        working_df["time_to_expiration_years"], errors="coerce"
    ).to_numpy(dtype=float)
    implied_volatility = pd.to_numeric(
        working_df["impliedVolatilityFinal"], errors="coerce"
    ).to_numpy(dtype=float)
    working_df["dividendYieldUsed"] = pd.to_numeric(
        working_df["dividendYieldUsed"], errors="coerce"
    )

    surface_option_prices = _black_scholes_prices_array(
        option_types=option_types.to_numpy(dtype=object),
        spot=underlying_price,
        strikes=strikes,
        time_to_expiration=time_to_expiration,
        risk_free_rate=risk_free_rate,
        volatility=implied_volatility,
        dividend_yield=working_df["dividendYieldUsed"].to_numpy(dtype=float),
    )
    working_df["surfaceOptionPrice"] = surface_option_prices
    working_df = working_df.dropna(subset=["surfaceOptionPrice"]).copy()
    working_df = working_df[working_df["surfaceOptionPrice"] > 0]
    if working_df.empty:
        return pd.DataFrame()

    working_df["forwardPrice"] = pd.to_numeric(
        working_df["forwardPrice"], errors="coerce"
    )

    strikes = working_df["strike"].to_numpy(dtype=float)
    forwards = working_df["forwardPrice"].to_numpy(dtype=float)
    option_types = working_df["optionType"].astype(str).str.lower()
    valid_forward = np.isfinite(forwards) & (forwards > 0) & np.isfinite(strikes) & (strikes > 0)
    log_moneyness = np.full(len(working_df), np.nan, dtype=float)
    log_moneyness[valid_forward] = np.log(strikes[valid_forward] / forwards[valid_forward])
    preferred = np.ones(len(working_df), dtype=bool)
    below_forward = valid_forward & (log_moneyness < -SURFACE_ATM_LOG_MONEYNESS_BAND)
    above_forward = valid_forward & (log_moneyness > SURFACE_ATM_LOG_MONEYNESS_BAND)
    preferred[below_forward] = option_types.to_numpy(dtype=object)[below_forward] == "put"
    preferred[above_forward] = option_types.to_numpy(dtype=object)[above_forward] == "call"
    working_df["surfaceQuotePreferred"] = preferred

    discounted_spot = underlying_price * np.exp(
        -working_df["dividendYieldUsed"].to_numpy(dtype=float)
        * working_df["time_to_expiration_years"].to_numpy(dtype=float)
    )
    discounted_strike = working_df["strike"].to_numpy(dtype=float) * np.exp(
        -risk_free_rate * working_df["time_to_expiration_years"].to_numpy(dtype=float)
    )
    surface_option_prices = working_df["surfaceOptionPrice"].to_numpy(dtype=float)
    call_equivalent_prices = surface_option_prices.copy()
    put_mask = option_types.to_numpy(dtype=object) == "put"
    call_equivalent_prices[put_mask] = (
        surface_option_prices[put_mask]
        + discounted_spot[put_mask]
        - discounted_strike[put_mask]
    )
    working_df["callEquivalentPrice"] = call_equivalent_prices
    working_df["surfaceWeight"] = np.clip(
        pd.to_numeric(working_df["surfaceWeight"], errors="coerce").to_numpy(dtype=float),
        0.05,
        None,
    )

    group_keys = ["days_to_expiration", "strike"]
    group_has_preferred = working_df.groupby(group_keys, sort=False)[
        "surfaceQuotePreferred"
    ].transform("any")
    selected_df = working_df[
        working_df["surfaceQuotePreferred"] | ~group_has_preferred
    ].copy()
    if selected_df.empty:
        return pd.DataFrame()

    selected_df["optionTypeLower"] = selected_df["optionType"].astype(str).str.lower()
    selected_df["weightedCallEquivalentPrice"] = (
        selected_df["callEquivalentPrice"] * selected_df["surfaceWeight"]
    )
    selected_df["volumeNumeric"] = pd.to_numeric(
        selected_df["volume"], errors="coerce"
    ).fillna(0.0)
    selected_df["openInterestNumeric"] = pd.to_numeric(
        selected_df["openInterest"], errors="coerce"
    ).fillna(0.0)

    grouped = selected_df.groupby(group_keys, sort=True, dropna=True)
    aggregated = grouped.agg(
        time_to_expiration_years=("time_to_expiration_years", "first"),
        forwardPrice=("forwardPrice", "first"),
        dividendYieldUsed=("dividendYieldUsed", "first"),
        weightedCallEquivalentPrice=("weightedCallEquivalentPrice", "sum"),
        surfaceWeight=("surfaceWeight", "sum"),
        selectedQuoteCount=("surfaceWeight", "size"),
        totalVolume=("volumeNumeric", "sum"),
        totalOpenInterest=("openInterestNumeric", "sum"),
    ).reset_index()
    aggregated["callEquivalentPrice"] = (
        aggregated["weightedCallEquivalentPrice"] / aggregated["surfaceWeight"]
    )
    selected_types = grouped["optionTypeLower"].agg(
        lambda values: "/".join(sorted(pd.unique(values)))
    ).reset_index(name="selectedOptionTypes")
    aggregated = aggregated.merge(selected_types, on=group_keys, how="left")

    node_log_moneyness = np.log(
        aggregated["strike"].to_numpy(dtype=float)
        / aggregated["forwardPrice"].to_numpy(dtype=float)
    )
    aggregated["preferredSurfaceSide"] = np.where(
        np.abs(node_log_moneyness) <= SURFACE_ATM_LOG_MONEYNESS_BAND,
        "both",
        np.where(node_log_moneyness < 0, "put", "call"),
    )
    aggregated = aggregated.drop(columns=["weightedCallEquivalentPrice"])

    return aggregated.sort_values(
        ["days_to_expiration", "strike"], ascending=[True, True]
    ).reset_index(drop=True)


def _build_arbitrage_free_surface(
    df: pd.DataFrame,
    underlying_price: float,
    risk_free_rate: float,
    dte_step: int = 1,
    build_smooth_grid: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    surface_nodes = _build_surface_nodes(
        df=df,
        underlying_price=underlying_price,
        risk_free_rate=risk_free_rate,
    )
    if surface_nodes.empty:
        raise ValueError("No usable quotes available for static-arbitrage-adjusted surface construction.")

    slice_projected_iv: Dict[Tuple[int, float], float] = {}
    slice_data = []
    for days_to_expiration, group in surface_nodes.groupby("days_to_expiration", sort=True):
        slice_df = group.sort_values("strike").copy()
        strikes = slice_df["strike"].to_numpy(dtype=float)
        weights = slice_df["surfaceWeight"].to_numpy(dtype=float)
        call_prices = slice_df["callEquivalentPrice"].to_numpy(dtype=float)
        time_to_expiration = float(slice_df["time_to_expiration_years"].iloc[0])
        slice_dividend_yield = float(slice_df["dividendYieldUsed"].iloc[0])
        projected_prices = _project_call_price_slice(
            strikes=strikes,
            call_prices=call_prices,
            weights=weights,
            spot=underlying_price,
            time_to_expiration=time_to_expiration,
            risk_free_rate=risk_free_rate,
            dividend_yield=slice_dividend_yield,
        )
        for strike, projected_price in zip(strikes, projected_prices):
            projected_iv = _implied_volatility(
                option_type="call",
                spot=underlying_price,
                strike=float(strike),
                time_to_expiration=time_to_expiration,
                risk_free_rate=risk_free_rate,
                market_price=float(projected_price),
                dividend_yield=slice_dividend_yield,
            )
            if np.isfinite(projected_iv):
                slice_projected_iv[(int(days_to_expiration), float(strike))] = float(projected_iv)

        if len(strikes) < 2:
            continue
        forward = float(slice_df["forwardPrice"].iloc[0])
        log_moneyness = np.log(strikes / forward)
        if (
            len(strikes) < MIN_SMOOTH_SLICE_NODE_COUNT
            or float(np.max(log_moneyness) - np.min(log_moneyness))
            < MIN_SMOOTH_SLICE_LOG_MONEYNESS_WIDTH
        ):
            continue
        slice_data.append(
            {
                "days_to_expiration": int(days_to_expiration),
                "time_to_expiration_years": time_to_expiration,
                "forward": forward,
                "dividend_yield": slice_dividend_yield,
                "strikes": strikes,
                "projected_prices": projected_prices,
                "weights": weights,
                "log_moneyness": log_moneyness,
            }
        )

    surface_nodes = surface_nodes.copy()
    surface_nodes["surfaceImpliedVolatility"] = surface_nodes.apply(
        lambda row: slice_projected_iv.get(
            (int(row["days_to_expiration"]), float(row["strike"])), float("nan")
        ),
        axis=1,
    )
    surface_nodes = surface_nodes.dropna(subset=["surfaceImpliedVolatility"]).copy()

    empty_grid = np.empty((0, 0), dtype=float)
    if not build_smooth_grid:
        return empty_grid, empty_grid, empty_grid, surface_nodes

    if len(slice_data) < MIN_SMOOTH_MATURITY_SLICE_COUNT:
        return empty_grid, empty_grid, empty_grid, surface_nodes

    k_min = max(float(np.min(item["log_moneyness"])) for item in slice_data)
    k_max = min(float(np.max(item["log_moneyness"])) for item in slice_data)
    if not np.isfinite(k_min) or not np.isfinite(k_max) or k_max <= k_min:
        return empty_grid, empty_grid, empty_grid, surface_nodes

    k_grid = np.linspace(k_min, k_max, ARBITRAGE_K_GRID_SIZE)
    dte_observed = np.array([item["days_to_expiration"] for item in slice_data], dtype=int)
    time_observed = np.array([item["time_to_expiration_years"] for item in slice_data], dtype=float)
    forward_observed = np.array([item["forward"] for item in slice_data], dtype=float)
    dividend_yield_observed = np.array(
        [item["dividend_yield"] for item in slice_data], dtype=float
    )
    observed_strike_grid = forward_observed[:, None] * np.exp(k_grid)

    total_variance = np.full((len(slice_data), len(k_grid)), np.nan)
    evaluation_weights = np.full((len(slice_data), len(k_grid)), 0.25)
    for row_index, item in enumerate(slice_data):
        evaluation_strikes = observed_strike_grid[row_index]
        interpolated_prices = np.interp(
            evaluation_strikes,
            item["strikes"],
            item["projected_prices"],
        )
        evaluation_weights[row_index] = np.interp(
            evaluation_strikes,
            item["strikes"],
            item["weights"],
            left=float(item["weights"][0]),
            right=float(item["weights"][-1]),
        )
        for column_index, strike in enumerate(evaluation_strikes):
            implied_volatility = _implied_volatility(
                option_type="call",
                spot=underlying_price,
                strike=float(strike),
                time_to_expiration=float(item["time_to_expiration_years"]),
                risk_free_rate=risk_free_rate,
                market_price=float(interpolated_prices[column_index]),
                dividend_yield=float(item["dividend_yield"]),
            )
            if np.isfinite(implied_volatility):
                total_variance[row_index, column_index] = float(
                    implied_volatility**2 * item["time_to_expiration_years"]
                )

    if not np.all(np.isfinite(total_variance)):
        return empty_grid, empty_grid, empty_grid, surface_nodes

    total_variance = _project_total_variance_grid(
        strike_grid=observed_strike_grid,
        time_to_expiration=time_observed,
        dividend_yields=dividend_yield_observed,
        weights=evaluation_weights,
        total_variance=total_variance,
        underlying_price=underlying_price,
        risk_free_rate=risk_free_rate,
    )

    dte_step = max(int(dte_step), 1)
    dte_dense = np.arange(int(dte_observed.min()), int(dte_observed.max()) + dte_step, dte_step, dtype=int)
    observed_max_dte = int(dte_observed.max())
    if dte_dense[-1] != observed_max_dte:
        dte_dense = np.append(dte_dense[dte_dense < observed_max_dte], observed_max_dte)
    time_dense = np.interp(dte_dense, dte_observed, time_observed)
    forward_dense = np.interp(dte_dense, dte_observed, forward_observed)
    dividend_yield_dense = risk_free_rate - np.log(forward_dense / underlying_price) / time_dense
    total_variance_dense = np.empty((len(dte_dense), len(k_grid)), dtype=float)
    evaluation_weights_dense = np.empty((len(dte_dense), len(k_grid)), dtype=float)
    for column_index in range(len(k_grid)):
        total_variance_dense[:, column_index] = np.interp(
            dte_dense, dte_observed, total_variance[:, column_index]
        )
        evaluation_weights_dense[:, column_index] = np.interp(
            dte_dense, dte_observed, evaluation_weights[:, column_index]
        )

    grid_strike = np.empty((len(dte_dense), len(k_grid)), dtype=float)
    for row_index, _ in enumerate(time_dense):
        grid_strike[row_index] = forward_dense[row_index] * np.exp(k_grid)

    total_variance_dense = _project_total_variance_grid(
        strike_grid=grid_strike,
        time_to_expiration=time_dense,
        dividend_yields=dividend_yield_dense,
        weights=evaluation_weights_dense,
        total_variance=total_variance_dense,
        underlying_price=underlying_price,
        risk_free_rate=risk_free_rate,
    )

    grid_dte = np.repeat(dte_dense[:, None], len(k_grid), axis=1)
    iv_grid = np.sqrt(
        np.maximum(total_variance_dense, 0.0) / np.maximum(time_dense[:, None], np.finfo(float).eps)
    )

    adjusted_nodes = dict(slice_projected_iv)
    for row_index, item in enumerate(slice_data):
        for strike in item["strikes"]:
            log_moneyness = float(np.log(float(strike) / item["forward"]))
            if log_moneyness < k_grid[0] or log_moneyness > k_grid[-1]:
                continue
            adjusted_total_variance = float(np.interp(log_moneyness, k_grid, total_variance[row_index]))
            adjusted_iv = float(
                np.sqrt(max(adjusted_total_variance, 0.0) / max(item["time_to_expiration_years"], np.finfo(float).eps))
            )
            adjusted_nodes[(int(item["days_to_expiration"]), float(strike))] = adjusted_iv

    surface_nodes["surfaceImpliedVolatility"] = surface_nodes.apply(
        lambda row: adjusted_nodes.get(
            (int(row["days_to_expiration"]), float(row["strike"])), float(row["surfaceImpliedVolatility"])
        ),
        axis=1,
    )
    return grid_strike, grid_dte, iv_grid, surface_nodes


def _build_unified_scatter_trace(surface_nodes: pd.DataFrame) -> go.Scatter3d:
    custom_data = np.column_stack(
        [
            surface_nodes["selectedOptionTypes"].fillna("none").astype(str).to_numpy(),
            surface_nodes["preferredSurfaceSide"].fillna("unknown").astype(str).to_numpy(),
            surface_nodes["selectedQuoteCount"].fillna(0).astype(int).to_numpy(),
            surface_nodes["totalVolume"].fillna(0.0).to_numpy(),
            surface_nodes["totalOpenInterest"].fillna(0.0).to_numpy(),
            surface_nodes["surfaceWeight"].fillna(0.0).to_numpy(),
        ]
    )

    return go.Scatter3d(
        x=surface_nodes["strike"],
        y=surface_nodes["days_to_expiration"],
        z=surface_nodes["surfaceImpliedVolatility"] * 100.0,
        mode="markers",
        name="Adjusted Surface Nodes",
        marker=dict(
            size=5,
            color=surface_nodes["surfaceImpliedVolatility"] * 100.0,
            colorscale="Viridis",
            opacity=0.9,
            showscale=False,
        ),
        customdata=custom_data,
        hovertemplate=(
            "Strike: %{x:.2f}<br>Days to Exp: %{y:.0f}<br>IV: %{z:.4f}%<br>Selected quotes: %{customdata[2]}"
            "<br>Quote types: %{customdata[0]}<br>Preferred side: %{customdata[1]}<br>Total volume: %{customdata[3]}"
            "<br>Total OI: %{customdata[4]}<br>Surface weight: %{customdata[5]:.3f}<extra></extra>"
        ),
    )


def _build_surface_trace(grid_strike: np.ndarray, grid_dte: np.ndarray, iv_grid: np.ndarray) -> go.Surface:
    z_display = iv_grid * 100.0

    return go.Surface(
        x=grid_strike,
        y=grid_dte,
        z=z_display,
        surfacecolor=z_display,
        colorscale="Viridis",
        opacity=0.80,
        name="Static-Arbitrage Adjusted Surface",
        showscale=True,
        colorbar=dict(title="IV (%)"),
        hovertemplate="Strike: %{x:.2f}<br>Days to Exp: %{y:.0f}<br>IV: %{z:.4f}%<extra></extra>",
    )


def _build_spot_line_trace(
    underlying_price: float, surface_nodes: pd.DataFrame, grid_strike: np.ndarray, grid_dte: np.ndarray, iv_grid: np.ndarray
) -> go.Scatter3d | None:
    dte_values = []
    iv_values = []

    def interpolate_at_spot(strikes, implied_volatilities):
        order = np.argsort(strikes)
        sorted_strikes = strikes[order]
        return float(
            np.interp(
                underlying_price,
                sorted_strikes,
                implied_volatilities[order],
            )
        )

    if grid_strike.size > 0 and grid_dte.size > 0 and iv_grid.size > 0:
        for strikes, dtes, implied_volatilities in zip(grid_strike, grid_dte, iv_grid):
            valid = (
                np.isfinite(strikes)
                & np.isfinite(dtes)
                & np.isfinite(implied_volatilities)
            )
            if not np.any(valid):
                continue
            strikes = strikes[valid].astype(float)
            implied_volatilities = implied_volatilities[valid].astype(float)
            dte_values.append(float(dtes[valid][0]))
            iv_values.append(interpolate_at_spot(strikes, implied_volatilities))
    else:
        for days_to_expiration, group in surface_nodes.groupby(
            "days_to_expiration", sort=True
        ):
            strikes = pd.to_numeric(group["strike"], errors="coerce").to_numpy(dtype=float)
            implied_volatilities = pd.to_numeric(
                group["surfaceImpliedVolatility"], errors="coerce"
            ).to_numpy(dtype=float)
            valid = np.isfinite(strikes) & np.isfinite(implied_volatilities)
            if not np.any(valid):
                continue
            strikes = strikes[valid]
            implied_volatilities = implied_volatilities[valid]
            dte_values.append(float(days_to_expiration))
            iv_values.append(interpolate_at_spot(strikes, implied_volatilities))

    if not dte_values:
        return None

    return go.Scatter3d(
        x=np.full(len(dte_values), float(underlying_price)),
        y=np.asarray(dte_values, dtype=float),
        z=np.asarray(iv_values, dtype=float) * 100.0,
        mode="lines",
        name="Current Spot",
        showlegend=False,
        visible=True,
        opacity=0.58,
        line=dict(color="#ffffff", width=6, dash="dot"),
        hovertemplate=f"Current spot: ${underlying_price:.2f}<br>Days to Exp: %{{y:.0f}}<br>IV: %{{z:.4f}}%<extra></extra>",
    )


def _axis_range(
    requested_range: Tuple[float, float] | Tuple[int, int] | None, observed_min: float | int, observed_max: float | int
) -> list[float | int]:
    if requested_range is None:
        return [observed_min, observed_max]
    requested_min, requested_max = requested_range
    return [min(requested_min, requested_max), max(requested_min, requested_max)]


def create_vol_surface(
    df: pd.DataFrame,
    smooth: bool = False,
    include_low_confidence: bool = False,
    underlying_price: float | None = None,
    risk_free_rate: float = 0.02,
    strike_range: Tuple[float, float] | None = None,
    dte_range: Tuple[int, int] | None = None,
):
    """Create an interactive raw-node or smoothed static-arbitrage-adjusted 3D volatility figure."""
    if df is None or df.empty:
        print("No data to plot")
        return go.Figure()
    if underlying_price is None or not np.isfinite(underlying_price) or float(underlying_price) <= 0:
        print("A positive underlying price is required to build a unified static-arbitrage-adjusted surface.")
        return go.Figure()

    working_df = df.copy()

    working_df = working_df.replace([np.inf, -np.inf], np.nan)
    working_df = working_df.dropna(
        subset=["strike", "days_to_expiration", "impliedVolatilityFinal", "optionType"]
    )
    working_df = working_df[working_df["impliedVolatilityFinal"] > 0]

    if working_df.empty:
        print("No rows left to render after filtering.")
        return go.Figure()

    if not include_low_confidence:
        surface_input_df = working_df[working_df["includeInSurface"].fillna(False).astype(bool)].copy()
    else:
        surface_input_df = working_df.copy()

    if surface_input_df.empty:
        print("No surface-eligible rows remain after quality filtering.")
        return go.Figure()

    observed_dte_min = int(working_df["days_to_expiration"].min())
    observed_dte_max = int(working_df["days_to_expiration"].max())
    observed_strike_min = float(working_df["strike"].min())
    observed_strike_max = float(working_df["strike"].max())
    strike_axis_range = _axis_range(strike_range, observed_strike_min, observed_strike_max)
    dte_axis_range = _axis_range(dte_range, observed_dte_min, observed_dte_max)

    dte_span = max(dte_axis_range[1] - dte_axis_range[0], observed_dte_max - observed_dte_min, 1)
    smooth_dte_step = max(1, int(np.ceil(dte_span / max(MAX_SMOOTH_DTE_GRID_POINTS - 1, 1))))

    try:
        grid_strike, grid_dte, iv_grid, surface_nodes = _build_arbitrage_free_surface(
            df=surface_input_df,
            underlying_price=float(underlying_price),
            risk_free_rate=risk_free_rate,
            dte_step=smooth_dte_step,
            build_smooth_grid=bool(smooth),
        )
    except Exception as exc:
        print(f"Static-arbitrage-adjusted surface construction failed: {exc}")
        return go.Figure()

    if surface_nodes.empty:
        print("No arbitrage-adjusted surface nodes could be constructed.")
        return go.Figure()

    traces: List[go.BaseTraceType] = []
    if smooth and grid_strike.size > 0:
        traces.append(_build_surface_trace(grid_strike, grid_dte, iv_grid))
    else:
        if smooth:
            print("Not enough stable nodes for a smoothed adjusted surface. Rendering adjusted nodes instead.")
        traces.append(_build_unified_scatter_trace(surface_nodes))

    spot_line_trace = _build_spot_line_trace(float(underlying_price), surface_nodes, grid_strike, grid_dte, iv_grid)
    if spot_line_trace is not None:
        traces.append(spot_line_trace)

    fig = go.Figure(data=traces)
    axis_style = dict(backgroundcolor="#0b1421", gridcolor="#2c3b50", zerolinecolor="#516275", color="#eeeeea")
    fig.update_layout(
        title=None,
        paper_bgcolor="#101a2a",
        plot_bgcolor="#101a2a",
        font=dict(color="#eeeeea", family="Inter, Segoe UI, Arial, sans-serif"),
        scene=dict(
            bgcolor="#101a2a",
            xaxis=dict(title="Strike Price", range=strike_axis_range, **axis_style),
            yaxis=dict(title="Days to Expiration", range=dte_axis_range, **axis_style),
            zaxis=dict(title="Implied Volatility (%)", **axis_style),
            camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.45, y=1.45, z=1.35)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0, font=dict(color="#eeeeea")),
        margin=dict(l=18, r=18, b=18, t=18),
        autosize=True,
    )
    return fig
