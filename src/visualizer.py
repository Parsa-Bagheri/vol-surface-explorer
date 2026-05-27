from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import Bounds, LinearConstraint, minimize

from src.data_cleaner import _black_scholes_price, _implied_volatility, _to_float


ARBITRAGE_K_GRID_SIZE = 81
ARBITRAGE_PROJECTION_ITERATIONS = 3
SURFACE_ATM_LOG_MONEYNESS_BAND = 0.02
MIN_SMOOTH_SLICE_NODE_COUNT = 3
MIN_SMOOTH_MATURITY_SLICE_COUNT = 3
MIN_SMOOTH_SLICE_LOG_MONEYNESS_WIDTH = 0.01
MAX_SMOOTH_DTE_GRID_POINTS = 80


def _forward_price(
    spot: float, time_to_expiration: float, risk_free_rate: float, dividend_yield: float
) -> float:
    return float(spot * np.exp((risk_free_rate - dividend_yield) * time_to_expiration))


def _call_equivalent_price(
    option_type: str,
    option_price: float,
    spot: float,
    strike: float,
    time_to_expiration: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> float:
    if option_type == "call":
        return float(option_price)
    discounted_spot = spot * np.exp(-dividend_yield * time_to_expiration)
    discounted_strike = strike * np.exp(-risk_free_rate * time_to_expiration)
    return float(option_price + discounted_spot - discounted_strike)


def _surface_option_price(
    row: pd.Series,
    underlying_price: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> float:
    option_type = str(row.get("optionType", "")).lower()
    strike = _to_float(row.get("strike"))
    time_to_expiration = _to_float(row.get("time_to_expiration_years"))
    implied_volatility = _to_float(row.get("impliedVolatilityFinal"))
    row_dividend_yield = _to_float(row.get("dividendYieldUsed"))
    if not np.isfinite(row_dividend_yield):
        row_dividend_yield = dividend_yield
    if (
        option_type not in {"call", "put"}
        or not np.isfinite(strike)
        or strike <= 0
        or not np.isfinite(time_to_expiration)
        or time_to_expiration <= 0
        or not np.isfinite(implied_volatility)
        or implied_volatility <= 0
    ):
        market_price = _to_float(row.get("marketPrice"))
        if np.isfinite(market_price) and market_price > 0:
            return float(market_price)
        return float("nan")

    # The selected IV is the source-of-truth for the surface. In Black-Scholes
    # mode this reprices back to the quote used for inversion; in provider-IV
    # mode it preserves provider-vs-recomputed differences for comparison.
    return _black_scholes_price(
        option_type=option_type,
        spot=underlying_price,
        strike=strike,
        time_to_expiration=time_to_expiration,
        risk_free_rate=risk_free_rate,
        volatility=implied_volatility,
        dividend_yield=row_dividend_yield,
    )


def _surface_quote_is_preferred(option_type: str, strike: float, forward: float) -> bool:
    if not np.isfinite(forward) or forward <= 0 or not np.isfinite(strike) or strike <= 0:
        return True

    log_moneyness = float(np.log(strike / forward))
    if abs(log_moneyness) <= SURFACE_ATM_LOG_MONEYNESS_BAND:
        return True
    if log_moneyness < 0:
        return option_type == "put"
    return option_type == "call"


def _surface_weights(df: pd.DataFrame) -> np.ndarray:
    if "surfaceWeight" in df.columns:
        weights = pd.to_numeric(df["surfaceWeight"], errors="coerce").fillna(0.25).to_numpy()
        return np.clip(weights.astype(float), 0.05, None)

    volume = pd.to_numeric(df.get("volume"), errors="coerce").fillna(0.0).to_numpy()
    open_interest = pd.to_numeric(df.get("openInterest"), errors="coerce").fillna(0.0).to_numpy()
    spread = pd.to_numeric(df.get("spreadRatio"), errors="coerce").fillna(0.0).to_numpy()
    confidence = df.get("confidenceLevel", pd.Series(["medium"] * len(df))).astype(str).str.lower()
    confidence_multiplier = confidence.map({"high": 1.0, "medium": 0.6, "low": 0.25}).fillna(0.4)
    liquidity = 0.5 * np.clip(np.log1p(volume) / np.log1p(100.0), 0.0, 1.0) + 0.5 * np.clip(
        np.log1p(open_interest) / np.log1p(500.0), 0.0, 1.0
    )
    spread_penalty = 1.0 / (1.0 + np.clip(spread, 0.0, None))
    weights = confidence_multiplier.to_numpy() * (0.5 + liquidity) * spread_penalty
    return np.clip(weights.astype(float), 0.05, None)


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

    constraints: List[LinearConstraint] = []
    monotone_matrix = []
    monotone_upper = []
    for index in range(len(strikes) - 1):
        row = np.zeros(len(strikes))
        row[index] = 1.0
        row[index + 1] = -1.0
        monotone_matrix.append(row)
        monotone_upper.append((strikes[index + 1] - strikes[index]) * discount_factor)
    if monotone_matrix:
        constraints.append(
            LinearConstraint(np.array(monotone_matrix), np.zeros(len(monotone_matrix)), np.array(monotone_upper))
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


def _build_surface_nodes(
    df: pd.DataFrame,
    underlying_price: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> pd.DataFrame:
    working_df = df.copy()
    working_df = working_df.replace([np.inf, -np.inf], np.nan)
    working_df = working_df.dropna(
        subset=["strike", "days_to_expiration", "time_to_expiration_years", "optionType"]
    )
    working_df = working_df[
        (working_df["strike"] > 0)
        & (working_df["days_to_expiration"] > 0)
        & (working_df["time_to_expiration_years"] > 0)
    ].copy()
    if working_df.empty:
        return pd.DataFrame()

    working_df["surfaceOptionPrice"] = working_df.apply(
        lambda row: _surface_option_price(
            row=row,
            underlying_price=underlying_price,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
        ),
        axis=1,
    )
    working_df = working_df.dropna(subset=["surfaceOptionPrice"]).copy()
    working_df = working_df[working_df["surfaceOptionPrice"] > 0]
    if working_df.empty:
        return pd.DataFrame()

    if "forwardPrice" in working_df.columns:
        working_df["forwardPrice"] = pd.to_numeric(
            working_df["forwardPrice"], errors="coerce"
        )
    else:
        working_df["forwardPrice"] = np.nan
    missing_forward = ~np.isfinite(working_df["forwardPrice"])
    if missing_forward.any():
        working_df.loc[missing_forward, "forwardPrice"] = working_df.loc[
            missing_forward
        ].apply(
            lambda row: _forward_price(
                spot=underlying_price,
                time_to_expiration=float(row["time_to_expiration_years"]),
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
            ),
            axis=1,
        )
    if "dividendYieldUsed" in working_df.columns:
        working_df["dividendYieldUsed"] = pd.to_numeric(
            working_df["dividendYieldUsed"], errors="coerce"
        )
    else:
        working_df["dividendYieldUsed"] = dividend_yield
    working_df["dividendYieldUsed"] = working_df["dividendYieldUsed"].fillna(dividend_yield)
    working_df["surfaceQuotePreferred"] = working_df.apply(
        lambda row: _surface_quote_is_preferred(
            option_type=str(row["optionType"]).lower(),
            strike=float(row["strike"]),
            forward=float(row["forwardPrice"]),
        ),
        axis=1,
    )
    working_df["callEquivalentPrice"] = working_df.apply(
        lambda row: _call_equivalent_price(
            option_type=str(row["optionType"]).lower(),
            option_price=float(row["surfaceOptionPrice"]),
            spot=underlying_price,
            strike=float(row["strike"]),
            time_to_expiration=float(row["time_to_expiration_years"]),
            risk_free_rate=risk_free_rate,
            dividend_yield=float(row["dividendYieldUsed"]),
        ),
        axis=1,
    )
    working_df["surfaceWeightWorking"] = _surface_weights(working_df)

    aggregated_rows = []
    for (days_to_expiration, strike), group in working_df.groupby(["days_to_expiration", "strike"], sort=True):
        preferred_group = group[group["surfaceQuotePreferred"]].copy()
        selected_group = preferred_group if not preferred_group.empty else group.copy()
        weights = selected_group["surfaceWeightWorking"].to_numpy(dtype=float)
        if len(weights) == 0 or not np.all(np.isfinite(weights)) or float(weights.sum()) <= 0:
            weights = np.ones(len(selected_group), dtype=float)

        time_to_expiration = float(selected_group["time_to_expiration_years"].iloc[0])
        forward = float(selected_group["forwardPrice"].iloc[0])
        selected_dividend_yield = float(selected_group["dividendYieldUsed"].iloc[0])
        log_moneyness = float(np.log(float(strike) / forward))
        if abs(log_moneyness) <= SURFACE_ATM_LOG_MONEYNESS_BAND:
            preferred_surface_side = "both"
        elif log_moneyness < 0:
            preferred_surface_side = "put"
        else:
            preferred_surface_side = "call"

        aggregated_rows.append(
            {
                "days_to_expiration": int(days_to_expiration),
                "strike": float(strike),
                "time_to_expiration_years": time_to_expiration,
                "forwardPrice": forward,
                "dividendYieldUsed": selected_dividend_yield,
                "callEquivalentPrice": float(
                    np.average(selected_group["callEquivalentPrice"].to_numpy(dtype=float), weights=weights)
                ),
                "surfaceWeight": float(np.sum(weights)),
                "selectedQuoteCount": int(len(selected_group)),
                "selectedOptionTypes": "/".join(sorted(selected_group["optionType"].astype(str).str.lower().unique())),
                "preferredSurfaceSide": preferred_surface_side,
                "totalVolume": float(
                    pd.to_numeric(selected_group.get("volume"), errors="coerce").fillna(0.0).sum()
                ),
                "totalOpenInterest": float(
                    pd.to_numeric(selected_group.get("openInterest"), errors="coerce").fillna(0.0).sum()
                ),
            }
        )

    if not aggregated_rows:
        return pd.DataFrame()

    return pd.DataFrame(aggregated_rows).sort_values(
        ["days_to_expiration", "strike"], ascending=[True, True]
    ).reset_index(drop=True)


def _build_arbitrage_free_surface(
    df: pd.DataFrame,
    underlying_price: float,
    risk_free_rate: float,
    dividend_yield: float,
    dte_step: int = 1,
    build_smooth_grid: bool = True,
    ) -> Tuple[Dict[Tuple[int, float], float], np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    surface_nodes = _build_surface_nodes(
        df=df,
        underlying_price=underlying_price,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
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
        return slice_projected_iv, empty_grid, empty_grid, empty_grid, surface_nodes

    if len(slice_data) < MIN_SMOOTH_MATURITY_SLICE_COUNT:
        return slice_projected_iv, empty_grid, empty_grid, empty_grid, surface_nodes

    k_min = max(float(np.min(item["log_moneyness"])) for item in slice_data)
    k_max = min(float(np.max(item["log_moneyness"])) for item in slice_data)
    if not np.isfinite(k_min) or not np.isfinite(k_max) or k_max <= k_min:
        return slice_projected_iv, empty_grid, empty_grid, empty_grid, surface_nodes

    k_grid = np.linspace(k_min, k_max, ARBITRAGE_K_GRID_SIZE)
    dte_observed = np.array([item["days_to_expiration"] for item in slice_data], dtype=int)
    time_observed = np.array([item["time_to_expiration_years"] for item in slice_data], dtype=float)
    forward_observed = np.array([item["forward"] for item in slice_data], dtype=float)

    total_variance = np.full((len(slice_data), len(k_grid)), np.nan)
    evaluation_weights = np.full((len(slice_data), len(k_grid)), 0.25)
    for row_index, item in enumerate(slice_data):
        evaluation_strikes = item["forward"] * np.exp(k_grid)
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
        return slice_projected_iv, empty_grid, empty_grid, empty_grid, surface_nodes

    for _ in range(ARBITRAGE_PROJECTION_ITERATIONS):
        for row_index, item in enumerate(slice_data):
            evaluation_strikes = item["forward"] * np.exp(k_grid)
            row_volatility = np.sqrt(np.maximum(total_variance[row_index], 0.0) / item["time_to_expiration_years"])
            call_prices = np.array(
                [
                    _black_scholes_price(
                        option_type="call",
                        spot=underlying_price,
                        strike=float(strike),
                        time_to_expiration=float(item["time_to_expiration_years"]),
                        risk_free_rate=risk_free_rate,
                        volatility=float(volatility),
                        dividend_yield=float(item["dividend_yield"]),
                    )
                    for strike, volatility in zip(evaluation_strikes, row_volatility)
                ],
                dtype=float,
            )
            projected_prices = _project_call_price_slice(
                strikes=evaluation_strikes,
                call_prices=call_prices,
                weights=evaluation_weights[row_index],
                spot=underlying_price,
                time_to_expiration=float(item["time_to_expiration_years"]),
                risk_free_rate=risk_free_rate,
                dividend_yield=float(item["dividend_yield"]),
            )
            updated_total_variance = []
            for strike, projected_price in zip(evaluation_strikes, projected_prices):
                implied_volatility = _implied_volatility(
                    option_type="call",
                    spot=underlying_price,
                    strike=float(strike),
                    time_to_expiration=float(item["time_to_expiration_years"]),
                    risk_free_rate=risk_free_rate,
                    market_price=float(projected_price),
                    dividend_yield=float(item["dividend_yield"]),
                )
                updated_total_variance.append(float(implied_volatility**2 * item["time_to_expiration_years"]))
            total_variance[row_index] = np.array(updated_total_variance, dtype=float)

        total_variance = np.maximum.accumulate(total_variance, axis=0)

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

    for _ in range(ARBITRAGE_PROJECTION_ITERATIONS):
        for row_index, time_to_expiration in enumerate(time_dense):
            row_volatility = np.sqrt(
                np.maximum(total_variance_dense[row_index], 0.0)
                / max(float(time_to_expiration), np.finfo(float).eps)
            )
            call_prices = np.array(
                [
                    _black_scholes_price(
                        option_type="call",
                        spot=underlying_price,
                        strike=float(strike),
                        time_to_expiration=float(time_to_expiration),
                        risk_free_rate=risk_free_rate,
                        volatility=float(volatility),
                        dividend_yield=float(dividend_yield_dense[row_index]),
                    )
                    for strike, volatility in zip(grid_strike[row_index], row_volatility)
                ],
                dtype=float,
            )
            projected_prices = _project_call_price_slice(
                strikes=grid_strike[row_index],
                call_prices=call_prices,
                weights=evaluation_weights_dense[row_index],
                spot=underlying_price,
                time_to_expiration=float(time_to_expiration),
                risk_free_rate=risk_free_rate,
                dividend_yield=float(dividend_yield_dense[row_index]),
            )
            updated_total_variance = []
            for strike, projected_price in zip(grid_strike[row_index], projected_prices):
                implied_volatility = _implied_volatility(
                    option_type="call",
                    spot=underlying_price,
                    strike=float(strike),
                    time_to_expiration=float(time_to_expiration),
                    risk_free_rate=risk_free_rate,
                    market_price=float(projected_price),
                    dividend_yield=float(dividend_yield_dense[row_index]),
                )
                updated_total_variance.append(float(implied_volatility**2 * time_to_expiration))
            total_variance_dense[row_index] = np.array(updated_total_variance, dtype=float)

        total_variance_dense = np.maximum.accumulate(total_variance_dense, axis=0)

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
    return adjusted_nodes, grid_strike, grid_dte, iv_grid, surface_nodes


def _build_unified_scatter_trace(surface_nodes: pd.DataFrame) -> go.Scatter3d:
    custom_data = np.stack(
        [
            surface_nodes["selectedOptionTypes"].fillna("none").astype(str).to_numpy(),
            surface_nodes["preferredSurfaceSide"].fillna("unknown").astype(str).to_numpy(),
            surface_nodes["selectedQuoteCount"].fillna(0).astype(int).to_numpy(),
            surface_nodes["totalVolume"].fillna(0.0).to_numpy(),
            surface_nodes["totalOpenInterest"].fillna(0.0).to_numpy(),
            surface_nodes["surfaceWeight"].fillna(0.0).to_numpy(),
        ],
        axis=-1,
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
            "Strike: %{x:.2f}<br>Days to Exp: %{y:.0f}<br>IV: %{z:.4f}%"
            + "<br>Selected quotes: %{customdata[2]}"
            + "<br>Quote types: %{customdata[0]}"
            + "<br>Preferred side: %{customdata[1]}"
            + "<br>Total volume: %{customdata[3]}"
            + "<br>Total OI: %{customdata[4]}"
            + "<br>Surface weight: %{customdata[5]:.3f}"
            + "<extra></extra>"
        ),
    )


def _build_surface_trace(
    grid_strike: np.ndarray,
    grid_dte: np.ndarray,
    iv_grid: np.ndarray,
) -> go.Surface:
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
        hovertemplate=(
            "Strike: %{x:.2f}<br>Days to Exp: %{y:.0f}<br>IV: %{z:.4f}%<extra></extra>"
        ),
    )


def create_vol_surface(
    df: pd.DataFrame,
    ticker: str,
    smooth: bool = False,
    include_low_confidence: bool = False,
    underlying_price: float | None = None,
    risk_free_rate: float = 0.02,
    dividend_yield: float = 0.0,
    dte_range: Tuple[int, int] | None = None,
):
    """
    Create an interactive 3D volatility figure.

    - Raw mode: unified static-arbitrage-adjusted surface nodes.
    - Smoothed mode: one unified surface projected onto discrete static no-arbitrage constraints.
    """
    if df is None or df.empty:
        print("No data to plot")
        return go.Figure()
    if underlying_price is None or not np.isfinite(underlying_price) or float(underlying_price) <= 0:
        print("A positive underlying price is required to build a unified static-arbitrage-adjusted surface.")
        return go.Figure()

    working_df = df.copy()

    if "impliedVolatilityFinal" not in working_df.columns and "impliedVolatility" in working_df.columns:
        working_df["impliedVolatilityFinal"] = working_df["impliedVolatility"]

    working_df = working_df.replace([np.inf, -np.inf], np.nan)
    working_df = working_df.dropna(
        subset=["strike", "days_to_expiration", "impliedVolatilityFinal", "optionType"]
    )
    working_df = working_df[working_df["impliedVolatilityFinal"] > 0]

    if working_df.empty:
        print("No rows left to render after filtering.")
        return go.Figure()

    if "includeInSurface" in working_df.columns and not include_low_confidence:
        surface_input_df = working_df[working_df["includeInSurface"].fillna(False).astype(bool)].copy()
    else:
        surface_input_df = working_df.copy()

    if surface_input_df.empty:
        print("No surface-eligible rows remain after quality filtering.")
        return go.Figure()

    observed_dte_min = int(working_df["days_to_expiration"].min())
    observed_dte_max = int(working_df["days_to_expiration"].max())
    dte_axis_range = None
    if dte_range is not None:
        requested_min, requested_max = int(dte_range[0]), int(dte_range[1])
        if requested_min > requested_max:
            requested_min, requested_max = requested_max, requested_min
        dte_axis_range = [requested_min, requested_max]
    else:
        dte_axis_range = [observed_dte_min, observed_dte_max]

    dte_span = max(dte_axis_range[1] - dte_axis_range[0], observed_dte_max - observed_dte_min, 1)
    smooth_dte_step = max(1, int(np.ceil(dte_span / max(MAX_SMOOTH_DTE_GRID_POINTS - 1, 1))))

    try:
        _, grid_strike, grid_dte, iv_grid, surface_nodes = _build_arbitrage_free_surface(
            df=surface_input_df,
            underlying_price=float(underlying_price),
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
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
        traces.append(
            _build_surface_trace(
                grid_strike=grid_strike,
                grid_dte=grid_dte,
                iv_grid=iv_grid,
            )
        )
    else:
        if smooth:
            print("Not enough stable nodes for a smoothed adjusted surface. Rendering adjusted nodes instead.")
        traces.append(_build_unified_scatter_trace(surface_nodes))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=None,
        paper_bgcolor="#101a2a",
        plot_bgcolor="#101a2a",
        font=dict(color="#eeeeea", family="Inter, Segoe UI, Arial, sans-serif"),
        scene=dict(
            bgcolor="#101a2a",
            xaxis=dict(
                title="Strike Price",
                backgroundcolor="#0b1421",
                gridcolor="#2c3b50",
                zerolinecolor="#516275",
                color="#eeeeea",
            ),
            yaxis=dict(
                title="Days to Expiration",
                range=dte_axis_range,
                backgroundcolor="#0b1421",
                gridcolor="#2c3b50",
                zerolinecolor="#516275",
                color="#eeeeea",
            ),
            zaxis=dict(
                title="Implied Volatility (%)",
                backgroundcolor="#0b1421",
                gridcolor="#2c3b50",
                zerolinecolor="#516275",
                color="#eeeeea",
            ),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.45, y=1.45, z=1.35),
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0.0,
            font=dict(color="#eeeeea"),
        ),
        margin=dict(l=18, r=18, b=18, t=18),
        autosize=True,
    )
    return fig
