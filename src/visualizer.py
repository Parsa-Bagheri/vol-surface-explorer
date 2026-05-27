from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.special import ndtr

from src.data_cleaner import (
    _black_scholes_price,
    _black_scholes_prices_array,
    _implied_volatility,
    _to_float,
)


ARBITRAGE_K_GRID_SIZE = 81
ARBITRAGE_PROJECTION_ITERATIONS = 3
SURFACE_ATM_LOG_MONEYNESS_BAND = 0.02
MIN_SMOOTH_SLICE_NODE_COUNT = 3
MIN_SMOOTH_MATURITY_SLICE_COUNT = 3
MIN_SMOOTH_SLICE_LOG_MONEYNESS_WIDTH = 0.01
MAX_SMOOTH_DTE_GRID_POINTS = 80
ARBITRAGE_FEASIBILITY_TOLERANCE = 1e-12


def _black_scholes_call_prices_array(
    spot: float,
    strikes: np.ndarray,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: np.ndarray,
    dividend_yield: float,
) -> np.ndarray:
    strikes = np.asarray(strikes, dtype=float)
    volatility = np.asarray(volatility, dtype=float)
    prices = np.full(strikes.shape, np.nan, dtype=float)
    if (
        time_to_expiration <= 0
        or spot <= 0
        or strikes.shape != volatility.shape
    ):
        return prices

    discounted_spot = spot * np.exp(-dividend_yield * time_to_expiration)
    discounted_strike = strikes * np.exp(-risk_free_rate * time_to_expiration)
    valid = np.isfinite(strikes) & (strikes > 0) & np.isfinite(volatility) & (volatility >= 0)
    zero_volatility = valid & (volatility == 0)
    if np.any(zero_volatility):
        prices[zero_volatility] = np.maximum(
            0.0,
            discounted_spot - discounted_strike[zero_volatility],
        )

    positive_volatility = valid & (volatility > 0)
    if np.any(positive_volatility):
        sqrt_t = np.sqrt(time_to_expiration)
        vol_sqrt_t = volatility[positive_volatility] * sqrt_t
        d1 = (
            np.log(spot / strikes[positive_volatility])
            + (
                risk_free_rate
                - dividend_yield
                + 0.5 * volatility[positive_volatility] ** 2
            )
            * time_to_expiration
        ) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
        prices[positive_volatility] = (
            discounted_spot * ndtr(d1)
            - discounted_strike[positive_volatility] * ndtr(d2)
        )
    return prices


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

    # The Black-Scholes IV is the source of truth for the surface, so each
    # node is repriced through the same model used for inversion.
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

    if "volume" in df.columns:
        volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).to_numpy()
    else:
        volume = np.zeros(len(df), dtype=float)
    if "openInterest" in df.columns:
        open_interest = pd.to_numeric(
            df["openInterest"], errors="coerce"
        ).fillna(0.0).to_numpy()
    else:
        open_interest = np.zeros(len(df), dtype=float)
    if "spreadRatio" in df.columns:
        spread = pd.to_numeric(df["spreadRatio"], errors="coerce").fillna(0.0).to_numpy()
    else:
        spread = np.zeros(len(df), dtype=float)
    confidence = df.get("confidenceLevel", pd.Series(["medium"] * len(df))).astype(str).str.lower()
    confidence_multiplier = confidence.map({"high": 1.0, "medium": 0.6, "low": 0.25}).fillna(0.4)
    liquidity = 0.5 * np.clip(np.log1p(volume) / np.log1p(100.0), 0.0, 1.0) + 0.5 * np.clip(
        np.log1p(open_interest) / np.log1p(500.0), 0.0, 1.0
    )
    spread_penalty = 1.0 / (1.0 + np.clip(spread, 0.0, None))
    weights = confidence_multiplier.to_numpy() * (0.5 + liquidity) * spread_penalty
    return np.clip(weights.astype(float), 0.05, None)


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

    option_types = working_df["optionType"].astype(str).str.lower()
    strikes = pd.to_numeric(working_df["strike"], errors="coerce").to_numpy(dtype=float)
    time_to_expiration = pd.to_numeric(
        working_df["time_to_expiration_years"], errors="coerce"
    ).to_numpy(dtype=float)
    implied_volatility = pd.to_numeric(
        working_df["impliedVolatilityFinal"], errors="coerce"
    ).to_numpy(dtype=float)
    if "dividendYieldUsed" in working_df.columns:
        dividend_yield_used = pd.to_numeric(
            working_df["dividendYieldUsed"], errors="coerce"
        ).fillna(dividend_yield)
    else:
        dividend_yield_used = pd.Series(dividend_yield, index=working_df.index)
    working_df["dividendYieldUsed"] = dividend_yield_used.astype(float)

    surface_option_prices = _black_scholes_prices_array(
        option_types=option_types.to_numpy(dtype=object),
        spot=underlying_price,
        strikes=strikes,
        time_to_expiration=time_to_expiration,
        risk_free_rate=risk_free_rate,
        volatility=implied_volatility,
        dividend_yield=working_df["dividendYieldUsed"].to_numpy(dtype=float),
    )
    invalid_surface_price = ~np.isfinite(surface_option_prices) | (surface_option_prices <= 0)
    if np.any(invalid_surface_price):
        if "marketPrice" in working_df.columns:
            market_prices = pd.to_numeric(
                working_df["marketPrice"], errors="coerce"
            ).to_numpy(dtype=float)
        else:
            market_prices = np.full(len(working_df), np.nan, dtype=float)
        fallback_mask = invalid_surface_price & np.isfinite(market_prices) & (market_prices > 0)
        surface_option_prices[fallback_mask] = market_prices[fallback_mask]
    working_df["surfaceOptionPrice"] = surface_option_prices
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
        fallback_forward = underlying_price * np.exp(
            (risk_free_rate - dividend_yield)
            * working_df.loc[missing_forward, "time_to_expiration_years"].to_numpy(dtype=float)
        )
        working_df.loc[missing_forward, "forwardPrice"] = fallback_forward

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
    working_df["surfaceWeightWorking"] = _surface_weights(working_df)

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
        selected_df["callEquivalentPrice"] * selected_df["surfaceWeightWorking"]
    )
    if "volume" in selected_df.columns:
        selected_df["volumeNumeric"] = pd.to_numeric(
            selected_df["volume"], errors="coerce"
        ).fillna(0.0)
    else:
        selected_df["volumeNumeric"] = 0.0
    if "openInterest" in selected_df.columns:
        selected_df["openInterestNumeric"] = pd.to_numeric(
            selected_df["openInterest"], errors="coerce"
        ).fillna(0.0)
    else:
        selected_df["openInterestNumeric"] = 0.0

    grouped = selected_df.groupby(group_keys, sort=True, dropna=True)
    aggregated = grouped.agg(
        time_to_expiration_years=("time_to_expiration_years", "first"),
        forwardPrice=("forwardPrice", "first"),
        dividendYieldUsed=("dividendYieldUsed", "first"),
        weightedCallEquivalentPrice=("weightedCallEquivalentPrice", "sum"),
        surfaceWeight=("surfaceWeightWorking", "sum"),
        selectedQuoteCount=("surfaceWeightWorking", "size"),
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
            call_prices = _black_scholes_call_prices_array(
                spot=underlying_price,
                strikes=evaluation_strikes,
                time_to_expiration=float(item["time_to_expiration_years"]),
                risk_free_rate=risk_free_rate,
                volatility=row_volatility,
                dividend_yield=float(item["dividend_yield"]),
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
            call_prices = _black_scholes_call_prices_array(
                spot=underlying_price,
                strikes=grid_strike[row_index],
                time_to_expiration=float(time_to_expiration),
                risk_free_rate=risk_free_rate,
                volatility=row_volatility,
                dividend_yield=float(dividend_yield_dense[row_index]),
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
    strike_range: Tuple[float, float] | None = None,
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
    observed_strike_min = float(working_df["strike"].min())
    observed_strike_max = float(working_df["strike"].max())
    strike_axis_range = None
    if strike_range is not None:
        requested_min, requested_max = float(strike_range[0]), float(strike_range[1])
        if requested_min > requested_max:
            requested_min, requested_max = requested_max, requested_min
        strike_axis_range = [requested_min, requested_max]
    else:
        strike_axis_range = [observed_strike_min, observed_strike_max]

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
                range=strike_axis_range,
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
