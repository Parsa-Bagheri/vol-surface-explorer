"""Optional batched native numerics, with an explicit SciPy reference backend.

Only a missing extension permits auto fallback. Import/load errors and numerical
failures never switch a calculation to a different implementation.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import IntEnum
from functools import lru_cache
import hashlib
import importlib
import math
import os
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


class Status(IntEnum):
    OK = 0
    ZERO_VOLATILITY = 1
    INVALID_INPUT = 2
    OUTSIDE_BOUNDS = 3
    UNBRACKETED = 4
    MAX_ITERATIONS = 5
    NONFINITE_RESULT = 6


class NumericalBackendError(ValueError):
    """Unavailable backend, native loading problem, or failed calculation."""


def _extension():
    try:
        return importlib.import_module("src._vol_native")
    except ModuleNotFoundError as exc:
        if exc.name != "src._vol_native":
            raise
        return None
    except ImportError as exc:
        raise NumericalBackendError(f"C++ extension could not be loaded: {exc}. Rebuild it for this Python.") from exc


def resolve_backend(backend: str | None = None) -> str:
    requested = backend if backend is not None else os.environ.get("VOL_NUMERICAL_BACKEND", "auto")
    if requested not in {"python", "cpp", "auto"}:
        raise ValueError("numerical_backend must be python, cpp or auto")
    if requested == "python":
        return "python"
    if _extension() is not None:
        return "cpp"
    if requested == "cpp":
        raise NumericalBackendError("C++ backend requested but src._vol_native is not installed. Build native/ with CMake or select python/auto.")
    return "python"


@dataclass
class Session:
    backend: str
    statuses: Counter = field(default_factory=Counter)
    calls: int = 0
    contracts: int = 0
    max_iterations: int = 0

    def diagnostics(self):
        return {**backend_identity(self.backend), "batch_calls": self.calls,
                "contracts_evaluated": self.contracts, "max_solver_iterations": self.max_iterations,
                "status_counts": dict(sorted(self.statuses.items()))}


_session: ContextVar[Session | None] = ContextVar("vol_numerical_session", default=None)


def current_backend():
    session = _session.get()
    return session.backend if session else resolve_backend()


@contextmanager
def numerical_session(backend=None):
    session = Session(resolve_backend(backend))
    token = _session.set(session)
    try:
        yield session
    finally:
        _session.reset(token)


@lru_cache(maxsize=4)
def _binary_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def backend_identity(backend=None):
    selected = resolve_backend(backend) if backend is not None else current_backend()
    if selected == "python":
        return {"backend": "python", "engine_version": "scipy-brentq-v1"}
    module = _extension()
    return {"backend": "cpp", "engine_version": module.engine_version,
            "source_sha256": module.source_sha256, "binary_sha256": _binary_digest(module.__file__),
            "compiler": module.compiler, "build_type": module.build_type}


@dataclass
class BatchResult:
    values: np.ndarray
    status: np.ndarray
    iterations: np.ndarray
    backend: str = "python"

    def checked_values(self):
        failed = np.isin(self.status, [Status.MAX_ITERATIONS, Status.NONFINITE_RESULT])
        if np.any(failed):
            indices = np.flatnonzero(failed)[:8].tolist()
            raise NumericalBackendError(f"{self.backend} numerical computation failed at batch indices {indices}; statuses={self.status[failed][:8].tolist()}. No backend fallback was attempted.")
        return self.values


def _inputs(option_types, spot, strikes, times, rate, values, dividends):
    strikes = np.asarray(strikes)
    if strikes.ndim != 1 or strikes.dtype.kind not in "fiu":
        raise ValueError("strikes must be a one dimensional numeric array")
    n = len(strikes)
    def numeric(value, name):
        array = np.asarray(value)
        if array.dtype.kind not in "fiu" or array.ndim > 1 or (array.ndim == 1 and len(array) != n):
            raise ValueError(f"{name} must be numeric and scalar or match strikes length")
        return np.array(np.broadcast_to(array, (n,)), dtype=np.float64, order="C", copy=True)
    types = np.asarray(option_types)
    if types.ndim > 1 or (types.ndim == 1 and len(types) != n):
        raise ValueError("option_types must be scalar or match strikes length")
    types = np.broadcast_to(types, (n,))
    kinds = np.where(types == "call", 1, np.where(types == "put", 0, 255)).astype(np.uint8)
    return (kinds, numeric(spot, "spot"), numeric(strikes, "strikes"), numeric(times, "times"),
            numeric(rate, "rate"), numeric(dividends, "dividends"), numeric(values, "values"))


def _reference(inputs, invert, maxiter, xtol):
    # Lazy import prevents a dependency cycle with the original reference module.
    from src.data_cleaner import _black_scholes_price, _black_scholes_price_from_terms, _no_arbitrage_bounds
    kinds, spots, strikes, times, rates, dividends, targets = inputs
    n = len(strikes)
    out = np.full(n, np.nan)
    status = np.full(n, Status.INVALID_INPUT, dtype=np.uint8)
    iterations = np.zeros(n, dtype=np.int32)
    for i, (kind, s, k, t, r, q, value) in enumerate(zip(*inputs)):
        s, k, t, r, q, value = map(float, (s, k, t, r, q, value))
        if kind > 1 or not all(math.isfinite(x) for x in (s, k, t, r, q, value)) or min(s, k, t) <= 0 or value < 0:
            continue
        side = "call" if kind else "put"
        try:
            if not invert:
                out[i] = _black_scholes_price(side, s, k, t, r, value, q)
                status[i] = Status.OK if value else Status.ZERO_VOLATILITY
                if not np.isfinite(out[i]):
                    status[i] = Status.NONFINITE_RESULT
                continue
            lower, upper = _no_arbitrage_bounds(side, s, k, t, r, q)
            if not math.isfinite(lower) or not math.isfinite(upper):
                status[i] = Status.NONFINITE_RESULT
                continue
            if abs(value - lower) <= 1e-8:
                out[i], status[i] = 0, Status.ZERO_VOLATILITY
                continue
            if value < lower - 1e-8 or value > upper + 1e-8:
                status[i] = Status.OUTSIDE_BOUNDS
                continue
            log_moneyness = math.log(s / k)
            discounted_spot = s * math.exp(-q * t)
            discounted_strike = k * math.exp(-r * t)
            sqrt_t = math.sqrt(t)
            def objective(sigma):
                return _black_scholes_price_from_terms(side, log_moneyness, discounted_spot, discounted_strike,
                    t, sqrt_t, r, q, sigma) - value
            a, b = objective(1e-6), objective(5.0)
            if not math.isfinite(a) or not math.isfinite(b):
                status[i] = Status.NONFINITE_RESULT
            elif a != 0 and b != 0 and math.copysign(1, a) == math.copysign(1, b):
                status[i] = Status.UNBRACKETED
            elif a == 0 or b == 0:
                out[i], status[i] = (1e-6 if a == 0 else 5.0), Status.OK
            else:
                root, info = brentq(objective, 1e-6, 5.0, maxiter=maxiter, xtol=xtol, full_output=True, disp=False)
                iterations[i] = info.iterations
                status[i] = Status.OK if info.converged else Status.MAX_ITERATIONS
                if info.converged:
                    out[i] = root
        except (OverflowError, ZeroDivisionError, ValueError):
            status[i] = Status.NONFINITE_RESULT
    return BatchResult(out, status, iterations)


def _batch(option_types, spot, strikes, times, rate, values, dividends, *, backend, invert, maxiter=200, xtol=1e-8):
    if not isinstance(maxiter, int) or maxiter < 0 or not math.isfinite(xtol) or xtol <= 0:
        raise ValueError("maxiter must be nonnegative and xtol must be finite and positive")
    selected = resolve_backend(backend) if backend is not None else current_backend()
    inputs = _inputs(option_types, spot, strikes, times, rate, values, dividends)
    if selected == "cpp":
        try:
            result = BatchResult(*_extension().batch(*inputs, invert, maxiter, xtol))
        except Exception as exc:
            raise NumericalBackendError(f"C++ batch computation failed: {exc}. No backend fallback was attempted.") from exc
    else:
        result = _reference(inputs, invert, maxiter, xtol)
    result.backend = selected
    _record(result)
    return result


def _record(result):
    session = _session.get()
    if session:
        session.calls += 1
        session.contracts += len(result.values)
        unique, counts = np.unique(result.status, return_counts=True)
        session.statuses.update({Status(int(s)).name.lower(): int(n) for s, n in zip(unique, counts)})
        session.max_iterations = max(session.max_iterations, int(result.iterations.max(initial=0)))


def prices(option_types, spot, strikes, times, rate, volatility, dividends=0.0, *, backend=None):
    selected = resolve_backend(backend) if backend is not None else current_backend()
    if selected == "python":
        # Keep the existing NumPy/SciPy compiled pricing path for a fair baseline.
        from src.data_cleaner import _black_scholes_prices_array
        inputs = _inputs(option_types, spot, strikes, times, rate, volatility, dividends)
        if np.ndim(spot) or np.ndim(rate):
            result = _reference(inputs, False, 200, 1e-8)
        else:
            values = (_black_scholes_prices_array(option_types, spot, np.asarray(strikes, dtype=float), times, rate, volatility, dividends)
                      if np.isfinite(rate) else np.full(len(inputs[0]), np.nan))
            valid = (inputs[0] <= 1) & np.all(np.isfinite(np.array(inputs[1:])), axis=0)
            valid &= (inputs[1] > 0) & (inputs[2] > 0) & (inputs[3] > 0) & (inputs[6] >= 0)
            values[~valid] = np.nan
            status = np.where(valid, np.where(inputs[6] == 0, Status.ZERO_VOLATILITY, Status.OK), Status.INVALID_INPUT).astype(np.uint8)
            status[valid & ~np.isfinite(values)] = Status.NONFINITE_RESULT
            result = BatchResult(values, status, np.zeros(len(values), dtype=np.int32))
        _record(result)
        return result.checked_values()
    return _batch(option_types, spot, strikes, times, rate, volatility, dividends,
                  backend=selected, invert=False).checked_values()


def implied_volatilities(option_types, spot, strikes, times, rate, market_prices, dividends=0.0, *, backend=None, maxiter=200, xtol=1e-8):
    return _batch(option_types, spot, strikes, times, rate, market_prices, dividends,
                  backend=backend, invert=True, maxiter=maxiter, xtol=xtol)
