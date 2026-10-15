"""Warm offline benchmark on an immutable, real SPY observation (no network)."""
from __future__ import annotations

import argparse
import contextlib
import cProfile
from dataclasses import replace
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import pstats
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.data_cleaner import _black_scholes_prices_array, _implied_volatility
from src.history_store import LocalObservationStore
from src.surface_config import SurfaceRequest
from src.surface_service import process_surface_inputs

SNAPSHOT = "6e471b7a5c5d8fe0695afacb"


def measure(action, repeats):
    with contextlib.redirect_stdout(io.StringIO()):
        action()  # Imports, caches and Plotly initialization are outside timings.
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            action()
            samples.append(time.perf_counter() - start)
    median = statistics.median(samples)
    return {"median_s": median, "min_s": min(samples), "max_s": max(samples),
            "mad_s": statistics.median(abs(x - median) for x in samples), "samples_s": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["python", "cpp"], default="python")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("Use at least three repeats")
    store = LocalObservationStore(ROOT / "history_bundle")
    observation, raw = store.load(SNAPSHOT)
    request = SurfaceRequest(**observation["parameters"])
    upgraded = "numerical_backend" in SurfaceRequest.__dataclass_fields__
    if upgraded:
        request = replace(request, numerical_backend=args.backend)
    elif args.backend != "python":
        parser.error("The baseline checkout only supports Python")
    def build(frame):
        return process_surface_inputs(request, frame, observation["spot"]["value"], observation["valuation_time_utc"])
    with contextlib.redirect_stdout(io.StringIO()):
        initial = build(raw)
    # Fixed original saved calculation inputs, even when the selected backend's
    # intermediate IVs differ by roundoff. Both backends receive identical bytes.
    processing_id = "baf2fff2ef39b0af91e9b2fe"
    artifact = store.load_artifact(SNAPSHOT, processing_id)
    cleaned = pd.read_parquet(store.artifact_path(SNAPSHOT, processing_id) / "cleaned.parquet")
    report = {"snapshot_id": SNAPSHOT, "source_kind": observation["source_kind"],
              "raw_sha256": observation["files"]["raw.parquet"]["sha256"],
              "valuation_time_utc": observation["valuation_time_utc"], "parameters": observation["parameters"],
              "backend": args.backend, "repeats": args.repeats,
              "calculation_input_sha256": artifact["files"]["cleaned.parquet"]["sha256"],
              "selected_engine": initial.diagnostics.get("numerical_engine", {"backend": "python", "implementation": "pre-upgrade reference"}),
              "environment": {"os": platform.platform(), "python": sys.version, "machine": platform.machine(),
                  "processor": platform.processor(), "logical_cpus": os.cpu_count(),
                  "dependencies": {n: importlib.metadata.version(n) for n in ("numpy", "scipy", "pandas", "plotly", "pyarrow")}},
              "calculation": [], "surface": [],
              "scope": "Warm computation including array conversion/bindings and Plotly figure construction; excludes file reads, network, browser rendering, startup and compilation. Subsets are deterministic samples of real saved quotes, not synthetic contracts."}
    if upgraded and args.backend == "cpp":
        from src.numerical_engine import backend_identity
        report["native_build"] = backend_identity("cpp")
        if report["native_build"]["build_type"] != "Release":
            raise SystemExit("Performance measurements require a Release native build")
    for count in (128, 512, len(cleaned)):
        frame = cleaned.sample(n=count, random_state=42).sort_index()
        types = frame.optionType.to_numpy()
        strikes = frame.strike.to_numpy(dtype=float)
        times = frame.time_to_expiration_years.to_numpy(dtype=float)
        dividends = frame.dividendYieldUsed.to_numpy(dtype=float)
        vol = frame.blackScholesImpliedVolatility.to_numpy(dtype=float)
        spot, rate = observation["spot"]["value"], request.risk_free_rate
        def calculation():
            # Same workload: one price and three inversions per retained contract.
            if upgraded:
                from src.numerical_engine import prices, implied_volatilities
                prices(types, spot, strikes, times, rate, vol, dividends, backend=args.backend)
                for column in ("marketPrice", "bid", "ask"):
                    implied_volatilities(types, spot, strikes, times, rate, frame[column].to_numpy(dtype=float), dividends, backend=args.backend)
            else:
                _black_scholes_prices_array(types, spot, strikes, times, rate, vol, dividends)
                for column in ("marketPrice", "bid", "ask"):
                    np.array([_implied_volatility(t, spot, k, dt, rate, p, q)
                              for t, k, dt, p, q in zip(types, strikes, times, frame[column].to_numpy(dtype=float), dividends)])
        report["calculation"].append({"contracts": count, **measure(calculation, args.repeats)})
    for count in (256, 1024, len(raw)):
        frame = raw.sample(n=count, random_state=42).sort_index() if count < len(raw) else raw
        report["surface"].append({"raw_contracts": count, **measure(lambda: build(frame), args.repeats)})
    if args.profile:
        profiler = cProfile.Profile()
        with contextlib.redirect_stdout(io.StringIO()):
            profiler.runcall(build, raw)
        output = io.StringIO()
        pstats.Stats(profiler, stream=output).strip_dirs().sort_stats("cumulative").print_stats(35)
        report["profile"] = output.getvalue()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
