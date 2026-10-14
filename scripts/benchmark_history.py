"""Reproducible local fixture workload. No market collection or AWS calls."""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.history_store import LocalObservationStore, processing_identity
from src.history_demo import seed_demo
from src.history_compare import compare_observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, help="A new fixture-only store directory")
    parser.add_argument("--output", default="docs/evidence/history-benchmark.json")
    args = parser.parse_args()
    store = LocalObservationStore(args.store)
    if store.root.exists():
        raise SystemExit("Use a new empty store path to measure first publication.")
    def timed(action):
        start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            result = action()
        return result, time.perf_counter() - start
    observations, total = timed(lambda: seed_demo(store))
    artifacts = [store.process(item["snapshot_id"]) for item in observations]
    replays, idempotent, query_times = [], [], []
    for _ in range(3):
        _, elapsed = timed(lambda: store.replay("synthetic-a", artifacts[0]["processing_id"]))
        replays.append(elapsed)
        _, elapsed = timed(lambda: store.process("synthetic-a"))
        idempotent.append(elapsed)
        rows, elapsed = timed(lambda: store.query("SELECT snapshot_id, partial, raw_rows, used_rows, excluded_rows FROM coverage ORDER BY snapshot_id"))
        query_times.append(elapsed)
    comparison, compare_seconds = timed(lambda: compare_observations(store, "synthetic-a", "synthetic-b",
                                                  artifacts[0]["processing_id"], artifacts[1]["processing_id"]))
    table_sizes = {str(path.relative_to(store.root)): path.stat().st_size for path in store.root.rglob("*.parquet")}
    report = {"workload": "Two synthetic BSM quote fixtures; four requested expirations; B has one simulated failed expiration.",
              "environment": {"python": platform.python_version(), "os": platform.platform(),
                              "machine": platform.machine(), "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "plotly", "pyarrow", "duckdb")}},
              "processor": processing_identity(), "seed_and_first_processing_seconds": total,
              "offline_replay_seconds": replays, "offline_replay_median_seconds": statistics.median(replays),
              "idempotent_process_seconds": idempotent, "query_seconds": query_times,
              "query_median_seconds": statistics.median(query_times), "coverage": rows.to_dict("records"),
              "comparison_seconds": compare_seconds,
              "comparison": {key: comparison[key] for key in ("supported_points", "unsupported_points", "domain", "mean_change_vol_points")},
              "parquet_bytes": sum(table_sizes.values()), "parquet_files": table_sizes,
              "market_collection": "Not executed; fixture ingestion only", "aws": "Prepared only; not deployed or queried",
              "limitations": "Warm local synthetic workload; no Yahoo latency, browser rendering, independent accuracy benchmark or production load."}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
