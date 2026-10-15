"""Durable collection and offline history CLI, separate from ephemeral web requests."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

from src.history_store import LocalObservationStore, ObservationError, json_ready
from src.surface_config import SurfaceRequest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="data/history", help="Persistent local observation lake")
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="One collection pass; schedule this on a persistent host")
    collect.add_argument("tickers", nargs="+", help="Small watchlist (maximum 10 per invocation)")
    collect.add_argument("--dte-min", type=int, default=1)
    collect.add_argument("--dte-max", type=int, default=60)
    collect.add_argument("--strike-min", type=float, default=.93)
    collect.add_argument("--strike-max", type=float, default=1.07)
    collect.add_argument("--label", default="")
    commands.add_parser("demo", help="Seed two explicitly synthetic fixtures, including a partial observation")
    commands.add_parser("list")
    for name in ("process", "replay", "inspect"):
        command = commands.add_parser(name)
        command.add_argument("snapshot_id")
        if name == "replay":
            command.add_argument("--processing-id")
        if name == "process":
            command.add_argument("--backend", choices=["python", "cpp", "auto"], default=None)
    query = commands.add_parser("query", help="Local DuckDB SQL over integrity-validated Parquet")
    query.add_argument("sql")
    compare = commands.add_parser("compare")
    compare.add_argument("left")
    compare.add_argument("right")
    compare.add_argument("--output", default="demo-output/comparison.html")
    export = commands.add_parser("export", help="Export a verified immutable observation and its processing artifacts")
    export.add_argument("snapshot_id")
    export.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    store = LocalObservationStore(args.store)
    started = time.perf_counter()
    try:
        if args.command == "demo":
            from src.history_demo import seed_demo
            result = [{"snapshot_id": item["snapshot_id"], "partial": item["partial"], "source_kind": item["source_kind"]}
                      for item in seed_demo(store)]
        elif args.command == "collect":
            if len(args.tickers) > 10:
                raise ObservationError("Limit watchlists to at most 10 tickers per collection pass.")
            from src.data_fetch import get_options_data, get_price_observation
            import pandas as pd
            result = []
            for ticker in args.tickers:
                try:
                    request = SurfaceRequest(ticker.upper(), strike_min_pct=args.strike_min,
                                             strike_max_pct=args.strike_max, dte_min=args.dte_min, dte_max=args.dte_max)
                    from src.surface_service import _validated_request
                    request = _validated_request(request)
                    valuation = pd.Timestamp.now(tz="UTC")
                    spot = get_price_observation(request.ticker)
                    raw = get_options_data(request.ticker, min_dte=request.dte_min, max_dte=request.dte_max, as_of_utc=valuation)
                    observation = store.save(request, raw, spot, valuation, label=args.label)
                    result.append({"snapshot_id": observation["snapshot_id"], "partial": observation["partial"],
                                   "raw_rows": observation["raw_row_count"], "ticker": request.ticker})
                except Exception as exc:
                    result.append({"ticker": ticker, "error": str(exc)})
            if any("error" in item for item in result):
                print(json.dumps(json_ready({"results": result, "elapsed_seconds": time.perf_counter() - started}), indent=2))
                return 1
        elif args.command == "list":
            observations, errors = store.list()
            result = {"observations": observations, "invalid_observations": errors, **store.publication_status()}
        elif args.command == "inspect":
            result = {"observation": store.load(args.snapshot_id)[0], **store.publication_status()}
        elif args.command == "process":
            result = store.process(args.snapshot_id, backend=args.backend)
        elif args.command == "replay":
            result = store.replay(args.snapshot_id, args.processing_id)
        elif args.command == "query":
            result = store.query(args.sql).to_dict("records")
        elif args.command == "compare":
            from src.history_compare import compare_observations
            left = store.process(args.left)
            right = store.process(args.right)
            comparison = compare_observations(store, args.left, args.right, left["processing_id"], right["processing_id"])
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            comparison["figure"].write_html(output, include_plotlyjs=True)
            result = {key: comparison[key] for key in ("supported_points", "unsupported_points", "domain", "mean_change_vol_points")}
            result["output"] = str(output.resolve())
        elif args.command == "export":
            from src.history_s3 import export_observation
            result = export_observation(store, args.snapshot_id, Path(args.output))
        print(json.dumps(json_ready({"result": result, "elapsed_seconds": time.perf_counter() - started}), indent=2))
        return 0
    except (ObservationError, ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
