"""Prepare a bounded, typed local Athena coverage projection; no AWS calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.history_store import LocalObservationStore, ObservationError, file_digest, local_path


def export_projection(store, snapshot_id, processing_id, root, *, versioned=False):
    observation = store.load(snapshot_id)[0]
    store.load_artifact(snapshot_id, processing_id)
    data = pd.read_parquet(store.artifact_path(snapshot_id, processing_id) / "coverage.parquet")
    data = data.drop(columns="ticker")
    data["valuation_time_utc"] = data.valuation_time_utc.dt.tz_convert("UTC").dt.tz_localize(None)
    directory = local_path(root) / f"date={observation['valuation_time_utc'][:10]}" / f"ticker={observation['ticker']}"
    if versioned:
        directory /= f"version={LocalObservationStore._id(processing_id)}"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{snapshot_id}.parquet"
    if output.exists():
        raise ObservationError("Projection already exists; choose a fresh root to explicitly select another processing version.")
    temporary = directory / f".{snapshot_id}.pending"
    try:
        data.to_parquet(temporary, index=False, coerce_timestamps="us", allow_truncated_timestamps=False)
        temporary.rename(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {"output": str(output), "bytes": output.stat().st_size, "sha256": file_digest(output),
            "snapshot_id": snapshot_id, "processing_id": processing_id, "aws_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_id")
    parser.add_argument("processing_id")
    parser.add_argument("--store", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(export_projection(LocalObservationStore(args.store), args.snapshot_id, args.processing_id, args.output), indent=2))


if __name__ == "__main__":
    main()
