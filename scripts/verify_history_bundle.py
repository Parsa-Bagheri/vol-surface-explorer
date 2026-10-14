"""Verify a prepared web release offline, including all lineage downloads."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import socket
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.history_reader import ReadOnlySnapshotStore
from src.history_compare import compare_observations
from src.history_store import digest
from web_app import create_app


def verify_bundle(root):
    store = ReadOnlySnapshotStore(root)
    def blocked(*args, **kwargs):
        raise AssertionError("Network or current-clock access prohibited during saved-input verification.")
    # Import provider modules before blocking so import-time cache configuration is separate.
    import src.surface_service as service
    import src.data_fetch as fetch
    records = []
    with ExitStack() as stack:
        for target, name in ((socket, "socket"), (pd.Timestamp, "now"), (service, "get_current_price"),
                             (service, "get_options_data"), (fetch.yf, "Ticker")):
            stack.enter_context(patch.object(target, name, blocked))
        client = create_app(history_root=str(root)).test_client()
        assert client.get("/").status_code == 200
        assert client.get("/history").status_code == 200
        for item in store.release["observations"]:
            snapshot_id, processing_id = item["snapshot_id"], item["processing_id"]
            observation = store.load(snapshot_id)[0]
            artifact = store.load_artifact(snapshot_id, processing_id)
            replay = store.replay(snapshot_id, processing_id)
            for filename in ("raw.parquet", "observation.parquet", "manifest.json", "cleaned.parquet", "selected.parquet",
                             "coverage.parquet", "surface.json", "diagnostics.json", "processing-manifest.json"):
                response = client.get(f"/history/{snapshot_id}/files/{filename}?processing={processing_id}")
                assert response.status_code == 200 and response.headers["Cache-Control"] == "no-store"
                source = ((store.observation_path(snapshot_id) / filename) if filename in {"raw.parquet", "observation.parquet", "manifest.json"}
                          else store.artifact_path(snapshot_id, processing_id) / ("manifest.json" if filename == "processing-manifest.json" else filename))
                assert response.data == source.read_bytes()
            records.append({"snapshot_id": snapshot_id, "processing_id": processing_id,
                            "valuation_time_utc": observation["valuation_time_utc"], "source_kind": observation["source_kind"],
                            "spot": observation["spot"], "partial": observation["partial"],
                            "expected_expirations": len(observation["expected_expirations"]),
                            "received_expirations": len(observation["received_expirations"]),
                            "outcomes": artifact["contract_outcomes"], "processor": artifact["processor"], "replay": replay})
        comparison = None
        if len(records) == 2:
            a, b = records
            result = compare_observations(store, a["snapshot_id"], b["snapshot_id"], a["processing_id"], b["processing_id"])
            response = client.get(f"/history?left={a['snapshot_id']}&right={b['snapshot_id']}")
            assert response.status_code == 200 and b"supported comparison points" in response.data
            comparison = {name: result[name] for name in ("supported_points", "unsupported_points", "domain", "mean_change_vol_points")}
    return {"verified": True, "release_sha256": digest(store.release), "observations": records,
            "comparison": comparison, "replay_network_and_current_clock_blocked": True,
            "all_lineage_downloads_match": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="history_bundle")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = verify_bundle(args.bundle)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"verified": True, "output": str(Path(args.output).resolve())}))


if __name__ == "__main__":
    main()
