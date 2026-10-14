"""Prepare an immutable web bundle from explicitly selected real observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.history_reader import ReadOnlySnapshotStore
from src.history_store import LocalObservationStore, ObservationError, canonical, digest, file_digest, local_path


def build_bundle(store, selections, output):
    output = local_path(output)
    if output.exists():
        raise ObservationError("Bundle destination already exists; prepare a new immutable release.")
    if not 1 <= len(selections) <= 10 or len({s for s, _ in selections}) != len(selections):
        raise ObservationError("Select 1–10 distinct observations and explicit processing versions.")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".pending-bundle-", dir=output.parent))
    try:
        observations, files = [], {}
        for snapshot_id, processing_id in selections:
            observation = store.load(snapshot_id)[0]
            store.load_artifact(snapshot_id, processing_id)
            if observation["source_kind"] != "collected":
                raise ObservationError("The public release bundle accepts collected inputs only.")
            for source in (store.observation_path(snapshot_id), store.artifact_path(snapshot_id, processing_id)):
                for path in source.iterdir():
                    if not path.is_file() or path.is_symlink():
                        raise ObservationError("Unexpected bundle file.")
                    name = path.relative_to(store.root).as_posix()
                    target = stage / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
                    files[name] = {"sha256": file_digest(target), "bytes": target.stat().st_size}
            observations.append({"snapshot_id": snapshot_id, "processing_id": processing_id,
                                 "valuation_time_utc": observation["valuation_time_utc"],
                                 "raw_sha256": observation["files"]["raw.parquet"]["sha256"]})
        if sum(m["bytes"] for m in files.values()) > 20_000_000:
            raise ObservationError("Web bundle exceeds the 20 MB first-release limit.")
        if any(m["bytes"] > 4_000_000 for m in files.values()):
            raise ObservationError("A history download exceeds the bounded host response size.")
        payload = {"schema_version": 1, "observations": observations, "files": files}
        envelope = {"payload": payload, "release_sha256": digest(payload)}
        (stage / "release.json").write_bytes(canonical(envelope))
        ReadOnlySnapshotStore(stage)
        stage.rename(output)
        return {"release_sha256": envelope["release_sha256"], "observations": observations,
                "bytes": sum(m["bytes"] for m in files.values()), "output": str(output)}
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def selections(values):
    pairs = [value.split(":") for value in values]
    if any(len(pair) != 2 for pair in pairs):
        raise ObservationError("Use SNAPSHOT_ID:PROCESSING_ID for each selection.")
    for pair in pairs:
        for value in pair:
            LocalObservationStore._id(value)
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("observations", nargs="+")
    args = parser.parse_args()
    print(json.dumps(build_bundle(LocalObservationStore(args.store), selections(args.observations), args.output), indent=2))


if __name__ == "__main__":
    main()
