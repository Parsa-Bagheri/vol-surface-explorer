"""Verified, read-only release bundles. No collection, network access or cache writes."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from src.history_store import LocalObservationStore, ObservationError, digest, file_digest


class ReadOnlySnapshotStore(LocalObservationStore):
    def __init__(self, root):
        super().__init__(root)
        try:
            envelope = json.loads((self.root / "release.json").read_text("utf-8"))
            self.release = envelope["payload"]
            if self.release["schema_version"] != 1 or digest(self.release) != envelope["release_sha256"]:
                raise ObservationError("Release bundle checksum or schema mismatch.")
            self.versions = {}
            for item in self.release["observations"]:
                snapshot_id = self._id(item["snapshot_id"])
                processing_id = self._id(item["processing_id"])
                if snapshot_id in self.versions:
                    raise ObservationError("Duplicate release observation.")
                self.versions[snapshot_id] = processing_id
            if not self.versions:
                raise ObservationError("Empty release bundle.")
            for name, metadata in self.release["files"].items():
                relative = PurePosixPath(name)
                if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
                    raise ObservationError("Unsafe release bundle path.")
                path = self._safe_path(self.root / relative)
                if not path.is_file() or path.stat().st_size != metadata["bytes"] or file_digest(path) != metadata["sha256"]:
                    raise ObservationError(f"Release file checksum mismatch: {name}.")
            for item in self.release["observations"]:
                observation = self.load(item["snapshot_id"])[0]
                if (observation["source_kind"] != "collected" or observation["valuation_time_utc"] != item["valuation_time_utc"]
                        or observation["files"]["raw.parquet"]["sha256"] != item["raw_sha256"]):
                    raise ObservationError("Release source identity mismatch.")
                self.load_artifact(item["snapshot_id"], item["processing_id"])
        except ObservationError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ObservationError(f"Invalid release bundle: {exc}") from exc

    def observation_path(self, snapshot_id):
        if snapshot_id not in self.versions:
            raise ObservationError("Observation is not in this release.")
        return super().observation_path(snapshot_id)

    def artifact_path(self, snapshot_id, processing_id):
        if self.versions.get(snapshot_id) != processing_id:
            raise ObservationError("Processing version is not in this release.")
        return super().artifact_path(snapshot_id, processing_id)

    def save(self, *args, **kwargs):
        raise ObservationError("The deployed snapshot bundle is read-only.")

    def process(self, *args, **kwargs):
        raise ObservationError("The deployed snapshot bundle is read-only; prepare versions on the collector host.")


def open_history_store(root, *, require_bundle=False):
    if require_bundle or (Path(root) / "release.json").is_file():
        return ReadOnlySnapshotStore(root)
    return LocalObservationStore(root)
