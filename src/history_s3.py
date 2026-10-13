"""Verified local exports and optional immutable S3 transport (not deployed)."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile

from src.history_store import LocalObservationStore, ObservationError, canonical, digest, file_digest, local_path


def export_observation(store: LocalObservationStore, snapshot_id: str, output: Path) -> dict:
    output = local_path(output)
    observation, _ = store.load(snapshot_id)
    if output.exists():
        raise ObservationError("Export destination already exists; choose a new immutable export path.")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".pending-export-", dir=output.parent))
    try:
        sources = [store.observation_path(snapshot_id)]
        for path in sorted((store.root / "processed" / snapshot_id).glob("*")):
            if path.is_dir() and not path.name.startswith("."):
                store.load_artifact(snapshot_id, path.name)
                sources.append(path)
        files = {}
        for source in sources:
            relative = source.relative_to(store.root)
            for path in source.iterdir():
                if not path.is_file() or path.is_symlink():
                    raise ObservationError("Unexpected export file.")
                target = stage / relative / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                files[target.relative_to(stage).as_posix()] = {"sha256": file_digest(target), "bytes": target.stat().st_size}
        payload = {"schema_version": 1, "snapshot_id": snapshot_id, "files": files,
                   "source_kind": observation["source_kind"]}
        bundle = {"payload": payload, "bundle_sha256": digest(payload)}
        (stage / "bundle.json").write_bytes(canonical(bundle))
        # Nested export files are flushed separately; publish only after the bundle validates.
        verify_export(stage)
        stage.rename(output)
        return {"output": str(output.resolve()), "snapshot_id": snapshot_id, "bundle_sha256": bundle["bundle_sha256"],
                "files": len(files), "bytes": sum(item["bytes"] for item in files.values())}
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def verify_export(root: Path) -> dict:
    root = local_path(root)
    try:
        bundle = json.loads((root / "bundle.json").read_text("utf-8"))
        payload = bundle["payload"]
        if payload.get("schema_version") != 1 or digest(payload) != bundle["bundle_sha256"]:
            raise ObservationError("Invalid export bundle checksum or schema.")
        for name, metadata in payload["files"].items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
                raise ObservationError("Unsafe export path.")
            path = root / relative
            if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
                raise ObservationError("Unsafe export file.")
            if file_digest(path) != metadata["sha256"] or path.stat().st_size != metadata["bytes"]:
                raise ObservationError(f"Export checksum mismatch: {name}.")
        temporary_store = LocalObservationStore(root)
        temporary_store.load(payload["snapshot_id"])
        for path in (root / "processed" / payload["snapshot_id"]).glob("*"):
            temporary_store.load_artifact(payload["snapshot_id"], path.name)
        return bundle
    except ObservationError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ObservationError(f"Invalid export bundle: {exc}") from exc


class S3ObservationArchive:
    """Same immutable bundle boundary on S3. Caller supplies authorized client/access.

    No AWS calls happen on import. The optional boto3 client is supplied by the
    operator after verifying account access and costs. Objects precede commit.json.
    """
    def __init__(self, client, bucket: str, prefix: str = "vol-history"):
        prefix = prefix.strip("/")
        if not bucket or ".." in PurePosixPath(prefix).parts or "\\" in prefix:
            raise ObservationError("Invalid archive bucket/prefix.")
        self.client, self.bucket, self.prefix = client, bucket, prefix

    def upload(self, export_root: Path) -> dict:
        export_root = local_path(export_root)
        bundle = verify_export(export_root)
        root_key = f"{self.prefix}/{bundle['payload']['snapshot_id']}/{bundle['bundle_sha256']}"
        for name, metadata in bundle["payload"]["files"].items():
            self._put_once(f"{root_key}/objects/{metadata['sha256']}", (export_root / name).read_bytes())
        # Single-object commit point: an interrupted upload exposes no committed archive.
        self._put_once(f"{root_key}/commit.json", canonical(bundle))
        return {"bucket": self.bucket, "commit_key": f"{root_key}/commit.json",
                "bundle_sha256": bundle["bundle_sha256"]}

    def _put_once(self, key: str, body: bytes) -> None:
        try:
            self.client.put_object(Bucket=self.bucket, Key=key, Body=body, IfNoneMatch="*",
                                   ServerSideEncryption="AES256")
        except Exception as exc:
            response = getattr(exc, "response", {})
            if response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 412:
                raise
            existing = self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
            if existing != body:
                raise ObservationError("Immutable S3 object conflict.") from exc

    def download(self, commit_key: str, output: Path) -> dict:
        output = local_path(output)
        if output.exists() or not commit_key.startswith(self.prefix + "/") or not commit_key.endswith("/commit.json"):
            raise ObservationError("Invalid archive commit key or existing destination.")
        if ".." in PurePosixPath(commit_key).parts or "\\" in commit_key:
            raise ObservationError("Unsafe archive key.")
        bundle = json.loads(self.client.get_object(Bucket=self.bucket, Key=commit_key)["Body"].read())
        if digest(bundle["payload"]) != bundle["bundle_sha256"]:
            raise ObservationError("S3 commit checksum mismatch.")
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".pending-download-", dir=output.parent))
        try:
            for name, metadata in bundle["payload"]["files"].items():
                relative = PurePosixPath(name)
                if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
                    raise ObservationError("Unsafe archive file path.")
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                key = commit_key.removesuffix("commit.json") + "objects/" + metadata["sha256"]
                target.write_bytes(self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read())
            (stage / "bundle.json").write_bytes(canonical(bundle))
            verify_export(stage)
            stage.rename(output)
            return {"output": str(output.resolve()), "bundle_sha256": bundle["bundle_sha256"]}
        finally:
            if stage.exists():
                shutil.rmtree(stage)
