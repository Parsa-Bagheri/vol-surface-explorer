"""Immutable local observation lake. Manifests are the publication boundary."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, date
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import tempfile
from typing import Any

import pandas as pd

from src.surface_config import SurfaceRequest
from src.surface_service import _validated_request, process_surface_inputs
from src.time_utils import normalize_as_of_utc

SCHEMA_VERSION = 1
PROCESSING_VERSION = "history-v1"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$")
CODE_FILES = ("data_cleaner.py", "visualizer.py", "contract_outcomes.py", "time_utils.py",
              "surface_service.py", "surface_config.py", "history_store.py")


class ObservationError(ValueError):
    """A saved observation is incomplete, corrupt, incompatible or invalid."""


def local_path(value: str | Path) -> Path:
    """Support deeply nested Windows checkouts without depending on OS long-path policy."""
    absolute = str(Path(value).resolve())
    if os.name == "nt" and not absolute.startswith("\\\\?\\"):
        absolute = "\\\\?\\UNC\\" + absolute[2:] if absolute.startswith("\\\\") else "\\\\?\\" + absolute
    return Path(absolute)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, Path, datetime, date)):
        return str(value)
    if hasattr(value, "item"):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical(value: Any) -> bytes:
    return json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def source_digest(path: Path) -> str:
    """Git's newline conversion must not change an otherwise identical processor."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def processing_identity() -> dict:
    root = Path(__file__).parent
    versions = {name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scipy", "plotly", "pyarrow")}
    return {"version": PROCESSING_VERSION,
            "code_sha256": digest({name: source_digest(root / name) for name in CODE_FILES}),
            "dependencies": versions}


def _utc(value: Any, label: str) -> str:
    try:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            raise ValueError("timezone required")
        return timestamp.tz_convert("UTC").isoformat()
    except (TypeError, ValueError) as exc:
        raise ObservationError(f"{label} must be a timezone-aware timestamp.") from exc


def _write_json(path: Path, payload: dict) -> None:
    path.write_bytes(canonical(payload))


def _manifest(stage: Path, payload: dict, names: list[str]) -> dict:
    payload = {**payload, "files": {name: {"sha256": file_digest(stage / name),
                                          "bytes": (stage / name).stat().st_size}
                                  for name in names}}
    envelope = {"payload": payload, "manifest_sha256": digest(payload)}
    _write_json(stage / "manifest.json", envelope)
    return payload


def _publish(stage: Path, destination: Path) -> None:
    # Flush files before publishing the complete directory in one rename.
    for path in stage.iterdir():
        with path.open("r+b") as handle:
            os.fsync(handle.fileno())
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.rename(stage, destination)  # No overwrite; a concurrent winning writer is inspected.


def _read_manifest(directory: Path, kind: str) -> dict:
    try:
        envelope = json.loads((directory / "manifest.json").read_text("utf-8"))
        payload = envelope["payload"]
        if envelope["manifest_sha256"] != digest(payload):
            raise ObservationError("Manifest checksum mismatch.")
        if payload["schema_version"] != SCHEMA_VERSION:
            raise ObservationError(f"Unsupported schema version: {payload['schema_version']}.")
        if payload["kind"] != kind:
            raise ObservationError("Unexpected manifest kind.")
        required = ({"raw.parquet", "observation.parquet"} if kind == "observation" else
                    {"cleaned.parquet", "selected.parquet", "coverage.parquet", "surface.json", "diagnostics.json"})
        if set(payload["files"]) != required:
            raise ObservationError("Manifest has an unexpected file set.")
        for name, metadata in payload["files"].items():
            path = directory / name
            if path.is_symlink() or not path.is_file():
                raise ObservationError(f"Missing or unsafe saved file: {name}.")
            if path.stat().st_size != metadata["bytes"] or file_digest(path) != metadata["sha256"]:
                raise ObservationError(f"File checksum mismatch: {name}.")
        return payload
    except ObservationError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ObservationError(f"Invalid {kind} manifest: {exc}") from exc


def _annotate_keys(raw: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, dict]:
    raw = raw.copy()
    raw["sourceRowJson"] = [canonical(row).decode("utf-8") for row in raw.to_dict("records")]
    for name in ("expirationDate", "optionType", "strike"):
        if name not in raw:
            raw[name] = pd.Series(dtype="object") if raw.empty else None
    dates = pd.to_datetime(raw.expirationDate, errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
    strikes = pd.to_numeric(raw.strike, errors="coerce").replace([float("inf"), float("-inf")], float("nan"))
    types = raw.optionType.astype(str).str.lower()
    invalid = dates.isna() | strikes.isna() | ~strikes.gt(0) | ~types.isin(["call", "put"])
    symbols = raw.get("contractSymbol", pd.Series("", index=raw.index)).fillna("").astype(str).str.strip()
    fallback = ticker + ":" + dates.fillna("invalid") + ":" + types + ":" + strikes.map(lambda x: format(x, ".12g"))
    raw["stableContractKey"] = symbols.where(symbols.ne(""), fallback)
    duplicate = raw.stableContractKey.duplicated(keep="first") & ~invalid
    raw["historyInvalidReason"] = ""
    raw.loc[invalid, "historyInvalidReason"] = "Malformed contract"
    raw.loc[duplicate, "historyInvalidReason"] = "Duplicate contract"
    # Parquet has a typed schema. Preserve the exact input values in sourceRowJson
    # while making invalid numeric/date fields nullable in the analytical columns.
    for name in ("strike", "bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility", "change", "percentChange"):
        if name in raw:
            raw[name] = pd.to_numeric(raw[name], errors="coerce").replace([float("inf"), float("-inf")], float("nan")).astype(float)
    for name in ("expirationDate", "lastTradeDate"):
        if name in raw:
            raw[name] = pd.to_datetime(raw[name], errors="coerce", utc=True)
    raw["optionType"] = types
    return raw, {"malformed_rows": int(invalid.sum()), "duplicate_rows": int(duplicate.sum())}


class LocalObservationStore:
    def __init__(self, root: str | Path = "data/history"):
        self.root = local_path(root)

    def _safe_path(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root) or path.is_symlink():
            raise ObservationError("Saved path escapes the observation store.")
        return path

    @staticmethod
    def _id(value: str) -> str:
        if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
            raise ObservationError("Invalid observation or processing ID.")
        return value

    def observation_path(self, snapshot_id: str) -> Path:
        snapshot_id = self._id(snapshot_id)
        matches = list((self.root / "observations").glob(f"date=*/ticker=*/{snapshot_id}"))
        if len(matches) != 1:
            raise ObservationError(f"Observation not found or ambiguous: {snapshot_id}.")
        return self._safe_path(matches[0])

    def save(self, request: SurfaceRequest, raw: pd.DataFrame, spot: dict,
             valuation_time_utc: Any, *, source_kind: str = "collected", label: str = "",
             snapshot_id: str | None = None) -> dict:
        request = _validated_request(request)
        valuation = _utc(valuation_time_utc, "valuation_time_utc")
        value = float(spot.get("value", 0))
        if not math.isfinite(value) or value <= 0:
            raise ObservationError("Spot must be finite and positive.")
        if source_kind not in {"collected", "synthetic_fixture", "imported"}:
            raise ObservationError("Unknown observation source kind.")
        spot = json_ready(spot)
        for name in ("collected_at_utc", "source_timestamp_utc"):
            if spot.get(name) is not None:
                spot[name] = _utc(spot[name], f"spot.{name}")
        if not spot.get("provider") or not spot.get("method") or not spot.get("collected_at_utc"):
            raise ObservationError("Spot provenance requires provider, method and collection time.")
        fetch = json_ready(raw.attrs.get("fetchDiagnostics", {}))
        raw, quality = _annotate_keys(raw, request.ticker)
        raw.attrs = {}
        raw["ticker"] = request.ticker
        # Stable order, while preserving the incoming per-build row IDs in raw storage.
        expected = sorted({str(item["expiration"]) for item in fetch.get("selected_expirations", [])})
        received = sorted(set(pd.to_datetime(raw.expirationDate, errors="coerce").dropna().dt.strftime("%Y-%m-%d")))
        if not expected:
            expected = received
        failed = sorted(set(expected) - set(received))
        outcome_failures = sorted({str(item["expiration"]) for item in fetch.get("outcomes", [])
                                   if item.get("status") in {"failed", "empty"}})
        payload = {"schema_version": SCHEMA_VERSION, "kind": "observation", "ticker": request.ticker,
                   "valuation_time_utc": valuation, "spot": spot, "parameters": asdict(request),
                   "source_kind": source_kind, "label": label, "processing_at_collection": processing_identity(),
                   "fetch": fetch, "expected_expirations": expected, "received_expirations": received,
                   "missing_expirations": failed, "failed_expirations": outcome_failures,
                   "partial": bool(failed or outcome_failures), "raw_row_count": len(raw),
                   "quality": quality}
        self.root.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".pending-", dir=self.root))
        lock = None
        try:
            raw.to_parquet(stage / "raw.parquet", index=False)
            identity = digest({**payload, "raw_sha256": file_digest(stage / "raw.parquet")})
            snapshot_id = self._id(snapshot_id or identity[:24])
            lock_parent = self._safe_path(self.root / ".locks")
            lock_parent.mkdir(exist_ok=True)
            candidate_lock = lock_parent / snapshot_id
            try:
                candidate_lock.mkdir()
            except FileExistsError as exc:
                raise ObservationError("Observation publication in progress or interrupted; inspect the ID lock before retrying.") from exc
            lock = candidate_lock
            _write_json(lock / "owner.json", {"pid": os.getpid(), "host": socket.gethostname(),
                                               "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat()})
            existing_paths = list((self.root / "observations").glob(f"date=*/ticker=*/{snapshot_id}"))
            if existing_paths:
                existing = self.load(snapshot_id)[0]
                if existing["identity_sha256"] != identity:
                    raise ObservationError("Observation IDs are globally immutable; conflicting date, ticker or inputs.")
                return existing
            raw["snapshot_id"] = snapshot_id
            raw.to_parquet(stage / "raw.parquet", index=False)
            payload["snapshot_id"] = snapshot_id
            payload["identity_sha256"] = identity
            summary = {"snapshot_id": snapshot_id, "ticker": request.ticker, "schema_version": SCHEMA_VERSION,
                       "valuation_time_utc": pd.Timestamp(valuation), "spot": value, "source_kind": source_kind,
                       "partial": payload["partial"], "expected_expirations": len(expected),
                       "received_expirations": len(received), "failed_expirations": len(outcome_failures),
                       "raw_rows": len(raw), **quality}
            pd.DataFrame([summary]).to_parquet(stage / "observation.parquet", index=False)
            payload = _manifest(stage, payload, ["raw.parquet", "observation.parquet"])
            destination = self._safe_path(self.root / "observations" / f"date={valuation[:10]}" /
                                          f"ticker={request.ticker}" / snapshot_id)
            if destination.exists():
                existing = self.load(snapshot_id)[0]
                if existing["identity_sha256"] != identity:
                    raise ObservationError("Observation IDs are immutable; conflicting inputs.")
                return existing
            try:
                _publish(stage, destination)
            except OSError:
                if not destination.exists() or self.load(snapshot_id)[0]["identity_sha256"] != identity:
                    raise
            return payload
        finally:
            if stage.exists():
                shutil.rmtree(stage)
            if lock is not None:
                (lock / "owner.json").unlink(missing_ok=True)
                lock.rmdir()

    def load(self, snapshot_id: str) -> tuple[dict, pd.DataFrame]:
        path = self.observation_path(snapshot_id)
        payload = _read_manifest(path, "observation")
        try:
            if payload["snapshot_id"] != snapshot_id:
                raise ObservationError("Observation ID does not match its directory.")
            _utc(payload["valuation_time_utc"], "valuation_time_utc")
            _validated_request(SurfaceRequest(**payload["parameters"]))
            spot = float(payload["spot"]["value"])
            if not math.isfinite(spot) or spot <= 0 or payload["parameters"]["ticker"] != payload["ticker"]:
                raise ObservationError("Invalid saved ticker or spot.")
            if path.parent.name != f"ticker={payload['ticker']}" or path.parent.parent.name != f"date={payload['valuation_time_utc'][:10]}":
                raise ObservationError("Observation partition does not match its metadata.")
            raw = pd.read_parquet(path / "raw.parquet")
            if len(raw) != payload["raw_row_count"] or not {"stableContractKey", "historyInvalidReason", "snapshot_id"}.issubset(raw.columns):
                raise ObservationError("Raw row count or required schema does not match the manifest.")
            if not raw.empty and (not raw.snapshot_id.eq(snapshot_id).all() or not raw.ticker.eq(payload["ticker"]).all()):
                raise ObservationError("Raw row lineage does not match the manifest.")
            raw.attrs["fetchDiagnostics"] = payload["fetch"]
            return payload, raw
        except ObservationError:
            raise
        except (ValueError, TypeError, KeyError, OSError) as exc:
            raise ObservationError(f"Invalid saved observation: {exc}") from exc

    def list(self) -> tuple[list[dict], list[dict]]:
        valid, errors = [], []
        for path in sorted((self.root / "observations").glob("date=*/ticker=*/*")):
            if path.name.startswith(".") or not path.is_dir():
                continue
            try:
                valid.append(self.load(path.name)[0])
            except ObservationError as exc:
                errors.append({"snapshot_id": path.name, "error": str(exc)})
        return sorted(valid, key=lambda item: item["valuation_time_utc"], reverse=True), errors

    def publication_status(self) -> dict:
        locks = []
        for path in (self.root / ".locks").glob("*"):
            try:
                owner = json.loads((self._safe_path(path) / "owner.json").read_text("utf-8"))
            except (OSError, ValueError):
                owner = {"owner": "unknown"}
            locks.append({"snapshot_id": path.name, "path": str(path), **owner})
        return {"publication_locks": locks, "unpublished_staging_directories": [str(path) for path in self.root.glob(".pending-*")]}

    def artifact_path(self, snapshot_id: str, processing_id: str) -> Path:
        self.observation_path(snapshot_id)
        return self._safe_path(self.root / "processed" / self._id(snapshot_id) / self._id(processing_id))

    def load_artifact(self, snapshot_id: str, processing_id: str) -> dict:
        observation, _ = self.load(snapshot_id)
        payload = _read_manifest(self.artifact_path(snapshot_id, processing_id), "processing")
        if (payload.get("snapshot_id") != snapshot_id or payload.get("processing_id") != processing_id
                or payload.get("raw_sha256") != observation["files"]["raw.parquet"]["sha256"]):
            raise ObservationError("Processing lineage mismatch.")
        expected_id = digest({"raw_sha256": payload["raw_sha256"], "parameters": observation["parameters"],
                              "processor": payload["processor"]})[:24]
        if expected_id != processing_id:
            raise ObservationError("Processing identity mismatch.")
        path = self.artifact_path(snapshot_id, processing_id)
        try:
            cleaned = pd.read_parquet(path / "cleaned.parquet")
            selected = pd.read_parquet(path / "selected.parquet")
            coverage = pd.read_parquet(path / "coverage.parquet")
            for frame in (cleaned, selected, coverage):
                if (not frame.snapshot_id.eq(snapshot_id).all() or not frame.processing_id.eq(processing_id).all()):
                    raise ObservationError("Processed row lineage mismatch.")
            outcomes = payload["contract_outcomes"]
            if (len(cleaned) != payload["cleaned_row_count"] or len(selected) != outcomes["used"]
                    or outcomes["fetched"] != observation["raw_row_count"]
                    or outcomes["fetched"] != outcomes["used"] + outcomes["excluded"]
                    or outcomes["excluded"] != sum(item["count"] for item in outcomes["reasons"])
                    or not selected.contractId.isin(cleaned.contractId).all()):
                raise ObservationError("Processed row counts or reconciliation mismatch.")
            if (len(coverage) != 1 or coverage.iloc[0].raw_rows != observation["raw_row_count"]
                    or coverage.iloc[0].cleaned_rows != len(cleaned) or coverage.iloc[0].used_rows != len(selected)
                    or coverage.iloc[0].excluded_rows != outcomes["excluded"]
                    or bool(coverage.iloc[0].partial) != observation["partial"]):
                raise ObservationError("Processing coverage metadata mismatch.")
        except ObservationError:
            raise
        except (OSError, ValueError, KeyError, AttributeError, TypeError) as exc:
            raise ObservationError(f"Invalid processed Parquet schema: {exc}") from exc
        return payload

    def process(self, snapshot_id: str) -> dict:
        observation, raw = self.load(snapshot_id)
        processor = processing_identity()
        processing_id = digest({"raw_sha256": observation["files"]["raw.parquet"]["sha256"],
                                "parameters": observation["parameters"], "processor": processor})[:24]
        destination = self.artifact_path(snapshot_id, processing_id)
        if destination.exists():
            return self.load_artifact(snapshot_id, processing_id)
        result = process_surface_inputs(SurfaceRequest(**observation["parameters"]), raw,
                                        observation["spot"]["value"], observation["valuation_time_utc"])
        selected = result.diagnostics["contract_outcomes"]
        if selected["fetched"] != selected["used"] + selected["excluded"]:
            raise ObservationError("Contract accounting did not reconcile.")
        from src.visualizer import select_surface_quotes
        eligible = result.cleaned_options_df
        if not result.request.include_low_confidence:
            eligible = eligible[eligible.includeInSurface.fillna(False).astype(bool)]
        quotes = select_surface_quotes(eligible, result.current_price, result.request.risk_free_rate,
                                       result.request.dividend_yield)
        self.root.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".pending-", dir=self.root))
        try:
            for frame, name in ((result.cleaned_options_df, "cleaned.parquet"), (quotes, "selected.parquet")):
                frame = frame.copy()
                frame.attrs = {}
                frame["snapshot_id"], frame["processing_id"] = snapshot_id, processing_id
                frame.to_parquet(stage / name, index=False)
            coverage = {"snapshot_id": snapshot_id, "processing_id": processing_id,
                        "schema_version": SCHEMA_VERSION, "ticker": observation["ticker"],
                        "valuation_time_utc": pd.Timestamp(observation["valuation_time_utc"]),
                        "source_kind": observation["source_kind"], "partial": observation["partial"],
                        "expected_expirations": len(observation["expected_expirations"]),
                        "received_expirations": len(observation["received_expirations"]),
                        "raw_rows": len(raw), "cleaned_rows": len(result.cleaned_options_df),
                        "used_rows": selected["used"], "excluded_rows": selected["excluded"],
                        **observation["quality"]}
            pd.DataFrame([coverage]).to_parquet(stage / "coverage.parquet", index=False)
            (stage / "surface.json").write_text(result.figure.to_json(), encoding="utf-8")
            _write_json(stage / "diagnostics.json", json_ready(result.diagnostics))
            payload = _manifest(stage, {"schema_version": SCHEMA_VERSION, "kind": "processing",
                                 "snapshot_id": snapshot_id, "processing_id": processing_id,
                                 "raw_sha256": observation["files"]["raw.parquet"]["sha256"],
                                 "processor": processor,
                                 "reproduction": processor == observation["processing_at_collection"],
                                 "contract_outcomes": selected, "cleaned_row_count": len(result.cleaned_options_df)},
                                ["cleaned.parquet", "selected.parquet", "coverage.parquet", "surface.json", "diagnostics.json"])
            try:
                _publish(stage, destination)
            except OSError:
                if not destination.exists():
                    raise
                return self.load_artifact(snapshot_id, processing_id)
            return payload
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def replay(self, snapshot_id: str, processing_id: str | None = None, *, tolerance: float = 1e-10) -> dict:
        import numpy as np
        from plotly.utils import PlotlyJSONEncoder
        import plotly.io as pio
        observation, raw = self.load(snapshot_id)
        if not processing_id:
            candidates = [path.name for path in (self.root / "processed" / snapshot_id).glob("*")
                          if path.is_dir() and not path.name.startswith(".")]
            if len(candidates) != 1:
                raise ObservationError("Replay requires one saved processing artifact or an explicit processing_id; run process first.")
            processing_id = candidates[0]
        artifact = self.load_artifact(snapshot_id, processing_id)
        if artifact["processor"] != processing_identity():
            raise ObservationError("Original processing code/dependencies unavailable; use process to explicitly reprocess.")
        result = process_surface_inputs(SurfaceRequest(**observation["parameters"]), raw,
                                        observation["spot"]["value"], observation["valuation_time_utc"])
        path = self.artifact_path(snapshot_id, artifact["processing_id"])
        saved = pd.read_parquet(path / "cleaned.parquet")
        columns = ["contractId", "days_to_expiration", "time_to_expiration_years", "impliedVolatilityFinal", "marketPrice"]
        for name in columns:
            if not np.allclose(saved[name].to_numpy(dtype=float), result.cleaned_options_df[name].to_numpy(dtype=float),
                               rtol=tolerance, atol=tolerance, equal_nan=True):
                raise ObservationError(f"Replay numerical mismatch: {name}.")
        saved_figure = pio.from_json((path / "surface.json").read_text("utf-8"))
        # Plotly templates can change representation on load; the plotted data are the numerical artifact.
        actual = json.loads(json.dumps(result.figure.to_plotly_json()["data"], cls=PlotlyJSONEncoder))
        original = json.loads(json.dumps(saved_figure.to_plotly_json()["data"], cls=PlotlyJSONEncoder))
        if digest(actual) != digest(original) or result.diagnostics["contract_outcomes"] != artifact["contract_outcomes"]:
            raise ObservationError("Replay surface or reconciliation mismatch.")
        return {"snapshot_id": snapshot_id, "processing_id": artifact["processing_id"],
                "matched": True, "tolerance": tolerance, "valuation_time_utc": observation["valuation_time_utc"],
                "spot": observation["spot"]["value"], "reproduction": artifact["reproduction"]}

    def query(self, sql: str) -> pd.DataFrame:
        import duckdb
        observations, errors = self.list()
        if errors:
            raise ObservationError(f"SQL catalog refused corrupt observations: {errors}")
        artifacts = []
        for observation in observations:
            for path in sorted((self.root / "processed" / observation["snapshot_id"]).glob("*")):
                if path.is_dir() and not path.name.startswith("."):
                    self.load_artifact(observation["snapshot_id"], path.name)
                    artifacts.append(path)
        if not observations:
            raise ObservationError("No validated saved observations.")
        with duckdb.connect(":memory:", config={"enable_external_access": "true"}) as connection:
            if artifacts:
                for view, filename in (("coverage", "coverage.parquet"), ("cleaned", "cleaned.parquet"),
                                       ("selected_quotes", "selected.parquet")):
                    connection.from_parquet([str(path / filename) for path in artifacts], hive_partitioning=False).create_view(view)
            connection.from_parquet([str(self.observation_path(item["snapshot_id"]) / "observation.parquet") for item in observations],
                                    hive_partitioning=False).create_view("observations")
            connection.from_parquet([str(self.observation_path(item["snapshot_id"]) / "raw.parquet") for item in observations],
                                    union_by_name=True, hive_partitioning=False).create_view("raw_quotes")
            return connection.execute(sql).fetchdf()
