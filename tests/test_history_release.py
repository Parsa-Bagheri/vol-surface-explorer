"""Release boundaries use generated test inputs and a mocked cloud, never market evidence."""
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import socket

import pytest

from scripts import publish_real_history as release
from scripts.build_history_bundle import build_bundle
from src.history_demo import fixture_quotes, seed_demo
from src.history_reader import ReadOnlySnapshotStore
from src.history_store import LocalObservationStore, ObservationError, canonical, digest
from src.surface_config import SurfaceRequest
from web_app import create_app

ACCOUNT = "123456789012"
AS_OF = "2026-10-12T18:00:00Z"


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("real-release-tests")
    store = LocalObservationStore(root / "lake")
    chosen = []
    for snapshot_id, clock in (("test-a", AS_OF), ("test-b", "2026-10-12T18:10:00Z")):
        raw = fixture_quotes(clock)
        if snapshot_id == "test-b":
            raw.attrs["fetchDiagnostics"]["outcomes"].append({"expiration": "2026-11-13", "status": "failed"})
        store.save(SurfaceRequest("DEMO", strike_min_pct=.9, strike_max_pct=1.1, dte_min=1, dte_max=30), raw,
                   {"value": 100, "provider": "mock collector", "method": "test input",
                    "collected_at_utc": clock, "source_timestamp_utc": clock}, clock,
                   source_kind="collected", snapshot_id=snapshot_id, label="Generated input for publisher tests only")
        chosen.append((snapshot_id, store.process(snapshot_id)["processing_id"]))
    baseline = Path(__file__).parents[1] / "infra/aws/history-demo.json"
    plan = release.prepare(store, chosen, root / "run", baseline)
    return store, chosen, root / "run", plan


class FakeCli:
    def __init__(self, root, plan):
        self.root, self.plan = root, plan
        self.objects, self.partitions, self.calls = {}, {}, []
        self.fail_stage = None
        self.query_rows = plan["local_duckdb_rows"]
        self.query_tokens = []
        self.baseline = json.loads((root / "baseline-stack.json").read_text())
        self.template = json.loads((root / "stack.json").read_text())
        self.current = deepcopy(self.template)
        self.outputs = {"ArchiveBucket": "test-bucket", "WorkGroup": "test-queries", "DatabaseName": "vol_history_demo",
                        "TableName": "volatility_coverage", "RealTableName": release.TABLE}
        self.table = deepcopy(self.template["Resources"]["RealCoverageTable"]["Properties"]["TableInput"])
        self.table["StorageDescriptor"]["Location"] = "s3://test-bucket/real/analytics/"

    def call(self, service, operation, payload=None, extra=()):
        payload = payload or {}
        self.calls.append((service, operation, deepcopy(payload)))
        if service == "sts":
            return {"Account": ACCOUNT}
        if service == "cloudformation":
            if operation == "describe-stacks":
                return {"Stacks": [{"StackId": f"arn:aws:cloudformation:us-east-2:{ACCOUNT}:stack/vol-history-demo/id",
                    "StackStatus": "UPDATE_COMPLETE", "Tags": [{"Key": k, "Value": v} for k, v in release.PROJECT_TAGS.items()],
                    "Parameters": [{"ParameterKey": "DatabaseName", "ParameterValue": "vol_history_demo"}],
                    "Outputs": [{"OutputKey": k, "OutputValue": v} for k, v in self.outputs.items()]}]}
            if operation == "get-template":
                return {"TemplateBody": self.current}
            if operation in {"validate-template", "update-stack"}:
                return {}
        if service == "s3api":
            key = extra[extra.index("--key") + 1]
            if operation == "put-object":
                if (self.fail_stage == "archive" and key.endswith("commit.json")
                        or self.fail_stage == "projection" and key.startswith("real/analytics/")):
                    raise OSError("Test publication interrupted")
                if key in self.objects:
                    raise release.AwsCliError("PreconditionFailed")
                assert payload["IfNoneMatch"] == "*"
                self.objects[key] = Path(extra[extra.index("--body") + 1]).read_bytes()
                return {}
            if operation == "get-object":
                Path(extra[-1]).write_bytes(self.objects[key])
                return {}
        if service == "glue":
            if operation == "get-table":
                return {"Table": self.table}
            if operation == "get-partition":
                values = tuple(payload["PartitionValues"])
                if values not in self.partitions:
                    raise release.AwsCliError("EntityNotFoundException")
                return {"Partition": self.partitions[values]}
            if operation == "create-partition":
                if self.fail_stage == "partition":
                    raise OSError("Test publication interrupted")
                partition = payload["PartitionInput"]
                self.partitions[tuple(partition["Values"])] = deepcopy(partition)
                return {}
        if service == "athena":
            if operation == "get-work-group":
                return {"WorkGroup": {"Configuration": {"BytesScannedCutoffPerQuery": release.SCAN_LIMIT,
                    "EnforceWorkGroupConfiguration": True, "ResultConfiguration": {"OutputLocation": "s3://test-bucket/results/"}}}}
            if operation == "start-query-execution":
                assert "processing_version=" in payload["QueryString"] and "snapshot_id=" in payload["QueryString"]
                self.query_tokens.append(payload["ClientRequestToken"])
                return {"QueryExecutionId": "stable-query"}
            if operation == "get-query-execution":
                return {"QueryExecution": {"Status": {"State": "SUCCEEDED"}, "Statistics": {"DataScannedInBytes": 1510}}}
            if operation == "get-query-results":
                names = release.SELECT_COLUMNS
                data = [names] + [[str(row[n]).lower() if isinstance(row[n], bool) else str(row[n]) for n in names]
                                  for row in self.query_rows]
                return {"ResultSet": {"ResultSetMetadata": {"ColumnInfo": [{"Name": n} for n in names]},
                        "Rows": [{"Data": [{"VarCharValue": value} for value in row]} for row in data]}}
        raise AssertionError((service, operation))


def test_template_only_adds_real_table_and_three_scoped_retention_rules(prepared):
    _, _, root, plan = prepared
    baseline = json.loads((root / "baseline-stack.json").read_text())
    template = json.loads((root / "stack.json").read_text())
    assert set(template["Resources"]) - set(baseline["Resources"]) == {"RealCoverageTable"}
    for name, resource in baseline["Resources"].items():
        if name != "ObservationArchive":
            assert template["Resources"][name] == resource
    rules = template["Resources"]["ObservationArchive"]["Properties"]["LifecycleConfiguration"]["Rules"]
    assert rules[:3] == baseline["Resources"]["ObservationArchive"]["Properties"]["LifecycleConfiguration"]["Rules"]
    assert [r["Prefix"] for r in rules[3:]] == ["real/archive/", "real/analytics/", "real/releases/"]
    assert all(r["ExpirationInDays"] == 365 for r in rules[3:])
    assert release.read_plan(root) == plan


@pytest.mark.parametrize("stage", ["archive", "projection", "partition"])
def test_partial_publication_visible_and_retry_is_immutable_and_idempotent(prepared, stage):
    _, _, root, plan = prepared
    cli = FakeCli(root, plan)
    cli.fail_stage = stage
    with pytest.raises(OSError, match="interrupted"):
        release.publish(cli, plan, ACCOUNT)
    report = json.loads((root / "publication.json").read_text())
    assert report["status"] == "partial_failure" and report["observations"][0]["stage"] == stage
    assert not cli.partitions and not any(k.startswith("real/releases/") for k in cli.objects)
    if stage == "archive":
        assert not any(k.endswith("commit.json") for k in cli.objects)
    saved = deepcopy(cli.objects)
    cli.fail_stage = None
    published = release.publish(cli, plan, ACCOUNT)
    assert published["status"] == "published" and len(cli.partitions) == 2
    assert all(cli.objects[k] == v for k, v in saved.items())
    before = deepcopy(cli.objects)
    assert release.publish(cli, plan, ACCOUNT) == published
    assert cli.objects == before


def test_round_trip_replay_query_reconciles_and_retry_keeps_same_query_token(prepared):
    _, _, root, plan = prepared
    cli = FakeCli(root, plan)
    release.publish(cli, plan, ACCOUNT)
    result = release.verify(cli, plan, ACCOUNT)
    assert result["verified"] and all(r["replay"]["matched"] for r in result["archives"])
    assert result["athena"]["scanned_bytes"] == 1510
    assert result["athena"]["estimated_billed_bytes"] == 10_000_000
    assert result["athena"]["rows"] == plan["local_duckdb_rows"]
    release.query_once(cli, plan, cli.outputs)
    assert cli.query_tokens[0] == cli.query_tokens[1]


def test_partition_conflict_refuses_overwrite_and_new_versions_use_new_paths(prepared):
    store, chosen, root, plan = prepared
    cli = FakeCli(root, plan)
    release.publish(cli, plan, ACCOUNT)
    key = next(iter(cli.partitions))
    cli.partitions[key]["StorageDescriptor"]["Location"] = "s3://wrong/"
    with pytest.raises(ObservationError, match="partition conflicts"):
        release.publish(cli, plan, ACCOUNT)
    from scripts.export_athena_projection import export_projection
    a = export_projection(store, *chosen[0], root / "another-version-test", versioned=True)
    assert f"version={chosen[0][1]}" in a["output"]
    changed = deepcopy(store.load(chosen[0][0])[0])
    assert release._projection_path(changed, "another-version") != plan["observations"][0]["projection_path"]


def test_wrong_account_template_drift_query_mismatch_and_unapproved_cloud_fail_closed(prepared, monkeypatch):
    _, _, root, plan = prepared
    cli = FakeCli(root, plan)
    with pytest.raises(ObservationError, match="account does not match"):
        release.check_account(cli, "999999999999")
    cli.current["Description"] = "drift"
    with pytest.raises(ObservationError, match="reviewed baseline"):
        release.update_stack(cli, plan, ACCOUNT)
    assert not any(call[1] == "update-stack" for call in cli.calls)
    cli.query_rows = []
    with pytest.raises(ObservationError, match="does not reconcile"):
        release.query_once(cli, plan, cli.outputs)
    monkeypatch.setattr(release.AwsCli, "call", lambda *a, **k: pytest.fail("unapproved AWS call"))
    assert release.main(["publish", "--output", str(root)]) == 1


def test_update_is_idempotent_and_uses_only_checked_baseline(prepared):
    _, _, root, plan = prepared
    cli = FakeCli(root, plan)
    assert not release.update_stack(cli, plan, ACCOUNT)["updated"]
    cli.current = cli.baseline
    assert release.update_stack(cli, plan, ACCOUNT)["status"] == "UPDATE_IN_PROGRESS"
    assert len([c for c in cli.calls if c[1] == "update-stack"]) == 1


def test_reprocessing_retains_older_versioned_projection(prepared, tmp_path, monkeypatch):
    store, chosen, _, _ = prepared
    from scripts.export_athena_projection import export_projection
    import src.history_store as storage
    snapshot_id, original_processing = chosen[0]
    first = export_projection(store, snapshot_id, original_processing, tmp_path, versioned=True)
    before = Path(first["output"]).read_bytes()
    identity = storage.processing_identity()
    monkeypatch.setattr(storage, "processing_identity", lambda: {**identity, "version": "explicit-reprocessing-test"})
    new = store.process(snapshot_id)
    second = export_projection(store, snapshot_id, new["processing_id"], tmp_path, versioned=True)
    assert first["output"] != second["output"] and Path(first["output"]).read_bytes() == before
    assert not new["reproduction"]


def test_verification_checks_remote_partition_and_partial_release_before_query(prepared):
    _, _, root, plan = prepared
    cli = FakeCli(root, plan)
    release.publish(cli, plan, ACCOUNT)
    next(iter(cli.partitions.values()))["StorageDescriptor"]["Location"] = "s3://wrong/"
    with pytest.raises(ObservationError, match="intended projection"):
        release.verify(cli, plan, ACCOUNT)
    assert not cli.query_tokens
    publication = json.loads((root / "publication.json").read_text())
    publication["observations"].pop()
    (root / "publication.json").write_text(json.dumps(publication))
    with pytest.raises(ObservationError, match="not been published"):
        release.verify(cli, plan, ACCOUNT)


def test_release_refuses_synthetic_exports_and_bundle(prepared, tmp_path):
    store = LocalObservationStore(tmp_path / "synthetic")
    seed_demo(store)
    chosen = [("synthetic-a", store.process("synthetic-a")["processing_id"])]
    with pytest.raises(ObservationError, match="collected inputs"):
        release.prepare(store, chosen, tmp_path / "release", Path(__file__).parents[1] / "infra/aws/history-demo.json")
    with pytest.raises(ObservationError, match="collected inputs"):
        build_bundle(store, chosen, tmp_path / "bundle")


def test_bundle_reader_serves_selected_lineage_on_read_only_filesystem_without_network(prepared, tmp_path, monkeypatch):
    store, chosen, _, _ = prepared
    output = tmp_path / "bundle"
    build_bundle(store, chosen, output)
    before = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    def prohibited(*args, **kwargs):
        raise AssertionError("Deployed history must neither write nor fetch")
    monkeypatch.setattr(Path, "mkdir", prohibited)
    monkeypatch.setattr(socket, "socket", prohibited)
    client = create_app(history_root=str(output)).test_client()
    assert client.get("/history").status_code == 200
    html = client.get("/history?left=test-a&right=test-b")
    assert b"supported comparison points" in html.data and b"PARTIAL" in html.data
    for filename in ("raw.parquet", "observation.parquet", "manifest.json", "cleaned.parquet", "selected.parquet",
                     "coverage.parquet", "diagnostics.json", "surface.json", "processing-manifest.json"):
        response = client.get(f"/history/test-a/files/{filename}?processing={chosen[0][1]}")
        assert response.status_code == 200 and response.headers["Cache-Control"] == "no-store"
    assert client.get("/history/test-a/files/coverage.parquet?processing=other").status_code == 404
    assert {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()} == before
    reader = ReadOnlySnapshotStore(output)
    with pytest.raises(ObservationError, match="read-only"):
        reader.process("test-a")
    # New app instance simulates a cold start/redeploy against the same shipped files.
    assert b"test-a" in create_app(history_root=str(output)).test_client().get("/history").data


def test_missing_or_corrupt_release_catalog_fails_closed(prepared, tmp_path, monkeypatch):
    store, chosen, _, _ = prepared
    output = tmp_path / "bundle"
    build_bundle(store, chosen, output)
    monkeypatch.setenv("VERCEL", "1")
    (output / "release.json").unlink()
    client = create_app(history_root=str(output)).test_client()
    assert b"Invalid release bundle" in client.get("/history").data
    assert client.get("/history/test-a/files/raw.parquet").status_code == 404
    output2 = tmp_path / "bundle2"
    build_bundle(store, chosen, output2)
    (next(output2.glob("processed/*/*/surface.json"))).write_text("corrupt")
    assert b"checksum mismatch" in create_app(history_root=str(output2)).test_client().get("/history").data
