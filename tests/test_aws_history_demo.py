from copy import deepcopy
import json
from pathlib import Path
import pytest

from src.history_store import LocalObservationStore, ObservationError
from src.history_demo import seed_demo
from scripts import aws_history_demo as demo

ACCOUNT = "123456789012"


def test_cli_streaming_download_uses_explicit_bucket_key_without_json_input(tmp_path, monkeypatch):
    from types import SimpleNamespace
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert command[command.index("--bucket") + 1] == "test-bucket"
        assert command[command.index("--key") + 1] == "archive/key with spaces"
        if command[2] == "get-object":
            assert "--cli-input-json" not in command
            Path(command[-1]).write_bytes(b"verified bytes")
        else:
            request = json.loads(Path(command[command.index("--cli-input-json") + 1][7:]).read_text())
            assert request["IfNoneMatch"] == "*"
            assert Path(command[command.index("--body") + 1]).read_bytes() == b"verified bytes"
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")
    monkeypatch.setattr(demo.subprocess, "run", run)
    client = demo.CliS3(demo.AwsCli(tmp_path))
    client.put_object(Bucket="test-bucket", Key="archive/key with spaces", Body=b"verified bytes", IfNoneMatch="*")
    assert client.get_object(Bucket="test-bucket", Key="archive/key with spaces")["Body"].read() == b"verified bytes"
    assert len(calls) == 2


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("aws-history")
    store = LocalObservationStore(root / "lake")
    seed_demo(store)
    plan = demo.prepare(store.root, root / "run", "vol-history-test")
    return root / "run", plan


def test_local_preparation_limits_resources_cost_and_uses_validated_duckdb_rows(prepared):
    root, plan = prepared
    assert not plan["aws_executed"] and plan["region"] == "us-east-2"
    assert plan["payload_bytes"] < 1_000_000
    template = json.loads((root / "stack.json").read_text("utf-8"))
    resources = template["Resources"]
    assert {item["Type"] for item in resources.values()} == {
        "AWS::S3::Bucket", "AWS::S3::BucketPolicy", "AWS::Athena::WorkGroup",
        "AWS::Glue::Database", "AWS::Glue::Table", "AWS::Glue::Partition"}
    config = resources["HistoryWorkGroup"]["Properties"]["WorkGroupConfiguration"]
    assert config["EnforceWorkGroupConfiguration"] and config["BytesScannedCutoffPerQuery"] == 10_000_000
    rules = resources["ObservationArchive"]["Properties"]["LifecycleConfiguration"]["Rules"]
    assert {rule["Prefix"]: rule["ExpirationInDays"] for rule in rules} == {"archive/": 30, "analytics/": 7, "results/": 1}
    assert len(plan["local_duckdb_rows"]) == 2 and all(row["raw_rows"] == row["used_rows"] + row["excluded_rows"] for row in plan["local_duckdb_rows"])
    assert demo.read_plan(root) == plan


def test_cloud_commands_require_execute_and_account_before_mutation(prepared, monkeypatch):
    root, _ = prepared
    def prohibited(*args, **kwargs):
        raise AssertionError("AWS CLI must not run without explicit execution/account")
    monkeypatch.setattr(demo.subprocess, "run", prohibited)
    assert demo.main(["deploy", "--output", str(root)]) == 1
    assert demo.main(["verify", "--output", str(root), "--execute"]) == 1


class FakeCli:
    def __init__(self, root, plan):
        self.root, self.plan = demo.local_path(root), plan
        self.calls, self.objects = [], {}
        self.stack_exists = True
        self.bucket_exists = True
        self.result_override = None
        self.config_override = None
        self.outputs = {"ArchiveBucket": "test-owned-bucket", "WorkGroup": "vol-history-test-queries",
                        "DatabaseName": plan["database_name"], "TableName": "volatility_coverage"}

    def call(self, service, operation, payload=None, extra=()):
        payload = payload or {}
        self.calls.append((service, operation, deepcopy(payload)))
        if service == "sts":
            return {"Account": ACCOUNT}
        if service == "cloudformation":
            if operation == "describe-stacks":
                if not self.stack_exists:
                    raise demo.AwsCliError("Stack does not exist")
                return {"Stacks": [{"StackId": f"arn:aws:cloudformation:us-east-2:{ACCOUNT}:stack/vol-history-test/id",
                         "StackStatus": "CREATE_COMPLETE", "Tags": [{"Key": k, "Value": v} for k, v in demo.PROJECT_TAGS.items()],
                         "Outputs": [{"OutputKey": k, "OutputValue": v} for k, v in self.outputs.items()]}]}
            if operation == "describe-stack-resources":
                return {"StackResources": []}
            if operation == "delete-stack":
                self.stack_exists = False
                return {}
        if service == "s3api":
            if operation == "put-object":
                key = extra[extra.index("--key") + 1]
                if key in self.objects:
                    raise demo.AwsCliError("PreconditionFailed")
                assert payload["IfNoneMatch"] == "*"
                self.objects[key] = Path(extra[extra.index("--body") + 1]).read_bytes()
                return {}
            if operation == "get-object":
                Path(extra[-1]).write_bytes(self.objects[extra[extra.index("--key") + 1]])
                return {}
            if operation == "get-bucket-versioning":
                return {}
            if operation == "get-bucket-tagging":
                return {"TagSet": [{"Key": k, "Value": v} for k, v in demo.PROJECT_TAGS.items()]}
            if operation == "list-objects-v2":
                return {"KeyCount": len(self.objects), "Contents": [{"Key": key} for key in self.objects]}
            if operation == "delete-objects":
                for item in payload["Delete"]["Objects"]:
                    del self.objects[item["Key"]]
                return {}
            if operation == "delete-bucket":
                self.bucket_exists = False
                return {}
            if operation == "head-bucket":
                if not self.bucket_exists:
                    raise demo.AwsCliError("404 Not Found")
                return {}
        if service == "athena":
            if operation == "get-work-group":
                config = {"BytesScannedCutoffPerQuery": 10_000_000, "EnforceWorkGroupConfiguration": True,
                          "ResultConfiguration": {"OutputLocation": "s3://test-owned-bucket/results/"}}
                return {"WorkGroup": {"Configuration": self.config_override or config}}
            if operation == "start-query-execution":
                assert "WHERE observation_date IN" in payload["QueryString"]
                return {"QueryExecutionId": "query-test"}
            if operation == "get-query-execution":
                return {"QueryExecution": {"Status": {"State": "SUCCEEDED"}, "Statistics": {"DataScannedInBytes": 1000},
                                           "EngineVersion": {"EffectiveEngineVersion": "test"}}}
            if operation == "get-query-results":
                names = demo.SELECT_COLUMNS
                records = self.result_override or self.plan["local_duckdb_rows"]
                data = [[name for name in names]] + [[str(record[name]).lower() if isinstance(record[name], bool) else str(record[name]) for name in names] for record in records]
                return {"ResultSet": {"ResultSetMetadata": {"ColumnInfo": [{"Name": name} for name in names]},
                                      "Rows": [{"Data": [{"VarCharValue": v} for v in row]} for row in data]}}
            if operation == "update-work-group":
                return {}
            if operation == "stop-query-execution":
                return {}
            if operation == "list-query-executions":
                return {"QueryExecutionIds": ["query-test"]}
        raise AssertionError((service, operation, payload))


def test_archive_read_replay_and_athena_reconciliation_then_scoped_cleanup(prepared):
    root, plan = prepared
    cli = FakeCli(root, plan)
    assert demo.check_account(cli, ACCOUNT) == ACCOUNT
    verified = demo.verify(cli, plan, ACCOUNT)
    assert verified["verified"] and verified["synthetic_only"]
    assert all(archive["checksum_matched"] and archive["replay"]["matched"] for archive in verified["archives"])
    assert verified["athena"]["reconciles"] and len(verified["athena"]["rows"]) == 2
    assert verified["estimated_athena_gross_usd"] == .00005
    assert len([call for call in cli.calls if call[:2] == ("athena", "start-query-execution")]) == 1
    cleaned = demo.teardown(cli, plan, ACCOUNT, plan["stack_name"])
    assert cleaned["cleanup_complete"] and not cli.objects and not cli.stack_exists and not cli.bucket_exists
    assert (root / "downloads" / "synthetic-a" / "bundle.json").is_file()


def test_account_or_workgroup_limit_mismatch_blocks_mutation(prepared):
    root, plan = prepared
    cli = FakeCli(root, plan)
    with pytest.raises(ObservationError, match="account does not match"):
        demo.check_account(cli, "999999999999")
    assert cli.calls == [("sts", "get-caller-identity", {})]
    cli = FakeCli(root, plan)
    cli.config_override = {"BytesScannedCutoffPerQuery": 100_000_000, "EnforceWorkGroupConfiguration": False,
                           "ResultConfiguration": {"OutputLocation": "s3://test-owned-bucket/results/"}}
    with pytest.raises(ObservationError, match="scan cutoff"):
        demo.verify(cli, plan, ACCOUNT)
    assert not any(call[:2] == ("s3api", "put-object") for call in cli.calls)


def test_athena_mismatched_results_fail_verification(prepared):
    root, plan = prepared
    cli = FakeCli(root, plan)
    cli.result_override = deepcopy(plan["local_duckdb_rows"])
    cli.result_override[0]["raw_rows"] = 999
    with pytest.raises(ObservationError, match="do not reconcile"):
        demo.bounded_query(cli, cli.outputs, plan)


def test_teardown_refuses_unrelated_objects_or_wrong_stack_confirmation(prepared):
    root, plan = prepared
    cli = FakeCli(root, plan)
    with pytest.raises(ObservationError, match="confirm-stack"):
        demo.teardown(cli, plan, ACCOUNT, "different-stack")
    assert not cli.calls
    cli.objects["unrelated/file"] = b"preserve"
    with pytest.raises(ObservationError, match="non-demo object prefix"):
        demo.teardown(cli, plan, ACCOUNT, plan["stack_name"])
    assert cli.objects["unrelated/file"] == b"preserve"
    assert not any(call[:2] == ("s3api", "delete-objects") for call in cli.calls)


def test_deploy_never_updates_existing_stack(prepared):
    root, plan = prepared
    cli = FakeCli(root, plan)
    with pytest.raises(ObservationError, match="never updates"):
        demo.deploy(cli, plan, ACCOUNT)
    assert not any(call[1] in {"create-stack", "update-stack"} for call in cli.calls)


def test_cli_request_is_structured_and_pagination_is_bounded(prepared, monkeypatch):
    from types import SimpleNamespace
    root, _ = prepared
    captured = {}
    def run(command, **kwargs):
        captured["command"] = command
        filename = command[command.index("--cli-input-json") + 1].removeprefix("file://")
        captured["payload"] = json.loads(Path(filename).read_text("utf-8"))
        return SimpleNamespace(returncode=0, stdout='{"Account":"123456789012"}', stderr="")
    monkeypatch.setattr(demo.subprocess, "run", run)
    result = demo.AwsCli(root, profile="project-upgrades").call("cloudformation", "describe-stacks", {"StackName": "bounded-demo"})
    assert result["Account"] == ACCOUNT and captured["payload"] == {"StackName": "bounded-demo"}
    assert "--no-paginate" in captured["command"]
    assert captured["command"][-2:] == ["--profile", "project-upgrades"]


def test_query_timeout_requests_cancellation(prepared, monkeypatch):
    root, plan = prepared
    cli = FakeCli(root, plan)
    clock = iter([0, 121])
    monkeypatch.setattr(demo.time, "monotonic", lambda: next(clock))
    with pytest.raises(ObservationError, match="cancellation requested"):
        demo.bounded_query(cli, cli.outputs, plan)
    assert any(call[:2] == ("athena", "stop-query-execution") for call in cli.calls)
