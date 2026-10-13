"""Prepare locally, then explicitly deploy/verify/teardown an isolated AWS fixture demo.

Uses an existing AWS CLI login; never creates IAM users/keys or changes the plan.
`prepare` performs no AWS calls. Cloud commands require --execute and account ID.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.history_store import LocalObservationStore, ObservationError, digest, file_digest, local_path, processing_identity
from src.history_s3 import S3ObservationArchive, export_observation, verify_export
from scripts.export_athena_projection import export_projection

PROJECT_TAGS = {"Project": "vol-surface-explorer", "Purpose": "synthetic-history-verification"}
REGION = "us-east-2"
SCAN_LIMIT = 10_000_000
SELECT_COLUMNS = ["snapshot_id", "processing_id", "partial", "expected_expirations", "received_expirations", "raw_rows", "used_rows", "excluded_rows"]


def write_json(path, value):
    path = local_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def read_plan(root):
    root = local_path(root)
    plan = json.loads((root / "plan.json").read_text("utf-8"))
    if plan["region"] != REGION or plan["project_tags"] != PROJECT_TAGS:
        raise ObservationError("Unrecognized AWS demo plan/region.")
    if file_digest(root / "stack.json") != plan["template_sha256"]:
        raise ObservationError("Prepared CloudFormation template checksum mismatch.")
    for item in plan["observations"]:
        LocalObservationStore._id(item["snapshot_id"])
        expected_projection = f"analytics/date={item['date']}/ticker={item['ticker']}/{item['snapshot_id']}.parquet"
        if item["export_path"] != f"exports/{item['snapshot_id']}" or item["projection_path"] != expected_projection:
            raise ObservationError("Prepared artifact path does not match its fixture lineage.")
        for name in ("export_path", "projection_path"):
            if not (root / item[name]).resolve().is_relative_to(root):
                raise ObservationError("Prepared artifact escapes the local run directory.")
        bundle = verify_export(root / item["export_path"])
        if bundle["payload"]["source_kind"] != "synthetic_fixture" or bundle["payload"]["snapshot_id"] != item["snapshot_id"]:
            raise ObservationError("Prepared archive must retain its synthetic fixture identity.")
        if file_digest(root / item["projection_path"]) != item["projection_sha256"]:
            raise ObservationError("Prepared projection checksum mismatch.")
    return plan


def prepare(store_root, root, stack_name, snapshot_ids=("synthetic-a", "synthetic-b")):
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,47}", stack_name):
        raise ObservationError("Use a lowercase stack name of at most 48 letters, digits or hyphens.")
    root = local_path(root)
    if root.exists():
        raise ObservationError("Preparation destination already exists; choose a fresh local output directory.")
    if len(snapshot_ids) != 2 or len(set(snapshot_ids)) != 2:
        raise ObservationError("This bounded demo requires exactly two distinct synthetic observations.")
    store = LocalObservationStore(store_root)
    template_path = Path(__file__).resolve().parents[1] / "infra/aws/history-demo.json"
    template = json.loads(template_path.read_text("utf-8"))
    table_descriptor = template["Resources"]["CoverageTable"]["Properties"]["TableInput"]["StorageDescriptor"]
    root.mkdir(parents=True)
    observations = []
    for index, snapshot_id in enumerate(snapshot_ids, 1):
        observation = store.load(snapshot_id)[0]
        if observation["source_kind"] != "synthetic_fixture":
            raise ObservationError("This deployment verifier uploads synthetic fixtures only.")
        artifacts = [store.load_artifact(snapshot_id, path.name) for path in (store.root / "processed" / snapshot_id).glob("*") if path.is_dir()]
        artifacts = [item for item in artifacts if item["processor"] == processing_identity()]
        if len(artifacts) != 1:
            raise ObservationError("Each fixture requires one artifact matching the current processor; run process first.")
        artifact = artifacts[0]
        export_path = Path("exports") / snapshot_id
        export_observation(store, snapshot_id, root / export_path)
        projection = export_projection(store, snapshot_id, artifact["processing_id"], root / "analytics")
        projection_path = Path(projection["output"]).relative_to(root).as_posix()
        date, ticker = observation["valuation_time_utc"][:10], observation["ticker"]
        descriptor = deepcopy(table_descriptor)
        descriptor["Location"] = {"Fn::Sub": f"s3://${{ObservationArchive}}/analytics/date={date}/ticker={ticker}/"}
        template["Resources"][f"FixturePartition{index}"] = {
            "Type": "AWS::Glue::Partition", "DependsOn": "CoverageTable",
            "Properties": {"CatalogId": {"Ref": "AWS::AccountId"}, "DatabaseName": {"Ref": "DatabaseName"},
                           "TableName": "volatility_coverage", "PartitionInput": {"Values": [date, ticker], "StorageDescriptor": descriptor}}}
        outcomes = artifact["contract_outcomes"]
        observations.append({"snapshot_id": snapshot_id, "processing_id": artifact["processing_id"],
                             "date": date, "ticker": ticker, "partial": observation["partial"],
                             "expected_expirations": len(observation["expected_expirations"]),
                             "received_expirations": len(observation["received_expirations"]),
                             "raw_rows": outcomes["fetched"], "used_rows": outcomes["used"], "excluded_rows": outcomes["excluded"],
                             "export_path": export_path.as_posix(), "projection_path": projection_path,
                             "projection_sha256": projection["sha256"], "projection_bytes": projection["bytes"]})
    write_json(root / "stack.json", template)
    identities = ", ".join("'" + item["processing_id"] + "'" for item in observations)
    local_rows = store.query(f"SELECT {', '.join(SELECT_COLUMNS)} FROM coverage WHERE processing_id IN ({identities}) ORDER BY snapshot_id").to_dict("records")
    if local_rows != sorted([{name: item[name] for name in SELECT_COLUMNS} for item in observations], key=lambda row: row["snapshot_id"]):
        raise ObservationError("Local DuckDB rows do not match the verified artifact manifests.")
    stored_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    if stored_bytes > 1_000_000:
        raise ObservationError("Prepared fixture payload exceeds the 1 MB demo limit.")
    plan = {"schema_version": 1, "aws_executed": False, "region": REGION, "stack_name": stack_name,
            "database_name": "vol_" + stack_name.replace("-", "_"), "project_tags": PROJECT_TAGS,
            "template_sha256": file_digest(root / "stack.json"), "observations": observations, "local_duckdb_rows": local_rows,
            "payload_bytes": stored_bytes, "scan_cutoff_bytes_per_query": SCAN_LIMIT,
            "query_count_per_verification": 1, "cost_estimate": {"gross_demo_usd_less_than": .01,
            "assumptions": "Under 1 MB stored, two fixture archives, two tiny projections, under 40 PUT/under 100 GET+LIST, one on-demand Athena query. Credits/account billing checked separately; no charge guarantee."}}
    write_json(root / "plan.json", plan)
    return plan


class AwsCliError(ObservationError):
    def __init__(self, message):
        super().__init__(message)
        self.response = {"ResponseMetadata": {"HTTPStatusCode": 412 if "PreconditionFailed" in message else 0}}


class AwsCli:
    def __init__(self, root, profile=None, executable="aws"):
        self.root, self.profile, self.executable = local_path(root), profile, executable

    def call(self, service, operation, payload=None, extra=()):
        # No shell evaluation; credentials stay inside AWS CLI's existing provider.
        with tempfile.TemporaryDirectory(prefix=".aws-request-", dir=self.root) as request_dir:
            filename = Path(request_dir) / "input.json"
            filename.write_text(json.dumps(payload or {}), encoding="utf-8")
            command = [self.executable, service, operation, "--region", REGION, "--no-cli-pager", "--no-paginate", "--output", "json"]
            # The CLI's streaming get-object command does not accept --cli-input-json.
            if payload:
                command += ["--cli-input-json", "file://" + str(filename)]
            command += list(map(str, extra))
            if self.profile:
                command += ["--profile", self.profile]
            result = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False)
            if result.returncode:
                raise AwsCliError(f"AWS CLI {service} {operation}: {result.stderr.strip() or 'request failed'}")
            return json.loads(result.stdout) if result.stdout.strip() else {}


class CliS3:
    def __init__(self, cli):
        self.cli = cli

    def put_object(self, **kwargs):
        body = kwargs.pop("Body")
        bucket, key = kwargs.pop("Bucket"), kwargs.pop("Key")
        with tempfile.TemporaryDirectory(prefix=".s3-body-", dir=self.cli.root) as temporary:
            filename = Path(temporary) / "object.bin"
            filename.write_bytes(body)
            return self.cli.call("s3api", "put-object", kwargs, ("--bucket", bucket, "--key", key, "--body", filename))

    def get_object(self, **kwargs):
        bucket, key = kwargs.pop("Bucket"), kwargs.pop("Key")
        with tempfile.TemporaryDirectory(prefix=".s3-read-", dir=self.cli.root) as temporary:
            filename = Path(temporary) / "object.bin"
            metadata = self.cli.call("s3api", "get-object", kwargs, ("--bucket", bucket, "--key", key, filename))
            return {**metadata, "Body": BytesIO(filename.read_bytes())}


def check_account(cli, expected_account):
    if not re.fullmatch(r"\d{12}", expected_account or ""):
        raise ObservationError("Cloud actions require the explicitly verified 12-digit account ID.")
    identity = cli.call("sts", "get-caller-identity")
    if identity["Account"] != expected_account:
        raise ObservationError("AWS login account does not match --expected-account.")
    return identity["Account"]


def stack_state(cli, plan, account):
    stack = cli.call("cloudformation", "describe-stacks", {"StackName": plan["stack_name"]})["Stacks"][0]
    arn = stack["StackId"].split(":")
    tags = {item["Key"]: item["Value"] for item in stack.get("Tags", [])}
    if arn[3] != REGION or arn[4] != account or any(tags.get(k) != v for k, v in PROJECT_TAGS.items()):
        raise ObservationError("Stack region/account/project ownership mismatch.")
    return stack, {item["OutputKey"]: item["OutputValue"] for item in stack.get("Outputs", [])}


def deploy(cli, plan, account):
    try:
        cli.call("cloudformation", "describe-stacks", {"StackName": plan["stack_name"]})
    except AwsCliError as exc:
        if "does not exist" not in str(exc):
            raise
    else:
        raise ObservationError("Stack already exists; this demo deployer never updates an existing stack.")
    template = (cli.root / "stack.json").read_text("utf-8")
    cli.call("cloudformation", "validate-template", {"TemplateBody": template})
    created = cli.call("cloudformation", "create-stack", {"StackName": plan["stack_name"], "TemplateBody": template,
                       "Parameters": [{"ParameterKey": "DatabaseName", "ParameterValue": plan["database_name"]}],
                       "Tags": [{"Key": k, "Value": v} for k, v in PROJECT_TAGS.items()]})
    write_json(cli.root / "deployment.json", {"aws_executed": True, "account": account, "region": REGION,
                                              "stack_id": created["StackId"], "status": "CREATE_IN_PROGRESS"})
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        stack, outputs = stack_state(cli, plan, account)
        if stack["StackStatus"] == "CREATE_COMPLETE":
            report = {"aws_executed": True, "account": account, "region": REGION, "stack_id": stack["StackId"],
                      "status": stack["StackStatus"], "outputs": outputs}
            write_json(cli.root / "deployment.json", report)
            return report
        if stack["StackStatus"] != "CREATE_IN_PROGRESS":
            raise ObservationError(f"Stack creation did not complete: {stack['StackStatus']}; inspect events and run scoped teardown.")
        time.sleep(3)
    raise ObservationError("CloudFormation create timed out; inspect deployment.json and stack events before retrying.")


def bounded_query(cli, outputs, plan):
    dates = sorted({item["date"] for item in plan["observations"]})
    tickers = sorted({item["ticker"] for item in plan["observations"]})
    def literals(values):
        return ", ".join("'" + value.replace("'", "''") + "'" for value in values)
    query = (f"SELECT {', '.join(SELECT_COLUMNS)} FROM {outputs['TableName']} "
             f"WHERE observation_date IN ({literals(dates)}) AND ticker IN ({literals(tickers)}) ORDER BY snapshot_id")
    started = cli.call("athena", "start-query-execution", {"QueryString": query,
                       "QueryExecutionContext": {"Database": outputs["DatabaseName"], "Catalog": "AwsDataCatalog"},
                       "WorkGroup": outputs["WorkGroup"], "ClientRequestToken": digest({"plan": plan, "time": time.time_ns()})})
    query_id = started["QueryExecutionId"]
    write_json(cli.root / "query-pending.json", {"query_id": query_id, "query": query})
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        execution = cli.call("athena", "get-query-execution", {"QueryExecutionId": query_id})["QueryExecution"]
        state = execution["Status"]["State"]
        if state == "SUCCEEDED":
            response = cli.call("athena", "get-query-results", {"QueryExecutionId": query_id, "MaxResults": 10})
            names = [item["Name"] for item in response["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]]
            result_rows = response["ResultSet"]["Rows"]
            data_rows = result_rows[1:] if result_rows and [item.get("VarCharValue") for item in result_rows[0]["Data"]] == names else result_rows
            rows = [dict(zip(names, [item.get("VarCharValue") for item in row["Data"]])) for row in data_rows]
            if response.get("NextToken"):
                raise ObservationError("Athena returned more rows than the bounded two-observation demo.")
            for row in rows:
                row["partial"] = row["partial"] == "true"
                for name in ("expected_expirations", "received_expirations", "raw_rows", "used_rows", "excluded_rows"):
                    row[name] = int(row[name])
            expected = plan["local_duckdb_rows"]
            if sorted(rows, key=lambda r: r["snapshot_id"]) != sorted(expected, key=lambda r: r["snapshot_id"]):
                raise ObservationError("Athena coverage/partial flags do not reconcile with the verified local fixtures.")
            return {"query_id": query_id, "query": query, "rows": rows,
                    "statistics": execution.get("Statistics", {}), "engine": execution.get("EngineVersion", {}),
                    "result_configuration": execution.get("ResultConfiguration", {}), "reconciles": True}
        if state in {"FAILED", "CANCELLED"}:
            raise ObservationError(f"Athena {state}: {execution['Status'].get('StateChangeReason', '')}; query {query_id}")
        time.sleep(2)
    cli.call("athena", "stop-query-execution", {"QueryExecutionId": query_id})
    raise ObservationError(f"Athena timeout; cancellation requested for {query_id}.")


def verify(cli, plan, account):
    stack, outputs = stack_state(cli, plan, account)
    if stack["StackStatus"] != "CREATE_COMPLETE":
        raise ObservationError("Verify requires a completed dedicated demo stack.")
    config = cli.call("athena", "get-work-group", {"WorkGroup": outputs["WorkGroup"]})["WorkGroup"]["Configuration"]
    expected_result = f"s3://{outputs['ArchiveBucket']}/results/"
    if (config.get("BytesScannedCutoffPerQuery") != SCAN_LIMIT or not config.get("EnforceWorkGroupConfiguration")
            or config["ResultConfiguration"]["OutputLocation"] != expected_result):
        raise ObservationError("Athena scan cutoff or result enforcement differs from the bounded plan.")
    s3 = CliS3(cli)
    archive = S3ObservationArchive(s3, outputs["ArchiveBucket"], "archive")
    archives, projections = [], []
    for item in plan["observations"]:
        upload = archive.upload(cli.root / item["export_path"])
        destination = cli.root / "downloads" / item["snapshot_id"]
        if destination.exists():
            bundle = verify_export(destination)
            if bundle["bundle_sha256"] != upload["bundle_sha256"]:
                raise ObservationError("Existing downloaded export has a different checksum.")
            destination = cli.root / "downloads" / f"{item['snapshot_id']}-{time.time_ns()}"
        downloaded = archive.download(upload["commit_key"], destination)
        replay = LocalObservationStore(destination).replay(item["snapshot_id"], item["processing_id"])
        archives.append({**upload, **downloaded, "replay": replay, "checksum_matched": True})
        path = cli.root / item["projection_path"]
        archive._put_once(item["projection_path"], path.read_bytes())
        returned = s3.get_object(Bucket=outputs["ArchiveBucket"], Key=item["projection_path"])["Body"].read()
        if returned != path.read_bytes():
            raise ObservationError("Uploaded analytics projection checksum mismatch.")
        projections.append({"key": item["projection_path"], "bytes": len(returned), "sha256": item["projection_sha256"]})
    query = bounded_query(cli, outputs, plan)
    scanned = query["statistics"].get("DataScannedInBytes", 0)
    report = {"aws_executed": True, "account": account, "region": REGION, "stack_id": stack["StackId"],
              "archives": archives, "projections": projections, "athena": query,
              "estimated_athena_gross_usd": max(10_000_000, scanned) * 5 / 1_000_000_000_000,
              "billing_status": "Not read by this verifier; inspect account credits/billing separately. Estimate is not an observed bill.",
              "synthetic_only": True, "verified": True}
    write_json(cli.root / "verification.json", report)
    return report


def teardown(cli, plan, account, confirm_stack):
    if confirm_stack != plan["stack_name"]:
        raise ObservationError("Teardown requires --confirm-stack matching the prepared stack exactly.")
    stack, outputs = stack_state(cli, plan, account)
    resources = cli.call("cloudformation", "describe-stack-resources", {"StackName": stack["StackId"]})["StackResources"]
    physical = {item["LogicalResourceId"]: item.get("PhysicalResourceId") for item in resources if item.get("ResourceStatus") != "DELETE_COMPLETE"}
    bucket = outputs.get("ArchiveBucket") or physical.get("ObservationArchive")
    workgroup = outputs.get("WorkGroup") or physical.get("HistoryWorkGroup")
    if workgroup:
        cli.call("athena", "update-work-group", {"WorkGroup": workgroup, "State": "DISABLED"})
        executions = cli.call("athena", "list-query-executions", {"WorkGroup": workgroup, "MaxResults": 50})
        if executions.get("NextToken"):
            raise ObservationError("More than 50 demo query executions; inspect the workgroup before scoped cleanup.")
        for query_id in executions["QueryExecutionIds"]:
            status = cli.call("athena", "get-query-execution", {"QueryExecutionId": query_id})["QueryExecution"]["Status"]["State"]
            if status in {"QUEUED", "RUNNING"}:
                cli.call("athena", "stop-query-execution", {"QueryExecutionId": query_id})
                raise ObservationError("Active query cancellation requested; retry teardown after it stops writing results.")
    deleted = []
    if bucket:
        versioning = cli.call("s3api", "get-bucket-versioning", {"Bucket": bucket})
        if versioning.get("Status"):
            raise ObservationError("Demo bucket versioning changed; inspect retained versions before deleting.")
        tags = cli.call("s3api", "get-bucket-tagging", {"Bucket": bucket})["TagSet"]
        if any({t["Key"]: t["Value"] for t in tags}.get(k) != v for k, v in PROJECT_TAGS.items()):
            raise ObservationError("S3 project ownership mismatch; refusing deletion.")
        listed = cli.call("s3api", "list-objects-v2", {"Bucket": bucket, "MaxKeys": 1000})
        if listed.get("IsTruncated"):
            raise ObservationError("More than 1000 demo objects; inspect bucket before deleting.")
        keys = [item["Key"] for item in listed.get("Contents", [])]
        if any(not key.startswith(("archive/", "analytics/", "results/")) for key in keys):
            raise ObservationError("Unexpected non-demo object prefix; refusing deletion.")
        if keys:
            result = cli.call("s3api", "delete-objects", {"Bucket": bucket, "Delete": {"Objects": [{"Key": key} for key in keys], "Quiet": True}})
            if result.get("Errors"):
                raise ObservationError(f"S3 cleanup errors: {result['Errors']}")
            deleted = keys
        if cli.call("s3api", "list-objects-v2", {"Bucket": bucket, "MaxKeys": 1}).get("KeyCount", 0):
            raise ObservationError("S3 bucket not empty after cleanup; stack retained for inspection.")
    cli.call("cloudformation", "delete-stack", {"StackName": stack["StackId"]})
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        try:
            status = cli.call("cloudformation", "describe-stacks", {"StackName": stack["StackId"]})["Stacks"][0]["StackStatus"]
        except AwsCliError as exc:
            if "does not exist" not in str(exc):
                raise
            break
        if status == "DELETE_FAILED":
            raise ObservationError("Stack deletion failed; inspect events. The retained bucket has not been deleted.")
        if status == "DELETE_COMPLETE":
            break  # Deleted-stack records remain addressable by ARN for a while.
        time.sleep(3)
    else:
        raise ObservationError("Stack deletion timed out; inspect AWS before retrying.")
    if bucket:
        cli.call("s3api", "delete-bucket", {"Bucket": bucket})
        try:
            cli.call("s3api", "head-bucket", {"Bucket": bucket})
        except AwsCliError as exc:
            if "404" not in str(exc) and "Not Found" not in str(exc) and "NoSuchBucket" not in str(exc):
                raise
        else:
            raise ObservationError("Retained S3 bucket still exists after delete.")
    report = {"aws_executed": True, "account": account, "region": REGION, "stack_id": stack["StackId"],
              "stack_absent": True, "retained_bucket_deleted": bool(bucket), "bucket": bucket,
              "deleted_objects": len(deleted), "local_exports_retained": True, "cleanup_complete": True}
    write_json(cli.root / "cleanup.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "deploy", "verify", "teardown"))
    parser.add_argument("--output", required=True, help="Local prepared run/evidence directory")
    parser.add_argument("--store", default="data/history")
    parser.add_argument("--stack-name", default="vol-history-demo")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-account")
    parser.add_argument("--profile")
    parser.add_argument("--aws-cli", default="aws")
    parser.add_argument("--confirm-stack")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.store, args.output, args.stack_name)
        else:
            if not args.execute:
                raise ObservationError("Cloud commands require --execute; prepare performs no cloud calls.")
            plan = read_plan(args.output)
            cli = AwsCli(args.output, args.profile, args.aws_cli)
            account = check_account(cli, args.expected_account)
            if args.command == "deploy":
                result = deploy(cli, plan, account)
            elif args.command == "verify":
                result = verify(cli, plan, account)
            else:
                result = teardown(cli, plan, account, args.confirm_stack)
        print(json.dumps(result, indent=2))
        return 0
    except (ObservationError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
