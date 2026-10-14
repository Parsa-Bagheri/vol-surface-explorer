"""Prepare, publish and verify a bounded real-data release on the existing stack.

Preparation is offline. Every cloud command requires --execute and the checked
account. This operator CLI is never imported by deployed web requests.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.aws_history_demo import (AwsCli, AwsCliError, CliS3, PROJECT_TAGS, REGION, SCAN_LIMIT,
                                     SELECT_COLUMNS, check_account, stack_state, write_json)
from scripts.build_history_bundle import selections
from scripts.export_athena_projection import export_projection
from src.history_s3 import S3ObservationArchive, export_observation, verify_export
from src.history_store import LocalObservationStore, ObservationError, canonical, digest, file_digest, local_path

TABLE = "volatility_real_coverage"
ARCHIVE_PREFIX = "real/archive"
ANALYTICS_PREFIX = "real/analytics"
RETENTION_DAYS = 365
MAX_BYTES = 20_000_000


def release_template(baseline):
    template = deepcopy(baseline)
    resources = template["Resources"]
    if "RealCoverageTable" in resources:
        raise ObservationError("Baseline already contains a real table; use its original prepared baseline.")
    table = deepcopy(resources["CoverageTable"])
    table_input = table["Properties"]["TableInput"]
    table_input["Name"] = TABLE
    table_input["Description"] = "Collected observations; explicit immutable processing-version partitions."
    table_input["PartitionKeys"].append({"Name": "processing_version", "Type": "string"})
    table_input["StorageDescriptor"]["Location"] = {"Fn::Sub": "s3://${ObservationArchive}/real/analytics/"}
    resources["RealCoverageTable"] = table
    rules = resources["ObservationArchive"]["Properties"]["LifecycleConfiguration"]["Rules"]
    rules.extend([{"Id": "RealArchiveRetention", "Status": "Enabled", "Prefix": ARCHIVE_PREFIX + "/",
                   "ExpirationInDays": RETENTION_DAYS, "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1}},
                  {"Id": "RealProjectionRetention", "Status": "Enabled", "Prefix": ANALYTICS_PREFIX + "/",
                   "ExpirationInDays": RETENTION_DAYS},
                  {"Id": "RealReleaseRetention", "Status": "Enabled", "Prefix": "real/releases/",
                   "ExpirationInDays": RETENTION_DAYS}])
    template["Outputs"]["RealTableName"] = {"Value": TABLE}
    return template


def _projection_path(observation, processing_id):
    return (f"{ANALYTICS_PREFIX}/date={observation['valuation_time_utc'][:10]}/ticker={observation['ticker']}"
            f"/version={processing_id}/{observation['snapshot_id']}.parquet")


def prepare(store, chosen, root, baseline_template, stack_name="vol-history-demo"):
    root = local_path(root)
    if root.exists():
        raise ObservationError("Preparation destination already exists; choose a fresh release directory.")
    if stack_name != "vol-history-demo":
        raise ObservationError("This release targets only the existing vol-history-demo stack.")
    if not 1 <= len(chosen) <= 10 or len({s for s, _ in chosen}) != len(chosen):
        raise ObservationError("Select 1–10 distinct real observations with explicit processing versions.")
    baseline = json.loads(Path(baseline_template).read_text("utf-8"))
    template = release_template(baseline)
    root.mkdir(parents=True)
    observations = []
    for snapshot_id, processing_id in chosen:
        observation = store.load(snapshot_id)[0]
        artifact = store.load_artifact(snapshot_id, processing_id)
        if observation["source_kind"] != "collected":
            raise ObservationError("Real publishing requires collected inputs; fixtures/imports are refused.")
        store.replay(snapshot_id, processing_id)
        exported = export_observation(store, snapshot_id, root / "exports" / snapshot_id)
        projection = export_projection(store, snapshot_id, processing_id, root / ANALYTICS_PREFIX, versioned=True)
        local_rows = store.query(f"SELECT {', '.join(SELECT_COLUMNS)} FROM coverage WHERE snapshot_id='{snapshot_id}' "
                                f"AND processing_id='{processing_id}'").to_dict("records")
        expected = {"snapshot_id": snapshot_id, "processing_id": processing_id, "partial": observation["partial"],
                    "expected_expirations": len(observation["expected_expirations"]),
                    "received_expirations": len(observation["received_expirations"]),
                    "raw_rows": artifact["contract_outcomes"]["fetched"],
                    "used_rows": artifact["contract_outcomes"]["used"],
                    "excluded_rows": artifact["contract_outcomes"]["excluded"]}
        if local_rows != [expected]:
            raise ObservationError("DuckDB rows do not match saved manifest counts.")
        observations.append({**expected, "date": observation["valuation_time_utc"][:10], "ticker": observation["ticker"],
                             "valuation_time_utc": observation["valuation_time_utc"],
                             "raw_sha256": observation["files"]["raw.parquet"]["sha256"],
                             "export_path": f"exports/{snapshot_id}", "bundle_sha256": exported["bundle_sha256"],
                             "projection_path": Path(projection["output"]).relative_to(root).as_posix(),
                             "projection_sha256": projection["sha256"], "projection_bytes": projection["bytes"]})
    write_json(root / "baseline-stack.json", baseline)
    write_json(root / "stack.json", template)
    payload_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    if payload_bytes > MAX_BYTES:
        raise ObservationError("Release exceeds the 20 MB payload limit.")
    plan = {"schema_version": 1, "region": REGION, "stack_name": stack_name, "project_tags": PROJECT_TAGS,
            "table_name": TABLE, "archive_prefix": ARCHIVE_PREFIX, "analytics_prefix": ANALYTICS_PREFIX,
            "retention_days": RETENTION_DAYS, "baseline_sha256": digest(baseline), "template_sha256": digest(template),
            "observations": observations, "payload_bytes": payload_bytes,
            "local_duckdb_rows": sorted([{name: item[name] for name in SELECT_COLUMNS} for item in observations],
                                      key=lambda r: r["snapshot_id"]),
            "scan_cutoff_bytes": SCAN_LIMIT, "query_count": 1}
    write_json(root / "plan.json", {"payload": plan, "plan_sha256": digest(plan)})
    read_plan(root)
    return plan


def read_plan(root):
    root = local_path(root)
    envelope = json.loads((root / "plan.json").read_text("utf-8"))
    plan = envelope["payload"]
    if (digest(plan) != envelope["plan_sha256"] or plan.get("schema_version") != 1
            or plan["region"] != REGION or plan["stack_name"] != "vol-history-demo"
            or plan["project_tags"] != PROJECT_TAGS or plan["table_name"] != TABLE
            or plan["archive_prefix"] != ARCHIVE_PREFIX or plan["analytics_prefix"] != ANALYTICS_PREFIX
            or plan["retention_days"] != RETENTION_DAYS or plan["payload_bytes"] > MAX_BYTES):
        raise ObservationError("Unrecognized or altered real-history release plan.")
    baseline = json.loads((root / "baseline-stack.json").read_text("utf-8"))
    template = json.loads((root / "stack.json").read_text("utf-8"))
    if (digest(baseline) != plan["baseline_sha256"] or digest(template) != plan["template_sha256"]
            or template != release_template(baseline)):
        raise ObservationError("Release template diff/checksum mismatch.")
    if (not 1 <= len(plan["observations"]) <= 10
            or len({i["snapshot_id"] for i in plan["observations"]}) != len(plan["observations"])
            or plan["scan_cutoff_bytes"] != SCAN_LIMIT or plan["query_count"] != 1):
        raise ObservationError("Unbounded release plan.")
    for item in plan["observations"]:
        snapshot_id, processing_id = item["snapshot_id"], item["processing_id"]
        LocalObservationStore._id(snapshot_id)
        LocalObservationStore._id(processing_id)
        if item["export_path"] != f"exports/{snapshot_id}":
            raise ObservationError("Release export identity mismatch.")
        bundle = verify_export(root / item["export_path"])
        store = LocalObservationStore(root / item["export_path"])
        observation = store.load(snapshot_id)[0]
        artifact = store.load_artifact(snapshot_id, processing_id)
        if (bundle["bundle_sha256"] != item["bundle_sha256"] or observation["source_kind"] != "collected"
                or item["date"] != observation["valuation_time_utc"][:10] or item["ticker"] != observation["ticker"]
                or item["valuation_time_utc"] != observation["valuation_time_utc"]
                or item["raw_sha256"] != observation["files"]["raw.parquet"]["sha256"]
                or item["projection_path"] != _projection_path(observation, processing_id)):
            raise ObservationError("Real observation/projection lineage mismatch.")
        path = root / item["projection_path"]
        if file_digest(path) != item["projection_sha256"] or path.stat().st_size != item["projection_bytes"]:
            raise ObservationError("Projection checksum mismatch.")
        import pandas as pd
        frame = pd.read_parquet(path)
        if len(frame) != 1:
            raise ObservationError("Projection must contain exactly one coverage row.")
        row = frame.iloc[0]
        expected = {name: item[name] for name in SELECT_COLUMNS}
        actual = {name: row[name] for name in SELECT_COLUMNS}
        if actual != expected or item["raw_rows"] != artifact["contract_outcomes"]["fetched"]:
            raise ObservationError("Projection counts or processing identity mismatch.")
    expected_rows = sorted([{name: item[name] for name in SELECT_COLUMNS} for item in plan["observations"]],
                           key=lambda r: r["snapshot_id"])
    if expected_rows != plan["local_duckdb_rows"]:
        raise ObservationError("Prepared expected query rows mismatch.")
    return plan


def _cloud_state(cli, plan, account, *, require_release=True):
    stack, outputs = stack_state(cli, plan, account)
    if stack["StackStatus"] not in {"CREATE_COMPLETE", "UPDATE_COMPLETE"}:
        raise ObservationError("The existing stack is not in a completed state.")
    if require_release and outputs.get("RealTableName") != TABLE:
        raise ObservationError("Real-data infrastructure update has not completed.")
    return stack, outputs


def update_stack(cli, plan, account):
    stack, _ = _cloud_state(cli, plan, account, require_release=False)
    current = cli.call("cloudformation", "get-template", {"StackName": stack["StackId"]})["TemplateBody"]
    if isinstance(current, str):
        current = json.loads(current)
    if digest(current) == plan["template_sha256"]:
        return {"updated": False, "status": stack["StackStatus"]}
    if digest(current) != plan["baseline_sha256"]:
        raise ObservationError("Current stack differs from the reviewed baseline; refusing update.")
    template = (cli.root / "stack.json").read_text("utf-8")
    cli.call("cloudformation", "validate-template", {"TemplateBody": template})
    cli.call("cloudformation", "update-stack", {"StackName": stack["StackId"], "TemplateBody": template,
             "Parameters": [{"ParameterKey": p["ParameterKey"], "UsePreviousValue": True} for p in stack.get("Parameters", [])],
             "ClientRequestToken": "vol-real-" + plan["template_sha256"][:24]})
    write_json(cli.root / "stack-update.json", {"status": "UPDATE_IN_PROGRESS", "stack_id": stack["StackId"]})
    # Do not block the operator while CloudFormation runs. Re-run after completion.
    return {"updated": True, "status": "UPDATE_IN_PROGRESS", "stack_id": stack["StackId"]}


def _partition(cli, outputs, item, descriptor):
    values = [item["date"], item["ticker"], item["processing_id"]]
    descriptor = deepcopy(descriptor)
    descriptor["Location"] = f"s3://{outputs['ArchiveBucket']}/" + item["projection_path"].rsplit("/", 1)[0] + "/"
    payload = {"CatalogId": outputs["Account"], "DatabaseName": outputs["DatabaseName"], "TableName": TABLE}
    try:
        current = cli.call("glue", "get-partition", {**payload, "PartitionValues": values})["Partition"]
    except AwsCliError as exc:
        if "EntityNotFoundException" not in str(exc):
            raise
        try:
            cli.call("glue", "create-partition", {**payload, "PartitionInput": {"Values": values, "StorageDescriptor": descriptor}})
        except AwsCliError as raced:
            if "AlreadyExistsException" not in str(raced):
                raise
        current = cli.call("glue", "get-partition", {**payload, "PartitionValues": values})["Partition"]
    actual = current["StorageDescriptor"]
    if (current["Values"] != values or any(actual.get(name) != descriptor.get(name)
            for name in ("Location", "Columns", "InputFormat", "OutputFormat", "SerdeInfo"))):
        raise ObservationError("Existing Glue partition conflicts with immutable projection lineage.")
    return {"values": values, "location": descriptor["Location"]}


def publish(cli, plan, account):
    stack, outputs = _cloud_state(cli, plan, account)
    outputs["Account"] = account
    table = cli.call("glue", "get-table", {"CatalogId": account, "DatabaseName": outputs["DatabaseName"], "Name": TABLE})["Table"]
    expected = json.loads((cli.root / "stack.json").read_text("utf-8"))["Resources"]["RealCoverageTable"]["Properties"]["TableInput"]
    if (table["PartitionKeys"] != expected["PartitionKeys"] or table["StorageDescriptor"]["Columns"] != expected["StorageDescriptor"]["Columns"]
            or table["StorageDescriptor"]["Location"] != f"s3://{outputs['ArchiveBucket']}/{ANALYTICS_PREFIX}/"):
        raise ObservationError("Real Glue table schema/location mismatch.")
    archive = S3ObservationArchive(CliS3(cli), outputs["ArchiveBucket"], ARCHIVE_PREFIX)
    report = {"plan_sha256": digest(plan), "stack_id": stack["StackId"], "status": "publishing", "observations": []}
    write_json(cli.root / "publication.json", report)
    try:
        for item in plan["observations"]:
            record = {"snapshot_id": item["snapshot_id"], "processing_id": item["processing_id"], "stage": "archive"}
            report["observations"].append(record)
            write_json(cli.root / "publication.json", report)
            record["archive"] = archive.upload(cli.root / item["export_path"])
            record["stage"] = "projection"
            write_json(cli.root / "publication.json", report)
            body = (cli.root / item["projection_path"]).read_bytes()
            archive._put_once(item["projection_path"], body)
            if archive.client.get_object(Bucket=outputs["ArchiveBucket"], Key=item["projection_path"])["Body"].read() != body:
                raise ObservationError("Projection read-back mismatch; partition was not registered.")
            record["stage"] = "partition"
            write_json(cli.root / "publication.json", report)
            record["partition"] = _partition(cli, outputs, item, table["StorageDescriptor"])
            record["stage"] = "published"
            write_json(cli.root / "publication.json", report)
        report["status"] = "published"
        report["release_commit_key"] = f"real/releases/{digest(plan)}.json"
        archive._put_once(report["release_commit_key"], canonical({"plan_sha256": digest(plan), "observations": report["observations"]}))
        write_json(cli.root / "publication.json", report)
        return report
    except Exception as exc:
        report["status"], report["error"] = "partial_failure", str(exc)
        write_json(cli.root / "publication.json", report)
        raise


def query_once(cli, plan, outputs):
    def literal(value):
        return "'" + value.replace("'", "''") + "'"
    predicates = [f"(observation_date={literal(i['date'])} AND ticker={literal(i['ticker'])} "
                  f"AND processing_version={literal(i['processing_id'])} AND snapshot_id={literal(i['snapshot_id'])})"
                  for i in plan["observations"]]
    query = f"SELECT {', '.join(SELECT_COLUMNS)} FROM {TABLE} WHERE " + " OR ".join(predicates) + " ORDER BY snapshot_id"
    query_id = cli.call("athena", "start-query-execution", {"QueryString": query,
        "QueryExecutionContext": {"Database": outputs["DatabaseName"], "Catalog": "AwsDataCatalog"},
        "WorkGroup": outputs["WorkGroup"], "ClientRequestToken": digest({"real_release": digest(plan), "query": query})})["QueryExecutionId"]
    write_json(cli.root / "query-pending.json", {"query_id": query_id, "query": query})
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        execution = cli.call("athena", "get-query-execution", {"QueryExecutionId": query_id})["QueryExecution"]
        state = execution["Status"]["State"]
        if state == "SUCCEEDED":
            response = cli.call("athena", "get-query-results", {"QueryExecutionId": query_id, "MaxResults": 11})
            names = [c["Name"] for c in response["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]]
            if names != SELECT_COLUMNS or response.get("NextToken"):
                raise ObservationError("Unexpected query columns or unbounded result.")
            records = response["ResultSet"]["Rows"]
            if records and [c.get("VarCharValue") for c in records[0]["Data"]] == names:
                records = records[1:]
            rows = [dict(zip(names, [c.get("VarCharValue") for c in r["Data"]])) for r in records]
            for row in rows:
                row["partial"] = row["partial"] == "true"
                for name in SELECT_COLUMNS[3:]:
                    row[name] = int(row[name])
            if rows != plan["local_duckdb_rows"]:
                raise ObservationError("Athena does not reconcile with local DuckDB.")
            scanned = execution.get("Statistics", {}).get("DataScannedInBytes", 0)
            billed = max(10_000_000, ((scanned + 999_999) // 1_000_000) * 1_000_000)
            return {"query_id": query_id, "query": query, "rows": rows, "reconciles": True,
                    "scanned_bytes": scanned, "estimated_billed_bytes": billed,
                    "estimated_gross_usd": billed * 5 / 1_000_000_000_000, "engine": execution.get("EngineVersion", {})}
        if state in {"FAILED", "CANCELLED"}:
            raise ObservationError(f"Athena {state}: {execution['Status'].get('StateChangeReason', '')}; query {query_id}")
        time.sleep(2)
    cli.call("athena", "stop-query-execution", {"QueryExecutionId": query_id})
    raise ObservationError("Athena timeout; cancellation requested. The stable request token prevents duplicate queries on retry.")


def verify(cli, plan, account):
    _, outputs = _cloud_state(cli, plan, account)
    publication = json.loads((cli.root / "publication.json").read_text("utf-8"))
    if (publication["status"] != "published" or publication["plan_sha256"] != digest(plan)
            or len(publication["observations"]) != len(plan["observations"])):
        raise ObservationError("The complete prepared release has not been published.")
    config = cli.call("athena", "get-work-group", {"WorkGroup": outputs["WorkGroup"]})["WorkGroup"]["Configuration"]
    if (config.get("BytesScannedCutoffPerQuery") != SCAN_LIMIT or not config.get("EnforceWorkGroupConfiguration")
            or config["ResultConfiguration"]["OutputLocation"] != f"s3://{outputs['ArchiveBucket']}/results/"):
        raise ObservationError("Athena workgroup differs from the bounded query plan.")
    archive = S3ObservationArchive(CliS3(cli), outputs["ArchiveBucket"], ARCHIVE_PREFIX)
    committed = json.loads(archive.client.get_object(Bucket=outputs["ArchiveBucket"], Key=publication["release_commit_key"])["Body"].read())
    if committed != {"plan_sha256": digest(plan), "observations": publication["observations"]}:
        raise ObservationError("Remote release commit mismatch.")
    report = {"real_data": True, "verified": False, "archives": []}
    for item, record in zip(plan["observations"], publication["observations"]):
        if record["snapshot_id"] != item["snapshot_id"] or record["archive"]["bundle_sha256"] != item["bundle_sha256"]:
            raise ObservationError("Published archive identity mismatch.")
        partition = cli.call("glue", "get-partition", {"CatalogId": account, "DatabaseName": outputs["DatabaseName"],
                             "TableName": TABLE, "PartitionValues": [item["date"], item["ticker"], item["processing_id"]]})["Partition"]
        expected_location = f"s3://{outputs['ArchiveBucket']}/" + item["projection_path"].rsplit("/", 1)[0] + "/"
        if (partition["Values"] != [item["date"], item["ticker"], item["processing_id"]]
                or partition["StorageDescriptor"]["Location"] != expected_location):
            raise ObservationError("Published Glue partition no longer points to the intended projection.")
        destination = cli.root / "downloads" / f"{item['snapshot_id']}-{time.time_ns()}"
        downloaded = archive.download(record["archive"]["commit_key"], destination)
        replay = LocalObservationStore(destination).replay(item["snapshot_id"], item["processing_id"])
        returned = archive.client.get_object(Bucket=outputs["ArchiveBucket"], Key=item["projection_path"])["Body"].read()
        if returned != (cli.root / item["projection_path"]).read_bytes():
            raise ObservationError("Remote projection checksum mismatch.")
        report["archives"].append({**downloaded, "snapshot_id": item["snapshot_id"], "checksum_matched": True, "replay": replay})
        write_json(cli.root / "verification.json", report)
    report["athena"] = query_once(cli, plan, outputs)
    report["verified"] = True
    write_json(cli.root / "verification.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "update-stack", "publish", "verify"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--store", default="data/history")
    parser.add_argument("--stack-template")
    parser.add_argument("--observations", nargs="+")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-account")
    parser.add_argument("--profile")
    parser.add_argument("--aws-cli", default="aws")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            if not args.stack_template or not args.observations:
                raise ObservationError("Prepare requires --stack-template and --observations SNAPSHOT:PROCESSING.")
            with redirect_stdout(sys.stderr):
                result = prepare(LocalObservationStore(args.store), selections(args.observations), args.output, args.stack_template)
        else:
            if not args.execute:
                raise ObservationError("Cloud commands require --execute after approval; prepare performs no cloud calls.")
            plan = read_plan(args.output)
            cli = AwsCli(args.output, args.profile, args.aws_cli)
            account = check_account(cli, args.expected_account)
            with redirect_stdout(sys.stderr):
                result = {"update-stack": update_stack, "publish": publish, "verify": verify}[args.command](cli, plan, account)
        print(json.dumps(result, indent=2))
        return 0
    except (ObservationError, OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
