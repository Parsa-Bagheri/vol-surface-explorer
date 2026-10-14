# Isolated S3 and Athena verification

This runbook uses the existing authorized AWS CLI login in **us-east-2**. It neither creates credentials nor changes the account plan. Preparation is local; only explicit `deploy`, `verify`, and `teardown` commands call AWS. The data are the same two labeled synthetic observations used by the local historical demo.

## Resources and expected usage

The generated CloudFormation stack creates one private encrypted S3 bucket with archive, analytics and result prefixes; a bucket policy requiring TLS; one on-demand SQL Athena workgroup; one Glue database/table; and two explicit date/ticker partitions. It includes no crawler, ETL job, Spark session, capacity reservation or always-on compute. The bucket is retained on stack deletion so cleanup must explicitly empty/delete it after preserving local evidence.

The workgroup enforces a **10,000,000-byte per-query scan cutoff**, AWS's [minimum accepted setting](https://docs.aws.amazon.com/athena/latest/APIReference/API_WorkGroupConfiguration.html), and its encrypted result location/expected account owner. Lifecycle rules expire archive objects after 30 days, analytics projections after seven days and results after one day. Lifecycle expiry is fallback retention; successful teardown removes these objects immediately.

The prepared payload is approximately **397 KB**, including two verified archives, two 9,082-byte Parquet coverage projections and the template. One verification makes roughly 22 uploads, reads both archives and projections back, replays both downloaded observations, and submits **one** partition-filtered Athena SELECT. It compares the actual Athena rows/partial flags to recorded local DuckDB rows: A reconciles 88 = 47 + 41 with 4/4 expirations; B reconciles 66 = 32 + 34 with 3/4 expirations.

At the published [$5/TB Athena rate](https://aws.amazon.com/athena/pricing/) and [10 MB minimum billing](https://aws.amazon.com/blogs/machine-learning/part-2-amazon-bedrock-cost-attribution-with-amazon-athena-and-cudos/), the tiny query is approximately **$0.00005 gross**. Assuming less than 1 MB stored, fewer than 40 PUT requests, fewer than 100 GET/LIST requests, and one query, the expected demonstration cost is **under $0.01 before credits**. At S3 Standard's indicative $0.023/GB-month, $0.005/1,000 writes/LIST and $0.0004/1,000 reads, these requests/storage contribute substantially less than one cent; region-specific rates are on [S3 pricing](https://aws.amazon.com/s3/pricing/). The handful of metadata objects fits [Glue Data Catalog's published first-million-object allowance](https://aws.amazon.com/glue/pricing/), assuming that account allowance is available. These are workload estimates, not an observed bill or an account spending cap. Verify current Free-plan/credit status separately; do not activate a paid plan to make this demo work. Local DuckDB remains the free alternative.

## Prepare and review locally

Run from the repository with history dependencies installed and a current AWS CLI available for later execution. No additional boto3 dependency is needed: the transport adapter invokes the existing CLI without shell evaluation, and credentials remain inside its provider.

```powershell
.\.venv\Scripts\python.exe scripts/aws_history_demo.py prepare --store demo-output/final-evidence-lake --output demo-output/aws-history-run --stack-name vol-history-demo
```

For a fresh checkout, first run `history.py --store demo-output/final-evidence-lake demo`. Preparation refuses real collected data, checks immutable source manifests, exports both fixtures, writes microsecond-timestamp Parquet projections and adds exact partitions to the template. It refuses an existing preparation directory. Review `plan.json`, `stack.json`, the two archives and projections. The plan records expected DuckDB rows, file checksums and the payload/scan bounds; preparation performs **zero AWS calls**.

The AWS caller needs the CloudFormation, S3, Athena and Glue permissions required by this dedicated stack and verification. No IAM resources or credential setup are included. A deployment permission failure should be inspected and scoped; it is not a reason to alter the account plan.

## Explicit deployment and verification

Replace `VERIFIED_ACCOUNT_ID` with the account independently checked through your CLI login. Keep account IDs, caller identities and detailed run JSON in private local evidence rather than public repository documentation. Use the existing profile; the examples use `project-upgrades`. `--aws-cli` accepts the installed executable path when it is absent from PATH.

```powershell
.\.venv\Scripts\python.exe scripts/aws_history_demo.py deploy --output demo-output/aws-history-run --execute --expected-account VERIFIED_ACCOUNT_ID --profile project-upgrades --aws-cli 'C:/Program Files/Amazon/AWSCLIV2/aws.exe'
.\.venv\Scripts\python.exe scripts/aws_history_demo.py verify --output demo-output/aws-history-run --execute --expected-account VERIFIED_ACCOUNT_ID --profile project-upgrades --aws-cli 'C:/Program Files/Amazon/AWSCLIV2/aws.exe'
```

Deployment checks the caller account, refuses an existing stack, validates the template, creates the dedicated stack and waits at most ten minutes. It never updates an existing stack. A failed stack remains inspectable; preserve events and run scoped cleanup if needed. Verification checks stack region/account/project tags and the enforced workgroup limit, then uses the existing immutable archive protocol. Content-addressed object writes are conditional and `commit.json` is published last. Each downloaded archive must pass bundle/file/manifests checksums and offline numerical replay. Projections are also read back byte-for-byte before querying.

The query is limited to the two prepared date/ticker partitions and selects coverage/count columns. Query polling has a two-minute deadline and requests cancellation if exceeded. CLI pagination is disabled; unexpectedly large responses are refused rather than silently scanning/paging additional data.

Evidence files remain in the local run directory:

- `deployment.json`: deployed stack/resources/account/region status.
- `downloads/`: verified local copies of both immutable exports.
- `verification.json`: S3 archive/projection checksum checks, replay results, actual Athena query ID, scan bytes, engine, result location and reconciled rows.
- `query-pending.json`: query ID for interruption/failure recovery.
- `cleanup.json`: verified cleanup after teardown.

The verifier estimates Athena cost from scan bytes and explicitly states that account billing/credits were not read by this script. Record the observed credit/charge status separately. Publish a redacted evidence note only after the real run; the original local evidence document must continue to distinguish synthetic data from real market collection and local tests from cloud execution.

## Explicit teardown

Keep the verified downloads and evidence first. Then run:

```powershell
.\.venv\Scripts\python.exe scripts/aws_history_demo.py teardown --output demo-output/aws-history-run --execute --expected-account VERIFIED_ACCOUNT_ID --confirm-stack vol-history-demo --profile project-upgrades --aws-cli 'C:/Program Files/Amazon/AWSCLIV2/aws.exe'
```

Teardown confirms the exact prepared stack and project ownership, disables its workgroup, and refuses to proceed while an active query might still write results. If one is active, cancellation is requested; retry after it stops. It refuses versioning changes, more than 1,000 objects or an unrelated object prefix. It deletes only the dedicated bucket's `archive/`, `analytics/`, and `results/` objects, verifies emptiness, deletes the stack/catalog/workgroup, then explicitly deletes and checks the retained bucket. Local exports remain untouched. Any failure is reported for inspection; do not use an account-wide deletion command as a workaround.

No scheduling or production integration is part of this verification. A future collector can use the same archive boundary after selecting approved persistence/cadence; this run remains a bounded two-observation cloud demonstration.
