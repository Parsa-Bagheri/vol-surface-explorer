# Optional S3 archive and Athena preparation

Status: prepared locally, not deployed. No AWS account was created, no S3 request or Athena query was executed, and no cloud cost was incurred. Tests use an in-memory fake S3 client. Reuse an existing approved account setup if one is available.

The free working implementation is local storage plus DuckDB. The optional archive uses the same verified export contract. `S3ObservationArchive` accepts an explicitly supplied client so importing or using local history does not load credentials or make AWS calls. Files are written under content-addressed object keys with conditional `IfNoneMatch='*'` writes; `commit.json` is written last. Repeated uploads compare existing bytes before reusing them. Downloads verify the bundle, file checksums and lake manifests in a staging directory before publication.

After account/access/cost review and explicit authorization, install `boto3` separately and supply a least-privilege client:

```python
from pathlib import Path
import boto3
from src.history_s3 import S3ObservationArchive

archive = S3ObservationArchive(boto3.client('s3'), 'AUTHORIZED_ISOLATED_BUCKET')
result = archive.upload(Path('demo-output/export-SNAPSHOT_ID'))
archive.download(result['commit_key'], Path('demo-output/verified-cloud-export'))
```

This example is not an instruction to execute before access/cost approval. The runtime needs scoped `s3:PutObject` and `s3:GetObject` on its archive prefix. Teardown needs separate delete/list permission. Do not make the bucket public. [S3 conditional writes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html) support the immutable object boundary.

`infra/aws/archive.json` prepares a private S3 bucket with server-side encryption, blocked public access, and a 30-day lifecycle for temporary demo objects. It retains the bucket on stack deletion to avoid unexpectedly losing evidence. Retention means teardown must explicitly empty/delete the bucket after verified local export; deleting the stack alone does not remove it. No crawler, always-on compute, managed orchestration service or account setup is included.

Athena uses a separate analytics projection containing only one explicitly selected processing version per snapshot, avoiding duplicate versions or archived JSON objects in the table's location. Prepare it locally with `python scripts/export_athena_projection.py SNAPSHOT_ID PROCESSING_ID --store STORE --output demo-output/analytics`. The script validates source lineage, removes the data-file ticker column (derived from the Athena partition), and writes UTC timestamps as Parquet microsecond timestamps. It refuses to overwrite an existing snapshot projection. Keep schema/version types as described in `infra/aws/athena.sql`. Register explicit date/ticker partitions rather than starting a crawler. The SQL file includes a date/ticker-filtered coverage query. This preparation follows AWS guidance on [Parquet/columnar storage](https://docs.aws.amazon.com/athena/latest/ug/columnar-storage.html) and [partitioned tables](https://docs.aws.amazon.com/athena/latest/ug/partitions.html).

Before any execution, verify the account's plan, eligibility and region, S3/Athena/Glue access, approved workload and storage retention. Present concrete costs based on observed exported bytes, collection cadence, request counts and query scan sizes, with local DuckDB as the free alternative. Athena's query result S3 prefix also needs retention and teardown. Use a dedicated workgroup, a small per-query bytes-scanned limit, a short result-retention window, and selected date/ticker predicates. Budget alerts are not a hard spending cap.

Deployment validation, when authorized, must record an actual upload/read checksum comparison, the Athena query ID and bytes scanned, account credit/charge status, and cleanup verification. Preserve local evidence before deleting the analytics projection, archive/result objects, Athena table/workgroup, and retained bucket. The current evidence contains none of those cloud results and makes no cloud deployment claim.
