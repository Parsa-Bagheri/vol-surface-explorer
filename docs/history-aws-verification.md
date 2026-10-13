# AWS verification — October 13, 2026

The isolated S3/Glue/Athena stack was deployed in us-east-2 using the existing AWS CLI login. Two explicitly synthetic observations were uploaded, downloaded, checksum-verified and replayed offline. Both replays matched at 1e-10 tolerance. This is cloud integration evidence, not real market history or independent financial validation.

One partition-filtered Athena query scanned 1,510 bytes and matched the local DuckDB results:

| Observation | Raw | Used | Excluded | Expirations | Partial |
| --- | ---: | ---: | ---: | --- | --- |
| synthetic-a | 88 | 47 | 41 | 4/4 | No |
| synthetic-b | 66 | 32 | 34 | 3/4 | Yes |

Both 9,082-byte Parquet projections also matched after S3 download. Athena engine version 3 completed the query successfully. Ten new deployment/transport tests pass; the previously verified numerical/history suite contains 119 tests.

The private encrypted bucket, catalog and on-demand workgroup remain deployed. There is no recurring collector or scheduled query. Archive objects expire after 30 days, projections after seven days and query results after one day. The workgroup enforces a 10 MB scan cutoff per query. The estimated Athena query cost is $0.00005 before credits, accounting for minimum billing; this is not an observed charge. Earlier transport retries also made S3 requests.

See the [runbook](history-aws-runbook.md) for reproduction, limits and scoped teardown. Detailed account/resource identifiers and run reports remain in private local evidence. Production behavior was not changed.
