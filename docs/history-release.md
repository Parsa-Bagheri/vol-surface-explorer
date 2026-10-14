# Real-observation release

Prepared October 13, 2026 (America/New_York). Local implementation and real-data verification are complete. AWS real-data execution and production promotion await the operator's approval. Earlier synthetic AWS evidence remains in [history-aws-verification.md](history-aws-verification.md).

## Data and web architecture

Two manual SPY collections use the existing collector with 1–30 DTE, 93%–107% strikes and the existing numerical settings. The collector retains all fetched rows before those filters. Each collection returned 2,604 contracts across 10/10 requested expirations; processing reconciles 2,604 = 1,067 used + 1,537 excluded. They were collected after the market close, about five minutes apart. A small difference in fitted values does not establish an intraday market movement: remaining pricing time and delayed provider inputs also affect the fit.

The local collector store keeps immutable raw inputs and processing artifacts. The prepared AWS release uses the existing private encrypted archive bucket, a separate `real/archive/` prefix, and a new Glue table `volatility_real_coverage`. Versioned projections use `real/analytics/date=UTC_DATE/ticker=TICKER/version=PROCESSING_ID/SNAPSHOT_ID.parquet`; Glue partitions include date, ticker and processing version. Selecting a new version adds a new projection and partition. Older versions remain queryable until retention expires. No crawler, Glue ETL job or persistent cloud compute is added.

Vercel serves the committed `history_bundle/` copy through a checksum-validating read-only reader. Bundle preparation selects explicit processing versions, verifies input identity, and rejects fixture/import sources, oversized bundles and downloads above 4 MB. Web requests perform no AWS authentication or collection. The default root resolves relative to `web_app.py`; `VOL_HISTORY_ROOT` can select another prepared bundle. On Vercel a missing/corrupt release catalog fails closed. A redeployment receives the same committed files rather than relying on a previous instance's temporary disk.

The bundled Python 3.12 runtime dependencies are pinned to the versions used for collection and processing, including PyArrow and DuckDB. The landing page retains deferred financial imports and caching, and the live calculation path uses the existing service. Local history operators can still point `VOL_HISTORY_ROOT` at a local store outside Vercel. Flask does not register a collector or schedule. No recurring collection is enabled.

## Reviewed change and permissions

The prepared CloudFormation diff preserves the existing bucket, policy, workgroup, database, synthetic table and fixture partitions. It adds one real-data Glue table, its output, and 365-day expiration rules for `real/archive/`, `real/analytics/` and `real/releases/`. Existing synthetic retention stays 30/7 days; query results remain one day. Account availability can end before the requested retention period; keep the verified local exports, code and committed bundle independently.

The existing CLI operator needs CloudFormation describe/get-template/validate/update on this stack; Glue get-table/get-partition/create-partition on its database and real table; S3 conditional put/get only on this bucket's `real/` objects; and Athena start/get-results/get-execution/stop on the existing workgroup with its result-prefix S3 permissions. No IAM users, keys or policies are created by the publisher. Deletion permissions are separate cleanup permissions. The deployed Flask app needs no AWS permissions or credentials. Other projects' resources are outside this release.

Preparation produces `plan.json`, baseline/proposed templates, checked exports and projections without AWS calls. The publisher verifies the account, stack ownership and completed state. It refuses unreviewed template drift, conflicting immutable objects and conflicting Glue locations. Files precede archive commits; read-back verification precedes partition registration. `publication.json` records each stage and a visible `partial_failure`; the final remote release commit appears only after every selected observation is published. Retrying reuses identical bytes and partitions.

## Bounded cost proposal

Prices checked October 13, 2026; all figures below are **USD before credits/tax**, estimates rather than observed charges. Private account inspection reports AWS Free/Active with $120 remaining credits and Vercel Hobby/Active. Credits do not authorize workloads. No account upgrade is proposed.

| Item | Proposed workload | Gross estimate |
| --- | --- | --- |
| S3 Standard storage, Ohio | Approximately 1.83 MB prepared payload; at most 20 MB enforced by preparation; retain real objects 365 days | About $0.000042/month at actual size; below $0.00046/month at the payload ceiling |
| S3 writes/reads | Two archives, two projections, one release commit and round-trip verification; allowance of 100 PUT/LIST and 200 GET requests | $0.00050 + $0.00008 |
| Athena SQL | One partition/version-filtered query, existing 10 MB scan cutoff, stable request token on retries | $0.00005 at 10 MB minimum billing |
| Glue Data Catalog | One table and two explicit partitions; no ETL or crawler | $0 within the account's first million stored objects/requests allowance |
| Data transfer | Bounded round-trip download, at most 20 MB assumed | Below $0.0018 using $0.09/GB conservatively, before any free allowance |
| Vercel | Preview/staged production deployment and bounded verification on existing Hobby project | $0 incremental within Hobby allowances; no upgrade or paid add-on |

Expected initial AWS execution is **less than $0.01 gross**, with no recurring collection/query. Actual stored payload retention is about **$0.0005/year** at current S3 Standard storage pricing. Payload bounds and the Athena cutoff are workload controls, not an account spending cap. Additional retries still make S3 requests; stop and inspect any unexpected failures rather than run an unbounded retry loop.

Sources: [S3 pricing](https://aws.amazon.com/s3/pricing/) and the [Ohio public price list](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonS3/current/us-east-2/index.json) (publication October 12, 2026), [Athena pricing](https://aws.amazon.com/athena/pricing/), [Glue pricing](https://aws.amazon.com/glue/pricing/), [Vercel Hobby](https://vercel.com/docs/plans/hobby). S3 rates used are $0.023/GB-month, $0.005/1,000 PUT/LIST and $0.004/10,000 GET; Athena is $5/TB with a 10 MB minimum.

## Execute after approval

Use a current AWS CLI and the operator's existing login. Keep account IDs, bucket names, caller identities, raw command output and detailed cloud reports in ignored private evidence. The prepared plan must name the already-reviewed stack. Replace the example selections with the exact snapshot/processing IDs in the release evidence.

```powershell
python scripts/publish_real_history.py prepare --store data/history --output demo-output/real-run --stack-template PRIVATE_BASELINE_TEMPLATE --observations SNAPSHOT_A:PROCESSING_A SNAPSHOT_B:PROCESSING_B
python scripts/build_history_bundle.py --store data/history --output history_bundle SNAPSHOT_A:PROCESSING_A SNAPSHOT_B:PROCESSING_B
python scripts/verify_history_bundle.py --bundle history_bundle --output demo-output/local-verification.json
python scripts/publish_real_history.py update-stack --output demo-output/real-run --execute --expected-account VERIFIED_ACCOUNT_ID --profile project-upgrades
# Inspect CloudFormation completion before publication; the update command does not block.
python scripts/publish_real_history.py publish --output demo-output/real-run --execute --expected-account VERIFIED_ACCOUNT_ID --profile project-upgrades
python scripts/publish_real_history.py verify --output demo-output/real-run --execute --expected-account VERIFIED_ACCOUNT_ID --profile project-upgrades
```

`--aws-cli` can name the installed executable. Re-running `update-stack` after completion is an idempotent no-op. `verify` checks remote partition locations, downloads archives, validates checksums/replay, and reconciles one Athena query with DuckDB. Its stable request token reuses the same query rather than launching another paid query on a retry. It records actual scanned bytes separately from estimated billed bytes. A failed/cancelled query needs explicit review before choosing a new query identity.

Update existing PR #4 with the reviewed source, bundle and redacted evidence. On the linked Vercel project, create a preview and verify `/`, `/history`, both selected versions, the common-domain comparison, all downloads and a bounded live SPY request. Stage production with `vercel deploy --prod --skip-domain`, verify that build, then promote it with `vercel promote DEPLOYMENT_URL`. Merge/promotion requires the user's specific approval. Preserve deployment protection. Python bundles above the host's default limit may require the large-functions setting; inspect the build result before changing it. No new paid plan is authorized by this procedure.

## Rollback and cleanup

Record the previous production deployment before release. Roll back the domain to that verified deployment if public history/live checks fail. Vercel rollback disables automatic assignment until a later promotion; account for that when continuing releases. Reverting the PR restores the earlier app without deleting history.

Keep verified downloads before cleanup. Remove only this release's exact real-table partitions and exact `real/` object keys after confirming their locations; never delete the bucket or the shared stack as a release rollback. Restore the original CloudFormation template only after those real partitions are removed, preserving synthetic resources. The old synthetic teardown intentionally refuses unexpected real prefixes; do not weaken that guard. Lifecycle expiration does not automatically remove Glue partition metadata. Remove expired real partitions during explicit operator cleanup. Keep the Git bundle/local exports until replacement evidence is verified.

Final commit/deployment identifiers, actual cloud results and public checks belong in [history-real-release-evidence.md](history-real-release-evidence.md) after execution.
