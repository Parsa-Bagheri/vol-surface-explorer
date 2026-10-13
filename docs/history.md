# Saved observations and reproducible surfaces

Historical mode captures the inputs to the existing financial pipeline, including the original UTC valuation clock and spot provenance. It does not infer an earlier market history from today's chains. Real history begins when a collector saves real observations. The demonstration uses generated model quotes and is labeled synthetic throughout.

## Local setup and demo

The existing live web and `main.py` paths keep their current dependencies. Install the optional history dependencies for local persistence and SQL:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe history.py --store demo-output/my-lake demo
.\.venv\Scripts\python.exe history.py --store demo-output/my-lake inspect synthetic-b
.\.venv\Scripts\python.exe history.py --store demo-output/my-lake replay synthetic-a
.\.venv\Scripts\python.exe history.py --store demo-output/my-lake query "SELECT snapshot_id, partial, raw_rows, used_rows, excluded_rows FROM coverage"
.\.venv\Scripts\python.exe history.py --store demo-output/my-lake compare synthetic-a synthetic-b --output demo-output/comparison.html
$env:VOL_HISTORY_ROOT='demo-output/my-lake'
$env:PORT='4313'
.\.venv\Scripts\python.exe web_app.py
```

Open `http://127.0.0.1:4313/history`, choose synthetic A and B, and select **View observations**. The single saved processing version is selected automatically; select versions explicitly when an observation has several. The raw data, observation manifest, processing manifest, cleaned data, and diagnostics are downloadable from each result.

A has 88 generated contracts across four expirations. B has 66 generated contracts across three of those expirations; the fourth has a simulated failed collection. The two observations have different original clocks, spots, and volatility assumptions. They demonstrate saved-input replay and partial coverage, not measured market changes.

Use a fresh demo store after changing collector/processing source. Fixed demo IDs are intentionally immutable; the tool will reject a different fixture or collection fingerprint under the same ID rather than replacing prior evidence.

## Real collection and scheduling

Collection and processing are separate durable operations:

```powershell
.\.venv\Scripts\python.exe history.py --store data/history collect SPY AAPL --dte-min 1 --dte-max 30 --label 'Daily watchlist'
.\.venv\Scripts\python.exe history.py --store data/history list
.\.venv\Scripts\python.exe history.py --store data/history process SNAPSHOT_ID
.\.venv\Scripts\python.exe history.py --store data/history replay SNAPSHOT_ID --processing-id PROCESSING_ID
```

The one-pass scheduling helper collects and processes each ticker, reports collection/processing times, keeps saved raw inputs if fitting fails, continues the remaining watchlist, and exits nonzero if any ticker failed:

```powershell
.\.venv\Scripts\python.exe scripts/collect_watchlist.py SPY AAPL --store data/history --dte-max 30
```

Configure an OS scheduler on a persistent host to run this once per weekday after market close, with the repository as its working directory, the full virtual-environment Python path, an absolute store path, and stdout/stderr redirected to a retained log. For Windows Task Scheduler: trigger weekly Monday–Friday at 16:15 New York time; action `REPO\.venv\Scripts\python.exe`; arguments `REPO\scripts\collect_watchlist.py SPY AAPL --store STORE --dte-max 30`; start in `REPO`. Use “Do not start a new instance” and record failed exit codes. This is a suggested configuration; no scheduler task was registered during implementation.

For a Linux persistent host, an equivalent cron entry is:

```cron
CRON_TZ=America/New_York
15 16 * * 1-5 cd /srv/vol-surface-explorer && .venv/bin/python scripts/collect_watchlist.py SPY AAPL --store /srv/vol-history --dte-max 30 >> /srv/vol-history-collector.log 2>&1
```

Cadence and watchlist are bounded by the operator; the CLI caps a pass at ten tickers. No background scheduler runs inside Flask. Yahoo can return delayed/fallback spot values, incomplete chains or rate-limit failures. Per-expiration outcomes and spot source timestamps make those limits visible. The underlying spot provenance identifies a history-close fallback and its bar timestamp, or marks the source timestamp unavailable for info fallbacks. A bar timestamp does not assert the exact last trade time of that spot.

Flask's deployed Vercel filesystem is ephemeral and is not the collection host. Historical mode is intended for a persistent local host or a separately prepared read-only data bundle. Configure `VOL_HISTORY_ROOT` to that store; do not assume a live web request durably schedules or saves data.

## Versioned contract, keys, and lineage

```text
STORE/
  observations/date=2026-10-12/ticker=DEMO/SNAPSHOT_ID/
    raw.parquet
    observation.parquet
    manifest.json
  processed/SNAPSHOT_ID/PROCESSING_ID/
    cleaned.parquet
    selected.parquet
    coverage.parquet
    surface.json
    diagnostics.json
    manifest.json
  .locks/SNAPSHOT_ID/owner.json          # only while publishing raw inputs
  .pending-RANDOM/                      # never part of the visible catalog
```

Schema version 1 observation manifests contain the snapshot ID, ticker, UTC valuation time, positive spot and its provider/method/source/collection timestamps, all `SurfaceRequest` parameters, source kind/label, expected/received/missing expirations, failed/empty expiration outcomes, fetch diagnostics, raw row count, malformed/duplicate counts, and the processing identity at collection. An envelope SHA-256 covers canonical manifest JSON; each file has its own SHA-256 and byte count. These checks detect accidental corruption and incomplete publication; they are not signatures proving provider authenticity.

`raw.parquet` retains source columns plus `sourceRowJson`, `ticker`, `snapshot_id`, `stableContractKey`, and `historyInvalidReason`. Analytical numeric fields are nullable double values; expiry and trade timestamps are timezone-aware UTC timestamps; textual identity fields are strings. `sourceRowJson` preserves original malformed textual values before analytical coercion. Missing/nonfinite JSON values are represented as null. Source contract symbols are retained and used as the stable key. If absent, the key is `TICKER:YYYY-MM-DD:call|put:STRIKE`, where strike is formatted with twelve significant digits. This fallback assumes standard equity/ETF contracts; it does not disambiguate adjusted contracts with missing provider symbols.

`contractId` is assigned by the shared numerical builder as a per-build row ID. Cross-observation joins use `stableContractKey`, never `contractId`. Duplicate stable keys retain the first row in deterministic input order; subsequent rows remain in immutable raw storage and receive **Duplicate contract** exclusions. Invalid expiry/type/strike values receive **Malformed contract** exclusions. The original fetched = used + excluded accounting includes both categories exactly once.

`observation.parquet` is one typed metadata row per raw snapshot, including source kind, original clock/spot, partial coverage and input-quality counts. It remains queryable even if all expiration fetches failed or no surface can be fitted. `cleaned.parquet` retains all cleaner output columns plus snapshot/processing lineage; `selected.parquet` contains the actual selected fitter quotes. `coverage.parquet` is one row per `(snapshot_id, processing_id)` with counts and reconciliation. Derived files point to the exact raw checksum.

The processing ID hashes the raw Parquet checksum, saved parameters, actual processing-source fingerprints, and NumPy/pandas/SciPy/Plotly/PyArrow versions. Source fingerprints cover the cleaner, visualizer, expiry utilities, outcomes, shared service, config, and history storage logic; CRLF/LF differences are normalized so Git newline conversion does not invalidate identical source. Rerunning unchanged processing validates and reuses its immutable artifact, including derived row lineage/counts and coverage reconciliation. Editing processing source or changing those dependencies publishes a separate version and marks it as reprocessing. Recording a fingerprint identifies the code that ran; the code and dependency environment must still be preserved in Git/release evidence to reproduce it later.

`replay` requires an existing artifact and the same source/dependency identity. It never substitutes the current clock, fetches current spot/chains, or silently reprocesses using another version. It checks cleaned IV/price/time/DTE/ID columns within `rtol=atol=1e-10`, exact plotted trace data on the recorded runtime, and exact exclusive reconciliation. To deliberately process a new code version, run `process`; then select that new processing ID. Matching runtime numerical behavior is an internal consistency check, not independent accuracy validation.

Unsupported schema versions are refused explicitly. There is no automatic migration that overwrites raw snapshots. Export old snapshots and retain the original code/runtime before a schema migration; implement a versioned import when a new schema is introduced.

## Integrity, interruptions, and recovery

Files and the manifest are built in an unpublished staging directory on the same filesystem. Files are flushed before a directory rename publishes the complete snapshot or processing artifact. Readers inspect only committed directories, then validate manifest/file checksums and metadata before reading Parquet. No complete-looking manifest is published before its files. On a normal failure the staging directory is removed; a killed process may leave unpublished staging directories or an ID lock.

Custom snapshot IDs are unique across the entire store, including date/ticker partitions. An atomic per-ID lock serializes raw publication, and conflicting inputs are rejected globally. A lock's `owner.json` records PID, host, and creation time. Processing versions use the same immutable-directory publication boundary; simultaneous identical work can reuse the winning artifact. These are local filesystem guarantees; cross-host writers against an eventually synchronized folder are not coordinated by this lock. Use one collector host and a persistent local filesystem for scheduled collection. This implementation does not claim a transaction spanning multiple machines or a power-loss guarantee for a network filesystem.

`list` and `inspect` report `publication_locks` and `unpublished_staging_directories`. For conservative recovery:

1. Stop the scheduled collector and confirm the recorded host/PID has no active publication. A PID alone can be reused; also inspect its command and start time. Do not remove a lock solely because it is old.
2. Validate any already committed observation with `inspect`. Its immutability and readability are independent of a stale publication lock.
3. Remove only that exact lock's `owner.json`, then its now-empty directory. Example after verifying ownership: `Remove-Item -LiteralPath 'STORE\.locks\SNAPSHOT_ID\owner.json'`; `Remove-Item -LiteralPath 'STORE\.locks\SNAPSHOT_ID'`. Review the resolved absolute path first.
4. Move or remove only confirmed orphan `.pending-*` staging directories after checking their resolved paths stay inside the store. They are never considered completed observations.
5. Retry collection/processing. Check corrupt committed data against a verified export or restore it from that export; do not edit a manifest checksum to hide a missing file.

A partial observation can be processed when sufficient usable quotes remain. The UI shows coverage and failed expirations. An empty or fully excluded observation remains inspectable/queryable and processing fails without publishing a surface artifact.

## SQL and comparisons

`history.py query` exposes DuckDB views `observations`, `raw_quotes`, `coverage`, `cleaned`, and `selected_quotes` over integrity-validated Parquet. Processing views are present when at least one artifact exists. Corrupt snapshots/artifacts make the CLI SQL catalog fail closed rather than silently changing a count. The historical UI instead lists the integrity issue and excludes that invalid item. SQL is a trusted local operator capability; it is not exposed as arbitrary SQL in the web interface.

See [history-queries.sql](history-queries.sql) for coverage, total failure, duplicate, reconciliation and stable-contract change queries. Multiple explicit reprocessing versions can appear in the processing views; filter by processing ID when computing per-snapshot aggregates. A rerun of the same processing identity does not add rows. Date/ticker partition names are stored in raw paths, while columns carry typed lineage for joins.

[DuckDB's Parquet reader](https://duckdb.org/docs/current/data/parquet/overview) supports direct Parquet scans. The catalog supplies explicit validated paths, rather than a glob that could include unpublished or corrupt files.

Comparisons require identical ticker, saved parameters and processing code/dependency identity. Each observation is built using its own original spot and time. The delta is B minus A in volatility percentage points, plotted by absolute strike and DTE relative to each observation. Comparisons are not a claim that the same calendar expiration has the same remaining maturity on both dates.

For grids, the difference evaluates only where both fitted surfaces support that strike/maturity. It interpolates between adjacent supported native strike/DTE grid cells, refuses extrapolation, and does not bridge NaN gaps. Unsupported points remain blank and their count is displayed. If both fits only produced adjusted nodes, comparison uses exactly matching strike/DTE nodes; a grid/node mismatch is refused. Gold markers show observed quote-derived IV, while surface/delta hover text identifies interpolated fitted values.

## Retention and export

```powershell
.\.venv\Scripts\python.exe history.py --store data/history export SNAPSHOT_ID --output demo-output/export-SNAPSHOT_ID
```

The export contains validated raw and all saved processing artifacts, preserving their lake-relative layout, with a checksummed `bundle.json`. Open the exported folder as a store to inspect/query/replay it under the matching runtime. Keep a copy of the code, environment versions, benchmark JSON, screenshots, and verified exports outside any temporary cloud account.

There is no implicit retention deletion. For an operator-selected retention window, export and verify each snapshot first, stop the scheduler, then remove only that snapshot's exact raw directory and its corresponding processed directory after checking resolved paths. Keep observation and derived retention consistent. Resume the scheduler and use `list` to confirm the remaining store. This avoids automatic deletion of evidence or user history.

## Tests and measurements

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/benchmark_history.py --store demo-output/new-benchmark-lake --output docs/evidence/history-benchmark.json
```

The golden numerical fixture was generated by the unmodified live service at commit `72e763b342aeea8a6ab014ae4ca00b7eec8ea815` using the same synthetic raw inputs, clock, spot and parameters. Regression tests compare the extracted processor's numerical columns and fitted x/y/z values to that baseline, independently of replaying artifacts produced by the new storage path.

The suite also blocks sockets, Yahoo objects, live fetch functions and `Timestamp.now` during replay; tests New York/UTC boundaries, both DST transitions and zero-day expiry; and checks corruption, schema/lineage mismatch, malformed/duplicate contracts, total/partial fetch failure, immutable reruns, global ID collisions, interruption before rename, stale locks, unsupported interpolation, UI/download lineage and simulated S3 transport failures.

Measured fixture results and their limitations are in [history-evidence.md](history-evidence.md), with machine-readable timings in [history-benchmark.json](evidence/history-benchmark.json). Optional cloud preparation is described in [history-aws.md](history-aws.md).
