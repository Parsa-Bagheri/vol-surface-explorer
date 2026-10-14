# Historical pipeline evidence

Recorded October 13, 2026. This feature implements local collection/persistence, replay, Parquet SQL and historical comparison. The execution evidence below uses synthetic fixtures; no real market history was collected, no AWS deployment was performed, and no production site was changed.

## Workload and environment

The fixture lake contains 154 generated European BSM quote rows. Synthetic A has original spot 100, valuation `2026-10-12T18:00:00Z`, and four received expirations. Synthetic B has spot 101, valuation `2026-10-13T18:00:00Z`, three received expirations and one simulated failed expiration. Both use strikes at 90%–110% of spot, DTE 0–60, smoothing, lenient quality rules, a 2% risk-free rate and zero configured dividend yield. Each original valuation clock is preserved; the labels are fixture times, not a claimed market backfill.

Environment: Windows 11, AMD64, Python 3.12.14; NumPy 2.5.4, pandas 2.3.3, SciPy 1.18.1, Plotly 6.9.0, PyArrow 26.0.0 and DuckDB 1.5.6. Exact source fingerprint and file sizes are in [history-benchmark.json](evidence/history-benchmark.json).

| Measurement | Result |
| --- | --- |
| Fixture generation, immutable capture and first processing of A+B | 3.50 seconds |
| A offline replay, three warm runs | 1.69, 1.50, 1.58 seconds; median 1.58 |
| A unchanged processing reuse | 0.054, 0.051, 0.057 seconds |
| Coverage SQL with integrity verification, three runs | 0.161, 0.117, 0.127 seconds; median 0.127 |
| Common-domain comparison | 0.182 seconds |
| Raw, metadata, cleaned, selected and coverage Parquet combined | 189,204 bytes across ten files |
| A reconciliation | 88 fetched = 47 used + 41 excluded |
| B reconciliation | 66 fetched = 32 used + 34 excluded |
| Supported delta grid | 1,088 points; 64 unsupported points excluded |
| Common fitted domain | strike $96.29–$105.83; DTE 4–24 |
| Mean fixture B-minus-A difference | 3.438793 volatility percentage points |
| Verified local A export | Nine files; 195,126 bytes, excluding bundle envelope |
| Prepared local Athena coverage projection | 9,082 bytes; explicit A processing version; no AWS execution |

These are small local warm-fixture measurements, including checksums and applicable file loads. They exclude Yahoo request latency, browser rendering, production throughput and independent IV accuracy. The seeded capture measurement includes first processing; it is not a live collection benchmark. The machine was shared with other local work, so these figures are descriptive rather than a dedicated performance study.

## Regression and failure evidence

The unchanged repository at commit `72e763b342aeea8a6ab014ae4ca00b7eec8ea815` passed its 77 regression tests. The final feature suite passed **119 tests in 52.31 seconds**, with four existing NumPy timedelta warnings; the output is retained in [pytest-history.txt](evidence/pytest-history.txt). Before the final projection, source-newline-portability and cleaned-count tests were added, the full 116-test suite was also independently rerun, with golden parity and the offline demo reproduced.

The original live service generated `tests/fixtures/original-live-bundle.json` from the same synthetic quote inputs and explicit clock/spot. The new processor matches original cleaned IV/price/time/DTE/row-ID fields and fitted x/y/z arrays at `rtol=atol=1e-10`; exclusive outcomes match exactly. Replay additionally verifies exact plotted trace data on the recorded runtime. Internal repricing and golden parity show numerical consistency and preservation of existing behavior; they are not independent market accuracy or predictive-ML results.

New tests cover:

- Network-free replay with sockets, Yahoo objects, live fetches and the current clock blocked.
- UTC/New York midnight boundaries, both DST transitions, and zero-day expiry with positive original pricing time.
- Same-input collection/processing retries without duplicate rows; changed code fingerprints produce explicit new processing versions; identical CRLF/LF source produces the same fingerprint.
- Cross-date/ticker custom-ID conflicts, visible active/stale ID locks, and interruption before atomic rename while prior observations stay readable.
- Missing/corrupt files, corrupt or incompatible manifests, schema changes, derived row/coverage count mismatch, lineage mismatch, and unsafe IDs/paths.
- Duplicate and malformed contracts, preservation of original malformed source values, and exclusive fetched/used/excluded reconciliation.
- Partial and total expiration failures, including failed outcomes alongside received rows and SQL over unprocessed empty captures.
- Compatible comparisons, unsupported domain cells, NaN gaps, UI labels, lineage downloads and invalid artifact requests.
- Local export validation, fake-S3 immutable upload/read replay, commit-last interrupted upload, and typed bounded Athena projection.

## Browser verification

The verified flow is observation selections → Flask history route → checksummed local manifests/Parquet → two fitted surfaces and the common-domain delta. The form was exercised in Chrome with the actual saved fixtures. Both original clocks/spots, A's 4/4 coverage, B's 3/4 partial coverage, 1,088/64 comparison support counts and all source/processing download links rendered. Three Plotly plots rendered on desktop and mobile; at 390px mobile width, document width was also 390px, so the page had no horizontal overflow.

The results page had zero JavaScript errors. Plotly emitted one Canvas2D readback performance warning. The empty history entry initially requested a nonexistent favicon; this did not affect the product flow. Screenshots are saved as [desktop](images/history-comparison-desktop.png) and [mobile](images/history-comparison-mobile.png). Browser screenshots require Plotly's CDN, as the existing live page does; replay and the CLI standalone comparison HTML can run offline.

## Reproduce and explain

Use the commands in [history.md](history.md) and the [five-minute walkthrough](history-demo.md). Run the benchmark with a new empty store to measure first publication. `docs/evidence/history-benchmark.json` records actual machine/runtime/source identity, individual timings, coverage rows, supported domain and exact Parquet sizes.

S3/Athena preparation is [documented separately](history-aws.md), with fake-client tests and local typed projection only. An actual S3 upload/read, Athena scan, account cost status and cleanup remain unexecuted. Scheduled real collection and ongoing retention are operator setup steps; no OS schedule was registered in this work.

## Factual resume drafts

- Built an immutable options-observation pipeline with UTC valuation and spot provenance, Parquet lineage, checksum validation and offline surface replay; preserved the existing numerical pipeline within 1e-10 tolerance and handled partial fetches, duplicate contracts and interrupted publication.
- Added DuckDB SQL and historical volatility-surface comparisons over a labeled 154-row synthetic workload; reconciled every contract, excluded 64 unsupported grid points, and measured 1.58-second median replay and 0.13-second coverage queries on a local Windows/Python environment.
