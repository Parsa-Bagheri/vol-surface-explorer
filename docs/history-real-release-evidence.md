# Real-data release evidence

Released October 13, 2026 (New York). **Local implementation, real AWS publication/verification and public release are complete.** [Public saved observations](https://www.volexplorer.ca/history). This record is separate from the earlier [synthetic AWS demonstration](history-aws-verification.md).

Two real yfinance/Yahoo SPY observations were collected manually on October 13 at **10:40:48 PM** and **10:45:43 PM EDT**. Their original UTC valuation clocks are October 14, 02:40:48.430117 and 02:45:43.381473. The UTC partition date is therefore October 14. No timestamp was backfilled or replaced with today's inputs during replay.

| Observation | Processing version | Fetched | Used | Excluded | Expirations |
| --- | --- | ---: | ---: | ---: | --- |
| `32635dd853246e6deb621938` | `1c73c45d8441de0ed2b55b00` | 2,604 | 1,067 | 1,537 | 10/10 |
| `6e471b7a5c5d8fe0695afacb` | `baf2fff2ef39b0af91e9b2fe` | 2,604 | 1,067 | 1,537 | 10/10 |

Both source kinds are `collected`; neither is synthetic/imported. Both retain a yfinance `history:1d:Close` spot of 778.010009765625, the provider's October 13 04:00 UTC daily-bar timestamp, their own spot collection time, per-expiration quote collection timestamps, source contract rows, requested settings and processing identity. A daily-bar timestamp is not the spot's last-trade time, and the spot is not claimed to be real-time. These after-hours collections may contain unchanged or delayed provider quotes.

Parameters: 1–30 DTE, strikes 93%–107%, smoothing enabled, existing risk-free/dividend/quality defaults. Code fingerprint is `39c3db92f021` (prefix); full fingerprint/dependency versions are in the downloadable processing manifests. Python 3.12 with NumPy 2.5.4, pandas 2.3.3, SciPy 1.18.1, Plotly 6.9.0, PyArrow 26.0.0; DuckDB 1.5.6 supplies local SQL.

Both offline replays matched with `rtol=atol=1e-10`, original spots/clocks and identical plotted-data/reconciliation checks. Network, provider calls and current-clock access were blocked during the prepared-bundle verification. This demonstrates reproducibility, not independent financial pricing accuracy.

The comparison renders **1,073 supported points**, leaves **79 unsupported points** blank and reports a shared supported domain of approximately $753–$808.58 strike and 1–24 DTE. The mean B-minus-A change is approximately 0.004475 IV percentage points. Elapsed pricing time and provider freshness affect this result; it is not evidence of a trading return or a clean isolated market-volatility movement.

The committed read-only web bundle contains **1,774,064 bytes** of selected artifact files; its release digest is `fe733a7e21ee5b12842ab2c7db8a0ccbf84d48e43819702eafaa33409aee3240`. The prepared cloud payload is **1,825,258 bytes**, including two 9,107-byte typed projections. All nine supported download types per observation matched the saved bytes locally. A fresh app instance retained the same bundle; corruption/missing-catalog tests fail closed.

The final full suite passed **142 tests** (129 existing + 13 release tests), with four existing NumPy timedelta deprecation warnings. Additional release tests cover generated/mock data for retry/partial publication, partition/version identity, source gating, account/template checks and the deployed read-only reader. Generated test inputs are not part of the market-collection evidence. Python compilation and whitespace checks also passed.

Local browser verification rendered both surfaces, observed quote markers and the supported-domain comparison without console warnings/errors. [Browser screenshot](images/real-history-local.png). The existing README's 2,687-contract September 28 benchmark and this representative 2,604-contract collection support a processing workload above 1,900 quotes; that figure is not inferred from the synthetic AWS fixtures or from summing repeated observations.

Durable design: local immutable collector store and approved S3 archive; the web deployment reads the committed verified bundle. Proposed real archive/projection/release retention is 365 days, query results one day. No recurring collector or query schedule has been enabled. See [release architecture, cost and rollback](history-release.md).

The approved existing CloudFormation stack update completed successfully. It retained the synthetic resources and added the real table and three scoped 365-day lifecycle rules. Both real archives/projections were published with conditional immutable writes, registered under the exact date/ticker/processing-version partitions and downloaded successfully. Checksums and both original-clock/spot replays matched at 1e-10. One Athena engine v3 query matched all DuckDB count/partial/version rows. It scanned **1,588 bytes**, separately from its **10,000,000 estimated billed bytes** (minimum billing), for an estimated **$0.00005 USD** Athena cost before credits. This is not an observed account charge.

The protected [verified preview](https://vol-surf-explorer-345z9l18y-parsa-bagheris-projects.vercel.app/history) is READY: `dpl_FiVtuKSHGnUh7WAqtnSy2jUyemgB`, source `dded5411db02d619b57efd1af72051a683699e6a`. The actual Linux Python 3.12 build passed using the documented large-functions beta setting on the existing Hobby plan. Its history listing, both explicitly selected processing versions and comparison passed; all **18 artifact downloads matched the committed files byte-for-byte**. There were 23 bounded history/landing/download checks and a separate successful live SPY surface request. History/live pages returned `no-store`; no AWS credentials were configured in Flask/Vercel.

The READY staged production build is `dpl_7fGQkPFgEsSpZCSm2NLLmSnbVhaj`, source `4bfbfcfb7d6c99538948114ed05c4a69e95d2c97`, at `vol-surf-explorer-eqwwjoftz-parsa-bagheris-projects.vercel.app`. It independently passed **24 bounded endpoint checks**, including a fresh live SPY surface and all 18 byte-matching downloads. Production and preview use `VERCEL_SUPPORT_LARGE_FUNCTIONS=1`; the account remains on Hobby. A second replay of both downloaded AWS exports passed with network/provider/current-clock access blocked. The post-run AWS account metadata still reported Free/Active with $120 credits; billing can lag and that is not proof of zero charges.

[PR #4](https://github.com/Parsa-Bagheri/vol-surface-explorer/pull/4) was merged under the user's approval: merge commit `06415be130dd9a01f1bbe884bb2d57f74b0ddd25` (PR head `b5cd54e117f6cbe3a5b6944aa21db4836ce71c89`). The exact tested staged build above was promoted to the existing public domain. Its application source is `4bfbfcf`; later commits contain release documentation only and do not change the deployed app.

Anonymous checks against `https://www.volexplorer.ca` passed **24 checks**, including the catalog, each selected version, comparison, all **18 byte-matching downloads**, and one real live SPY surface. The browser rendered all three figures with no console warnings/errors. The bounded 15-minute post-release runtime-error query returned no error records. [Full public screenshot](images/real-history-public.png) and [comparison screenshot](images/real-history-comparison.png). Preview and the newly created production deployment independently retained the same committed bundle, including the original provenance and byte-identical downloads. Temporary instance storage is not the durable source.

Rollback candidate recorded before release: `dpl_9J3d2MGwXER8wJj3ubNmAAJerRdr`, the earlier READY production application. No rollback was needed. The release retains normal Vercel protection on deployment URLs; custom production domains are public as before. No recurring collection/query, external monitoring service or log drain was installed; runtime logs and explicit bounded checks supplied this release's operational evidence.

Release identifiers/status:

- Initial implementation commit: `dded5411db02d619b57efd1af72051a683699e6a`.
- Real AWS archive/partition/query verification: passed; detailed query/resource identifiers stay private.
- Staged production deployment/checks: passed at the deployment/source above.
- Public `/history`/live checks and production promotion/PR merge: passed/completed under explicit approval.

Detailed account/resource identifiers and raw cloud outputs remain in ignored private evidence rather than this public note.
