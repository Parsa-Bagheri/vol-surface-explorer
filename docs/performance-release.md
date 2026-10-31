# Surface performance release

The release addresses GitHub issues #6–#11 on six stacked feature branches.
All six changes received independent reviews from gpt-6.1-sol at high reasoning.
The original American-options checkout is excluded.

| Issue | Change / reviewed commit | Reviewer verdict |
|---|---|---|
| #7 | Keep raw diagnostics while clearing private calculation attrs; 0caf7f8 | No findings; 56 focused tests and exact Python/C++ smooth/raw parity |
| #8 | Private per-worker expiry map reuse; 6fe5fc4 | No findings; 16 tests and sequential/concurrent/retry probes against yfinance 0.2.66 |
| #6 | Prepare each Vercel worker before serving surfaces; dcb7cd0 | Cleared with nested-error phase fix in 6fe1b56 and a regression test |
| #9 | Successful public responses cached by Vercel for 30 seconds; 6fe1b56 | No findings; 27 web/timing tests; actual HIT/expiry still require deployment verification |
| #10 | Reuse quote records and avoid per-node Series allocation; fe22919 | No findings; independent 102-case cleaner parity matrix and 61 focused regression tests |
| #11 | Preload the installed package's matching Plotly.js; 65e6237 | No findings; 29 web/timing and 2 history UI tests |

The CPU optimization preserves exact cleaned/raw frames, diagnostics and figure
JSON for saved SPY/TSLA inputs with Python/C++ and smoothing on/off. Offline,
three alternating-order pairs reduced median SPY processing from 2.023 to
1.590 seconds and TSLA from 1.595 to 1.532 seconds. These are offline CPU
measurements, not deployed click-to-chart results.

[Fixed-input parity and timing samples](evidence/surface-cpu-parity.json).

The worker readiness path imports the financial stack, initializes the chosen
backend and Plotly validators once under a lock, and never fetches market data.
Preparation moves work before Build when the worker is already initialized;
a newly allocated worker can still impose startup latency. Runtime logs separate
web import and preparation times.

Successful surface GETs use a shared Vercel CDN entry for 30 seconds, with no
stale-while-revalidate extension. Query parameters distinguish the entry. The
page shows the build time and offers Fetch latest, which is never cacheable.
Errors, empty charts, preparation and history remain uncached as live surfaces.
Cached phase headers describe the original build; actual browser click-to-chart
is recorded for every Build click independently.

The install hook regenerates the Plotly browser-version manifest from the
installed Python package. The homepage preloads that URL without blocking the
form; the chart includes one matching script. The existing CSP remains intact.

Before production promotion, require the full regression suite and platform CI.
After deployment, verify C++ Release/source provenance, startup readiness, real
SPY/TSLA 2–60-day click-to-chart times, query isolation, cache HIT/MISS, normal
expiry after 30 seconds, and Fetch latest. Preserve the last timing-only
production deployment dpl_H4LdKzsqNwzzDhkkXxhcayx8YcEb as rollback target.
