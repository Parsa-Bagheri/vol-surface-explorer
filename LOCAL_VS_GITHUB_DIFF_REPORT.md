# Local vs GitHub Difference Report

Generated: 2026-05-26

Compared against GitHub repository `Parsa-Bagheri/vol-surface-explorer`, default branch `main`, commit `532f6039edd21cb336dfc8147597ea573c663eab`.

The local committed `HEAD` is the same commit as GitHub `main`; the differences below are the local working-tree changes and untracked local files.

## Executive Summary

The local version materially changes the data and surface-construction pipeline. The GitHub version already has a unified call/put, static-arbitrage-projected surface, but the local version makes the pipeline more precise and stricter:

- Fetching now filters expirations by requested DTE before downloading chains.
- DTE and time-to-expiration now use a shared valuation timestamp and the real 4pm New York option close, including daylight-saving behavior.
- Forward prices and dividend yields are now estimated per expiry from put-call parity when enough paired quotes exist.
- Black-Scholes recomputed IV is now gated by bid/ask width, bid/ask-implied-IV width, stale-trade checks, one-sided quote checks, and recompute-specific reliability flags.
- `auto` now prefers reliable quote-derived Black-Scholes IV and uses Yahoo provider IV only as fallback.
- Surface construction now uses the selected final IV as the source of truth, instead of preferring raw market price whenever present.
- Smoothed interpolation is stricter, denser in log-forward moneyness, capped in DTE grid density, and uses expiry-specific forwards/dividend yields.
- The web app adds input clamping, security headers, safer error handling, and Vercel deployment files.

The most important adjustment/interpolation difference is this: the local version no longer treats the selected option `marketPrice` as the primary surface input whenever it exists. It reprices from `impliedVolatilityFinal`, so `yfinance`, `auto`, and `black-scholes` modes can actually produce different surfaces.

## Local Pipeline Overview

The local end-to-end path is:

1. `src.surface_service.build_surface_bundle` validates the request, normalizes ticker/mode/rates/ranges, captures one UTC valuation timestamp, fetches spot, fetches options, cleans them, builds diagnostics, and creates the Plotly figure.
2. `src.data_fetch.get_current_price` uses `yfinance.Ticker(...).history(period="1d")["Close"][-1]`, falling back to `info["currentPrice"]` or `info["previousClose"]`.
3. `src.data_fetch.get_options_data` reads available expirations from Yahoo Finance, computes DTE using the shared valuation time and New York option close, fetches only expirations inside the requested DTE range, retries poor-quality expirations, and stores fetch diagnostics in `DataFrame.attrs`.
4. `src.data_cleaner.prepare_options_data` normalizes columns, filters by option type, strike, date, DTE, and positive time-to-expiration, estimates per-expiry forwards, resolves final IV by mode, assigns confidence and quality flags, and computes surface weights.
5. `src.visualizer.create_vol_surface` filters surface-eligible rows unless low-confidence rows are explicitly included, converts calls and puts into call-equivalent prices, prefers OTM options around the forward, aggregates call/put quotes into one node per strike/DTE, projects each slice to monotone/convex call prices, and optionally interpolates total variance over log-forward moneyness and DTE.

## File-Level Differences

| Area | GitHub `main` | Local version |
| --- | --- | --- |
| `.gitignore` | Ignores generated `*_vol_surface_*.html` files. | Also ignores `.vercel`. |
| `README.md` | Documents the existing unified arbitrage-free pipeline. | Adds new quality flags, DTE behavior, parity forward estimation, selected-IV surface input behavior, and clarifies that the implementation is direct static projection, not SVI calibration. |
| `main.py` | Prints "unified call/put arbitrage-free surface". | Renames console wording to "static-arbitrage-adjusted surface". |
| `src/time_utils.py` | Not present. DTE logic is embedded in cleaner/fetch code. | New shared time utilities for UTC valuation time, New York 4pm expiration close, displayed DTE, and year fraction. |
| `src/data_fetch.py` | Fetches every available expiration from Yahoo Finance, then downstream cleaning filters DTE. Diagnostics report requested expirations as all available dates. | Filters expirations before fetching based on `min_dte`, `max_dte`, and `as_of_utc`. Adds `expirationCloseUtc`, `expirationDteAtFetch`, selected/available expiration diagnostics, and `as_of_utc`. |
| `src/data_cleaner.py` | Cleans rows, computes/reuses IV, applies simpler quote/liquidity flags, and uses configured dividend yield throughout. | Adds parity forward estimation, expiry-level dividend yield, bid/ask-IV-width checks, recompute reliability gates, more flags, DTE diagnostics, and per-row dividend yield in validation/repricing. |
| `src/surface_service.py` | Validates only the basics and fetches all option chains. | Adds ticker regex, finite/range validation, mode validation, DTE/rate/trade-age bounds, shared valuation time, fetch-time DTE filtering, and passes requested DTE to diagnostics/plot axis. |
| `src/visualizer.py` | Builds a unified projected surface from market prices when available, uses 41-point log-moneyness grid, requires only 2 maturity slices for smoothing, and uses configured dividend yield/forward. | Uses final selected IV as surface input, uses parity-derived forwards/dividend yields, has an 81-point log-moneyness grid, stricter smoothing prerequisites, DTE-grid cap, requested DTE axis range, and low-confidence inclusion support. |
| `web_app.py` | Lightweight Flask UI with minimal parsing and debug always on when run directly. | Adds security headers, no-store cache, max content length, bounded query parsing, safer unexpected-error handling, DTE summary cards, and env-controlled debug. |
| `templates/index.html` | Uses "Unified Arbitrage-Free Surface" label and "smoothed arbitrage-free surface" wording. | Rewords the UI around "visualizing and exploring" and "smoothed adjusted surface"; also replaces the curly apostrophe in the error title. |
| `api/index.py` | Not present. | New Vercel entrypoint importing the Flask app. |
| `.vercelignore`, `vercel.json`, `requirements.txt` | Not present. | New Vercel deployment config and Python dependency constraints. |
| Tests | Cover the previous fetch/clean/visualizer/web behavior. | Add coverage for DTE fetch filtering, time utilities, stricter IV reliability, selected-IV surface input, low-confidence inclusion, request DTE axis, web security, and service validation wiring. |

## Data Fetching Differences

### GitHub version

- Pulls `stock.options` from Yahoo Finance.
- Iterates over every listed expiration.
- For each expiration, fetches calls and puts with `stock.option_chain(date)`.
- Adds `expirationDate`, `optionType`, snapshot timestamp, attempt count, and per-expiration quality ratios.
- Retries once when an expiration has a high zero bid/ask ratio and many placeholder IVs.
- Defers strike and DTE filtering to `prepare_options_data`.

### Local version

- Uses the new `time_utils` path to compute DTE before fetching an expiration.
- Only downloads expirations whose DTE is inside the requested range.
- Records all available expirations and selected expirations separately.
- Adds `expirationCloseUtc` and `expirationDteAtFetch` to each row.
- Uses a shared valuation timestamp so fetch filtering, cleaning, validation, and plotting agree on DTE.

Impact: local runs should download fewer option chains for narrow DTE requests, and diagnostics can now distinguish "not listed by Yahoo" from "filtered out by the request." The displayed/filtered DTE also no longer depends on a hard-coded `+21 hours` UTC close.

## Cleaning, Filtering, And IV Adjustment Differences

### DTE and time-to-expiration

GitHub computes option close as `expiration_utc + 21 hours`. That approximates 4pm ET during standard time but is one hour off during daylight-saving time.

Local computes expiration close as 4pm in `America/New_York`, then converts to UTC. This affects near-expiry DTE and year fraction, especially around daylight-saving periods.

### Forward and dividend adjustment

GitHub uses the configured dividend yield everywhere, so the forward is:

```text
F = S * exp((r - q) * T)
```

Local starts with the same fallback, but tries to estimate an expiry-level forward from put-call parity:

```text
F_candidate = K + (call_mid - put_mid) / exp(-rT)
q_implied = r - log(F / S) / T
```

The local parity estimate is used only when there are at least 3 call/put strike pairs, quotes are two-sided, strikes are positive, paired strikes are within `abs(log(K / S)) <= 0.15`, and the implied dividend yield is between `-25%` and `25%`. It uses the median of up to 7 closest-to-ATM forward candidates. Otherwise it falls back to the configured dividend yield.

Impact: local moneyness, put-call conversion, no-arbitrage bounds, IV inversion, and interpolation can all differ by expiration because they now use `forwardPrice` and `dividendYieldUsed`.

### Market price selection

Both versions prefer a midpoint when bid/ask are valid and fall back to last price or one-sided quotes. The local version adds extra treatment for wide quotes:

- Flags `wide_recompute_spread` when a midpoint quote is too wide for reliable recomputation.
- Checks whether a recent last trade sits inside or outside that wide quote.
- Flags `recent_trade_inside_wide_quote`, `last_trade_outside_quote`, or `stale_last_trade`.
- Flags one-sided bid/ask fallbacks with `one_sided_quote_price`.

### Recomputed Black-Scholes IV reliability

GitHub accepts a positive finite Black-Scholes IV as long as the selected market price is inside no-arbitrage bounds.

Local computes Black-Scholes IV only for `auto` and `black-scholes` modes, and then marks it usable only when:

- the option has no no-arbitrage violation,
- the quote is not stale,
- the quote is not one-sided,
- the recompute spread is not too wide,
- the bid/ask-implied-IV interval is available and not too wide,
- the price source is `mid` or `lastPrice`.

Missing two-sided quotes can still be used from a recent `lastPrice`, but those rows are flagged with `no_two_sided_quote` and `recomputed_iv_low_quote_support` and receive medium confidence. If the IV can be solved but fails the reliability gates, local adds `recomputed_iv_unreliable`; `black-scholes` mode then leaves the row without final IV.

### Provider IV behavior

GitHub `auto` uses provider IV only when quote sanity is clean, otherwise it falls back to Black-Scholes IV if possible.

Local `auto` now uses reliable Black-Scholes IV first. It falls back to valid provider IV only when the recomputed IV is unavailable or unreliable, and it adds `provider_used_without_quote_sanity` when quote or recompute checks are questionable.

Impact: local `auto` is more quote-derived than both the previous local behavior and the GitHub baseline. This matters because current Yahoo provider IV snapshots can contain repeated, coarse values that create deceptively smooth or symmetric surfaces.

### Confidence and inclusion

GitHub marks some missing two-sided quote cases as medium if liquidity is otherwise acceptable.

Local treats reliable recomputed IV from recent last prices as medium confidence when two-sided quote support is missing, treats `recomputed_iv_unreliable` as low, and adds recompute-spread and IV-width flags as medium/low drivers. Strict mode still includes only high-confidence rows; balanced and lenient include high and medium confidence rows.

## Surface Construction And Interpolation Differences

### Surface input source

This is the largest local surface difference.

GitHub `_surface_option_price` uses `marketPrice` first whenever it is present and positive. Only if market price is unavailable does it reprice from `impliedVolatilityFinal`.

Local `_surface_option_price` uses `impliedVolatilityFinal` first and reprices it through Black-Scholes-Merton. It falls back to `marketPrice` only if final IV is invalid.

Impact: in the GitHub version, `yfinance` mode can still build a surface from selected market prices rather than provider IVs. In the local version, the selected IV mode actually drives the surface:

- `black-scholes` mode reprices back to the quote used for inversion.
- `yfinance` mode reprices from provider IV.
- `auto` mode reprices from whichever final IV source was selected.

### Call/put unification

Both versions build one surface rather than separate call and put surfaces:

- Convert puts into call-equivalent prices using put-call parity.
- Prefer OTM puts below the forward and OTM calls above the forward.
- Treat a small ATM band as eligible for either side.
- Aggregate selected quote(s) at each strike/DTE with surface weights.

Local changes the forward/dividend values used by this conversion from configured constants to per-expiry estimates when possible.

### Static no-arbitrage projection

Both versions project each expiry slice to call-equivalent prices with:

- price bounds,
- monotone call-price constraints in strike,
- convex call-price constraints in strike,
- weighted least-squares objective solved by SLSQP.

Local keeps the same core method, but applies it with per-expiry `dividendYieldUsed` and stricter quote inputs.

### Smoothed interpolation grid

| Detail | GitHub `main` | Local version |
| --- | --- | --- |
| Log-forward-moneyness grid size | 41 | 81 |
| Minimum maturity slices for smooth grid | 2 | 3 |
| Minimum nodes per smooth slice | No explicit threshold beyond having at least 2 strikes | 3 |
| Minimum log-moneyness width per smooth slice | None | `0.01` |
| Raw mode behavior | Builds the smooth grid even if only scatter nodes are needed. | Skips smooth-grid construction when `smooth=False`. |
| DTE step | Always daily step of 1. | Chosen from requested/observed span to cap the dense grid near 80 DTE rows. |
| Last observed DTE | Could miss the observed maximum if the step did not land on it. | Ensures the observed maximum DTE is included. |
| Forward interpolation | Recomputes forward from configured `q`. | Interpolates observed forwards and derives dense-row dividend yields. |
| Calendar adjustment | Uses `maximum.accumulate` over total variance. | Same, but on the stricter, per-forward grid. |

Impact: the local smoothed surface is more stable and more precise, but may fall back to adjusted scatter nodes more often because it requires more maturity/strike support. It is also bounded against very dense DTE requests.

## Web And Deployment Differences

Local adds:

- `requirements.txt` with Flask, NumPy, pandas, Plotly, SciPy, and yfinance constraints.
- Vercel `api/index.py`, `vercel.json`, and `.vercelignore`.
- Security headers including CSP, frame protection, no-referrer, no-sniff, and HSTS.
- Query parameter clamping for strike and DTE sliders.
- `MAX_CONTENT_LENGTH = 1024`.
- Generic handling for unexpected surface-build failures.
- Requested/included DTE metric cards.

The web UI now exposes the same IV-source choices as the CLI: `auto`, `yfinance`, and `black-scholes`.

## Test Status

Ran locally with the bundled Codex Python runtime:

```text
C:\Users\Parsa\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m pytest
32 passed in 59.78s
```

The normal shell PATH did not have `pytest`, `python`, or `py`, so the bundled runtime was required.
