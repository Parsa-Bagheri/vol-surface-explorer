# vol-surface-project
Constructing and analyzing IV surfaces for equity options.

## Example Visualization
<img width="1865" height="983" alt="image" src="https://github.com/user-attachments/assets/f427e8f6-4cf4-49c9-8f57-0bd7ff08668b" />

## Features
- Fetch options data from Yahoo Finance with per-expiration snapshot metadata.
- Compute IV using:
  - `auto` (default): quote-derived IV first, provider IV fallback
  - `yfinance`: provider IV comparison
  - `black-scholes`: quote-derived IV only
- Use the dividend-adjusted Black-Scholes-Merton formula for repricing and IV inversion.
- Attach quality/confidence flags per contract:
  - `placeholder_iv`
  - `no_two_sided_quote`
  - `stale_last_trade`
  - `arbitrage_violation`
  - `oi_zero_or_missing`
  - `low_open_interest`
  - `volume_zero_or_missing`
  - `low_volume`
  - `wide_spread`
  - `wide_recompute_spread`
  - `wide_iv_bid_ask`
  - `recomputed_iv_unreliable`
  - `provider_quote_iv_disagreement`
  - `recent_trade_inside_wide_quote`
  - `last_trade_outside_quote`
  - `one_sided_quote_price`
- Build one unified surface from call/put quotes after put-call-parity conversion, with an OTM quote preference at each strike/maturity node.
- Downweight illiquid contracts and project the fitted surface to satisfy discrete static no-arbitrage constraints.
- Emit diagnostics (internal validation, static-arbitrage certification scope, optional external benchmark comparison).

## Usage

### Web App
```bash
python web_app.py
```

Then open `http://127.0.0.1:5000` and use the UI to:
- enter a ticker
- move strike-range sliders in percent of spot
- move DTE min/max sliders
- switch between quote-derived IV and provider-IV comparison

The web UI is built with `Flask` plus server-rendered HTML/CSS/JS so it stays lightweight and reuses the same backend pipeline as the CLI.

### Basic Usage (Auto IV + Quality Metadata)
```bash
python main.py TICKER
```

### Smoothed Surface
```bash
python main.py TICKER --smooth
```

### Advanced Example
```bash
python main.py TICKER --smooth --dte_max 60 --quality_mode lenient --diagnostics_json diagnostics.json
```

## CLI Options
- `--strike_min_pct`: Min strike as percent of spot. Default: `0.93`.
- `--strike_max_pct`: Max strike as percent of spot. Default: `1.07`.
- `--dte_max`: Max DTE in days. Default: `60`.
- `--dte_min`: Min DTE in days. Default: `1`.
- `--smooth`: Show the interpolated adjusted grid instead of adjusted surface nodes only.
- `--output_dir`: Output directory for HTML file.
- `--iv_source {auto,yfinance,black-scholes}`: IV mode. `auto` is quote-first. Default: `auto`.
- `--risk_free_rate`: Annualized risk-free rate for Black-Scholes. Default: `0.02`.
- `--dividend_yield`: Continuous dividend yield for Black-Scholes. Default: `0.0`.
- `--quality_mode {strict,balanced,lenient}`: Surface-inclusion confidence profile. Default: `lenient`.
- `--max_trade_age_hours`: Last-trade recency threshold for fallback pricing. Default: `72`.
- `--diagnostics_json`: Optional path for diagnostics JSON output.
- `--benchmark_csv`: Optional CSV for external IV comparison.
- `--include_low_confidence`: Include low-confidence rows in the surface fit instead of using the default confidence filter.

## Diagnostics
Each run computes:
- Flag counts and exclusion reasons
- IV source usage split, quote-derived fraction, and provider fraction
- Per-DTE confidence summary
- Observed and surface-included strike ranges, so mode-specific quality filtering is visible
- DTE coverage: requested window, fetched listed expirations, surface-included expirations, and next available expiry after the requested max
- Internal repricing validation (MAE/RMSE against selected market prices)
- Static-arbitrage diagnostics under `diagnostics["static_arbitrage"]`

Static-arbitrage diagnostics include call price bounds, strike monotonicity, convexity / butterfly checks, total-variance calendar checks, projection error versus original call-equivalent mids, bid/ask interval violations, and a certification scope of `grid`, `nodes_only`, or `not_certified`.

If `--benchmark_csv` is provided, additional external metrics are produced:
- Matched rows
- MAE / RMSE in vol points
- Per-option-type error summary

### Benchmark CSV Format
CSV must include:
- `optionType` (`call` or `put`)
- `strike`
- `days_to_expiration`
- `iv_ref`

## Notes on Surface Accuracy
- This is a research-grade visualizer, not a pricing-grade engine or executable arbitrage detector.
- US equity options are represented with European-equivalent Black-Scholes-Merton IV. American exercise and discrete dividend modeling are intentionally out of scope for this pass.
- `auto` mode suppresses provider IV placeholders (for example `0.00001`) and prefers reliable quote-derived IV over provider IV. Provider IV remains available as fallback metadata and as `yfinance` comparison mode.
- Quote-derived IV is only used when the selected option price is reliable enough to define a point estimate. Very wide bid/ask quotes, one-sided quotes, and wide bid/ask-implied-IV intervals are retained for diagnostics but excluded from the quote-derived surface fit.
- Expiration times are evaluated at the standard 4pm New York option close, including daylight-saving transitions. The requested DTE range controls the plot axis and fetch filter, but the app does not invent expirations; if XLY only has listed expirations at 12, 19, 26, 33, and 39 DTE inside a 7-60 request, those are the only maturities with plotted quotes.
- The Yahoo Finance provider IV fields for calls and puts at the same strike and expiry are not guaranteed to be parity-consistent, so the plotted surface is not built by linearly joining raw provider-IV points.
- The forward used for moneyness, put-call conversion, and recomputed IV is estimated by expiry from put-call parity when enough paired quotes are available; otherwise it falls back to the configured dividend yield.
- The selected IV source drives the surface input. In `black-scholes` mode, IV is inverted from the selected quote price and repriced through Black-Scholes-Merton before projection. In `yfinance` mode, the provider IV is repriced through the same model before projection, so provider and quote-derived surfaces can be compared directly.
- Provider-comparison mode uses the same quote-support gate for surface inclusion as quote-derived mode. This keeps strike coverage comparable across modes instead of allowing provider IV to extend into strikes where quote-derived IV is not reliable.
- The implementation is not an SVI calibration. It uses direct static no-arbitrage projection on call-equivalent prices and interpolated total variance.
- The surface uses a single smile/surface, not separate call and put surfaces. At each node the fit prefers out-of-the-money puts below the forward and out-of-the-money calls above the forward, which is standard market practice because those quotes are usually more liquid and less distorted.
- In node mode, adjusted surface nodes are checked for discrete price bounds, strike monotonicity, convexity, and any available overlapping-moneyness calendar constraints.
- In smoothed mode, the rendered interpolated grid is projected and checked on the construction grid: call-equivalent price slices are enforced to be monotone and convex in strike (no butterfly arbitrage), and total variance is enforced to be non-decreasing in maturity on an overlapping log-forward-moneyness grid (calendar-spread control).
- The app reports only discrete certification on generated nodes or grid points. It does not claim continuous arbitrage freedom between or outside those points.
- If there are not enough stable nodes to build the smoothed grid, the app falls back to rendering the adjusted raw surface nodes and reports `nodes_only` certification when those checks pass.
- Low-confidence rows are retained for diagnostics but excluded from surfaces by default.
- If the requested DTE maximum is not represented in the plot, check `diagnostics["dte_coverage"]`. The app only plots listed expirations returned by the provider unless an interpolated grid can be built between listed expirations; it does not invent maturities beyond the last fetched expiry.

## Literature Basis
- Static arbitrage is treated as butterfly arbitrage plus calendar-spread arbitrage, following Gatheral and Jacquier's arbitrage-free SVI framework.
- Butterfly checks use the Breeden-Litzenberger price-space implication that call prices should be decreasing and convex in strike.
- The direct shape-constrained projection approach is consistent with Fengler's arbitrage-free smoothing literature.
- The OTM put/call quote preference is consistent with common volatility-index construction practice, including Cboe VIX methodology.
