# vol-surface-project
Constructing and analyzing IV surfaces for equity options.

## Features
- Fetch options data from Yahoo Finance with per-expiration snapshot metadata.
- Compute IV using:
  - `auto` (default): provider IV with Black-Scholes fallback
  - `yfinance`: provider IV only
  - `black-scholes`: recomputed IV only
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
- Build one unified surface from call/put quotes after put-call-parity conversion, with an OTM quote preference at each strike/maturity node.
- Downweight illiquid contracts and project the fitted surface to satisfy discrete static no-arbitrage constraints.
- Emit diagnostics (internal validation + optional external benchmark comparison).

## Usage

### Web App
```bash
python web_app.py
```

Then open `http://127.0.0.1:5000` and use the UI to:
- enter a ticker
- move strike-range sliders in percent of spot
- move DTE min/max sliders
- switch between `yfinance` IV and recomputed `black-scholes` IV

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
- `--smooth`: Build a smoothed unified surface.
- `--output_dir`: Output directory for HTML file.
- `--iv_source {auto,yfinance,black-scholes}`: IV mode. Default: `auto`.
- `--risk_free_rate`: Annualized risk-free rate for Black-Scholes. Default: `0.02`.
- `--dividend_yield`: Continuous dividend yield for Black-Scholes. Default: `0.0`.
- `--quality_mode {strict,balanced,lenient}`: Surface-inclusion confidence profile. Default: `lenient`.
- `--max_trade_age_hours`: Last-trade recency threshold for fallback pricing. Default: `72`.
- `--diagnostics_json`: Optional path for diagnostics JSON output.
- `--benchmark_csv`: Optional CSV for external IV comparison.
- `--include_low_confidence`: Keep low-confidence rows in diagnostics output, but not in the arbitrage-free surface fit.

## Diagnostics
Each run computes:
- Flag counts and exclusion reasons
- IV source usage split and fallback fraction
- Per-DTE confidence summary
- Internal repricing validation (MAE/RMSE against selected market prices)

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
- `auto` mode suppresses provider IV placeholders (for example `0.00001`) by recomputing IV from prices when possible.
- The Yahoo Finance provider IV fields for calls and puts at the same strike and expiry are not guaranteed to be parity-consistent, so the plotted surface is not built by linearly joining raw provider-IV points.
- The surface uses a single smile/surface, not separate call and put surfaces. At each node the fit prefers out-of-the-money puts below the forward and out-of-the-money calls above the forward, which is standard market practice because those quotes are usually more liquid and less distorted.
- In smoothed mode, the rendered surface is projected to be free of discrete static arbitrage: call-equivalent price slices are enforced to be monotone and convex in strike (no butterfly arbitrage), and total variance is enforced to be non-decreasing in maturity on an overlapping log-forward-moneyness grid (no calendar arbitrage).
- Low-confidence rows are retained for diagnostics but excluded from surfaces by default.
