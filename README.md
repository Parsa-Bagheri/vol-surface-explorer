# vol-surface-project
Constructing and analyzing IV surfaces for equity options.

## Example Visualization
<img width="1865" height="983" alt="image" src="https://github.com/user-attachments/assets/f427e8f6-4cf4-49c9-8f57-0bd7ff08668b" />

## Features
- Fetch options data from Yahoo Finance with per-expiration snapshot metadata.
- Compute IV by inverting Black-Scholes-Merton from the selected quote price.
- Use the dividend-adjusted Black-Scholes-Merton formula for repricing and IV inversion.
- Attach quality/confidence flags per contract:
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
  - `recent_trade_inside_wide_quote`
  - `last_trade_outside_quote`
- Build one unified surface from call/put quotes after put-call-parity conversion, with an OTM quote preference at each strike/maturity node.
- Downweight illiquid contracts and project the fitted surface to satisfy discrete static no-arbitrage constraints.
- Emit diagnostics (internal validation + optional external benchmark comparison).

## Usage

### Recommended Setup
Create a local virtual environment and install the runtime dependencies from `requirements.txt`:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Then install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The `environment.yml` file is available if you prefer Conda or want a broader local data-science environment with Jupyter.

### Web App
```bash
python web_app.py
```

Then open `http://127.0.0.1:5000` and use the UI to:
- enter a ticker
- move strike-range sliders in percent of spot
- move DTE min/max sliders

The web UI is built with `Flask` plus server-rendered HTML/CSS/JS so it stays lightweight and reuses the same backend pipeline as the CLI.

### Vercel Deployment

The repository includes `vercel.json`, `.vercelignore`, `api/index.py`, and `requirements.txt` for deploying the Flask app on Vercel. The deployment entrypoint imports the shared Flask app, while `.vercelignore` keeps local caches, tests, and development-only files out of the uploaded deployment bundle.

### Basic Usage
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
- `--risk_free_rate`: Annualized risk-free rate for Black-Scholes. Default: `0.02`.
- `--dividend_yield`: Continuous dividend yield for Black-Scholes. Default: `0.0`.
- `--quality_mode {strict,balanced,lenient}`: Surface-inclusion confidence profile. Default: `lenient`.
- `--max_trade_age_hours`: Last-trade recency threshold for fallback pricing. Default: `120`.
- `--diagnostics_json`: Optional path for diagnostics JSON output.
- `--benchmark_csv`: Optional CSV for external IV comparison.
- `--include_low_confidence`: Include low-confidence rows in the surface fit instead of using the default confidence filter.

## Diagnostics
Each run computes:
- Flag counts and exclusion reasons
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
- Black-Scholes-Merton IV is inverted from the selected option price when the quote is reliable enough to define a point estimate. Very wide bid/ask quotes, one-sided quotes, stale last prices, and wide bid/ask-implied-IV intervals are retained for diagnostics but excluded from the surface fit.
- Expiration times are evaluated at the standard 4pm New York option close, including daylight-saving transitions. The requested DTE range controls the plot axis and fetch filter, but the app does not invent expirations; if XLY only has listed expirations at 12, 19, 26, 33, and 39 DTE inside a 7-60 request, those are the only maturities with plotted quotes.
- The forward used for moneyness, put-call conversion, and recomputed IV is estimated by expiry from put-call parity when enough paired quotes are available; otherwise it falls back to the configured dividend yield.
- The Black-Scholes-Merton IV drives the surface input: IV is inverted from the selected quote price and repriced through the same model before projection.
- The implementation is not an SVI calibration. It uses direct static no-arbitrage projection on call-equivalent prices and interpolated total variance.
- The surface uses a single smile/surface, not separate call and put surfaces. At each node the fit prefers out-of-the-money puts below the forward and out-of-the-money calls above the forward, which is standard market practice because those quotes are usually more liquid and less distorted.
- In smoothed mode, the rendered surface is projected to remove discrete static arbitrage on the construction grid: call-equivalent price slices are enforced to be monotone and convex in strike (no butterfly arbitrage), and total variance is enforced to be non-decreasing in maturity on an overlapping log-forward-moneyness grid (calendar-spread control).
- If there are not enough stable nodes to build the smoothed surface, the app falls back to rendering the adjusted raw surface nodes.
- Low-confidence rows are retained for diagnostics but excluded from surfaces by default.
