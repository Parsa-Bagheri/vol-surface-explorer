# vol-surface-project
Constructing and analyzing IV surfaces for equity options.

## Example Visualization
<img width="1865" height="983" alt="image" src="https://github.com/user-attachments/assets/f427e8f6-4cf4-49c9-8f57-0bd7ff08668b" />

## Features
- Fetch filtered Yahoo Finance option chains concurrently by expiration.
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
- Emit contract-count and exclusion diagnostics.

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

The `environment.yml` file is available if you prefer Conda.

### Web App
```bash
python web_app.py
```

Then open `http://127.0.0.1:5000` and use the UI to:
- enter a ticker
- move strike-range sliders in percent of spot
- move DTE min/max sliders

The web UI is built with `Flask` plus server-rendered HTML/CSS/JS so it stays lightweight and reuses the shared backend pipeline.

### Vercel Deployment

The repository includes `vercel.json`, `.vercelignore`, `api/index.py`, and `requirements.txt` for deploying the Flask app on Vercel. The deployment entrypoint imports the shared Flask app, while `.vercelignore` keeps local caches, tests, and development-only files out of the uploaded deployment bundle.

## Diagnostics
Each run computes:
- Retained, surface-included, and excluded contract counts
- One primary exclusion reason per excluded contract

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
