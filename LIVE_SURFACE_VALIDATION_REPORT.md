# Live Surface Validation Report

Generated: 2026-05-26

## Question Checked

The concern was whether the displayed nodes and interpolated surface were directionally sound, or whether the app was fitting, optimizing, or inventing a weirdly symmetric placeholder surface.

## Finding

The app was not fabricating placeholder surface data. The weird symmetric shape came from Yahoo Finance provider `impliedVolatility` values in explicit `yfinance` mode and in the previous provider-first `auto` behavior.

Across multiple tickers, Yahoo's provider IV field returned a small set of repeated, coarse values such as `0.03125969` and `0.12500875`. Those values were especially implausible for higher-volatility names like NVDA and TSLA.

The local `auto` mode has been changed to prefer reliable quote-derived Black-Scholes IV first. It now uses provider IV only as fallback when quote-derived IV is unavailable or unreliable.

## Live Checks

All checks used:

- Strike range: 93% to 107% of spot
- DTE range: 7 to 60
- Quality mode: lenient
- Smooth surface: on
- Max last-trade age: 120 hours

### SPY

| Mode | Included rows | Source used | Final IV min | Final IV median | Final IV max | Rounded unique IVs |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `auto` | 1,569 | Black-Scholes | 0.066420 | 0.157205 | 0.418585 | 1,569 |
| `black-scholes` | 1,569 | Black-Scholes | 0.066422 | 0.157212 | 0.418599 | 1,568 |
| `yfinance` | 1,011 | Yahoo provider IV | 0.000498 | 0.031260 | 0.125009 | 9 |

Yahoo provider IV in the cleaned SPY range had only 10 rounded unique raw values. The most common were:

| Provider IV | Count |
| ---: | ---: |
| 0.00001000 | 705 |
| 0.03125969 | 379 |
| 0.06250938 | 244 |
| 0.01563484 | 204 |
| 0.00782242 | 98 |

### Other Tickers

| Ticker | Mode | Included rows | Source used | Final IV min | Final IV median | Final IV max | Rounded unique IVs |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| AAPL | `auto` | 169 | Black-Scholes | 0.173049 | 0.235501 | 0.397283 | 169 |
| AAPL | `black-scholes` | 169 | Black-Scholes | 0.173049 | 0.235501 | 0.397283 | 169 |
| AAPL | `yfinance` | 88 | Yahoo provider IV | 0.003916 | 0.031260 | 0.125009 | 6 |
| NVDA | `auto` | 160 | Black-Scholes | 0.340467 | 0.403346 | 0.507698 | 160 |
| NVDA | `black-scholes` | 160 | Black-Scholes | 0.340467 | 0.403346 | 0.507698 | 160 |
| NVDA | `yfinance` | 80 | Yahoo provider IV | 0.000987 | 0.031260 | 0.125009 | 8 |
| TSLA | `auto` | 222 | Black-Scholes | 0.258175 | 0.470608 | 0.549067 | 222 |
| TSLA | `black-scholes` | 222 | Black-Scholes | 0.258175 | 0.470608 | 0.549067 | 222 |
| TSLA | `yfinance` | 112 | Yahoo provider IV | 0.001963 | 0.031260 | 0.125009 | 7 |

## Soundness Interpretation

The quote-derived `auto` and `black-scholes` nodes are directionally plausible across the tested tickers:

- SPY produced a median IV around 15.7%.
- AAPL produced a median IV around 23.6%.
- NVDA produced a median IV around 40.3%.
- TSLA produced a median IV around 47.1%.

The explicit `yfinance` surfaces are not directionally reliable in this live snapshot. They cluster around repeated provider IV values and should be treated as provider-data inspection, not the default research surface.

## Construction And Interpolation

The corrected `auto` path now feeds quote-derived IV nodes into the same construction pipeline:

1. Select a market price from midpoint when available, otherwise from recent `lastPrice`.
2. Reject stale last prices beyond `max_trade_age_hours`.
3. Invert Black-Scholes-Merton IV from the selected price.
4. Convert puts to call-equivalent prices and prefer OTM quotes around the forward.
5. Aggregate one weighted node per strike and DTE.
6. Project each expiry slice to monotone and convex call prices.
7. Interpolate total variance over log-forward moneyness and DTE.
8. Reproject the smoothed grid for discrete static no-arbitrage constraints.

The interpolation is therefore connecting the selected nodes; it is not making up a symmetric surface. When explicit `yfinance` mode looks symmetric, the symmetry is already present in the provider IV inputs.

## Important Data Timing Note

The live checks ran early on Tuesday, 2026-05-26 UTC, after the Monday, 2026-05-25 US market holiday. Many recent option `lastTradeDate` values were from the prior Friday close. The default last-trade threshold was changed from 72 hours to 120 hours so the app does not incorrectly discard the prior trading session after a long weekend, while still excluding genuinely stale last-price rows.
