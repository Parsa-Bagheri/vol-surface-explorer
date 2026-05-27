# Local Mode Differences Report

Generated: 2026-05-26

The local code exposes two different three-choice mode families:

- IV source modes: `auto`, `yfinance`, `black-scholes`
- Quality modes: `strict`, `balanced`, `lenient`

The CLI exposes both families. The local web app exposes the `auto`, `yfinance`, and `black-scholes` IV source choices, hard-codes `quality_mode="lenient"`, and uses a separate `smooth` checkbox.

## IV Source Modes

These modes decide where `impliedVolatilityFinal` comes from. That final IV is the source of truth for the local surface construction.

| Mode | Uses provider IV? | Recomputes Black-Scholes IV? | Fallback behavior | Best use |
| --- | --- | --- | --- | --- |
| `auto` | Yes, only as fallback when quote-derived IV is unavailable or unreliable. | Yes. | Uses reliable recomputed IV first. If recomputed IV is unavailable or unreliable, can fall back to valid provider IV with quality flags. | Default exploration when you want the surface anchored to current/recent option prices while still retaining some coverage. |
| `yfinance` | Yes. | No recompute-specific IV is attached. | No Black-Scholes fallback. Invalid provider IV leaves final IV unavailable. | Inspecting Yahoo Finance's provider IV surface directly. |
| `black-scholes` | No. | Yes. | No provider fallback. If recomputed IV is unavailable, stale, or unreliable, final IV is unavailable. | Quote-implied surface based on selected market prices. |

### `auto`

`auto` is the CLI and web default. It now tries to use a reliable Black-Scholes IV from the selected option price before using Yahoo Finance's provider IV. This avoids anchoring the default surface to repeated or vendor-smoothed provider IV values when current/recent option prices can support a quote-derived IV.

That recomputed IV must pass the local reliability gates:

- no no-arbitrage violation,
- no stale last-trade problem,
- no one-sided quote fallback,
- no wide recompute spread,
- no wide bid/ask-implied-IV interval,
- price source is `mid` or `lastPrice`.

If the recomputed IV is not reliable and provider IV is finite, positive, above the placeholder threshold, and <= 5.0, `auto` falls back to provider IV and adds `provider_used_without_quote_sanity` when quote/recompute checks are questionable.

Practical effect: `auto` is no longer provider-first. It usually tracks `black-scholes` where there are enough usable recent prices, and only uses provider IV for coverage when quote-derived IV is not usable.

### `yfinance`

`yfinance` uses only Yahoo Finance's `impliedVolatility` field after basic validity checks:

- IV must be finite.
- IV must be greater than `1e-4`.
- IV must be <= `5.0`.

The local cleaner still computes market price source, spread ratio, liquidity flags, and confidence, but it does not attach `blackScholesImpliedVolatility`, bid IV, ask IV, or recompute-specific width flags in this mode.

Practical effect: this mode is useful for comparing against the provider surface. It may preserve provider quirks, call/put inconsistencies, stale values, or vendor smoothing that are not explained by current bid/ask quotes.

### `black-scholes`

`black-scholes` ignores provider IV and inverts Black-Scholes-Merton from the selected market price:

1. Prefer bid/ask midpoint when the quote is two-sided.
2. Otherwise fall back to `lastPrice` if it is recent enough.
3. Check no-arbitrage bounds using the local row's `dividendYieldUsed`.
4. Solve IV with Brent root finding between `1e-6` and `5.0`.
5. Accept the IV only if the recomputed value is reliable.

Practical effect: this mode is stricter and may produce fewer included rows. It is the cleanest choice when you want the surface to reflect currently selected quote prices rather than provider IV fields. Missing two-sided quotes can still be used from a recent `lastPrice`, but they are medium confidence and flagged as low quote support.

## Quality Modes

Quality modes decide how strict the row confidence and inclusion gates are. They affect flags, confidence, `includeInSurface`, and `surfaceWeight`.

| Quality mode | Spread max | Recompute spread max | IV bid/ask width max | IV width ratio max | Min volume | Min open interest | Surface inclusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `strict` | 0.50 | 0.20 | 0.06 | 0.35 | 10 | 50 | High-confidence rows only |
| `balanced` | 1.00 | 0.35 | 0.08 | 0.50 | 5 | 25 | High and medium confidence |
| `lenient` | 2.00 | 0.50 | 0.10 | 0.75 | 1 | 5 | High and medium confidence |

### `strict`

`strict` is the cleanest profile. It excludes medium-confidence rows from the surface and requires tighter liquidity/spread/IV-width behavior.

Use it when you care more about quote reliability than surface coverage.

### `balanced`

`balanced` allows medium-confidence rows but keeps tighter limits than lenient mode.

Use it when strict mode is too sparse but you still want meaningful quote-quality pressure.

### `lenient`

`lenient` allows the widest spreads and lowest liquidity thresholds. It still excludes low-confidence rows from the default surface, but medium-confidence rows remain eligible.

Use it when options chains are sparse or when the web app needs to produce a surface for common ticker/range requests. The local web app currently uses this mode.

## Related Local Toggles

### `smooth`

`smooth=False` renders adjusted surface nodes. These nodes are still static-arbitrage-adjusted per expiry slice.

`smooth=True` tries to build a dense surface by interpolating projected total variance over an overlapping log-forward-moneyness grid and DTE grid, then reprojecting for static no-arbitrage. If there are not enough stable slices, it falls back to adjusted nodes.

### `include_low_confidence`

By default, low-confidence rows stay in diagnostics but do not enter the surface fit.

When enabled, valid low-confidence rows are allowed into the visualizer's surface input. This can increase coverage, but it can also let stale, one-sided, or unreliable quote-derived points influence the surface.

## Practical Recommendations

- Use `black-scholes` plus `strict` for the cleanest quote-implied surface.
- Use `auto` plus `balanced` for a good research default in the CLI.
- Use `yfinance` when the goal is provider-IV comparison, not quote-derived IV.
- Use `lenient` for thin chains or web-app usability, then inspect diagnostics before trusting the shape.
