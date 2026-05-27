# IV Mode Soundness Report

Generated: 2026-05-26

## Short Answer

The most mathematically sound mode is **`black-scholes`**.

The best practical default is usually **`auto`**, because it now uses the same quote-derived Black-Scholes IV first and only falls back to Yahoo provider IV when the quote-derived IV cannot be trusted or cannot be computed.

The least sound mode for a research surface is **`yfinance`**. It is useful for inspecting Yahoo Finance's provider IV field, but it should not be treated as the most accurate surface input.

## Important Definition

"Accurate" and "mathematically sound" are not identical.

- **Accuracy** would require an external ground truth, such as exchange-grade option quotes, a professional vendor IV surface, or observed tradable mid/bid/ask data at the same timestamp.
- **Mathematical soundness** means the app's inputs are transparent, internally consistent, tied to observed option prices, and processed through defensible no-arbitrage constraints.

By that standard, `black-scholes` is the cleanest mode because every retained node is derived from the selected option price and then passed through the same deterministic surface construction.

## What All Three Modes Share

All three modes use the same broad pipeline after `impliedVolatilityFinal` is chosen:

1. Fetch option chains and spot from Yahoo Finance.
2. Filter by requested strike and DTE.
3. Estimate expiry-level forwards/dividend yields from put-call parity when enough paired quotes exist.
4. Choose one final IV per row according to the selected mode.
5. Reprice that final IV through Black-Scholes-Merton to get a surface option price.
6. Convert puts to call-equivalent prices.
7. Prefer OTM puts below the forward and OTM calls above the forward.
8. Aggregate to one weighted node per strike and DTE.
9. Project each expiry slice to satisfy discrete call-price bounds, monotonicity, and convexity.
10. If smoothed, interpolate total variance over log-forward moneyness and DTE.
11. Reproject the smoothed grid for discrete static no-arbitrage constraints.

So the key difference is not the interpolation engine. The key difference is the source and reliability of the IV nodes fed into that engine.

## Mode 1: `black-scholes`

### What It Does

`black-scholes` ignores Yahoo's provider IV field. It computes IV by inverting Black-Scholes-Merton from the selected option price:

- Prefer a two-sided bid/ask midpoint.
- If there is no usable two-sided quote, use a recent `lastPrice`.
- Reject stale last prices.
- Reject arbitrage-bound violations.
- Reject one-sided quote fallbacks.
- Reject too-wide recompute spreads.
- Reject too-wide bid/ask-implied-IV intervals.
- Accept only finite positive IV below the maximum reasonable IV cap.

### Why It Is Most Sound

This mode is most mathematically sound because the surface is anchored to observed option prices rather than an opaque vendor IV field. The transformation is explainable:

```text
observed option price -> no-arbitrage bound check -> BSM IV inversion -> BSM repricing -> surface node
```

That chain is internally consistent. If a row cannot pass the reliability gates, `black-scholes` does not silently replace it with provider IV. It leaves the row without final IV and excludes it from the surface fit.

### Weaknesses

`black-scholes` is not magic. It can still be wrong if:

- Yahoo's quote data is stale or bad.
- `lastPrice` is not representative of the current market.
- Bid/ask spreads are wide but still pass the selected quality mode.
- The configured risk-free rate or dividend assumptions are off.
- The Black-Scholes-Merton assumptions are too simple for the product or market condition.

But those weaknesses are visible and bounded by the app's flags. That makes the mode auditable.

## Mode 2: `auto`

### What It Does

`auto` now tries to use reliable Black-Scholes IV first. If that fails, it can fall back to Yahoo provider IV when the provider IV is finite, above the placeholder threshold, and not out of range.

In practice:

```text
reliable quote-derived IV -> use Black-Scholes IV
otherwise valid provider IV -> use Yahoo IV with quality flags
otherwise -> no final IV
```

### Why It Is A Good Default

`auto` is the best default for usability because it often matches `black-scholes` when quote-derived IV is available, while preserving some coverage when option quotes are too sparse to invert safely.

For liquid names and normal ranges, current live checks showed `auto` matching `black-scholes` closely because every included node came from Black-Scholes IV.

### Why It Is Not The Most Sound

`auto` is slightly less mathematically pure than `black-scholes` because it can mix two different IV regimes:

- Quote-derived IV from observed prices.
- Provider IV from Yahoo's opaque `impliedVolatility` field.

That fallback improves coverage, but it also means a surface can contain some nodes whose source is less transparent. If provider fallback rows appear, the surface is still constructed consistently afterward, but the input nodes are less defensible.

## Mode 3: `yfinance`

### What It Does

`yfinance` uses Yahoo Finance's `impliedVolatility` field directly after basic validity checks:

- IV must be finite.
- IV must be greater than the placeholder threshold.
- IV must be no greater than the maximum reasonable IV cap.

It does not compute Black-Scholes IV, bid IV, ask IV, or recompute-specific width flags in this mode.

### Why It Is Least Sound

This mode is least mathematically sound because the provider IV is opaque. The app does not know:

- Whether Yahoo's IV was computed from current bid/ask, last trade, or stale data.
- Whether Yahoo smoothed or rounded the values.
- Whether calls and puts are parity-consistent.
- Whether the same assumptions for rate, dividend, and timestamp were used.

Recent live checks also showed that Yahoo provider IVs could be highly repeated and implausibly low across SPY, AAPL, NVDA, and TSLA. For example, `yfinance` mode produced only a handful of rounded unique IV values in several tested surfaces, while quote-derived modes produced a distinct IV at nearly every included row.

That does not mean `yfinance` is useless. It means it is a provider-data inspection mode, not the mode I would choose for the most defensible surface.

## Ranking

| Rank | Mode | Soundness | Why |
| ---: | --- | --- | --- |
| 1 | `black-scholes` | Highest | Fully quote-derived, transparent, reliability-gated, no provider fallback. |
| 2 | `auto` | High practical soundness | Uses `black-scholes` first, but can mix in provider IV fallback rows. |
| 3 | `yfinance` | Lowest | Depends on opaque provider IV values and can preserve stale, rounded, or vendor-smoothed artifacts. |

## Final Recommendation

Use **`black-scholes`** when you want the most mathematically sound surface.

Use **`auto`** when you want the best practical default and are comfortable with provider fallback rows when quote-derived IV is unavailable.

Use **`yfinance`** only when your goal is to inspect or compare Yahoo's provider IV field directly.

For the strongest workflow:

1. Start with `black-scholes`.
2. Check whether enough surface rows are included.
3. If coverage is too sparse, switch to `auto`.
4. Use `yfinance` as a diagnostic comparison, not as the primary answer.

## Bottom Line

If I had to pick one mode as the most accurate/mathematically sound, I would pick:

```text
black-scholes
```

The reason is not that Black-Scholes is a perfect model. It is that this mode has the clearest provenance, the most auditable reliability gates, the least opaque vendor dependence, and the most internally consistent relationship between option prices, IV nodes, and the final no-arbitrage-adjusted surface.
