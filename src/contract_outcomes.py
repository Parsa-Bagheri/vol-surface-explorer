from collections import Counter

import numpy as np
import pandas as pd

from src.visualizer import select_surface_quotes


def _exclusion_reasons(rows):
    flags = rows.get("qualityFlags", pd.Series("none", index=rows.index)).fillna("none").astype(str)

    def has_flags(*names):
        return flags.str.contains(
            r"(?:^|;)(?:" + "|".join(names) + r")(?:;|$)", regex=True
        ).to_numpy(dtype=bool)

    prices = pd.to_numeric(
        rows.get("marketPrice", pd.Series(np.nan, index=rows.index)), errors="coerce"
    )
    ivs = pd.to_numeric(
        rows.get("impliedVolatilityFinal", pd.Series(np.nan, index=rows.index)),
        errors="coerce",
    )
    return pd.Series(
        np.select(
            [
                prices.isna() | (prices <= 0) | has_flags("stale_last_trade", "one_sided_quote_price"),
                has_flags("arbitrage_violation"),
                has_flags("wide_recompute_spread", "wide_iv_bid_ask", "iv_bid_ask_width_unavailable", "iv_outlier"),
                ivs.isna() | (ivs <= 0),
                has_flags("volume_zero_or_missing", "low_volume", "oi_zero_or_missing", "low_open_interest"),
            ],
            [
                "Missing usable price",
                "Price outside model bounds",
                "Quote too uncertain",
                "Calculation failed",
                "Insufficient liquidity",
            ],
            default="Calculation failed",
        ),
        index=rows.index,
    )


def exclusion_reason(row):
    return _exclusion_reasons(pd.DataFrame([row])).iloc[0]


def build_contract_outcomes(raw, cleaned, request, spot, selected_quotes=None):
    if selected_quotes is None:
        eligible = cleaned
        if not request.include_low_confidence:
            eligible = cleaned[cleaned["includeInSurface"].fillna(False).astype(bool)]
        selected_quotes = select_surface_quotes(
            eligible, spot, request.risk_free_rate, request.dividend_yield
        )
    used_ids = selected_quotes.get("contractId", pd.Series(dtype=int))
    outside = raw.loc[~raw["contractId"].isin(cleaned["contractId"])]
    strikes = pd.to_numeric(
        outside.get("strike", pd.Series(np.nan, index=outside.index)), errors="coerce"
    )
    expiries = pd.to_datetime(
        outside.get("expirationDate", pd.Series(pd.NaT, index=outside.index)),
        errors="coerce", utc=True,
    )
    missing_fields = strikes.isna() | (strikes <= 0) | expiries.isna()
    outside_reasons = pd.Series(
        np.where(missing_fields, "Missing contract fields", "Outside selected range"),
        index=outside.index,
    )
    excluded = cleaned.loc[~cleaned["contractId"].isin(used_ids)]
    excluded_reasons = _exclusion_reasons(excluded)
    alternative = excluded["includeInSurface"].fillna(False).astype(bool)
    excluded_reasons.loc[alternative] = "Alternative quote used"

    # Preserve first-occurrence ordering while assigning exactly one reason per contract.
    reasons = Counter(pd.concat([outside_reasons, excluded_reasons]).to_list())
    accepted_trade_count = int(
        selected_quotes.get("priceSourceUsed", pd.Series(dtype=str)).eq("lastPrice").sum()
    )
    used_count = len(selected_quotes)
    assert len(raw) == used_count + sum(reasons.values())
    return {
        "fetched": len(raw),
        "used": used_count,
        "excluded": len(raw) - used_count,
        "reasons": [{"label": label, "count": count} for label, count in reasons.items()],
        "accepted_trade_count": accepted_trade_count,
    }
