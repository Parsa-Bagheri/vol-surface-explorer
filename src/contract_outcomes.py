from collections import Counter

import pandas as pd

from src.visualizer import select_surface_quotes


def exclusion_reason(row):
    flags = set(str(row.get("qualityFlags", "none")).split(";"))
    price = row.get("marketPrice")
    if pd.isna(price) or price <= 0 or flags & {"stale_last_trade", "one_sided_quote_price"}:
        return "Missing usable price"
    if "arbitrage_violation" in flags:
        return "Price outside model bounds"
    if flags & {"wide_recompute_spread", "wide_iv_bid_ask", "iv_bid_ask_width_unavailable", "iv_outlier"}:
        return "Quote too uncertain"
    if pd.isna(row.get("impliedVolatilityFinal")) or row.get("impliedVolatilityFinal", 0) <= 0:
        return "Calculation failed"
    if flags & {"volume_zero_or_missing", "low_volume", "oi_zero_or_missing", "low_open_interest"}:
        return "Insufficient liquidity"
    return "Calculation failed"


def build_contract_outcomes(raw, cleaned, request, spot):
    eligible = cleaned[cleaned["includeInSurface"].fillna(False).astype(bool)]
    selected = select_surface_quotes(
        eligible, spot, request.risk_free_rate, request.dividend_yield
    )
    used_ids = set(selected.get("contractId", []))
    cleaned_ids = set(cleaned["contractId"])
    reasons = Counter()
    for _, row in raw.iterrows():
        if row["contractId"] in cleaned_ids:
            continue
        strike = row.get("strike")
        expiry = row.get("expirationDate")
        if pd.isna(strike) or strike <= 0 or pd.isna(expiry):
            reasons["Missing contract fields"] += 1
        else:
            reasons["Outside selected range"] += 1
    for _, row in cleaned.iterrows():
        if row["contractId"] in used_ids:
            continue
        reason = (
            "Alternative quote used"
            if bool(row["includeInSurface"])
            else exclusion_reason(row)
        )
        reasons[reason] += 1
    accepted_trade_count = sum(
        row.get("priceSourceUsed") == "lastPrice"
        for _, row in selected.iterrows()
    )
    assert len(raw) == len(selected) + sum(reasons.values())
    return {
        "fetched": len(raw),
        "used": len(selected),
        "excluded": len(raw) - len(selected),
        "reasons": [{"label": label, "count": count} for label, count in reasons.items()],
        "accepted_trade_count": accepted_trade_count,
    }
