import re


TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,11}$")


def validate_ticker_symbol(ticker: str) -> str:
    normalized = (ticker or "").strip().upper()
    if normalized in {"VIX", "^VIX"}:
        raise ValueError(
            "VIX options exist, but this app does not yet support them. "
            "They require expiry-specific VIX forward prices and VIX settlement "
            "timing; using the spot VIX with the stock-option model would produce "
            "misleading implied volatility."
        )
    if not normalized:
        raise ValueError("A ticker symbol is required.")
    if not TICKER_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Ticker symbols may only contain letters, numbers, dots, and hyphens, "
            "and must be 12 characters or fewer."
        )
    return normalized
