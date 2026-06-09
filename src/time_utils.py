from __future__ import annotations

from datetime import datetime, time
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd


OPTION_EXCHANGE_TZ = ZoneInfo("America/New_York")
OPTION_EXPIRATION_CLOSE = time(16, 0)


def normalize_as_of_utc(as_of_utc: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """Return a timezone-aware UTC valuation timestamp."""
    if as_of_utc is None:
        return pd.Timestamp.now(tz="UTC")

    timestamp = pd.Timestamp(as_of_utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def expiration_close_utc(expiration_date) -> pd.Timestamp:
    """Return the standard 4pm New York option-expiration timestamp in UTC."""
    expiration = pd.to_datetime(expiration_date, errors="coerce")
    if pd.isna(expiration):
        return pd.NaT

    expiration_day = expiration.date()
    local_close = pd.Timestamp(
        datetime.combine(expiration_day, OPTION_EXPIRATION_CLOSE),
        tz=OPTION_EXCHANGE_TZ,
    )
    return local_close.tz_convert("UTC")


def expiration_dte(
    expiration_date,
    as_of_utc: Optional[pd.Timestamp] = None,
) -> Optional[int]:
    """Compute exchange-calendar DTE while excluding contracts past their close."""
    valuation_time = normalize_as_of_utc(as_of_utc)
    expiration_close = expiration_close_utc(expiration_date)
    if pd.isna(expiration_close):
        return None

    if expiration_close <= valuation_time:
        return -1

    valuation_day = valuation_time.tz_convert(OPTION_EXCHANGE_TZ).date()
    expiration_day = expiration_close.tz_convert(OPTION_EXCHANGE_TZ).date()
    return int((expiration_day - valuation_day).days)
