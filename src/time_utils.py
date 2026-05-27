from __future__ import annotations

from datetime import datetime, time
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
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


def expiration_dte(expiration_date, as_of_utc: Optional[pd.Timestamp] = None) -> Optional[int]:
    """Compute displayed DTE as ceil calendar days to 4pm New York expiration."""
    valuation_time = normalize_as_of_utc(as_of_utc)
    expiration_close = expiration_close_utc(expiration_date)
    if pd.isna(expiration_close):
        return None

    remaining_days = (
        expiration_close - valuation_time
    ).total_seconds() / (24.0 * 60.0 * 60.0)
    if not np.isfinite(remaining_days):
        return None
    return int(np.ceil(remaining_days))


def time_to_expiration_years(
    expiration_date,
    as_of_utc: Optional[pd.Timestamp] = None,
) -> float:
    valuation_time = normalize_as_of_utc(as_of_utc)
    expiration_close = expiration_close_utc(expiration_date)
    if pd.isna(expiration_close):
        return float("nan")

    remaining_years = (
        expiration_close - valuation_time
    ).total_seconds() / (365.25 * 24.0 * 60.0 * 60.0)
    if not np.isfinite(remaining_years):
        return float("nan")
    return float(remaining_years)
