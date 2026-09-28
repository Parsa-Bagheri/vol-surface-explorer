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
    """Return calendar DTE in New York, independently of time remaining."""
    value = expiration_dtes(pd.Series([expiration_date]), as_of_utc).iloc[0]
    return None if pd.isna(value) else int(value)


def expiration_dtes(
    expiration_dates: pd.Series,
    as_of_utc: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """Use the same exchange-calendar date for fetching and cleaning."""
    valuation_day = (
        normalize_as_of_utc(as_of_utc)
        .tz_convert(OPTION_EXCHANGE_TZ)
        .tz_localize(None)
        .normalize()
    )
    # Expiration dates are date labels, not instants to convert to New York.
    expiration_days = (
        pd.to_datetime(expiration_dates, errors="coerce", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    return (expiration_days - valuation_day).dt.days.astype("Int64")


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
