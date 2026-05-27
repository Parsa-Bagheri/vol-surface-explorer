from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import tempfile
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.time_utils import expiration_close_utc, expiration_dte, normalize_as_of_utc


MAX_OPTION_FETCH_WORKERS = 12


def _configure_yfinance_cache() -> None:
    configured_dir = os.environ.get("YFINANCE_CACHE_DIR")
    candidate_dirs = [configured_dir] if configured_dir else [
        os.path.join(os.getcwd(), ".cache", "yfinance"),
        os.path.join(tempfile.gettempdir(), "vol-surface-explorer-yfinance"),
    ]

    last_error = None
    for cache_dir in candidate_dirs:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            yf.set_tz_cache_location(cache_dir)
            return
        except Exception as error:
            last_error = error

    print(f"Warning: Could not configure yfinance cache: {last_error}")


_configure_yfinance_cache()


def _positive_float(value) -> Optional[float]:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if np.isfinite(converted) and converted > 0:
        return converted
    return None


def _latest_close_from_history(stock, ticker_symbol: str) -> Optional[float]:
    for period in ("1d", "5d", "1mo"):
        try:
            history = stock.history(period=period)
        except Exception as error:
            print(f"Could not fetch {period} price history for {ticker_symbol}: {error}")
            continue
        if history is None or history.empty or "Close" not in history.columns:
            continue

        closes = pd.to_numeric(history["Close"], errors="coerce").dropna()
        closes = closes[closes > 0]
        if not closes.empty:
            return float(closes.iloc[-1])
    return None


def _price_from_fast_info(stock) -> Optional[float]:
    try:
        fast_info = stock.fast_info
    except Exception:
        return None
    if fast_info is None:
        return None

    for key in (
        "last_price",
        "regular_market_price",
        "previous_close",
        "last_close",
    ):
        try:
            price = _positive_float(fast_info.get(key))
        except Exception:
            continue
        if price is not None:
            return price
    return None


def _price_from_info(stock) -> Optional[float]:
    try:
        info = stock.info
    except Exception:
        return None
    if not isinstance(info, dict):
        return None

    for key in (
        "currentPrice",
        "regularMarketPrice",
        "previousClose",
        "regularMarketPreviousClose",
        "open",
    ):
        price = _positive_float(info.get(key))
        if price is not None:
            return price
    return None


def get_current_price(ticker_symbol: str) -> Optional[float]:
    """Fetches the current market price for a given stock ticker."""
    stock = yf.Ticker(ticker_symbol)
    current_price = (
        _latest_close_from_history(stock, ticker_symbol)
        or _price_from_fast_info(stock)
        or _price_from_info(stock)
    )
    if current_price is None:
        print(f"Could not determine current price for {ticker_symbol} from available data.")
        return None
    return float(current_price)


def _compute_chain_health(
    chain_df: pd.DataFrame,
    poor_quality_zero_quote_ratio: float,
) -> Dict[str, float]:
    """Compute lightweight per-expiration quality metrics."""
    if chain_df is None or chain_df.empty:
        return {
            "contracts": 0,
            "zero_bid_ask_ratio": 1.0,
            "zero_open_interest_ratio": 1.0,
            "poor_quality": True,
        }

    bid = pd.to_numeric(chain_df.get("bid"), errors="coerce")
    ask = pd.to_numeric(chain_df.get("ask"), errors="coerce")
    oi = pd.to_numeric(chain_df.get("openInterest"), errors="coerce")

    zero_quote_mask = bid.isna() | ask.isna() | (bid <= 0) | (ask <= 0) | (ask < bid)
    zero_oi_mask = oi.fillna(0) <= 0

    contracts = float(len(chain_df))
    zero_bid_ask_ratio = float(zero_quote_mask.sum() / contracts)
    zero_open_interest_ratio = float(zero_oi_mask.sum() / contracts)

    poor_quality = bool(zero_bid_ask_ratio >= poor_quality_zero_quote_ratio)

    return {
        "contracts": int(contracts),
        "zero_bid_ask_ratio": zero_bid_ask_ratio,
        "zero_open_interest_ratio": zero_open_interest_ratio,
        "poor_quality": poor_quality,
    }


def _expiration_dte(expiration_date: str, now_utc: pd.Timestamp) -> Optional[int]:
    return expiration_dte(expiration_date, now_utc)


def _fetch_expiration_chain(
    ticker_symbol: str,
    expiration_date: str,
    retry_on_poor_quality: bool,
    max_fetch_attempts: int,
    poor_quality_zero_quote_ratio: float,
    retry_wait_seconds: float,
) -> tuple[str, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    stock = yf.Ticker(ticker_symbol)
    chosen_calls = pd.DataFrame()
    chosen_puts = pd.DataFrame()
    chosen_health = None

    for attempt in range(1, max_fetch_attempts + 1):
        snapshot_timestamp = pd.Timestamp.now(tz="UTC")
        options_chain = stock.option_chain(expiration_date)
        calls = options_chain.calls.copy()
        puts = options_chain.puts.copy()

        chain_for_health = pd.concat([calls, puts], ignore_index=True)
        health = _compute_chain_health(chain_for_health, poor_quality_zero_quote_ratio)
        health["expiration"] = expiration_date
        health["attempt"] = attempt
        health["snapshot_timestamp_utc"] = snapshot_timestamp.isoformat()

        should_retry = bool(
            retry_on_poor_quality
            and health["poor_quality"]
            and attempt < max_fetch_attempts
        )
        if should_retry:
            time.sleep(retry_wait_seconds)
            continue

        chosen_calls = calls
        chosen_puts = puts
        chosen_health = health
        break

    if chosen_health is None:
        chosen_health = {
            "expiration": expiration_date,
            "attempt": 0,
            "snapshot_timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "contracts": 0,
            "zero_bid_ask_ratio": 1.0,
            "zero_open_interest_ratio": 1.0,
            "poor_quality": True,
        }

    return expiration_date, chosen_calls, chosen_puts, chosen_health


def get_options_data(
    ticker_symbol: str,
    retry_on_poor_quality: bool = True,
    max_fetch_attempts: int = 2,
    poor_quality_zero_quote_ratio: float = 0.97,
    retry_wait_seconds: float = 0.75,
    min_dte: Optional[int] = None,
    max_dte: Optional[int] = None,
    as_of_utc: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Fetches all available call and put options data for a given stock ticker.
    Strike range filtering will be applied in the data cleaning step.
    """
    if max_fetch_attempts < 1:
        max_fetch_attempts = 1

    stock = yf.Ticker(ticker_symbol)
    options_data_list = []
    fetch_diagnostics = []
    available_dates = stock.options # Get all available expiration dates

    if not available_dates:
        print(f"No option expiration dates found for {ticker_symbol}.")
        return pd.DataFrame()

    print(f"Fetching options for {ticker_symbol} for {len(available_dates)} expiration dates...")

    now_utc = normalize_as_of_utc(as_of_utc)
    available_expirations = []
    selected_dates = []
    selected_dte_by_date = {}
    for date in available_dates:
        expiration_days = _expiration_dte(date, now_utc)
        available_expirations.append(
            {
                "expiration": date,
                "days_to_expiration": expiration_days,
            }
        )
        if expiration_days is None or expiration_days <= 0:
            continue
        if min_dte is not None and expiration_days < int(min_dte):
            continue
        if max_dte is not None and expiration_days > int(max_dte):
            continue
        selected_dates.append(date)
        selected_dte_by_date[date] = expiration_days

    if not selected_dates:
        print(f"No option expiration dates matched the requested DTE range for {ticker_symbol}.")
        empty_df = pd.DataFrame()
        empty_df.attrs["fetchDiagnostics"] = {
            "ticker": ticker_symbol,
            "expirations_available": len(available_dates),
            "expirations_requested": 0,
            "expirations_fetched": 0,
            "expirations_flagged_poor_quality": 0,
            "max_attempt_used": 0,
            "as_of_utc": now_utc.isoformat(),
            "available_expirations": available_expirations,
            "selected_expirations": [],
            "details": [],
        }
        return empty_df

    print(
        f"Fetching {len(selected_dates)} filtered expiration dates for {ticker_symbol} "
        f"from {len(available_dates)} available dates."
    )

    fetched_by_date = {}
    worker_count = min(MAX_OPTION_FETCH_WORKERS, len(selected_dates))
    if worker_count <= 1:
        for date in selected_dates:
            try:
                fetched_by_date[date] = _fetch_expiration_chain(
                    ticker_symbol=ticker_symbol,
                    expiration_date=date,
                    retry_on_poor_quality=retry_on_poor_quality,
                    max_fetch_attempts=max_fetch_attempts,
                    poor_quality_zero_quote_ratio=poor_quality_zero_quote_ratio,
                    retry_wait_seconds=retry_wait_seconds,
                )
            except Exception as error:
                print(f"Could not fetch options for {ticker_symbol} on {date}: {error}")
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_by_date = {
                executor.submit(
                    _fetch_expiration_chain,
                    ticker_symbol,
                    date,
                    retry_on_poor_quality,
                    max_fetch_attempts,
                    poor_quality_zero_quote_ratio,
                    retry_wait_seconds,
                ): date
                for date in selected_dates
            }
            for future in as_completed(future_by_date):
                date = future_by_date[future]
                try:
                    fetched_by_date[date] = future.result()
                except Exception as error:
                    print(f"Could not fetch options for {ticker_symbol} on {date}: {error}")

    for date in selected_dates:
        fetched = fetched_by_date.get(date)
        if fetched is None:
            continue

        try:
            _, chosen_calls, chosen_puts, chosen_health = fetched

            # Add expirationDate, optionType, and snapshot metadata to calls
            if not chosen_calls.empty:
                chosen_calls["expirationDate"] = pd.to_datetime(date)
                chosen_calls["optionType"] = "call"
                chosen_calls["expirationCloseUtc"] = expiration_close_utc(date)
                chosen_calls["expirationDteAtFetch"] = selected_dte_by_date.get(date)
                chosen_calls["snapshotTimestampUtc"] = chosen_health[
                    "snapshot_timestamp_utc"
                ]
                chosen_calls["expirationFetchAttempt"] = chosen_health["attempt"]
                chosen_calls["expirationZeroBidAskRatio"] = chosen_health[
                    "zero_bid_ask_ratio"
                ]
                chosen_calls["expirationZeroOpenInterestRatio"] = chosen_health[
                    "zero_open_interest_ratio"
                ]
                chosen_calls["expirationPoorQuality"] = chosen_health["poor_quality"]
                options_data_list.append(chosen_calls)

            # Add expirationDate, optionType, and snapshot metadata to puts
            if not chosen_puts.empty:
                chosen_puts["expirationDate"] = pd.to_datetime(date)
                chosen_puts["optionType"] = "put"
                chosen_puts["expirationCloseUtc"] = expiration_close_utc(date)
                chosen_puts["expirationDteAtFetch"] = selected_dte_by_date.get(date)
                chosen_puts["snapshotTimestampUtc"] = chosen_health[
                    "snapshot_timestamp_utc"
                ]
                chosen_puts["expirationFetchAttempt"] = chosen_health["attempt"]
                chosen_puts["expirationZeroBidAskRatio"] = chosen_health[
                    "zero_bid_ask_ratio"
                ]
                chosen_puts["expirationZeroOpenInterestRatio"] = chosen_health[
                    "zero_open_interest_ratio"
                ]
                chosen_puts["expirationPoorQuality"] = chosen_health["poor_quality"]
                options_data_list.append(chosen_puts)

            fetch_diagnostics.append(chosen_health)
        except Exception as error:
            print(f"Could not prepare options for {ticker_symbol} on {date}: {error}")
            continue # Skip to next date if an error occurs

    if not options_data_list:
        print(f"No options data could be compiled for {ticker_symbol}.")
        return pd.DataFrame()

    combined_options_df = pd.concat(options_data_list, ignore_index=True)
    
    if fetch_diagnostics:
        poor_quality_count = sum(
            1 for item in fetch_diagnostics if item.get("poor_quality")
        )
        max_attempt_used = max(item.get("attempt", 1) for item in fetch_diagnostics)
        combined_options_df.attrs["fetchDiagnostics"] = {
            "ticker": ticker_symbol,
            "expirations_available": len(available_dates),
            "expirations_requested": len(selected_dates),
            "expirations_fetched": len(fetch_diagnostics),
            "expirations_flagged_poor_quality": poor_quality_count,
            "max_attempt_used": max_attempt_used,
            "as_of_utc": now_utc.isoformat(),
            "available_expirations": available_expirations,
            "selected_expirations": [
                {
                    "expiration": date,
                    "days_to_expiration": selected_dte_by_date.get(date),
                }
                for date in selected_dates
            ],
            "details": fetch_diagnostics,
        }

    print(f"Successfully fetched {len(combined_options_df)} total option contracts for {ticker_symbol}.")
    return combined_options_df
