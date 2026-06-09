from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import tempfile
import time
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.time_utils import expiration_dte, normalize_as_of_utc


MAX_OPTION_FETCH_WORKERS = 12
OPTION_INPUT_COLUMNS = (
    "strike",
    "volume",
    "openInterest",
    "bid",
    "ask",
    "lastPrice",
    "lastTradeDate",
)


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


def _has_poor_quote_quality(
    chain_df: pd.DataFrame,
    poor_quality_zero_quote_ratio: float,
) -> bool:
    if chain_df is None or chain_df.empty:
        return True

    bid = pd.to_numeric(chain_df.get("bid"), errors="coerce")
    ask = pd.to_numeric(chain_df.get("ask"), errors="coerce")
    zero_quote_mask = bid.isna() | ask.isna() | (bid <= 0) | (ask <= 0) | (ask < bid)
    return bool(float(zero_quote_mask.mean()) >= poor_quality_zero_quote_ratio)


def _fetch_expiration_chain(
    ticker_symbol: str,
    expiration_date: str,
    retry_on_poor_quality: bool,
    max_fetch_attempts: int,
    poor_quality_zero_quote_ratio: float,
    retry_wait_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stock = yf.Ticker(ticker_symbol)

    for attempt in range(1, max_fetch_attempts + 1):
        options_chain = stock.option_chain(expiration_date)
        calls = options_chain.calls.copy()
        puts = options_chain.puts.copy()

        chain_for_health = pd.concat([calls, puts], ignore_index=True)
        should_retry = bool(
            retry_on_poor_quality
            and _has_poor_quote_quality(chain_for_health, poor_quality_zero_quote_ratio)
            and attempt < max_fetch_attempts
        )
        if should_retry:
            time.sleep(retry_wait_seconds)
            continue

        return calls, puts

    return pd.DataFrame(), pd.DataFrame()


def _prepare_option_rows(
    chain_df: pd.DataFrame,
    expiration_date: str,
    option_type: str,
) -> pd.DataFrame:
    prepared = chain_df.reindex(columns=OPTION_INPUT_COLUMNS).copy()
    prepared["expirationDate"] = pd.to_datetime(expiration_date)
    prepared["optionType"] = option_type
    return prepared


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
    available_dates = stock.options

    if not available_dates:
        print(f"No option expiration dates found for {ticker_symbol}.")
        return pd.DataFrame()

    print(f"Fetching options for {ticker_symbol} for {len(available_dates)} expiration dates...")

    now_utc = normalize_as_of_utc(as_of_utc)
    selected_dates = []
    for date in available_dates:
        expiration_days = expiration_dte(date, now_utc)
        if expiration_days is None or expiration_days < 0:
            continue
        if min_dte is not None and expiration_days < int(min_dte):
            continue
        if max_dte is not None and expiration_days > int(max_dte):
            continue
        selected_dates.append(date)

    if not selected_dates:
        print(f"No option expiration dates matched the requested DTE range for {ticker_symbol}.")
        return pd.DataFrame()

    print(
        f"Fetching {len(selected_dates)} filtered expiration dates for {ticker_symbol} "
        f"from {len(available_dates)} available dates."
    )

    worker_count = min(MAX_OPTION_FETCH_WORKERS, len(selected_dates))
    fetched_by_date = {}
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

        chosen_calls, chosen_puts = fetched
        for option_type, chain_df in (("call", chosen_calls), ("put", chosen_puts)):
            if not chain_df.empty:
                options_data_list.append(_prepare_option_rows(chain_df, date, option_type))

    if not options_data_list:
        print(f"No options data could be compiled for {ticker_symbol}.")
        return pd.DataFrame()

    combined_options_df = pd.concat(options_data_list, ignore_index=True)

    print(f"Successfully fetched {len(combined_options_df)} total option contracts for {ticker_symbol}.")
    return combined_options_df
