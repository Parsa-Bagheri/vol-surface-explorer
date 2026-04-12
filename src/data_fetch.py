import time
from typing import Dict, Optional

import pandas as pd
import yfinance as yf


IV_PLACEHOLDER_THRESHOLD = 1e-4

def get_current_price(ticker_symbol: str) -> Optional[float]:
    """Fetches the current market price for a given stock ticker."""
    stock = yf.Ticker(ticker_symbol)
    try:
        # Attempt to get the most recent closing price
        current_price = stock.history(period="1d")['Close'].iloc[-1]
        if pd.isna(current_price):
            # Fallback to currentPrice or previousClose from info if history is NaN
            info = stock.info
            current_price = info.get('currentPrice') or info.get('previousClose')
        
        if current_price is None:
            print(f"Could not determine current price for {ticker_symbol} from available data.")
            return None
        return float(current_price)
    except Exception as e:
        print(f"Error fetching current price for {ticker_symbol}: {e}")
        return None


def _compute_chain_health(
    chain_df: pd.DataFrame,
    poor_quality_zero_quote_ratio: float,
) -> Dict[str, float]:
    """Compute lightweight per-expiration quality metrics."""
    if chain_df is None or chain_df.empty:
        return {
            "contracts": 0,
            "zero_bid_ask_ratio": 1.0,
            "placeholder_iv_ratio": 1.0,
            "zero_open_interest_ratio": 1.0,
            "poor_quality": True,
        }

    bid = pd.to_numeric(chain_df.get("bid"), errors="coerce")
    ask = pd.to_numeric(chain_df.get("ask"), errors="coerce")
    iv = pd.to_numeric(chain_df.get("impliedVolatility"), errors="coerce")
    oi = pd.to_numeric(chain_df.get("openInterest"), errors="coerce")

    zero_quote_mask = bid.isna() | ask.isna() | (bid <= 0) | (ask <= 0) | (ask < bid)
    placeholder_iv_mask = iv.isna() | (iv <= IV_PLACEHOLDER_THRESHOLD)
    zero_oi_mask = oi.fillna(0) <= 0

    contracts = float(len(chain_df))
    zero_bid_ask_ratio = float(zero_quote_mask.sum() / contracts)
    placeholder_iv_ratio = float(placeholder_iv_mask.sum() / contracts)
    zero_open_interest_ratio = float(zero_oi_mask.sum() / contracts)

    poor_quality = bool(
        zero_bid_ask_ratio >= poor_quality_zero_quote_ratio
        and placeholder_iv_ratio >= 0.5
    )

    return {
        "contracts": int(contracts),
        "zero_bid_ask_ratio": zero_bid_ask_ratio,
        "placeholder_iv_ratio": placeholder_iv_ratio,
        "zero_open_interest_ratio": zero_open_interest_ratio,
        "poor_quality": poor_quality,
    }


def get_options_data(
    ticker_symbol: str,
    retry_on_poor_quality: bool = True,
    max_fetch_attempts: int = 2,
    poor_quality_zero_quote_ratio: float = 0.97,
    retry_wait_seconds: float = 0.75,
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

    for date in available_dates:
        try:
            chosen_calls = None
            chosen_puts = None
            chosen_health = None

            for attempt in range(1, max_fetch_attempts + 1):
                snapshot_timestamp = pd.Timestamp.now(tz="UTC")
                options_chain = stock.option_chain(date)
                calls = options_chain.calls.copy()
                puts = options_chain.puts.copy()

                chain_for_health = pd.concat([calls, puts], ignore_index=True)
                health = _compute_chain_health(
                    chain_for_health, poor_quality_zero_quote_ratio
                )
                health["expiration"] = date
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
                continue

            # Add expirationDate, optionType, and snapshot metadata to calls
            if not chosen_calls.empty:
                chosen_calls["expirationDate"] = pd.to_datetime(date)
                chosen_calls["optionType"] = "call"
                chosen_calls["snapshotTimestampUtc"] = chosen_health[
                    "snapshot_timestamp_utc"
                ]
                chosen_calls["expirationFetchAttempt"] = chosen_health["attempt"]
                chosen_calls["expirationZeroBidAskRatio"] = chosen_health[
                    "zero_bid_ask_ratio"
                ]
                chosen_calls["expirationPlaceholderIvRatio"] = chosen_health[
                    "placeholder_iv_ratio"
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
                chosen_puts["snapshotTimestampUtc"] = chosen_health[
                    "snapshot_timestamp_utc"
                ]
                chosen_puts["expirationFetchAttempt"] = chosen_health["attempt"]
                chosen_puts["expirationZeroBidAskRatio"] = chosen_health[
                    "zero_bid_ask_ratio"
                ]
                chosen_puts["expirationPlaceholderIvRatio"] = chosen_health[
                    "placeholder_iv_ratio"
                ]
                chosen_puts["expirationZeroOpenInterestRatio"] = chosen_health[
                    "zero_open_interest_ratio"
                ]
                chosen_puts["expirationPoorQuality"] = chosen_health["poor_quality"]
                options_data_list.append(chosen_puts)

            fetch_diagnostics.append(chosen_health)
        except Exception as e:
            print(f"Could not fetch options for {ticker_symbol} on {date}: {e}")
            continue # Skip to next date if an error occurs

    if not options_data_list:
        print(f"No options data could be compiled for {ticker_symbol}.")
        return pd.DataFrame()

    combined_options_df = pd.concat(options_data_list, ignore_index=True)
    
    # Ensure 'impliedVolatility' is numeric, coercing errors
    if 'impliedVolatility' in combined_options_df.columns:
        combined_options_df['impliedVolatility'] = pd.to_numeric(combined_options_df['impliedVolatility'], errors='coerce')

    if fetch_diagnostics:
        poor_quality_count = sum(
            1 for item in fetch_diagnostics if item.get("poor_quality")
        )
        max_attempt_used = max(item.get("attempt", 1) for item in fetch_diagnostics)
        combined_options_df.attrs["fetchDiagnostics"] = {
            "ticker": ticker_symbol,
            "expirations_requested": len(available_dates),
            "expirations_fetched": len(fetch_diagnostics),
            "expirations_flagged_poor_quality": poor_quality_count,
            "max_attempt_used": max_attempt_used,
            "details": fetch_diagnostics,
        }

    print(f"Successfully fetched {len(combined_options_df)} total option contracts for {ticker_symbol}.")
    return combined_options_df
