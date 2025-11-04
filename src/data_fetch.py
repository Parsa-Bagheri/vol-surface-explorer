import yfinance as yf
import pandas as pd
from typing import Optional, Tuple

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

def get_options_data(ticker_symbol: str) -> pd.DataFrame:
    """
    Fetches all available call and put options data for a given stock ticker.
    Strike range filtering will be applied in the data cleaning step.
    """
    stock = yf.Ticker(ticker_symbol)
    options_data_list = []
    available_dates = stock.options # Get all available expiration dates

    if not available_dates:
        print(f"No option expiration dates found for {ticker_symbol}.")
        return pd.DataFrame()

    print(f"Fetching options for {ticker_symbol} for {len(available_dates)} expiration dates...")

    for date in available_dates:
        try:
            options_chain = stock.option_chain(date)
            
            # Add expirationDate and optionType to calls
            calls = options_chain.calls
            if not calls.empty:
                calls['expirationDate'] = pd.to_datetime(date)
                calls['optionType'] = 'call'
                options_data_list.append(calls)

            # Add expirationDate and optionType to puts
            puts = options_chain.puts
            if not puts.empty:
                puts['expirationDate'] = pd.to_datetime(date)
                puts['optionType'] = 'put'
                options_data_list.append(puts)
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

    print(f"Successfully fetched {len(combined_options_df)} total option contracts for {ticker_symbol}.")
    return combined_options_df
