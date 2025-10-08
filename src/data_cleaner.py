import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.optimize import brentq


def _black_scholes_price(option_type: str, spot: float, strike: float, time_to_expiration: float, risk_free_rate: float, volatility: float, dividend_yield: float = 0.0) -> float:
    """Compute the Black-Scholes theoretical price for a European option."""
    if time_to_expiration <= 0 or volatility <= 0 or spot <= 0 or strike <= 0:
        return float("nan")

    sqrt_t = np.sqrt(time_to_expiration)
    forward = spot * np.exp(-dividend_yield * time_to_expiration)

    d1 = (
        np.log(forward / strike)
        + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiration
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    discount_factor = np.exp(-risk_free_rate * time_to_expiration)

    if option_type == "call":
        price = forward * norm.cdf(d1) - strike * discount_factor * norm.cdf(d2)
    else:
        price = strike * discount_factor * norm.cdf(-d2) - forward * norm.cdf(-d1)

    return float(price)


def _implied_volatility(option_type: str, spot: float, strike: float, time_to_expiration: float, risk_free_rate: float, market_price: float, dividend_yield: float = 0.0) -> float:
    """Solve for implied volatility using the Black-Scholes formula."""
    if any(val is None for val in (spot, strike, time_to_expiration, market_price)):
        return float("nan")

    if time_to_expiration <= 0 or market_price <= 0:
        return float("nan")

    intrinsic_value = max(0.0, spot * np.exp(-dividend_yield * time_to_expiration) - strike) if option_type == "call" else max(0.0, strike - spot * np.exp(-dividend_yield * time_to_expiration))
    if market_price < intrinsic_value:
        return float("nan")

    def objective(vol: float) -> float:
        return _black_scholes_price(option_type, spot, strike, time_to_expiration, risk_free_rate, vol, dividend_yield) - market_price

    try:
        return float(brentq(objective, 1e-6, 5.0, maxiter=100, xtol=1e-6))
    except ValueError:
        return float("nan")


def _select_market_price(row: pd.Series) -> float:
    """Choose the best available market price for implied volatility inversion."""
    bid = row.get("bid")
    ask = row.get("ask")
    last_price = row.get("lastPrice")

    prices = []
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        prices.append((bid + ask) / 2)
    if last_price is not None and last_price > 0:
        prices.append(last_price)
    if bid is not None and bid > 0:
        prices.append(bid)
    if ask is not None and ask > 0:
        prices.append(ask)

    return float(prices[0]) if prices else float("nan")

def prepare_options_data(
    df: pd.DataFrame,
    min_strike: float = None,
    max_strike: float = None,
    min_date: str = None,
    max_date: str = None,
    option_type_to_plot: str = 'both',  # New parameter: 'call', 'put', or 'both'
    max_dte: int = None,  # New parameter for maximum days to expiration
    iv_source: str = 'yfinance',
    underlying_price: float = None,
    risk_free_rate: float = 0.02,
    dividend_yield: float = 0.0
):
    """
    Clean and filter options data for visualization.
    
    Args:
        df: Raw options DataFrame from data_fetch
        min_strike: Minimum strike price to include
        max_strike: Maximum strike price to include
        min_date: Minimum date (YYYY-MM-DD)
        max_date: Maximum date (YYYY-MM-DD)
        option_type_to_plot (str): 'call', 'put', or 'both' to filter by. Default is 'both'.
        max_dte: Maximum days to expiration to include.
        iv_source: 'yfinance' (default) to use the provider's IV or 'black-scholes' to recompute.
        underlying_price: Current underlying price, required when iv_source='black-scholes'.
        risk_free_rate: Annualized risk-free rate used for Black-Scholes IV calculation.
        dividend_yield: Continuous dividend yield assumption used in Black-Scholes.
    
    Returns:
        Cleaned DataFrame with necessary columns for volatility surface
    """
    if df is None or df.empty:
        print("No data to clean")
        return pd.DataFrame()

    # Create working copy
    clean_df = df.copy()

    # Filter by option type if specified before other processing
    if option_type_to_plot.lower() == 'call':
        clean_df = clean_df[clean_df['optionType'] == 'call'].copy()
        print(f"Filtering for CALL options. Rows remaining: {len(clean_df)}")
    elif option_type_to_plot.lower() == 'put':
        clean_df = clean_df[clean_df['optionType'] == 'put'].copy()
        print(f"Filtering for PUT options. Rows remaining: {len(clean_df)}")
    elif option_type_to_plot.lower() != 'both':
        print(f"Warning: Invalid option_type_to_plot '{option_type_to_plot}'. Using 'both'.")
    
    if clean_df.empty:
        print(f"No data remaining after filtering for option type: {option_type_to_plot}")
        return pd.DataFrame()

    # Convert dates to datetime
    clean_df['expirationDate'] = pd.to_datetime(clean_df['expirationDate'])
    
    # Calculate days to expiration
    today = pd.Timestamp(datetime.now().date())
    clean_df['days_to_expiration'] = (clean_df['expirationDate'] - today).dt.days
    
    # Filter by strike price
    if min_strike:
        clean_df = clean_df[clean_df['strike'] >= min_strike]
    if max_strike:
        clean_df = clean_df[clean_df['strike'] <= max_strike]
    
    # Filter by date
    if min_date:
        min_date = pd.Timestamp(min_date)
        clean_df = clean_df[clean_df['expirationDate'] >= min_date]
    if max_date:
        max_date = pd.Timestamp(max_date)
        clean_df = clean_df[clean_df['expirationDate'] <= max_date]
    
    # Filter by max_dte if provided
    if max_dte is not None:
        clean_df = clean_df[clean_df['days_to_expiration'] <= max_dte]
        print(f"Filtering for DTE <= {max_dte}. Rows remaining: {len(clean_df)}")

    iv_source_normalized = (iv_source or 'yfinance').strip().lower()
    if iv_source_normalized not in {'yfinance', 'black-scholes'}:
        print(f"Warning: Invalid iv_source '{iv_source}'. Defaulting to 'yfinance'.")
        iv_source_normalized = 'yfinance'

    if iv_source_normalized == 'black-scholes':
        if underlying_price is None or underlying_price <= 0:
            raise ValueError("underlying_price must be provided and positive when iv_source='black-scholes'.")

        print("Recomputing implied volatility using Black-Scholes model...")
        clean_df = clean_df.copy()
        clean_df['marketPrice'] = clean_df.apply(_select_market_price, axis=1)
        clean_df['time_to_expiration_years'] = clean_df['days_to_expiration'] / 365.25

        clean_df['impliedVolatility'] = clean_df.apply(
            lambda row: _implied_volatility(
                row['optionType'],
                underlying_price,
                row['strike'],
                row['time_to_expiration_years'],
                risk_free_rate,
                row['marketPrice'],
                dividend_yield
            ),
            axis=1
        )

        before_drop = len(clean_df)
        clean_df = clean_df.replace([np.inf, -np.inf], np.nan)
        clean_df = clean_df.dropna(subset=['impliedVolatility'])
        print(f"Black-Scholes IV computation succeeded for {len(clean_df)} contracts (dropped {before_drop - len(clean_df)} with invalid prices).")

    # Remove invalid data
    clean_df = clean_df[
        (clean_df['impliedVolatility'] > 0) &  # Remove zero/negative volatility
        (clean_df['days_to_expiration'] > 0)    # Remove expired options
    ]
    
    # Remove illiquid options (low volume AND low open interest)
    VOLUME_THRESHOLD = 5       # Minimum acceptable volume
    OI_THRESHOLD = 10         # Minimum acceptable open interest
    clean_df = clean_df[
        ~((clean_df['volume'] < VOLUME_THRESHOLD) & 
          (clean_df['openInterest'] < OI_THRESHOLD))
    ]

    # Select and rename columns for visualization
    final_df = clean_df[[
        'strike',
        'days_to_expiration',
        'impliedVolatility',
        'expirationDate',
        'optionType',
        'volume',
        'openInterest',
        'bid',
        'ask',
        'lastPrice'
    ]].copy()
    final_df['ivComputationMethod'] = 'black-scholes' if iv_source_normalized == 'black-scholes' else 'yfinance'

    return final_df

if __name__ == "__main__":
    # Test the cleaner
    try:
        # Read the CSV we created in data_fetch
        input_file = 'options_data.csv'
        raw_data = pd.read_csv(input_file)
        print(f"Loaded {len(raw_data)} rows of raw data from {input_file}")

        # Define date range for testing (next 3 months)
        today_date = datetime.now()
        three_months_later = today_date + timedelta(days=90)
        min_test_date = today_date.strftime('%Y-%m-%d')
        max_test_date = three_months_later.strftime('%Y-%m-%d')
        
        # Clean the data
        clean_data = prepare_options_data(
            raw_data,
            min_strike=raw_data['strike'].min(), # Or a more specific test range
            max_strike=raw_data['strike'].max(), # Or a more specific test range
            # min_date=min_test_date, # Using DTE for date range now
            # max_date=max_test_date,
            max_dte=90, # Test with 90 days DTE
            option_type_to_plot='call',  # Specify 'call' for testing
            iv_source='black-scholes',
            underlying_price=raw_data['underlyingPrice'].iloc[0] if 'underlyingPrice' in raw_data.columns else None
        )
        
        if not clean_data.empty:
            print("\nCleaning successful!")
            print(f"Cleaned data shape: {clean_data.shape}")
            print("\nSample of cleaned data:")
            print(clean_data.head())
            
            # Save cleaned data
            output_file = 'cleaned_options_data.csv'
            clean_data.to_csv(output_file, index=False)
            print(f"\nCleaned data saved to {output_file}")
            
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")