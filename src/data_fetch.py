import yfinance as yf
import pandas as pd

def get_options_data(ticker_symbol: str):
    """
    Fetches options chain data for a given stock ticker.
      Args:
        ticker_symbol (str): Stock ticker (e.g., 'XLY')
        
    Returns:
        pd.DataFrame | None: DataFrame with options data or None if fetch fails
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        expiration_dates = stock.options
        
        all_options_data = []
        for date in expiration_dates:
            option_chain = stock.option_chain(date)
            
            # Process calls
            calls = option_chain.calls
            calls['expirationDate'] = pd.to_datetime(date)
            calls['optionType'] = 'call'
            
            # Process puts
            puts = option_chain.puts
            puts['expirationDate'] = pd.to_datetime(date)
            puts['optionType'] = 'put'
            
            all_options_data.extend([calls, puts])
            
        return pd.concat(all_options_data, ignore_index=True)
        
    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {e}")
        return None


if __name__ == "__main__":
    # Test configuration
    ticker = 'XLY'
    
    print(f"\nAttempting to fetch data for {ticker}...")
    
    try:
        # Get current stock price
        stock = yf.Ticker(ticker)
        current_price = stock.info['regularMarketPrice']
        print(f"Current {ticker} price: ${current_price:.2f}")
        
        # Set strike range: ±20% around current price
        min_strike = int(current_price * 0.8)
        max_strike = int(current_price * 1.2)
        print(f"Strike range: ${min_strike} to ${max_strike}")
        
        # Fetch options data
        test_data = get_options_data(ticker)
        
        if test_data is not None and not test_data.empty:
            # Filter data by strike range
            mask = (test_data['strike'] >= min_strike) & (test_data['strike'] <= max_strike)
            test_data = test_data[mask]
            
            print("\nData fetch successful!")
            print(f"Total rows (after filtering): {len(test_data)}")
            print(f"Total expiration dates: {test_data['expirationDate'].nunique()}")
            
            print("\nFirst 5 rows of filtered data:")
            print(test_data.head())
            
            print("\nSample of available data:")
            print(f"Strike price range: ${test_data['strike'].min():.2f} to ${test_data['strike'].max():.2f}")
            print(f"Expiration dates: {sorted(test_data['expirationDate'].unique()[:3])}")

            # Save to CSV
            output_file = 'options_data.csv'
            test_data.to_csv(output_file, index=False)
            print(f"\nFull data saved to {output_file}")
    
        else:
            print(f"\nNo data returned for {ticker}. Check if:")
            print("1. Your internet connection is working")
            print("2. The ticker symbol is valid")
            print("3. The market is open and options data is available")
    
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        print("Make sure you have required packages installed:")
        print("conda install yfinance pandas")