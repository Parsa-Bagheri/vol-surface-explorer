import pandas as pd
from datetime import datetime, timedelta

def prepare_options_data(
    df: pd.DataFrame,
    min_strike: float = None,
    max_strike: float = None,
    min_date: str = None,
    max_date: str = None,
    option_type_to_plot: str = 'both',  # New parameter: 'call', 'put', or 'both'
    max_dte: int = None  # New parameter for maximum days to expiration
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
            option_type_to_plot='call'  # Specify 'call' for testing
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