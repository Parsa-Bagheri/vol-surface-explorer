import argparse
import pandas as pd
from datetime import datetime
import os

from src.data_fetch import get_options_data, get_current_price
from src.data_cleaner import prepare_options_data
from src.visualizer import create_vol_surface

def main():
    parser = argparse.ArgumentParser(description="Fetch, clean, and visualize equity option volatility surfaces.")
    parser.add_argument("ticker", type=str, help="Stock ticker symbol (e.g., AAPL, SPY).")
    parser.add_argument(
        "--option_type", 
        type=str, 
        default="both", 
        choices=["call", "put", "both"], 
        help="Type of options to plot: 'call', 'put', or 'both'. Default is 'both'."
    )
    parser.add_argument(
        "--strike_min_pct", 
        type=float, 
        default=0.90, 
        help="Minimum strike price as a percentage of the current stock price (e.g., 0.9 for -10%%). Default is 0.90."
    )
    parser.add_argument(
        "--strike_max_pct", 
        type=float, 
        default=1.10, 
        help="Maximum strike price as a percentage of the current stock price (e.g., 1.1 for +10%%). Default is 1.10."
    )
    parser.add_argument(
        "--dte_max", 
        type=int, 
        default=90, 
        help="Maximum days to expiration for options to include. Default is 90 days."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save the output HTML file. Default is the current directory."
    )

    args = parser.parse_args()

    print(f"Processing ticker: {args.ticker}")
    
    # 1. Fetch current stock price
    current_price = get_current_price(args.ticker)
    if current_price is None:
        print(f"Error: Could not fetch current price for {args.ticker}. Exiting.")
        return

    print(f"Current price for {args.ticker}: ${current_price:.2f}")

    min_strike_abs = current_price * args.strike_min_pct
    max_strike_abs = current_price * args.strike_max_pct

    print(f"Target strike range: ${min_strike_abs:.2f} to ${max_strike_abs:.2f}")
    print(f"Max DTE: {args.dte_max} days")
    print(f"Option type: {args.option_type}")

    # 2. Fetch options data (broadly first, then filter in cleaning)
    raw_options_df = get_options_data(args.ticker) # Fetches all available options

    if raw_options_df.empty:
        print(f"No options data found for {args.ticker}. Check the ticker symbol.")
        return
    
    print(f"Fetched {len(raw_options_df)} raw option contracts initially.")

    # 3. Clean and prepare options data using all specified filters
    cleaned_options_df = prepare_options_data(
        raw_options_df,
        min_strike=min_strike_abs,
        max_strike=max_strike_abs,
        max_dte=args.dte_max,
        option_type_to_plot=args.option_type 
    )

    if cleaned_options_df.empty:
        print(f"No suitable options data found for {args.ticker} after cleaning with the specified parameters.")
        print("Consider adjusting strike range (--strike_min_pct, --strike_max_pct), DTE (--dte_max), or option type (--option_type).")
        return
        
    print(f"Prepared {len(cleaned_options_df)} option contracts for visualization.")

    # 4. Create volatility surface plot
    fig = create_vol_surface(cleaned_options_df, args.ticker, args.option_type)

    # 5. Save the plot
    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
            print(f"Created output directory: {args.output_dir}")
        except OSError as e:
            print(f"Error creating output directory {args.output_dir}: {e}. Saving to current directory instead.")
            args.output_dir = "."

    output_filename = f"{args.ticker}_{args.option_type}_vol_surface_{datetime.now().strftime('%Y%m%d-%H%M%S')}.html"
    output_path = os.path.join(args.output_dir, output_filename)
    
    try:
        fig.write_html(output_path)
        print(f"Volatility surface plot saved to: {output_path}")
    except Exception as e:
        print(f"Error saving plot to {output_path}: {e}")

if __name__ == "__main__":
    main()
