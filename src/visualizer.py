import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import griddata

def _smooth_surface(df: pd.DataFrame, strike_step: float = 1.0, dte_step: int = 1):
    """
    Interpolate implied volatility values onto a dense grid for smoothing.
    
    Args:
        df: DataFrame with 'strike', 'days_to_expiration', and 'impliedVolatility' columns
        strike_step: Step size for the strike grid
        dte_step: Step size for the days to expiration grid
        
    Returns:
        Tuple of (grid_strikes, grid_dtes, interpolated_ivs) as flattened arrays
    """
    # Add padding to avoid edge effects - reduce grid range slightly
    strike_min = df['strike'].min()
    strike_max = df['strike'].max()
    dte_min = df['days_to_expiration'].min()
    dte_max = df['days_to_expiration'].max()
    
    # Create grid that stays within data bounds (no extrapolation)
    strikes = np.arange(strike_min, strike_max + strike_step, strike_step)
    dtes = np.arange(dte_min, dte_max + dte_step, dte_step)
    grid_strike, grid_dte = np.meshgrid(strikes, dtes)
    
    iv_grid = griddata(
        points=df[['strike', 'days_to_expiration']].to_numpy(),
        values=df['impliedVolatility'].to_numpy(),
        xi=(grid_strike, grid_dte),
        method='linear',
        fill_value=np.nan  # Use NaN for points outside convex hull (prevents extrapolation spikes)
    )
    
    return grid_strike.ravel(), grid_dte.ravel(), iv_grid.ravel()

def create_vol_surface(df: pd.DataFrame, ticker: str, option_type: str = "both", smooth: bool = False):
    """
    Create an interactive 3D volatility surface plot.
    
    Args:
        df: Cleaned DataFrame with options data
        ticker: Stock ticker symbol for the title
        option_type: Type of the option ('call', 'put', or 'both') to filter the data
        smooth: Whether to apply interpolation smoothing to the surface
    """
    if df.empty:
        print("No data to plot")
        return
    
    iv_method = None
    if 'ivComputationMethod' in df.columns and not df.empty:
        iv_method = df['ivComputationMethod'].iloc[0]

    # Filter data based on option type
    if option_type in ['call', 'put']:
        df = df[df['optionType'] == option_type]
    
    # Apply smoothing if requested
    if smooth:
        try:
            print("Applying surface smoothing via interpolation...")
            x_data, y_data, z_data = _smooth_surface(df)
            
            # Remove NaN values that may result from interpolation
            mask = ~np.isnan(z_data)
            x_data = x_data[mask]
            y_data = y_data[mask]
            z_data = z_data[mask]
            
            # Create hover text for smoothed data (without volume/OI info)
            hovertext = []
            for s, d, iv in zip(x_data, y_data, z_data):
                text = (
                    f'Strike: ${s:.2f}<br>'
                    f'Days to Exp: {d}<br>'
                    f'IV: {iv*100:.2f}%'
                )
                if iv_method:
                    text += f'<br>IV Source: {iv_method}'
                hovertext.append(text)
            
            print(f"Smoothing successful: {len(x_data)} interpolated points")
        except Exception as e:
            print(f"Smoothing failed: {e}. Falling back to raw data.")
            smooth = False
    
    # Use raw data if smoothing is disabled or failed
    if not smooth:
        x_data = df['strike']
        y_data = df['days_to_expiration']
        z_data = df['impliedVolatility']
        hovertext = []
        for s, d, iv, v, oi in zip(
            df['strike'],
            df['days_to_expiration'],
            df['impliedVolatility'],
            df['volume'],
            df['openInterest']
        ):
            text = (
                f'Strike: ${s:.2f}<br>'
                f'Days to Exp: {d}<br>'
                f'IV: {iv*100:.2f}%<br>'
                f'Volume: {v}<br>'
                f'OI: {oi}'
            )
            if iv_method:
                text += f'<br>IV Source: {iv_method}'
            hovertext.append(text)
    
    # Convert IV from decimal to percentage for display (0.25 -> 25)
    z_data_display = z_data * 100 if hasattr(z_data, '__iter__') else z_data * 100
    
    # Create the 3D surface plot
    fig = go.Figure(data=[
        go.Mesh3d(
            x=x_data,
            y=y_data,
            z=z_data_display,
            intensity=z_data_display,
            colorscale='Viridis',
            opacity=0.8,
            name='Vol Surface',
            showscale=True,
            hovertext=hovertext,
            hoverinfo='text'
        )
    ])

    # Update the layout
    iv_method_suffix = f" | IV Source: {iv_method}" if iv_method else ""

    fig.update_layout(
        title=f'{ticker} {option_type.capitalize() if option_type != "both" else "Call and Put"} Option Volatility Surface{iv_method_suffix}',
        scene=dict(
            xaxis_title='Strike Price',
            yaxis_title='Days to Expiration',
            zaxis_title='Implied Volatility (%)',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        margin=dict(l=65, r=50, b=65, t=90)
    )

    return fig

# Test the function
if __name__ == "__main__":
    # This test assumes you have a 'cleaned_options_data.csv' file from data_cleaner.py
    # or you can load and clean data here directly for a self-contained test.
    try:
        cleaned_data = pd.read_csv("cleaned_options_data.csv")
        if not cleaned_data.empty:
            # Ensure 'optionType' column exists for the title, or provide a default
            option_type_for_title = cleaned_data['optionType'].iloc[0] if 'optionType' in cleaned_data.columns and not cleaned_data.empty else "both"
            
            # For testing, let's assume a ticker and use the first option type found or 'both'
            test_ticker = "SPY" # Example ticker
            
            fig = create_vol_surface(cleaned_data, test_ticker, option_type_for_title)
            fig.write_html("volatility_surface.html")
            print(f"Volatility surface plot saved to volatility_surface.html for {test_ticker}")
        else:
            print("Cleaned data is empty, skipping plot generation.")
    except FileNotFoundError:
        print("cleaned_options_data.csv not found. Run data_cleaner.py to generate it.")
    except Exception as e:
        print(f"An error occurred during visualizer test: {e}")


