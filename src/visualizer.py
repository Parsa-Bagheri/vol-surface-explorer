import pandas as pd
import plotly.graph_objects as go

def create_vol_surface(df: pd.DataFrame, ticker: str, option_type: str = "both"):
    """
    Create an interactive 3D volatility surface plot.
    
    Args:
        df: Cleaned DataFrame with options data
        ticker: Stock ticker symbol for the title
        option_type: Type of the option ('call', 'put', or 'both') to filter the data
    """
    if df.empty:
        print("No data to plot")
        return
    
    # Filter data based on option type
    if option_type in ['call', 'put']:
        df = df[df['optionType'] == option_type]
    
    # Create the 3D surface plot
    fig = go.Figure(data=[
        go.Mesh3d(
            x=df['strike'],
            y=df['days_to_expiration'],
            z=df['impliedVolatility'],
            intensity=df['impliedVolatility'],
            colorscale='Viridis',
            opacity=0.8,
            name='Vol Surface',
            showscale=True,
            hovertext=[
                f'Strike: ${s:.2f}<br>'
                f'Days to Exp: {d}<br>'
                f'IV: {iv:.2%}<br>'
                f'Volume: {v}<br>'
                f'OI: {oi}'
                for s, d, iv, v, oi in zip(
                    df['strike'],
                    df['days_to_expiration'],
                    df['impliedVolatility'],
                    df['volume'],
                    df['openInterest']
                )
            ],
            hoverinfo='text'
        )
    ])

    # Update the layout
    fig.update_layout(
        title=f'{ticker} {option_type.capitalize() if option_type != "both" else "Call and Put"} Option Volatility Surface',
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


