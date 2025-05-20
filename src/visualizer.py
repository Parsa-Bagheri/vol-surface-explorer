import pandas as pd
import plotly.graph_objects as go

def create_vol_surface(df: pd.DataFrame, ticker: str):
    """
    Create an interactive 3D volatility surface plot.
    
    Args:
        df: Cleaned DataFrame with options data
        ticker: Stock ticker symbol for the title
    """
    if df.empty:
        print("No data to plot")
        return
    
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
        title=f'Implied Volatility Surface - {ticker}',
        scene=dict(
            xaxis_title='Strike Price ($)',
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

if __name__ == "__main__":
    try:
        # Read the cleaned data
        input_file = 'cleaned_options_data.csv'
        data = pd.read_csv(input_file)
        print(f"Loaded {len(data)} rows of cleaned data")
          # Create the plot and save to HTML
        fig = create_vol_surface(data, "XLY")
        if fig:
            output_file = 'volatility_surface.html'
            fig.write_html(output_file)
            print(f"Plot saved to {output_file}")
            print("Open this file in your web browser to view the interactive plot")
            
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")