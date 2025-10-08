# vol-surface-project
Constructing and analyzing IV surfaces for equity options

## Features

- Fetch options data from Yahoo Finance
- Clean and filter options by strike price, expiration date, and option type
- Visualize implied volatility surfaces in interactive 3D plots
- **NEW**: Surface smoothing via interpolation for better visualization of illiquid data
- **NEW**: Recompute implied volatility with the Black-Scholes model on demand

## Usage

### Basic Usage

```bash
python main.py TICKER
```

### With Smoothing

Apply interpolation smoothing to create a smoother surface, especially useful for illiquid strikes:

```bash
python main.py TICKER --smooth
```

### Additional Options

```bash
python main.py TICKER --option_type call --smooth --dte_max 60
```

Options:
- `--option_type {call,put,both}`: Filter by option type (default: both)
- `--strike_min_pct`: Minimum strike as % of current price (default: 0.93)
- `--strike_max_pct`: Maximum strike as % of current price (default: 1.07)
- `--dte_max`: Maximum days to expiration (default: 60)
- `--smooth`: Apply interpolation smoothing to the surface
- `--output_dir`: Directory to save output HTML (default: current directory)
- `--iv_source {yfinance,black-scholes}`: Choose the implied volatility source (default: yfinance)
- `--risk_free_rate`: Annualized risk-free rate for Black-Scholes (default: 0.02)
- `--dividend_yield`: Continuous dividend yield for Black-Scholes (default: 0.0)

### Selecting Implied Volatility Source

Use the `--iv_source black-scholes` flag to recompute implied volatilities from option prices using the Black-Scholes model:

```bash
python main.py TICKER --iv_source black-scholes --risk_free_rate 0.03
```

When choosing the Black-Scholes mode, the pipeline uses the underlying price fetched at runtime, filters the option chain, solves for implied volatility with Brent's method, and labels the output so the visualizer displays the chosen source in the hover text and chart title.

## Smoothing Feature

The `--smooth` flag enables linear interpolation of implied volatility values across a dense strike/DTE grid. This helps:
- Visualize smoother surfaces in areas with sparse data
- Better identify patterns in illiquid strikes
- Create more visually appealing surfaces

The smoothing gracefully falls back to raw data if interpolation fails (e.g., insufficient data points).

## For AI Coding Assistants

This project includes specific guidance for AI tools in `.github/copilot-instructions.md`.
GitHub Copilot users: This is automatically loaded.
Other tools: See the instructions file for project-specific patterns.

