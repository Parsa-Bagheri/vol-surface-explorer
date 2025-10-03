# vol-surface-project
Constructing and analyzing IV surfaces for equity options

## Features

- Fetch options data from Yahoo Finance
- Clean and filter options by strike price, expiration date, and option type
- Visualize implied volatility surfaces in interactive 3D plots
- **NEW**: Surface smoothing via interpolation for better visualization of illiquid data

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
- `--strike_min_pct`: Minimum strike as % of current price (default: 0.90)
- `--strike_max_pct`: Maximum strike as % of current price (default: 1.10)
- `--dte_max`: Maximum days to expiration (default: 90)
- `--smooth`: Apply interpolation smoothing to the surface
- `--output_dir`: Directory to save output HTML (default: current directory)

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

