# Copilot Instructions for vol-surface-project

## Project Overview
This is a financial visualization tool that fetches equity options data from Yahoo Finance and creates interactive 3D implied volatility surfaces. The project uses a modular pipeline architecture: fetch → clean → visualize.

## Architecture & Data Flow

**Pipeline**: `main.py` orchestrates the three-stage process:
1. **Fetch** (`src/data_fetch.py`): Downloads all options chains via `yfinance`, adds `expirationDate` and `optionType` columns
2. **Clean** (`src/data_cleaner.py`): Filters by strike range, DTE, option type, and removes illiquid contracts (volume < 5 AND OI < 20)
3. **Visualize** (`src/visualizer.py`): Renders 3D surface using Plotly's `Mesh3d` with optional interpolation smoothing

**Key Data Transformations**:
- IV values are converted from decimals (0.25) to percentages (25%) in `prepare_options_data()`
- DTE is calculated dynamically: `(expirationDate - today).days`
- Smoothing creates a dense grid (~7.6x more points) via `scipy.interpolate.griddata` with cubic method

## Critical Developer Workflows

**Run the tool**:
```bash
python main.py TICKER [--option_type {call,put,both}] [--smooth] [--dte_max 90]
```

**Environment setup** (conda):
```bash
conda env create -f environment.yml
conda activate vol-surface-project
```

**Testing individual modules**: Each `src/*.py` file has a `if __name__ == "__main__"` block for standalone testing. They assume intermediate CSV files exist (e.g., `options_data.csv`, `cleaned_options_data.csv`).

## Project-Specific Conventions

**Filtering Philosophy**: Fetch broadly, filter narrowly. `get_options_data()` retrieves ALL available chains; `prepare_options_data()` applies user-specified filters (strike %, DTE, type).

**Liquidity Threshold**: Options with BOTH `volume < 5` AND `openInterest < 20` are removed. This is hardcoded in `data_cleaner.py` lines 77-81.

**Smoothing Fallback**: If interpolation fails (e.g., insufficient data points), `create_vol_surface()` gracefully falls back to raw data plotting. Smoothed plots omit volume/OI from hover text since interpolated points don't have this metadata.

**Output Naming**: Files are timestamped: `{ticker}_{option_type}_vol_surface_{YYYYMMDD-HHMMSS}.html`

## Key Files & Patterns

**`src/data_fetch.py`**: 
- Always returns a DataFrame with `expirationDate` (datetime) and `optionType` (string) columns
- Coerces `impliedVolatility` to numeric, setting errors to NaN

**`src/data_cleaner.py`**:
- The `option_type_to_plot` parameter filters BEFORE other operations (line 32-42)
- Returns a DataFrame with exactly these columns: `strike`, `days_to_expiration`, `impliedVolatility`, `expirationDate`, `optionType`, `volume`, `openInterest`, `bid`, `ask`, `lastPrice`

**`src/visualizer.py`**:
- `_smooth_surface()` is a private helper; don't call it directly
- `smooth=True` creates a uniform grid with `strike_step=1.0` and `dte_step=1`, which may need tuning for wide strike ranges or long DTEs

## Integration Points

**External Dependency**: `yfinance` is the sole data source. If Yahoo Finance changes their API or data structure, update `data_fetch.py` accordingly.

**Plotly Output**: The `create_vol_surface()` function returns a `plotly.graph_objects.Figure` object. To display inline (e.g., in Jupyter), call `.show()`. To save, call `.write_html()`.

## Recent Changes (PR #1)

The smoothing feature was added via `--smooth` flag. It uses cubic interpolation to create denser surfaces for illiquid data. Gaussian filtering was intentionally omitted. If extending smoothing:
- Adjust grid density via `strike_step` and `dte_step` parameters in `_smooth_surface()`
- Consider adding error handling for edge cases (e.g., single expiration date, moneyness extremes)
