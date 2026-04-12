import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.surface_service import SurfaceRequest, build_surface_bundle


def _json_ready(value: Any):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def build_external_benchmark_report(surface_df: pd.DataFrame, benchmark_csv: str) -> Dict[str, Any]:
    report: Dict[str, Any] = {"benchmark_csv": benchmark_csv}
    if not benchmark_csv:
        report["error"] = "No benchmark CSV path provided."
        return report
    if not os.path.exists(benchmark_csv):
        report["error"] = f"Benchmark CSV not found: {benchmark_csv}"
        return report

    benchmark_df = pd.read_csv(benchmark_csv)
    required_cols = {"optionType", "strike", "days_to_expiration", "iv_ref"}
    missing_cols = sorted(required_cols - set(benchmark_df.columns))
    if missing_cols:
        report["error"] = f"Benchmark CSV missing required columns: {', '.join(missing_cols)}"
        return report

    working_surface = surface_df.copy()
    if "impliedVolatilityFinal" not in working_surface.columns:
        report["error"] = "Surface dataframe missing impliedVolatilityFinal."
        return report

    working_surface = working_surface.dropna(
        subset=["optionType", "strike", "days_to_expiration", "impliedVolatilityFinal"]
    ).copy()
    if working_surface.empty:
        report["matched_rows"] = 0
        report["error"] = "No valid surface rows to compare."
        return report

    working_surface["optionType"] = working_surface["optionType"].astype(str).str.lower()
    working_surface["strike_join"] = pd.to_numeric(
        working_surface["strike"], errors="coerce"
    ).round(4)
    working_surface["days_to_expiration"] = pd.to_numeric(
        working_surface["days_to_expiration"], errors="coerce"
    ).astype("Int64")

    benchmark_working = benchmark_df.copy()
    benchmark_working["optionType"] = benchmark_working["optionType"].astype(str).str.lower()
    benchmark_working["strike_join"] = pd.to_numeric(
        benchmark_working["strike"], errors="coerce"
    ).round(4)
    benchmark_working["days_to_expiration"] = pd.to_numeric(
        benchmark_working["days_to_expiration"], errors="coerce"
    ).astype("Int64")
    benchmark_working["iv_ref"] = pd.to_numeric(benchmark_working["iv_ref"], errors="coerce")
    benchmark_working = benchmark_working.dropna(
        subset=["optionType", "strike_join", "days_to_expiration", "iv_ref"]
    )

    merged = working_surface.merge(
        benchmark_working[["optionType", "strike_join", "days_to_expiration", "iv_ref"]],
        on=["optionType", "strike_join", "days_to_expiration"],
        how="inner",
    )

    if merged.empty:
        report["matched_rows"] = 0
        report["error"] = "No benchmark matches found by optionType/strike/DTE."
        return report

    # Vol points means percentage points (0.01 == 1 vol point)
    merged["error_vol_points"] = (merged["impliedVolatilityFinal"] - merged["iv_ref"]) * 100.0
    abs_error = merged["error_vol_points"].abs()
    report["matched_rows"] = int(len(merged))
    report["mae_vol_points"] = float(abs_error.mean())
    report["rmse_vol_points"] = float(np.sqrt(np.mean(merged["error_vol_points"] ** 2)))
    report["p95_abs_error_vol_points"] = float(np.quantile(abs_error, 0.95))

    by_type = {}
    for side, group in merged.groupby("optionType"):
        group_abs_error = group["error_vol_points"].abs()
        by_type[str(side)] = {
            "matched_rows": int(len(group)),
            "mae_vol_points": float(group_abs_error.mean()),
            "rmse_vol_points": float(np.sqrt(np.mean(group["error_vol_points"] ** 2))),
            "p95_abs_error_vol_points": float(np.quantile(group_abs_error, 0.95)),
        }
    report["by_option_type"] = by_type
    return report


def _print_diagnostics_summary(diagnostics: Dict[str, Any]) -> None:
    rows_retained = diagnostics.get("rows_retained", 0)
    rows_included = diagnostics.get("rows_surface_included", 0)
    rows_excluded = diagnostics.get("rows_surface_excluded", 0)
    fallback_fraction = diagnostics.get("fallback_iv_fraction", 0.0) or 0.0
    print(
        "Diagnostics summary: "
        f"retained={rows_retained}, included={rows_included}, excluded={rows_excluded}, "
        f"fallback_iv_fraction={fallback_fraction:.3f}"
    )

    flag_counts = diagnostics.get("flag_counts", {})
    if flag_counts:
        top_flags = sorted(flag_counts.items(), key=lambda item: item[1], reverse=True)[:5]
        top_flags_text = ", ".join(f"{name}:{count}" for name, count in top_flags)
        print(f"Top quality flags: {top_flags_text}")

    internal = diagnostics.get("internal_validation", {})
    if internal:
        mae = internal.get("repricing_mae")
        rmse = internal.get("repricing_rmse")
        rows_market = internal.get("rows_with_market_price")
        print(
            "Internal validation: "
            f"rows_with_market_price={rows_market}, repricing_mae={mae}, repricing_rmse={rmse}"
        )

    external = diagnostics.get("external_benchmark")
    if external:
        if "error" in external:
            print(f"External benchmark: {external['error']}")
        else:
            print(
                "External benchmark: "
                f"matched_rows={external.get('matched_rows')}, "
                f"mae_vol_points={external.get('mae_vol_points')}, "
                f"rmse_vol_points={external.get('rmse_vol_points')}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Fetch, clean, and visualize equity option volatility surfaces."
    )
    parser.add_argument("ticker", type=str, help="Stock ticker symbol (e.g., AAPL, SPY).")
    parser.add_argument(
        "--strike_min_pct",
        type=float,
        default=0.93,
        help="Minimum strike as a percentage of current stock price.",
    )
    parser.add_argument(
        "--strike_max_pct",
        type=float,
        default=1.07,
        help="Maximum strike as a percentage of current stock price.",
    )
    parser.add_argument(
        "--dte_min",
        type=int,
        default=1,
        help="Minimum days-to-expiration to include.",
    )
    parser.add_argument(
        "--dte_max",
        type=int,
        default=60,
        help="Maximum days-to-expiration to include.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory for output HTML.",
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Apply arbitrage-aware smoothing to build the unified volatility surface.",
    )
    parser.add_argument(
        "--iv_source",
        type=str,
        default="auto",
        choices=["auto", "yfinance", "black-scholes"],
        help="IV mode: 'auto' (recommended), 'yfinance', or 'black-scholes'.",
    )
    parser.add_argument(
        "--risk_free_rate",
        type=float,
        default=0.02,
        help="Annualized risk-free rate for Black-Scholes calculations.",
    )
    parser.add_argument(
        "--dividend_yield",
        type=float,
        default=0.0,
        help="Continuous dividend yield for Black-Scholes calculations.",
    )
    parser.add_argument(
        "--quality_mode",
        type=str,
        default="lenient",
        choices=["strict", "balanced", "lenient"],
        help="Quality gate profile used for confidence scoring and surface inclusion.",
    )
    parser.add_argument(
        "--max_trade_age_hours",
        type=float,
        default=72.0,
        help="Max age for last-trade fallback pricing before stale flagging.",
    )
    parser.add_argument(
        "--diagnostics_json",
        type=str,
        default=None,
        help="Optional path to save diagnostics JSON report.",
    )
    parser.add_argument(
        "--benchmark_csv",
        type=str,
        default=None,
        help="Optional benchmark CSV path with columns: optionType,strike,days_to_expiration,iv_ref.",
    )
    parser.add_argument(
        "--include_low_confidence",
        action="store_true",
        help="Include low-confidence rows in visualization output.",
    )

    args = parser.parse_args()

    print(f"Processing ticker: {args.ticker}")

    try:
        result = build_surface_bundle(
            SurfaceRequest(
                ticker=args.ticker,
                strike_min_pct=args.strike_min_pct,
                strike_max_pct=args.strike_max_pct,
                dte_min=args.dte_min,
                dte_max=args.dte_max,
                smooth=args.smooth,
                iv_source=args.iv_source,
                risk_free_rate=args.risk_free_rate,
                dividend_yield=args.dividend_yield,
                quality_mode=args.quality_mode,
                max_trade_age_hours=args.max_trade_age_hours,
                include_low_confidence=args.include_low_confidence,
            )
        )
    except ValueError as error:
        print(f"Error: {error}")
        return

    current_price = result.current_price
    raw_options_df = result.raw_options_df
    cleaned_options_df = result.cleaned_options_df
    diagnostics = result.diagnostics
    fig = result.figure

    print(f"Current price for {args.ticker}: ${current_price:.2f}")
    print(
        f"Target strike range: ${current_price * args.strike_min_pct:.2f} "
        f"to ${current_price * args.strike_max_pct:.2f}"
    )
    print(f"DTE range: {args.dte_min} to {args.dte_max} days")
    print(f"IV source: {args.iv_source}")
    print(f"Quality mode: {args.quality_mode}")
    print("Surface construction: unified call/put arbitrage-free surface")
    print(f"Fetched {len(raw_options_df)} raw option contracts initially.")
    print(f"Prepared {len(cleaned_options_df)} option contracts with quality metadata.")

    if args.benchmark_csv:
        diagnostics["external_benchmark"] = build_external_benchmark_report(
            cleaned_options_df, args.benchmark_csv
        )

    _print_diagnostics_summary(diagnostics)

    if args.diagnostics_json:
        diagnostics_dir = os.path.dirname(args.diagnostics_json)
        if diagnostics_dir:
            os.makedirs(diagnostics_dir, exist_ok=True)
        with open(args.diagnostics_json, "w", encoding="utf-8") as handle:
            json.dump(_json_ready(diagnostics), handle, indent=2)
        print(f"Diagnostics JSON saved to: {args.diagnostics_json}")

    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
            print(f"Created output directory: {args.output_dir}")
        except OSError as error:
            print(f"Error creating output directory {args.output_dir}: {error}. Saving locally.")
            args.output_dir = "."

    output_filename = (
        f"{args.ticker}_unified_vol_surface_{datetime.now().strftime('%Y%m%d-%H%M%S')}.html"
    )
    output_path = os.path.join(args.output_dir, output_filename)
    try:
        fig.write_html(output_path)
        print(f"Volatility surface plot saved to: {output_path}")
    except Exception as error:
        print(f"Error saving plot to {output_path}: {error}")


if __name__ == "__main__":
    main()
