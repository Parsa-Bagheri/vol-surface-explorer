from datetime import datetime, timedelta, timezone
import json
import sys

import pandas as pd
import plotly.graph_objects as go

import main as main_module
from src.surface_service import SurfaceBuildResult


def _mock_options_chain_df():
    now = datetime.now(timezone.utc)
    rows = []
    for option_type, base_iv in [("call", 0.23), ("put", 0.26)]:
        for strike in [95.0, 100.0, 105.0]:
            for dte in [10, 20]:
                rows.append(
                    {
                        "strike": strike,
                        "expirationDate": (now + timedelta(days=dte)).date(),
                        "optionType": option_type,
                        "impliedVolatility": base_iv + (strike - 100.0) * 0.001,
                        "volume": 100.0,
                        "openInterest": 250.0,
                        "bid": 1.0 + (strike - 100.0) * 0.02,
                        "ask": 1.2 + (strike - 100.0) * 0.02,
                        "lastPrice": 1.1 + (strike - 100.0) * 0.02,
                        "lastTradeDate": now.isoformat(),
                    }
                )
    df = pd.DataFrame(rows)
    df.attrs["fetchDiagnostics"] = {"expirations_fetched": 2, "max_attempt_used": 1}
    return df


def test_main_cli_generates_html_and_diagnostics(tmp_path, monkeypatch):
    mock_df = _mock_options_chain_df()
    diagnostics_path = tmp_path / "diagnostics.json"
    cleaned_df = mock_df.copy()
    cleaned_df["days_to_expiration"] = 10
    cleaned_df["impliedVolatilityFinal"] = cleaned_df["impliedVolatility"]
    figure = go.Figure()
    figure.add_scatter(x=[1, 2], y=[1, 2])

    def fake_build_surface_bundle(request):
        diagnostics = {
            "request": {
                "ticker": request.ticker,
                "strike_min_pct": request.strike_min_pct,
                "strike_max_pct": request.strike_max_pct,
                "dte_min": request.dte_min,
                "dte_max": request.dte_max,
                "iv_source": request.iv_source,
                "quality_mode": request.quality_mode,
                "max_trade_age_hours": request.max_trade_age_hours,
                "smooth": bool(request.smooth),
                "include_low_confidence": bool(request.include_low_confidence),
            },
            "flag_counts": {},
            "internal_validation": {},
            "rows_retained": len(cleaned_df),
            "rows_surface_included": len(cleaned_df),
            "rows_surface_excluded": 0,
            "fallback_iv_fraction": 0.0,
        }
        return SurfaceBuildResult(
            request=request,
            current_price=100.0,
            raw_options_df=mock_df.copy(),
            cleaned_options_df=cleaned_df.copy(),
            diagnostics=diagnostics,
            figure=figure,
        )

    monkeypatch.setattr(main_module, "build_surface_bundle", fake_build_surface_bundle)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "TEST",
            "--smooth",
            "--output_dir",
            str(tmp_path),
            "--diagnostics_json",
            str(diagnostics_path),
            "--iv_source",
            "auto",
        ],
    )

    main_module.main()

    html_files = list(tmp_path.glob("TEST_unified_vol_surface_*.html"))
    assert html_files, "Expected output HTML file"
    assert diagnostics_path.exists(), "Expected diagnostics JSON file"

    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert "internal_validation" in diagnostics
    assert "flag_counts" in diagnostics
    assert diagnostics["request"]["iv_source"] == "auto"


def test_external_benchmark_report_computes_metrics(tmp_path):
    surface_df = pd.DataFrame(
        [
            {
                "optionType": "call",
                "strike": 100.0,
                "days_to_expiration": 10,
                "impliedVolatilityFinal": 0.25,
            },
            {
                "optionType": "put",
                "strike": 100.0,
                "days_to_expiration": 10,
                "impliedVolatilityFinal": 0.30,
            },
        ]
    )
    benchmark_df = pd.DataFrame(
        [
            {"optionType": "call", "strike": 100.0, "days_to_expiration": 10, "iv_ref": 0.24},
            {"optionType": "put", "strike": 100.0, "days_to_expiration": 10, "iv_ref": 0.31},
        ]
    )
    benchmark_path = tmp_path / "benchmark.csv"
    benchmark_df.to_csv(benchmark_path, index=False)

    report = main_module.build_external_benchmark_report(surface_df, str(benchmark_path))
    assert report["matched_rows"] == 2
    assert "mae_vol_points" in report
    assert "by_option_type" in report
