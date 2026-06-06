from __future__ import annotations

import os
from typing import Any, Callable, Dict

from flask import Flask, render_template, request

from src.surface_service import SurfaceBuildResult, SurfaceRequest, build_surface_bundle


DEFAULT_FORM_VALUES = {
    "ticker": "",
    "strike_min_pct": 93,
    "strike_max_pct": 107,
    "dte_min": 7,
    "dte_max": 60,
    "iv_source": "black-scholes",
    "smooth": True,
}

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.plot.ly; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "font-src 'self' data:; "
        "connect-src 'self'; "
        "worker-src 'self' blob:; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "frame-ancestors 'none'; "
        "form-action 'self'"
    ),
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()",
    "Referrer-Policy": "no-referrer",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}


def _parse_int_arg(
    name: str,
    default: int,
    min_value: int,
    max_value: int,
) -> int:
    raw_value = request.args.get(name, default)
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        return int(default)
    return max(min_value, min(max_value, parsed))


def _parse_bool_arg(name: str, default: bool) -> bool:
    raw_value = request.args.get(name)
    if raw_value is None:
        return bool(default)
    return str(raw_value).lower() in {"1", "true", "on", "yes"}


def _form_values() -> Dict[str, Any]:
    ticker = (request.args.get("ticker") or DEFAULT_FORM_VALUES["ticker"]).strip().upper()
    iv_source = (request.args.get("iv_source") or DEFAULT_FORM_VALUES["iv_source"]).strip().lower()
    if iv_source not in {"yfinance", "black-scholes"}:
        iv_source = str(DEFAULT_FORM_VALUES["iv_source"])

    return {
        "ticker": ticker,
        "strike_min_pct": _parse_int_arg(
            "strike_min_pct", int(DEFAULT_FORM_VALUES["strike_min_pct"]), 50, 100
        ),
        "strike_max_pct": _parse_int_arg(
            "strike_max_pct", int(DEFAULT_FORM_VALUES["strike_max_pct"]), 100, 150
        ),
        "dte_min": _parse_int_arg("dte_min", int(DEFAULT_FORM_VALUES["dte_min"]), 1, 180),
        "dte_max": _parse_int_arg("dte_max", int(DEFAULT_FORM_VALUES["dte_max"]), 1, 365),
        "iv_source": iv_source,
        "smooth": _parse_bool_arg("smooth", bool(DEFAULT_FORM_VALUES["smooth"])),
    }


def _summary_cards(result: SurfaceBuildResult) -> list[dict[str, str]]:
    diagnostics = result.diagnostics
    internal_validation = diagnostics.get("internal_validation", {})
    request_diagnostics = diagnostics.get("request", {})
    dte_coverage = diagnostics.get("dte_coverage", {})
    return [
        {"label": "Spot", "value": f"${result.current_price:,.2f}"},
        {"label": "Raw Contracts", "value": f"{len(result.raw_options_df):,}"},
        {
            "label": "Surface Quotes",
            "value": f"{diagnostics.get('rows_surface_included', 0):,}",
        },
        {
            "label": "Requested DTE",
            "value": (
                f"{request_diagnostics.get('dte_min')}-{request_diagnostics.get('dte_max')}"
                if request_diagnostics.get("dte_min") is not None
                and request_diagnostics.get("dte_max") is not None
                else "n/a"
            ),
        },
        {
            "label": "Included DTE",
            "value": (
                f"{diagnostics.get('surface_dte_min')}-{diagnostics.get('surface_dte_max')}"
                if diagnostics.get("surface_dte_min") is not None
                and diagnostics.get("surface_dte_max") is not None
                else "n/a"
            ),
        },
        {
            "label": "Surface Strikes",
            "value": (
                f"{diagnostics.get('surface_strike_min'):.0f}-{diagnostics.get('surface_strike_max'):.0f}"
                if diagnostics.get("surface_strike_min") is not None
                and diagnostics.get("surface_strike_max") is not None
                else "n/a"
            ),
        },
        {
            "label": "Fetched DTE",
            "value": (
                f"{dte_coverage.get('selected_min')}-{dte_coverage.get('selected_max')}"
                if dte_coverage.get("selected_min") is not None
                and dte_coverage.get("selected_max") is not None
                else "n/a"
            ),
        },
        {
            "label": "Next DTE",
            "value": (
                str(dte_coverage.get("next_available_after_requested_max"))
                if dte_coverage.get("next_available_after_requested_max") is not None
                else "n/a"
            ),
        },
        {
            "label": "Quote IV Share",
            "value": f"{100.0 * float(diagnostics.get('quote_derived_iv_fraction', diagnostics.get('fallback_iv_fraction', 0.0))):.1f}%",
        },
        {
            "label": "Arb Scope",
            "value": str(
                diagnostics.get("static_arbitrage", {}).get(
                    "certification_scope", "n/a"
                )
            ),
        },
        {
            "label": "Repricing MAE",
            "value": (
                f"{float(internal_validation.get('repricing_mae')):.4f}"
                if internal_validation.get("repricing_mae") is not None
                else "n/a"
            ),
        },
    ]


def create_app(
    surface_builder: Callable[[SurfaceRequest], SurfaceBuildResult] = build_surface_bundle,
) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 1024

    @app.after_request
    def apply_security_headers(response):
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        if request.endpoint == "index":
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.route("/", methods=["GET"])
    def index():
        values = _form_values()
        error_message = None
        plot_html = None
        diagnostics = None
        cards = []

        if values["ticker"]:
            try:
                result = surface_builder(
                    SurfaceRequest(
                        ticker=str(values["ticker"]),
                        strike_min_pct=float(values["strike_min_pct"]) / 100.0,
                        strike_max_pct=float(values["strike_max_pct"]) / 100.0,
                        dte_min=int(values["dte_min"]),
                        dte_max=int(values["dte_max"]),
                        smooth=bool(values["smooth"]),
                        iv_source=str(values["iv_source"]),
                        quality_mode="lenient",
                    )
                )
                plot_html = result.figure.to_html(
                    full_html=False,
                    include_plotlyjs="cdn",
                    config={"responsive": True, "displaylogo": False},
                )
                diagnostics = result.diagnostics
                cards = _summary_cards(result)
            except ValueError as exc:
                error_message = str(exc)
            except Exception:
                app.logger.exception("Unexpected surface build failure")
                error_message = (
                    "The surface could not be built right now. Try another ticker or range."
                )

        return render_template(
            "index.html",
            values=values,
            plot_html=plot_html,
            diagnostics=diagnostics,
            cards=cards,
            error_message=error_message,
        )

    return app


app = create_app()


if __name__ == "__main__":
    debug_enabled = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(host="127.0.0.1", port=5000, debug=debug_enabled)
