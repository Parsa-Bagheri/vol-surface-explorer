from __future__ import annotations

import os
import hashlib
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict

from flask import Flask, g, render_template, request, send_file, url_for
from datetime import datetime, timezone
from src.ticker_validation import validate_ticker_symbol as _validate_ticker_symbol
from src.request_timing import install_request_timing, measure_phase, browser_timing_head, browser_timing_after_plot
from src.surface_config import (
    DEFAULT_WEB_DTE_MIN, MAX_DTE, MAX_STRIKE_PCT, MIN_DTE, MIN_STRIKE_PCT, SurfaceRequest,
)


DEFAULT_FORM_VALUES = {
    "ticker": "",
    "strike_min_pct": round(SurfaceRequest.strike_min_pct * 100),
    "strike_max_pct": round(SurfaceRequest.strike_max_pct * 100),
    "dte_min": DEFAULT_WEB_DTE_MIN,
    "dte_max": SurfaceRequest.dte_max,
    "smooth": SurfaceRequest.smooth,
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

_runtime_prepare_lock = Lock()
_runtime_prepared = False


def _prepare_surface_runtime() -> None:
    global _runtime_prepared
    with _runtime_prepare_lock:
        if _runtime_prepared:
            return
        from src import surface_service
        from plotly.graph_objects import Figure, Surface
        from src.numerical_engine import resolve_backend

        # Fail readiness if an explicitly requested backend is unavailable.
        # Prepare code/validators only: never fetch a quote or fit a surface.
        resolve_backend(None)
        figure = Figure(data=[Surface(z=[[0.2, 0.2], [0.2, 0.2]],
                                      colorbar={"title": "IV (%)"})])
        figure.update_layout(scene={"xaxis": {"title": "Strike Price"}})
        figure.to_json()
        _runtime_prepared = True


def _default_surface_builder(request: Any) -> Any:
    with measure_phase("imports"):
        from src.surface_service import build_surface_bundle

    return build_surface_bundle(request)


def _surface_request(**kwargs: Any) -> Any:
    return SurfaceRequest(**kwargs)


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
    raw_values = request.args.getlist(name)
    if not raw_values:
        return bool(default)
    return any(str(value).lower() in {"1", "true", "on", "yes"} for value in raw_values)


def _form_values() -> Dict[str, Any]:
    ticker = (request.args.get("ticker") or DEFAULT_FORM_VALUES["ticker"]).strip().upper()

    return {
        "ticker": ticker,
        "strike_min_pct": _parse_int_arg(
            "strike_min_pct", int(DEFAULT_FORM_VALUES["strike_min_pct"]),
            round(MIN_STRIKE_PCT * 100), round(MAX_STRIKE_PCT * 100),
        ),
        "strike_max_pct": _parse_int_arg(
            "strike_max_pct", int(DEFAULT_FORM_VALUES["strike_max_pct"]),
            round(MIN_STRIKE_PCT * 100), round(MAX_STRIKE_PCT * 100),
        ),
        "dte_min": _parse_int_arg("dte_min", int(DEFAULT_FORM_VALUES["dte_min"]), MIN_DTE, MAX_DTE),
        "dte_max": _parse_int_arg("dte_max", int(DEFAULT_FORM_VALUES["dte_max"]), MIN_DTE, MAX_DTE),
        "smooth": _parse_bool_arg("smooth", bool(DEFAULT_FORM_VALUES["smooth"])),
    }


def _dte_range_label(min_value: Any, max_value: Any) -> str:
    if min_value is None or max_value is None:
        return "n/a"
    return f"{min_value}-{max_value}"


def _surface_metric_cards(result: Any) -> list[dict[str, str]]:
    diagnostics = result.diagnostics
    request_diagnostics = diagnostics.get("request", {})
    return [
        {"label": "Spot", "value": f"${result.current_price:,.2f}"},
        {
            "label": "Requested DTE",
            "value": _dte_range_label(
                request_diagnostics.get("dte_min"),
                request_diagnostics.get("dte_max"),
            ),
        },
        {
            "label": "Available DTE",
            "value": _dte_range_label(
                diagnostics.get("surface_dte_min"),
                diagnostics.get("surface_dte_max"),
            ),
        },
    ]


def _advanced_metric_cards(result: Any) -> list[dict[str, str]]:
    diagnostics = result.diagnostics
    outcomes = diagnostics.get("contract_outcomes", {})
    return [
        {"label": "Contracts fetched", "value": f"{len(result.raw_options_df):,}"},
        {
            "label": "Contracts used",
            "value": f"{outcomes.get('used', 0):,}",
        },
    ]


def create_app(
    surface_builder: Callable[[Any], Any] = _default_surface_builder,
    history_root: str | None = None,
    runtime_preparer: Callable[[], None] = _prepare_surface_runtime,
) -> Flask:
    app = Flask(__name__)
    install_request_timing(app)
    app.config["MAX_CONTENT_LENGTH"] = 1024
    configured_history = history_root or os.environ.get("VOL_HISTORY_ROOT")
    app.config["HISTORY_ROOT"] = str((Path(__file__).parent / (configured_history or "history_bundle")).resolve())
    app.config["HISTORY_BUNDLE_REQUIRED"] = not configured_history or bool(os.environ.get("VERCEL"))
    asset_versions = {
        name: hashlib.sha256((Path(app.static_folder) / name).read_bytes()).hexdigest()[:12]
        for name in ("app.css", "app.js", "history.css")
    }

    @app.after_request
    def apply_security_headers(response):
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        if request.endpoint == "index":
            if (request.args.get("ticker") or "").strip():
                if getattr(g, "surface_cacheable", False):
                    response.headers["Cache-Control"] = "public, max-age=0, must-revalidate"
                    response.headers["Vercel-CDN-Cache-Control"] = "public, s-maxage=30"
                    response.headers["X-Surface-Built-At"] = g.surface_built_at
                    response.headers["X-Surface-Timing-Scope"] = "original-build"
                else:
                    response.headers["Cache-Control"] = "no-store"
                    response.headers["Vercel-CDN-Cache-Control"] = "no-store"
            else:
                response.headers["Cache-Control"] = "public, max-age=0"
                response.headers["Vercel-CDN-Cache-Control"] = (
                    "public, s-maxage=3600, stale-while-revalidate=86400"
                )
        elif request.endpoint in {"history", "history_file"}:
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.post("/prepare-surface")
    def prepare_surface():
        runtime_preparer()
        return "", 204, {"Cache-Control": "no-store"}

    @app.route("/", methods=["GET"])
    def index():
        values = _form_values()
        error_message = None
        plot_html = None
        diagnostics = None
        surface_cards = []
        advanced_cards = []
        built_at = None

        if values["ticker"]:
            try:
                ticker = _validate_ticker_symbol(str(values["ticker"]))
                result = surface_builder(
                    _surface_request(
                        ticker=ticker,
                        strike_min_pct=float(values["strike_min_pct"]) / 100.0,
                        strike_max_pct=float(values["strike_max_pct"]) / 100.0,
                        dte_min=int(values["dte_min"]),
                        dte_max=int(values["dte_max"]),
                        smooth=bool(values["smooth"]),
                        quality_mode="lenient",
                    )
                )
                if not result.figure.data:
                    raise ValueError("No usable surface could be constructed. Try another range.")
                built_at = datetime.now(timezone.utc).isoformat()
                with measure_phase("html"):
                    plot_html = result.figure.to_html(
                        full_html=False,
                        post_script=browser_timing_after_plot(built_at),
                        include_plotlyjs="cdn",
                        default_width="100%",
                        default_height="100%",
                        config={"responsive": True, "displaylogo": False},
                    )
                diagnostics = result.diagnostics
                surface_cards = _surface_metric_cards(result)
                advanced_cards = _advanced_metric_cards(result)
                # The CDN keys by URL including the complete query. Never retain
                # errors, empty figures or an explicit fresh-data request.
                g.surface_cacheable = "fresh" not in request.args
                g.surface_built_at = built_at
            except ValueError as exc:
                error_message = str(exc)
            except Exception:
                app.logger.exception("Unexpected surface build failure")
                error_message = (
                    "The surface could not be built right now. Try another ticker or range."
                )

        return render_template(
            "index.html",
            request_timing_script=browser_timing_head(),
            built_at=built_at,
            fresh_url=url_for("index", **values, fresh="1"),
            values=values,
            min_dte=MIN_DTE,
            max_dte=MAX_DTE,
            asset_versions=asset_versions,
            plot_html=plot_html,
            diagnostics=diagnostics,
            surface_cards=surface_cards,
            advanced_cards=advanced_cards,
            outcomes=diagnostics.get("contract_outcomes", {}) if diagnostics else {},
            error_message=error_message,
        )

    @app.get("/history")
    def history():
        # Keep the live landing page's deferred financial imports intact.
        choices, invalid, selected, plots, comparison = [], [], [], [], None
        error = None
        left_id, right_id = request.args.get("left", ""), request.args.get("right", "")
        try:
            from src.history_store import ObservationError
            from src.history_reader import open_history_store
            from src.history_compare import compare_observations, saved_figure
            store = open_history_store(app.config["HISTORY_ROOT"], require_bundle=app.config["HISTORY_BUNDLE_REQUIRED"])
            observations, invalid = store.list()
            for observation in observations:
                artifacts = []
                for path in sorted((store.root / "processed" / observation["snapshot_id"]).glob("*")):
                    if path.is_dir() and not path.name.startswith("."):
                        try:
                            artifacts.append(store.load_artifact(observation["snapshot_id"], path.name))
                        except ObservationError as exc:
                            invalid.append({"snapshot_id": observation["snapshot_id"], "error": str(exc)})
                choices.append({"observation": observation, "artifacts": artifacts})
            for side, snapshot_id in (("left", left_id), ("right", right_id)):
                if not snapshot_id:
                    continue
                entry = next((item for item in choices if item["observation"]["snapshot_id"] == snapshot_id), None)
                if entry is None:
                    raise ObservationError("Requested observation is unavailable or invalid.")
                processing_id = request.args.get(f"{side}_processing", "")
                if not processing_id:
                    if len(entry["artifacts"]) != 1:
                        raise ObservationError("Select a saved processing version for this observation.")
                    processing_id = entry["artifacts"][0]["processing_id"]
                artifact = store.load_artifact(snapshot_id, processing_id)
                selected.append({"observation": entry["observation"], "artifact": artifact, "side": side})
                plots.append(saved_figure(store, snapshot_id, processing_id).to_html(full_html=False,
                             include_plotlyjs="cdn" if not plots else False,
                             config={"responsive": True, "displaylogo": False}, default_height="480px"))
            if len(selected) == 2:
                comparison = compare_observations(store, left_id, right_id,
                             selected[0]["artifact"]["processing_id"], selected[1]["artifact"]["processing_id"])
                comparison["plot_html"] = comparison["figure"].to_html(full_html=False, include_plotlyjs=False,
                              config={"responsive": True, "displaylogo": False}, default_height="440px")
        except ImportError:
            error = "History requires the optional local dependencies in requirements-history.txt."
        except ValueError as exc:
            error = str(exc)
        return render_template("history.html", choices=choices, invalid=invalid, selected=selected, plots=plots,
                               comparison=comparison, error_message=error, left_id=left_id, right_id=right_id,
                               asset_versions=asset_versions)

    @app.get("/history/<snapshot_id>/files/<name>")
    def history_file(snapshot_id, name):
        from src.history_reader import open_history_store
        try:
            store = open_history_store(app.config["HISTORY_ROOT"], require_bundle=app.config["HISTORY_BUNDLE_REQUIRED"])
            store.load(snapshot_id)
            if name in {"raw.parquet", "observation.parquet", "manifest.json"}:
                path = store.observation_path(snapshot_id) / name
            elif name in {"cleaned.parquet", "selected.parquet", "coverage.parquet", "diagnostics.json", "surface.json", "processing-manifest.json"}:
                processing_id = request.args.get("processing", "")
                store.load_artifact(snapshot_id, processing_id)
                path = store.artifact_path(snapshot_id, processing_id) / ("manifest.json" if name == "processing-manifest.json" else name)
            else:
                return "Unknown saved artifact", 404
            return send_file(path, as_attachment=True, download_name=f"{snapshot_id}-{name}")
        except ValueError as exc:
            return str(exc), 404

    return app


app = create_app()


def _server_port() -> int:
    try:
        return int(os.environ.get("PORT", "5000"))
    except ValueError:
        return 5000


if __name__ == "__main__":
    debug_enabled = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(host="127.0.0.1", port=_server_port(), debug=debug_enabled)
