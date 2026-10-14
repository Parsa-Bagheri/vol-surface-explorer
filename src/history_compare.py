"""Compare saved fitted surfaces only inside both supported domains."""
from __future__ import annotations

import base64
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from src.history_store import LocalObservationStore, ObservationError


def _array(value) -> np.ndarray:
    if isinstance(value, dict) and "bdata" in value:
        array = np.frombuffer(base64.b64decode(value["bdata"]), dtype=value["dtype"])
        if value.get("shape"):
            array = array.reshape(tuple(int(item) for item in value["shape"].split(",")))
        return array
    return np.asarray(value, dtype=float)


def saved_figure(store: LocalObservationStore, snapshot_id: str, processing_id: str) -> go.Figure:
    store.load_artifact(snapshot_id, processing_id)
    path = store.artifact_path(snapshot_id, processing_id)
    figure = pio.from_json((path / "surface.json").read_text("utf-8"))
    for trace in figure.data:
        if trace.type == "surface":
            trace.name = "Interpolated adjusted surface"
            trace.showlegend = True
            trace.hovertemplate += "<br>Interpolated; not an observed chain"
    quotes = pd.read_parquet(path / "selected.parquet")
    if not quotes.empty:
        figure.add_scatter3d(x=quotes.strike, y=quotes.days_to_expiration,
                            z=quotes.impliedVolatilityFinal * 100, mode="markers",
                            name="Observed quote-derived IV", marker={"size": 3, "color": "#f3cf75"},
                            hovertemplate="Observed quote-derived IV<br>Strike %{x:.2f}<br>DTE %{y:.0f}<br>IV %{z:.4f}%<extra></extra>")
    figure.update_layout(showlegend=True, legend={"orientation": "h", "y": 1.05, "x": 0})
    return figure


def _grid(path: Path):
    figure = pio.from_json((path / "surface.json").read_text("utf-8"))
    if not figure.data:
        raise ObservationError("Saved observation has no supported fitted surface.")
    trace = figure.data[0]
    return trace.type, _array(trace.x), _array(trace.y), _array(trace.z)


def _evaluate(x: np.ndarray, y: np.ndarray, z: np.ndarray, strike: float, dte: float) -> float:
    times = y[:, 0]
    upper = int(np.searchsorted(times, dte, side="left"))
    if upper >= len(times) or dte < times[0]:
        return float("nan")
    lower = upper if times[upper] == dte else upper - 1
    values = []
    for row in (lower, upper):
        order = np.argsort(x[row])
        strikes, values_row = x[row][order], z[row][order]
        if len(strikes) < 2 or strike < strikes.min() or strike > strikes.max():
            return float("nan")
        hi = int(np.searchsorted(strikes, strike, side="left"))
        lo = hi if strikes[hi] == strike else hi - 1
        if not np.isfinite(values_row[[lo, hi]]).all():
            return float("nan")  # Never bridge an unsupported interior grid cell.
        values.append(float(np.interp(strike, strikes[[lo, hi]], values_row[[lo, hi]])))
    if lower == upper:
        return values[0]
    weight = (dte - times[lower]) / (times[upper] - times[lower])
    return values[0] + weight * (values[1] - values[0])


def compare_observations(store: LocalObservationStore, left_id: str, right_id: str,
                         left_processing: str, right_processing: str) -> dict:
    left = store.load(left_id)[0]
    right = store.load(right_id)[0]
    a = store.load_artifact(left_id, left_processing)
    b = store.load_artifact(right_id, right_processing)
    if left["parameters"] != right["parameters"] or a["processor"] != b["processor"]:
        raise ObservationError("Comparison requires the same ticker, saved parameters and processing code/dependencies. Reprocess explicitly to align versions.")
    kind_a, xa, ya, za = _grid(store.artifact_path(left_id, left_processing))
    kind_b, xb, yb, zb = _grid(store.artifact_path(right_id, right_processing))
    if kind_a != kind_b:
        raise ObservationError("One observation has only adjusted nodes; use two supported grids or two node observations.")
    rows = []
    if kind_a == "surface":
        strikes = np.linspace(max(xa.min(), xb.min()), min(xa.max(), xb.max()), 36)
        dtes = np.linspace(max(ya.min(), yb.min()), min(ya.max(), yb.max()), 32)
        if strikes[0] >= strikes[-1] or dtes[0] >= dtes[-1]:
            raise ObservationError("Observations have no overlapping supported strike and DTE domain.")
        delta = np.full((len(dtes), len(strikes)), np.nan)
        for i, dte in enumerate(dtes):
            for j, strike in enumerate(strikes):
                va, vb = _evaluate(xa, ya, za, strike, dte), _evaluate(xb, yb, zb, strike, dte)
                if np.isfinite(va) and np.isfinite(vb):
                    delta[i, j] = vb - va
                    rows.append({"strike": float(strike), "dte": float(dte), "left_iv_pct": va,
                                 "right_iv_pct": vb, "change_vol_points": vb - va, "kind": "interpolated"})
        figure = go.Figure(go.Heatmap(x=strikes, y=dtes, z=delta, colorscale="RdBu_r", zmid=0,
                           colorbar={"title": "IV points"}, hoverongaps=False,
                           hovertemplate="Interpolated common domain<br>Strike %{x:.2f}<br>DTE %{y:.2f}<br>Change %{z:.4f} IV points<extra></extra>"))
        unsupported = int(np.isnan(delta).sum())
    else:
        points_b = {(float(x), float(y)): float(z) for x, y, z in zip(xb, yb, zb)}
        for x, y, z in zip(xa, ya, za):
            if (float(x), float(y)) in points_b:
                vb = points_b[(float(x), float(y))]
                rows.append({"strike": float(x), "dte": float(y), "left_iv_pct": float(z),
                             "right_iv_pct": vb, "change_vol_points": vb - float(z), "kind": "adjusted_node"})
        figure = go.Figure(go.Scatter(x=[row["strike"] for row in rows], y=[row["change_vol_points"] for row in rows],
                                     mode="markers", name="Matching adjusted nodes"))
        unsupported = len(xa) + len(xb) - 2 * len(rows)
    if not rows:
        raise ObservationError("No common supported values; extrapolation is excluded.")
    figure.update_layout(template="plotly_dark", xaxis_title="Strike ($)",
                         yaxis_title="DTE" if kind_a == "surface" else "Change (IV points)",
                         paper_bgcolor="#101a2a", plot_bgcolor="#101a2a", margin={"t": 25})
    return {"left": left, "right": right, "figure": figure, "rows": rows,
            "supported_points": len(rows), "unsupported_points": unsupported,
            "mean_change_vol_points": float(np.mean([row["change_vol_points"] for row in rows])),
            "domain": {"strike_min": min(row["strike"] for row in rows), "strike_max": max(row["strike"] for row in rows),
                       "dte_min": min(row["dte"] for row in rows), "dte_max": max(row["dte"] for row in rows)}}
