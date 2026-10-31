"""Opt-in request timings. Logs phases and a worker token, never quote/auth data."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import os
import re
from threading import Lock
from time import perf_counter, thread_time

_active = ContextVar("surface_request_timing", default=None)
_runtime_id = os.urandom(6).hex()
_sequence = 0
_sequence_lock = Lock()


def browser_timing_head():
    if os.environ.get("VOL_REQUEST_TIMING") != "1":
        return ""
    return """<script>
document.addEventListener('click', function(event) {
  if (!event.target.closest('.primary-button')) return;
  try { sessionStorage.setItem('volexplorer.buildClick', JSON.stringify({epochMs:performance.timeOrigin+performance.now()})); } catch (error) {}
}, true);
</script>"""


def browser_timing_after_plot(built_at=None):
    if os.environ.get("VOL_REQUEST_TIMING") != "1":
        return None
    state = _active.get()
    sequence = state["surface_request_number"] if state else 0
    script = """requestAnimationFrame(function() { requestAnimationFrame(function() {
  try {
    const stored = sessionStorage.getItem('volexplorer.buildClick');
    sessionStorage.removeItem('volexplorer.buildClick');
    if (!stored) return;
    const click = JSON.parse(stored);
    const finished = performance.timeOrigin + performance.now();
    if (finished-click.epochMs > 120000) return;
    const nav = performance.getEntriesByType('navigation')[0];
    console.info('VOLEXPLORER_BROWSER_TIMING', JSON.stringify({
      runtime_id:__RUNTIME_ID__, surface_request_number:__SEQUENCE__,
      surface_built_utc:__BUILT_AT__, server_timing_scope:'original-build',
      clicked_utc:new Date(click.epochMs).toISOString(), click_to_chart_ms:finished-click.epochMs,
      request_to_first_byte_ms:nav ? nav.responseStart-nav.requestStart : null,
      server_timing:nav ? nav.serverTiming.map(function(x) {return {name:x.name,duration_ms:x.duration};}) : []
    }));
  } catch (error) {}
}); });"""
    return script.replace("__RUNTIME_ID__", json.dumps(_runtime_id)).replace("__SEQUENCE__", str(sequence)).replace("__BUILT_AT__", json.dumps(built_at))


def _emit(record):
    try:
        print("VOLEXPLORER_TIMING " + json.dumps(record, separators=(",", ":")), flush=True)
    except (OSError, ValueError):
        pass


def log_runtime_ready(wall_ms, thread_cpu_ms, **preparation):
    if os.environ.get("VOL_REQUEST_TIMING") == "1":
        _emit({"event": "runtime_ready", "runtime_id": _runtime_id,
               "web_app_import_ms": round(wall_ms, 3), "thread_cpu_ms": round(thread_cpu_ms, 3),
               **{key: round(value, 3) for key, value in preparation.items()}})


@contextmanager
def measure_phase(name):
    state = _active.get()
    if state is None:
        yield
        return
    started, cpu = perf_counter(), thread_time()
    try:
        yield
    except Exception as error:
        state.setdefault("error_phase", name)
        state.setdefault("error_type", type(error).__name__)
        raise
    finally:
        row = state["phases"].setdefault(name, {"wall_ms": 0.0, "thread_cpu_ms": 0.0})
        row["wall_ms"] += (perf_counter() - started) * 1000
        row["thread_cpu_ms"] += (thread_time() - cpu) * 1000


def install_request_timing(app):
    if os.environ.get("VOL_REQUEST_TIMING") != "1":
        return
    from flask import g, request

    @app.before_request
    def begin_timing():
        global _sequence
        if request.path != "/" or not (request.args.get("ticker") or "").strip():
            return
        with _sequence_lock:
            _sequence += 1
            number = _sequence
        ticker = (request.args.get("ticker") or "").strip().upper()
        state = {"event": "surface_request", "runtime_id": _runtime_id,
                 "surface_request_number": number,
                 "ticker": ticker if re.fullmatch(r"[A-Z0-9.^=-]{1,16}", ticker) else None,
                 "started_utc": datetime.now(timezone.utc).isoformat(),
                 "started": perf_counter(), "phases": {}}
        g.surface_timing_token = _active.set(state)

    @app.after_request
    def finish_timing(response):
        state = _active.get()
        if state is None:
            return response
        elapsed = (perf_counter() - state["started"]) * 1000
        phases = state["phases"]
        values = [f"{name};dur={row['wall_ms']:.3f}" for name, row in phases.items()]
        values.append(f"app;dur={elapsed:.3f}")
        response.headers.add("Server-Timing", ", ".join(values))
        response.headers["X-Surface-Runtime"] = _runtime_id
        response.headers["X-Surface-Request"] = str(state["surface_request_number"])
        record = {key: value for key, value in state.items() if key != "started"}
        record.update(app_ms=round(elapsed, 3), status=response.status_code)
        _emit(record)
        return response

    @app.teardown_request
    def clear_timing(error):
        token = g.pop("surface_timing_token", None)
        if token is not None:
            _active.reset(token)
