import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

from flask import Flask, request
import pytest

from src.request_timing import install_request_timing, measure_phase


def _records(capsys):
    return [json.loads(line.removeprefix("VOLEXPLORER_TIMING "))
            for line in capsys.readouterr().out.splitlines() if line.startswith("VOLEXPLORER_TIMING ")]


def test_disabled_timing_preserves_response_and_silence(monkeypatch, capsys):
    monkeypatch.delenv("VOL_REQUEST_TIMING", raising=False)
    app = Flask(__name__)
    install_request_timing(app)
    @app.get("/")
    def index():
        with measure_phase("spot"):
            return "unchanged", 201
    response = app.test_client().get("/?ticker=SPY")
    assert response.status_code == 201 and response.data == b"unchanged"
    assert "Server-Timing" not in response.headers
    assert _records(capsys) == []


def test_landing_page_stays_cacheable_without_timing(monkeypatch, capsys):
    monkeypatch.setenv("VOL_REQUEST_TIMING", "1")
    from web_app import create_app
    response = create_app().test_client().get("/")
    assert response.status_code == 200
    assert "Server-Timing" not in response.headers
    assert "s-maxage=3600" in response.headers["Vercel-CDN-Cache-Control"]
    assert _records(capsys) == []


def test_failed_stage_keeps_error_response_and_omits_message(monkeypatch, capsys):
    monkeypatch.setenv("VOL_REQUEST_TIMING", "1")
    app = Flask(__name__)
    install_request_timing(app)
    @app.get("/")
    def index():
        with measure_phase("spot"):
            raise ValueError("private upstream message")
    @app.errorhandler(ValueError)
    def failed(error):
        return "unavailable", 503
    response = app.test_client().get("/?ticker=SPY", headers={"Authorization": "Bearer never-log-me", "Cookie": "session=never-log-me"})
    assert response.status_code == 503 and response.data == b"unavailable"
    assert "spot;dur=" in response.headers["Server-Timing"]
    records = _records(capsys)
    assert len(records) == 1
    assert records[0]["error_phase"] == "spot" and records[0]["error_type"] == "ValueError"
    assert "private upstream message" not in json.dumps(records)
    assert "never-log-me" not in json.dumps(records)


def test_concurrent_requests_and_followup_do_not_share_phases(monkeypatch, capsys):
    monkeypatch.setenv("VOL_REQUEST_TIMING", "1")
    app = Flask(__name__)
    install_request_timing(app)
    barrier = Barrier(2)
    @app.get("/")
    def index():
        phase = "spot" if request.args["ticker"] == "SPY" else "options"
        with measure_phase(phase):
            if request.args.get("wait"):
                barrier.wait(timeout=5)
            return request.args["ticker"]
    def fetch(ticker):
        return app.test_client().get(f"/?ticker={ticker}&wait=1")
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(fetch, ["SPY", "TSLA"]))
    assert all(r.status_code == 200 for r in responses)
    assert "spot;dur=" in responses[0].headers["Server-Timing"]
    assert "options;dur=" not in responses[0].headers["Server-Timing"]
    assert "options;dur=" in responses[1].headers["Server-Timing"]
    assert "spot;dur=" not in responses[1].headers["Server-Timing"]
    later = app.test_client().get("/?ticker=SPY")
    assert "options;dur=" not in later.headers["Server-Timing"]
    records = _records(capsys)
    assert len(records) == 3
    assert len({r["surface_request_number"] for r in records}) == 3
    assert len({r["runtime_id"] for r in records}) == 1
    assert all(set(r["phases"]) == ({"spot"} if r["ticker"] == "SPY" else {"options"}) for r in records)


def test_web_surface_response_keeps_body_and_measures_html(monkeypatch, capsys):
    monkeypatch.setenv("VOL_REQUEST_TIMING", "1")
    from web_app import create_app
    def builder(surface_request):
        return SimpleNamespace(current_price=123.45, raw_options_df=[1],
            figure=SimpleNamespace(data=[1], to_html=lambda **kwargs: '<div id="test-chart">chart</div>'),
            diagnostics={"request": {"dte_min": surface_request.dte_min, "dte_max": surface_request.dte_max},
                         "surface_dte_min": 2, "surface_dte_max": 60, "contract_outcomes": {"used": 1}})
    response = create_app(surface_builder=builder).test_client().get("/?ticker=SPY&dte_min=2&dte_max=60")
    assert response.status_code == 200
    assert b'id="test-chart"' in response.data
    assert response.headers["Cache-Control"] == "public, max-age=0, must-revalidate"
    assert "html;dur=" in response.headers["Server-Timing"]
    assert _records(capsys)[0]["ticker"] == "SPY"


def test_phase_without_web_request_preserves_exception():
    with pytest.raises(RuntimeError, match="same error"):
        with measure_phase("surface"):
            raise RuntimeError("same error")


def test_nested_failure_preserves_deepest_phase(monkeypatch, capsys):
    monkeypatch.setenv("VOL_REQUEST_TIMING", "1")
    app = Flask(__name__)
    install_request_timing(app)
    @app.get("/")
    def index():
        with measure_phase("options"):
            with measure_phase("expiry_discovery"):
                raise ValueError("not logged")
    @app.errorhandler(ValueError)
    def failed(error):
        return "unavailable", 503
    response = app.test_client().get("/?ticker=SPY")
    assert response.status_code == 503
    assert _records(capsys)[0]["error_phase"] == "expiry_discovery"
