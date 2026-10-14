import json
from pathlib import Path
import socket

import numpy as np
import pandas as pd
import pytest

import src.history_store as history_store
from src.history_store import LocalObservationStore, ObservationError, canonical, digest
from src.history_demo import fixture_quotes, seed_demo
from src.surface_config import SurfaceRequest
from src.surface_service import process_surface_inputs
from src.history_compare import compare_observations, _evaluate, _array
from src.time_utils import time_to_expiration_years, expiration_dte
from src.data_cleaner import _black_scholes_price
from web_app import create_app


AS_OF = "2026-10-12T18:00:00Z"
REQUEST = SurfaceRequest("DEMO", strike_min_pct=.9, strike_max_pct=1.1, dte_min=0, dte_max=60, smooth=True)


def save(store, raw=None, snapshot_id="sample", as_of=AS_OF, request=REQUEST):
    if raw is None:
        raw = fixture_quotes(as_of)
    return store.save(request, raw, {"value": 100, "provider": "test_fixture", "method": "fixture",
                                    "collected_at_utc": as_of, "source_timestamp_utc": as_of},
                      as_of, source_kind="synthetic_fixture", snapshot_id=snapshot_id)


def rewrite_manifest(path, mutate):
    envelope = json.loads(path.read_text("utf-8"))
    mutate(envelope["payload"])
    envelope["manifest_sha256"] = digest(envelope["payload"])
    path.write_bytes(canonical(envelope))


def test_source_fingerprint_is_equal_for_lf_and_crlf_checkouts(tmp_path):
    lf, crlf = tmp_path / "lf.py", tmp_path / "crlf.py"
    lf.write_bytes(b"def processor():\n    return 42\n")
    crlf.write_bytes(b"def processor():\r\n    return 42\r\n")
    assert history_store.source_digest(lf) == history_store.source_digest(crlf)
    crlf.write_bytes(b"def processor():\r\n    return 43\r\n")
    assert history_store.source_digest(lf) != history_store.source_digest(crlf)


def test_offline_processor_matches_original_live_pipeline_golden():
    golden = json.loads((Path(__file__).parent / "fixtures/original-live-bundle.json").read_text("utf-8"))
    assert golden["baseline_commit"] == "72e763b342aeea8a6ab014ae4ca00b7eec8ea815"
    result = process_surface_inputs(REQUEST, fixture_quotes(AS_OF), golden["spot"], golden["valuation_time_utc"])
    np.testing.assert_allclose(result.cleaned_options_df[golden["columns"]].to_numpy(dtype=float),
                               pd.DataFrame(golden["rows"])[golden["columns"]].to_numpy(dtype=float),
                               atol=1e-10, rtol=1e-10, equal_nan=True)
    assert result.diagnostics["contract_outcomes"] == golden["contract_outcomes"]
    actual = json.loads(result.figure.to_json())["data"]
    for a, b in zip(actual, golden["figure_data"]):
        for field in ("x", "y", "z"):
            np.testing.assert_allclose(_array(a[field]), _array(b[field]), rtol=1e-10, atol=1e-10, equal_nan=True)


def test_replay_preserves_original_spot_time_and_makes_no_network_calls(tmp_path, monkeypatch):
    store = LocalObservationStore(tmp_path)
    save(store)
    artifact = store.process("sample")
    def blocked(*args, **kwargs):
        raise AssertionError("Network/live clock access is prohibited during replay")
    import src.surface_service as service
    import src.data_fetch as fetch
    monkeypatch.setattr(service, "get_current_price", blocked)
    monkeypatch.setattr(service, "get_options_data", blocked)
    monkeypatch.setattr(fetch.yf, "Ticker", blocked)
    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(pd.Timestamp, "now", blocked)
    replay = store.replay("sample", artifact["processing_id"])
    assert replay["matched"] and replay["reproduction"]
    assert replay["spot"] == 100 and replay["valuation_time_utc"] == "2026-10-12T18:00:00+00:00"


def test_processing_and_collection_reruns_are_idempotent(tmp_path):
    store = LocalObservationStore(tmp_path)
    first = save(store)
    assert save(store) == first
    a = store.process("sample")
    b = store.process("sample")
    assert a == b
    assert store.query("SELECT COUNT(*) n FROM coverage").iloc[0].n == 1
    assert store.query("SELECT COUNT(*) n FROM raw_quotes").iloc[0].n == 66


@pytest.mark.parametrize("change", ["date", "ticker", "rows"])
def test_observation_ids_are_globally_immutable(tmp_path, change):
    store = LocalObservationStore(tmp_path)
    save(store)
    with pytest.raises(ObservationError, match="globally immutable"):
        if change == "date":
            save(store, as_of="2026-10-13T18:00:00Z")
        elif change == "ticker":
            save(store, request=SurfaceRequest("OTHER"))
        else:
            save(store, raw=fixture_quotes(AS_OF).iloc[:-1])
    assert len(store.list()[0]) == 1
    assert store.load("sample")[0]["ticker"] == "DEMO"


def test_interruptions_never_publish_complete_missing_data_and_keep_original_readable(tmp_path, monkeypatch):
    store = LocalObservationStore(tmp_path)
    save(store)
    def interrupted(*args):
        raise OSError("Simulated power interruption before atomic rename")
    monkeypatch.setattr(history_store, "_publish", interrupted)
    with pytest.raises(OSError, match="interruption"):
        save(store, snapshot_id="interrupted")
    with pytest.raises(OSError, match="interruption"):
        store.process("sample")
    assert store.load("sample")[0]["raw_row_count"] == 66
    assert [item["snapshot_id"] for item in store.list()[0]] == ["sample"]
    assert not list(tmp_path.glob(".pending-*"))
    assert not list((tmp_path / "processed").glob("*/*/manifest.json"))


def test_stale_or_concurrent_id_lock_is_visible_without_breaking_existing_data(tmp_path):
    store = LocalObservationStore(tmp_path)
    save(store)
    (tmp_path / ".locks" / "sample").mkdir()
    with pytest.raises(ObservationError, match="publication in progress or interrupted"):
        save(store)
    assert store.load("sample")[0]["raw_row_count"] == 66


@pytest.mark.parametrize("corruption", ["raw", "manifest", "missing", "schema", "partition", "lineage"])
def test_corrupt_or_incompatible_observations_are_rejected(tmp_path, corruption):
    store = LocalObservationStore(tmp_path)
    save(store)
    directory = store.observation_path("sample")
    if corruption == "raw":
        with (directory / "raw.parquet").open("ab") as handle:
            handle.write(b"corrupt")
    elif corruption == "manifest":
        envelope = json.loads((directory / "manifest.json").read_text("utf-8"))
        envelope["payload"]["spot"]["value"] = 999
        (directory / "manifest.json").write_bytes(canonical(envelope))
    elif corruption == "missing":
        (directory / "raw.parquet").unlink()
    elif corruption == "schema":
        rewrite_manifest(directory / "manifest.json", lambda p: p.update(schema_version=2))
    elif corruption == "partition":
        rewrite_manifest(directory / "manifest.json", lambda p: p.update(valuation_time_utc="2026-10-14T18:00:00Z"))
    else:
        rewrite_manifest(directory / "manifest.json", lambda p: p.update(snapshot_id="other"))
    with pytest.raises(ObservationError):
        store.load("sample")
    valid, invalid = store.list()
    assert not valid and len(invalid) == 1


def test_changed_processing_code_creates_explicit_reprocessing_version(tmp_path, monkeypatch):
    store = LocalObservationStore(tmp_path)
    save(store)
    a = store.process("sample")
    identity = history_store.processing_identity()
    monkeypatch.setattr(history_store, "processing_identity", lambda: {**identity, "code_sha256": "changed"})
    with pytest.raises(ObservationError, match="Original processing"):
        store.replay("sample", a["processing_id"])
    b = store.process("sample")
    assert a["processing_id"] != b["processing_id"] and not b["reproduction"]
    assert store.load_artifact("sample", a["processing_id"]) == a
    assert store.query("SELECT COUNT(*) n FROM coverage").iloc[0].n == 2


def test_duplicate_and_malformed_rows_preserve_raw_values_and_reconcile(tmp_path):
    store = LocalObservationStore(tmp_path)
    raw = fixture_quotes(AS_OF)
    original = raw.iloc[0].to_dict()
    bad = {**original, "contractSymbol": "bad", "strike": "not a strike", "optionType": "invalid"}
    raw = pd.concat([raw, pd.DataFrame([original, bad])], ignore_index=True)
    observation = save(store, raw=raw)
    assert observation["quality"] == {"duplicate_rows": 1, "malformed_rows": 1}
    preserved = store.load("sample")[1]
    assert json.loads(preserved.iloc[-1].sourceRowJson)["strike"] == "not a strike"
    artifact = store.process("sample")
    outcomes = artifact["contract_outcomes"]
    reasons = {item["label"]: item["count"] for item in outcomes["reasons"]}
    assert outcomes["fetched"] == 68 == outcomes["used"] + outcomes["excluded"]
    assert reasons["Duplicate contract"] == reasons["Malformed contract"] == 1
    assert store.query("SELECT COUNT(*) n FROM raw_quotes WHERE historyInvalidReason='Duplicate contract'").iloc[0].n == 1


def test_partial_outcomes_remain_visible_even_with_received_rows(tmp_path):
    store = LocalObservationStore(tmp_path)
    raw = fixture_quotes(AS_OF)
    raw.attrs["fetchDiagnostics"]["outcomes"][0]["status"] = "failed"
    observation = save(store, raw=raw)
    assert observation["partial"] and not observation["missing_expirations"]
    assert observation["failed_expirations"] == ["2026-10-16"]
    assert store.query("SELECT snapshot_id FROM observations WHERE partial").iloc[0].snapshot_id == "sample"


def test_total_expiration_failure_is_saved_queryable_and_cannot_publish_a_surface(tmp_path):
    store = LocalObservationStore(tmp_path)
    raw = pd.DataFrame()
    raw.attrs["fetchDiagnostics"] = {"selected_expirations": [{"expiration": "2026-10-16"}],
                                     "outcomes": [{"expiration": "2026-10-16", "status": "failed"}]}
    observation = save(store, raw=raw)
    assert observation["partial"] and observation["raw_row_count"] == 0
    assert store.query("SELECT raw_rows FROM observations WHERE partial").iloc[0].raw_rows == 0
    with pytest.raises(ValueError, match="No options data"):
        store.process("sample")


@pytest.mark.parametrize("as_of,expiry,expected_dte", [
    ("2026-09-29T01:00:00Z", "2026-09-29", 1),
    ("2026-12-15T01:00:00Z", "2026-12-15", 1),
    ("2026-03-07T18:00:00Z", "2026-03-09", 2),
    ("2026-10-31T18:00:00Z", "2026-11-02", 2),
    ("2026-09-28T19:00:00Z", "2026-09-28", 0),
    ("2026-12-14T20:00:00Z", "2026-12-14", 0),
])
def test_original_clock_survives_replay_calendar_dst_and_zero_day(tmp_path, as_of, expiry, expected_dte):
    store = LocalObservationStore(tmp_path)
    request = SurfaceRequest("DEMO", dte_min=expected_dte, dte_max=expected_dte, smooth=False)
    years = time_to_expiration_years(expiry, pd.Timestamp(as_of))
    price = _black_scholes_price("call", 100, 100, years, .02, .30)
    raw = pd.DataFrame([{"strike": 100, "expirationDate": expiry, "optionType": "call", "bid": price * .995,
                         "ask": price * 1.005, "lastPrice": price, "lastTradeDate": pd.Timestamp(as_of),
                         "volume": 100, "openInterest": 500}])
    save(store, raw=raw, as_of=as_of, request=request)
    artifact = store.process("sample")
    assert store.replay("sample", artifact["processing_id"])["matched"]
    saved = pd.read_parquet(store.artifact_path("sample", artifact["processing_id"]) / "cleaned.parquet")
    assert saved.iloc[0].days_to_expiration == expected_dte == expiration_dte(expiry, pd.Timestamp(as_of))
    assert saved.iloc[0].time_to_expiration_years == pytest.approx(years)
    assert saved.iloc[0].impliedVolatilityFinal == pytest.approx(.30, abs=1e-7)


@pytest.mark.parametrize("snapshot_id", ["../sample", "sample/../../", "sample\\child", "", "*", "C:sample"])
def test_saved_paths_reject_traversal_and_globs(tmp_path, snapshot_id):
    with pytest.raises(ObservationError, match="Invalid observation"):
        LocalObservationStore(tmp_path).load(snapshot_id)


def test_compare_masks_unsupported_domain_and_ui_links_lineage(tmp_path):
    store = LocalObservationStore(tmp_path)
    seed_demo(store)
    a, b = store.process("synthetic-a"), store.process("synthetic-b")
    comparison = compare_observations(store, "synthetic-a", "synthetic-b", a["processing_id"], b["processing_id"])
    assert comparison["supported_points"] > 100 and comparison["unsupported_points"] > 0
    assert comparison["domain"]["dte_max"] <= 24
    assert all(row["kind"] == "interpolated" for row in comparison["rows"])
    app = create_app(history_root=str(tmp_path))
    response = app.test_client().get("/history?left=synthetic-a&right=synthetic-b")
    assert response.status_code == 200 and response.headers["Cache-Control"] == "no-store"
    for text in (b"Synthetic fixture", b"PARTIAL", b"3/4", b"no extrapolation", b"Observed quote-derived IV", b"Processing manifest"):
        assert text in response.data
    download = app.test_client().get(f"/history/synthetic-a/files/cleaned.parquet?processing={a['processing_id']}")
    assert download.status_code == 200 and "attachment" in download.headers["Content-Disposition"]
    assert app.test_client().get("/history/synthetic-a/files/unknown").status_code == 404
    assert app.test_client().get("/history/synthetic-a/files/cleaned.parquet?processing=..%2Fx").status_code == 404


def test_compare_does_not_interpolate_through_interior_nan_gap():
    x = np.array([[90., 100., 110.], [90., 100., 110.]])
    y = np.array([[10., 10., 10.], [20., 20., 20.]])
    z = np.array([[20., np.nan, 22.], [21., np.nan, 23.]])
    assert np.isnan(_evaluate(x, y, z, 95, 15))
    assert np.isnan(_evaluate(x, y, z, 105, 15))
    assert _evaluate(x, y, z, 90, 15) == 20.5


def test_compare_rejects_incompatible_method_parameters(tmp_path):
    store = LocalObservationStore(tmp_path)
    save(store, snapshot_id="a")
    save(store, snapshot_id="b", request=SurfaceRequest("DEMO", strike_min_pct=.9, strike_max_pct=1.1,
                                                       dte_min=0, dte_max=60, smooth=True, risk_free_rate=.04))
    a, b = store.process("a"), store.process("b")
    with pytest.raises(ObservationError, match="same ticker, saved parameters"):
        compare_observations(store, "a", "b", a["processing_id"], b["processing_id"])


def test_processing_artifact_corruption_and_lineage_are_rejected(tmp_path):
    store = LocalObservationStore(tmp_path)
    save(store)
    artifact = store.process("sample")
    path = store.artifact_path("sample", artifact["processing_id"])
    rewrite_manifest(path / "manifest.json", lambda p: p.update(raw_sha256="wrong"))
    with pytest.raises(ObservationError, match="lineage"):
        store.load_artifact("sample", artifact["processing_id"])
    with pytest.raises(ObservationError):
        store.query("SELECT * FROM coverage")


def test_processing_manifest_cleaned_count_matches_file_and_rejects_inconsistency(tmp_path):
    store = LocalObservationStore(tmp_path)
    save(store)
    artifact = store.process("sample")
    path = store.artifact_path("sample", artifact["processing_id"])
    assert artifact["cleaned_row_count"] == len(pd.read_parquet(path / "cleaned.parquet"))
    store.load_artifact("sample", artifact["processing_id"])
    rewrite_manifest(path / "manifest.json", lambda p: p.update(cleaned_row_count=0))
    with pytest.raises(ObservationError, match="row counts"):
        store.load_artifact("sample", artifact["processing_id"])
