"""Native/reference parity and the application boundary, without network access."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from src import numerical_engine as engine
from src.data_cleaner import _black_scholes_price, _implied_volatility, _no_arbitrage_bounds, prepare_options_data
from src.history_demo import fixture_quotes
from src.history_store import LocalObservationStore
from src.surface_config import SurfaceRequest
from src.surface_service import process_surface_inputs

AS_OF = "2026-10-12T18:00:00Z"
REQUEST = SurfaceRequest("DEMO", strike_min_pct=.9, strike_max_pct=1.1, dte_min=0, dte_max=60, smooth=True)


@pytest.fixture
def native():
    return pytest.importorskip("src._vol_native")


def test_compiled_source_identity_matches_checkout(native):
    root = Path(__file__).resolve().parents[1] / "native"
    names = ("engine.hpp", "engine.cpp", "bindings.cpp", "CMakeLists.txt", "build_info.hpp.in")
    content = b"".join(name.encode() + b"\n" + (root / name).read_bytes().replace(b"\r\n", b"\n") + b"\n" for name in names)
    assert native.source_sha256 == hashlib.sha256(content).hexdigest(), "Native source changed; rebuild the extension"


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_price_iv_reference_grid(backend):
    if backend == "cpp":
        pytest.importorskip("src._vol_native")
    # Includes negative carry, short time, ITM/OTM, and very high volatility.
    rng = np.random.default_rng(731)
    n = 1500
    types = np.where(rng.random(n) > .5, "call", "put")
    strikes = rng.uniform(60, 140, n)
    times = np.exp(rng.uniform(np.log(1e-5), np.log(2), n))
    dividends = rng.uniform(-.03, .08, n)
    sigma = rng.uniform(.02, 4.9, n)
    expected_prices = np.array([_black_scholes_price(t, 100, k, dt, .02, v, q)
                                for t, k, dt, v, q in zip(types, strikes, times, sigma, dividends)])
    actual = engine.prices(types, 100., strikes, times, .02, sigma, dividends, backend=backend)
    np.testing.assert_allclose(actual, expected_prices, rtol=2e-13, atol=2e-12)
    reference = np.array([_implied_volatility(t, 100, k, dt, .02, p, q)
                         for t, k, dt, p, q in zip(types, strikes, times, expected_prices, dividends)])
    result = engine.implied_volatilities(types, 100, strikes, times, .02, expected_prices, dividends, backend=backend)
    # Existing Brent absolute IV tolerance is 1e-8. Zero/lower-bound results must match exactly.
    np.testing.assert_allclose(result.values, reference, rtol=0, atol=1e-8, equal_nan=True)
    np.testing.assert_array_equal(result.values[reference == 0], 0.)
    assert result.iterations.max() <= 200


@pytest.mark.parametrize("side", ["call", "put"])
@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_model_bounds_and_zero_volatility(side, backend):
    if backend == "cpp":
        pytest.importorskip("src._vol_native")
    strike = 95 if side == "call" else 105
    lo, hi = _no_arbitrage_bounds(side, 100, strike, .1, .02, .01)
    prices = np.array([lo, lo + 5e-9, lo + 2e-8, lo - 2e-8, hi, hi - 1e-9, hi + 2e-8])
    result = engine.implied_volatilities(side, 100, np.full(len(prices), strike, dtype=float), .1, .02, prices, .01, backend=backend)
    assert result.values[0] == result.values[1] == 0
    assert result.status[3] == result.status[-1] == engine.Status.OUTSIDE_BOUNDS
    assert result.status[4] == result.status[5] == engine.Status.UNBRACKETED
    reference = [_implied_volatility(side, 100, strike, .1, .02, p, .01) for p in prices]
    np.testing.assert_allclose(result.values, reference, atol=1e-8, rtol=0, equal_nan=True)
    zero = engine.prices(side, 100, np.array([float(strike)]), .1, .02, 0., .01, backend=backend)
    assert zero[0] == lo


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_invalid_nonfinite_and_convergence_statuses(backend):
    if backend == "cpp":
        pytest.importorskip("src._vol_native")
    strikes = np.array([0., -1., np.nan, np.inf, 100., 100.])
    result = engine.implied_volatilities(["call"] * 5 + ["bad"], 100, strikes, [1, 1, 1, 1, 0, 1], .02, 10., backend=backend)
    assert np.isnan(result.values).all()
    assert (result.status == engine.Status.INVALID_INPUT).all()
    for target in (np.nan, np.inf, -1):
        result = engine.implied_volatilities("call", 100, [100.], 1, .02, target, backend=backend)
        assert result.status[0] == engine.Status.INVALID_INPUT
    for rate in (np.nan, np.inf):
        assert engine.implied_volatilities("call", 100, [100.], 1, rate, 10., backend=backend).status[0] == engine.Status.INVALID_INPUT
    result = engine.implied_volatilities("call", 100, [100.], 1, 0., 10., backend=backend, maxiter=1)
    assert np.isnan(result.values[0]) and result.iterations[0] == 1
    assert result.status[0] == engine.Status.MAX_ITERATIONS
    with pytest.raises(engine.NumericalBackendError, match="No backend fallback"):
        result.checked_values()
    overflow = engine.implied_volatilities("put", 100, [100.], 1, -1000., 10., backend=backend)
    assert overflow.status[0] == engine.Status.NONFINITE_RESULT


def test_backend_selection_and_missing_extension(monkeypatch):
    monkeypatch.setattr(engine, "_extension", lambda: None)
    monkeypatch.setenv("VOL_NUMERICAL_BACKEND", "auto")
    assert engine.resolve_backend() == engine.resolve_backend("auto") == "python"
    assert engine.resolve_backend("python") == "python"
    with pytest.raises(engine.NumericalBackendError, match="not installed"):
        engine.resolve_backend("cpp")
    with pytest.raises(ValueError, match="python, cpp or auto"):
        engine.resolve_backend("typo")
    result = process_surface_inputs(REQUEST, fixture_quotes(AS_OF), 100, AS_OF)
    assert result.figure.data and result.diagnostics["numerical_engine"]["backend"] == "python"


def test_broken_native_import_is_not_auto_fallback(monkeypatch):
    def broken(name):
        raise ImportError("DLL load failed")
    monkeypatch.setattr(engine.importlib, "import_module", broken)
    with pytest.raises(engine.NumericalBackendError, match="DLL load failed"):
        engine.resolve_backend("auto")
    assert engine.resolve_backend("python") == "python"


def test_native_exception_is_not_fallback(native, monkeypatch):
    def broken(*args):
        raise RuntimeError("native failure")
    monkeypatch.setattr(native, "batch", broken)
    with pytest.raises(engine.NumericalBackendError, match="native failure"):
        engine.implied_volatilities("call", 100, [100.], 1, 0, 10., backend="auto")


@pytest.mark.parametrize("changes", [
    {"strikes": [[100.]]}, {"strikes": ["100"]}, {"times": [1., 2.]},
    {"option_types": ["call", "put"]}, {"volatility": ["bad"]}, {"dividends": [[0.]]},
])
def test_public_shape_and_type_validation(changes):
    args = dict(option_types="call", spot=100, strikes=[100.], times=1, rate=0, volatility=.2)
    args.update(changes)
    with pytest.raises(ValueError):
        engine.prices(**args, backend="python")


def test_native_boundary_ownership_and_validation(native):
    kinds = np.array([1, 0], dtype=np.uint8)
    doubles = [np.array(x, dtype=np.float64) for x in ([100, 100], [100, 110], [.1, .2], [.02, .02], [0, .01], [.2, .3])]
    args = [kinds, *doubles]
    out, status, iterations = native.batch(*args, False)
    expected = out.copy()
    assert out.flags.owndata and status.flags.owndata and iterations.flags.owndata
    doubles[-1][:] = 10
    np.testing.assert_array_equal(out, expected)
    with pytest.raises(TypeError):
        native.batch(kinds.astype(np.int32), *doubles, False)
    with pytest.raises(ValueError):
        native.batch(kinds, doubles[0][:1], *doubles[1:], False)
    with pytest.raises(ValueError):
        native.batch(kinds.reshape(1, 2), *doubles, False)
    with pytest.raises(ValueError):
        native.batch(kinds, np.ones(4)[::2], *doubles[1:], False)
    with pytest.raises(TypeError):
        native.batch(kinds, doubles[0].astype(np.float32), *doubles[1:], False)
    assert native.batch(np.empty(0, dtype=np.uint8), *[np.empty(0)] * 6, True)[0].size == 0


def test_noncontiguous_public_arrays_and_parallel_request_isolation(native):
    strikes = np.linspace(90., 110., 100)[::2]
    def work(backend):
        with engine.numerical_session(backend) as session:
            prices = engine.prices("call", 100, strikes, 1., .02, .2)
            values = engine.implied_volatilities("call", 100, strikes, 1., .02, prices).checked_values()
            return values, session.diagnostics()
    with ThreadPoolExecutor(max_workers=2) as pool:
        python, cpp = list(pool.map(work, ["python", "cpp"]))
    np.testing.assert_allclose(cpp[0], python[0], atol=1e-8, rtol=0)
    assert cpp[1]["backend"] == "cpp" and python[1]["backend"] == "python"
    assert cpp[1]["batch_calls"] == 2


def test_real_saved_input_preserves_quality_accounting_and_surface(native):
    root = Path(__file__).resolve().parents[1]
    observation, raw = LocalObservationStore(root / "history_bundle").load("6e471b7a5c5d8fe0695afacb")
    request = SurfaceRequest(**observation["parameters"])
    builds = [process_surface_inputs(replace(request, numerical_backend=b), raw, observation["spot"]["value"], observation["valuation_time_utc"])
              for b in ("python", "cpp")]
    python, cpp = builds
    # Existing quality policy is exact; only calculated floating point values have tolerance.
    numeric = python.cleaned_options_df.select_dtypes(include="number").columns
    np.testing.assert_allclose(cpp.cleaned_options_df[numeric], python.cleaned_options_df[numeric], atol=1e-8, rtol=1e-10, equal_nan=True)
    pd.testing.assert_frame_equal(cpp.cleaned_options_df.drop(columns=numeric), python.cleaned_options_df.drop(columns=numeric))
    assert cpp.diagnostics["contract_outcomes"] == python.diagnostics["contract_outcomes"]
    assert cpp.diagnostics["flag_counts"] == python.diagnostics["flag_counts"]
    from src.history_compare import _array
    a, b = json.loads(python.figure.to_json())["data"], json.loads(cpp.figure.to_json())["data"]
    assert len(a) == len(b) > 0
    for x, y in zip(a, b):
        for key in ("x", "y", "z"):
            # Surface IV is displayed in percentage points; optimizer is sensitive
            # to roundoff. Enforce a small, explicit derived-output tolerance.
            np.testing.assert_allclose(_array(x[key]), _array(y[key]), atol=1e-5, rtol=1e-7)
    assert cpp.diagnostics["request"]["valuation_time_utc"] == observation["valuation_time_utc"]
    assert cpp.diagnostics["numerical_engine"]["backend"] == "cpp"


def test_history_versions_and_replay_use_artifact_backend(native, tmp_path, monkeypatch):
    store = LocalObservationStore(tmp_path)
    store.save(replace(REQUEST, numerical_backend="python"), fixture_quotes(AS_OF),
               {"value": 100, "provider": "test_fixture", "method": "fixture", "collected_at_utc": AS_OF},
               AS_OF, snapshot_id="numerics", source_kind="synthetic_fixture")
    python = store.process("numerics", backend="python")
    cpp = store.process("numerics", backend="cpp")
    assert python["processing_id"] != cpp["processing_id"]
    assert cpp["processor"]["numerical_engine"]["backend"] == "cpp"
    assert not cpp["reproduction"] and python["reproduction"]
    monkeypatch.setenv("VOL_NUMERICAL_BACKEND", "python")
    def blocked(*args, **kwargs):
        raise AssertionError("Replay must be offline and use the original clock")
    import src.surface_service as service
    monkeypatch.setattr(service, "get_current_price", blocked)
    monkeypatch.setattr(service, "get_options_data", blocked)
    monkeypatch.setattr(pd.Timestamp, "now", blocked)
    assert store.replay("numerics", cpp["processing_id"])["matched"]
    monkeypatch.setenv("VOL_NUMERICAL_BACKEND", "cpp")
    assert store.replay("numerics", python["processing_id"])["matched"]


def test_flask_and_cli_use_forced_cpp_with_saved_inputs(native, monkeypatch, tmp_path, capsys):
    import src.surface_service as service
    import main as cli
    from web_app import create_app
    import sys
    monkeypatch.setenv("VOL_NUMERICAL_BACKEND", "cpp")
    monkeypatch.setattr(service, "get_current_price", lambda ticker: 100.)
    monkeypatch.setattr(service, "get_options_data", lambda *args, **kwargs: fixture_quotes(AS_OF))
    monkeypatch.setattr(service, "normalize_as_of_utc", lambda value=None: pd.Timestamp(value or AS_OF))
    captured = []
    def builder(request):
        result = service.build_surface_bundle(request)
        captured.append(result)
        return result
    app = create_app(surface_builder=builder)
    response = app.test_client().get("/?ticker=DEMO&dte_min=0&dte_max=60")
    assert response.status_code == 200 and captured
    assert captured[0].figure.data
    assert captured[0].diagnostics["numerical_engine"]["backend"] == "cpp"
    assert b"Numerical backend: <strong>cpp</strong>" in response.data
    monkeypatch.setattr(sys, "argv", ["main.py", "DEMO", "--backend", "cpp", "--smooth", "--dte_min", "0", "--output_dir", str(tmp_path)])
    cli.main()
    assert list(tmp_path.glob("*.html"))
    assert "Numerical backend: cpp" in capsys.readouterr().out


@pytest.mark.parametrize("quality", ["strict", "balanced", "lenient"])
def test_mixed_quote_failures_and_duplicate_indices_preserve_policy(native, quality):
    raw = fixture_quotes(AS_OF)
    raw.loc[0, ["bid", "ask", "lastPrice"]] = [0, 0, 1.]
    raw.loc[0, "lastTradeDate"] = pd.Timestamp("2026-09-01T00:00:00Z")
    raw.loc[1, ["bid", "ask", "lastPrice"]] = [149., 151., 150.]
    raw.loc[2, ["bid", "ask"]] = [.01, 10.]
    raw.loc[3, ["bid", "ask", "lastPrice"]] = [0, 0, 0]
    raw.loc[4, ["volume", "openInterest"]] = [0, 0]
    raw.index = np.zeros(len(raw), dtype=int)
    args = dict(underlying_price=100., as_of_utc=pd.Timestamp(AS_OF), quality_mode=quality)
    python = prepare_options_data(raw, numerical_backend="python", **args)
    cpp = prepare_options_data(raw, numerical_backend="cpp", **args)
    pd.testing.assert_frame_equal(cpp, python, atol=1e-8, rtol=1e-10)


def test_per_contract_spots_and_rates(native):
    args = (["call", "put"], [100., 120.], [105., 110.], [.1, .2], [.02, -.01], [.2, .3], [.01, .04])
    python = engine.prices(*args, backend="python")
    cpp = engine.prices(*args, backend="cpp")
    np.testing.assert_allclose(cpp, python, atol=1e-12, rtol=1e-13)
    for backend in ("python", "cpp"):
        result = engine.implied_volatilities(*args[:5], python, args[-1], backend=backend)
        np.testing.assert_allclose(result.values, args[5], atol=1e-8, rtol=0)


def test_missing_cpp_fails_before_network_or_saved_artifact_reads_require_it(native, tmp_path, monkeypatch):
    import src.surface_service as service
    store = LocalObservationStore(tmp_path)
    store.save(replace(REQUEST, numerical_backend="cpp"), fixture_quotes(AS_OF),
               {"value": 100, "provider": "test", "method": "fixture", "collected_at_utc": AS_OF},
               AS_OF, snapshot_id="readable", source_kind="synthetic_fixture")
    artifact = store.process("readable")
    monkeypatch.setattr(engine, "_extension", lambda: None)
    def blocked(*args, **kwargs):
        raise AssertionError("No live fetch is needed to report an unavailable backend")
    monkeypatch.setattr(service, "get_current_price", blocked)
    with pytest.raises(engine.NumericalBackendError, match="not installed"):
        service.build_surface_bundle(replace(REQUEST, numerical_backend="cpp"))
    assert store.load_artifact("readable", artifact["processing_id"]) == artifact
    with pytest.raises(engine.NumericalBackendError, match="not installed"):
        store.replay("readable", artifact["processing_id"])


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_pricing_nonfinite_and_invalid_inputs(backend):
    if backend == "cpp":
        pytest.importorskip("src._vol_native")
    args = dict(option_types="call", spot=100., strikes=[100.], times=1., rate=.02, volatility=.2, dividends=.01)
    for name in ("spot", "strikes", "times", "rate", "volatility", "dividends"):
        for value in (float("nan"), float("inf"), -float("inf")):
            changed = {**args, name: [value] if name == "strikes" else value}
            assert np.isnan(engine.prices(**changed, backend=backend)).all(), (name, value)
    for name in ("spot", "strikes", "times"):
        changed = {**args, name: [0.] if name == "strikes" else 0.}
        assert np.isnan(engine.prices(**changed, backend=backend)).all()
    assert np.isnan(engine.prices(**{**args, "volatility": -.1}, backend=backend)).all()
