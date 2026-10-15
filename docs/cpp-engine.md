# Optional C++ numerical engine

The feature branch is `cpp-numerical-engine`, based on main commit
`34ecc60`. The main checkout was clean before work. No production deployment,
market-data collection, remote CI run or AWS operation was performed.

## Boundary and implementation

`native/engine.cpp` implements European Black-Scholes-Merton call/put pricing
with continuous dividend yield and a bounded Brent-Dekker IV solver. It uses
double precision, `erfc` normal probabilities, cached log-moneyness, discounted
spot/strike and square-root maturity. Interpolation (secant or inverse quadratic)
is accepted only when it improves a bounded step; bisection provides the safeguard.
The module contains the calculations, rather than delegating them to a pricing
library. It is single threaded.

Python still owns quote selection, forward estimation, quality rules, contract
accounting, interpolation, static-arbitrage optimization and plotting. These are
policy or existing NumPy/SciPy operations; the repeated Python callback from IV
inversion was the useful boundary to remove. The cleaner submits midpoint, bid
and ask prices together in **one native call**, independent of contract count.
The visualizer batches each projected maturity slice instead of calling C++ per
node. Selection and validation repricing also use batch pricing.

The native quote prepass converts just four quote fields to records rather than
constructing a Pandas Series per contract. Profiling found that `iterrows()`
otherwise deep-copied the attached fetch diagnostics for every temporary row.
Removing that conversion cost keeps the existing quote-selection function and
quality policy intact; it is part of preparing the batch, not a new pricing model.

The scalar helpers in `src/data_cleaner.py` remain the original reference.
`src/numerical_engine.py` supplies the common interface and SciPy reference batch
adapter. Python pricing retains the existing NumPy/SciPy vectorized path for
scalar spot/rate; array spot/rate inputs use the scalar reference.

The public batch interface accepts one-dimensional numeric strikes and scalar
or equal-length spot, maturity, rate, volatility/price and dividend inputs.
Option types are `call` or `put`; unknown types produce invalid-input results.
Malformed shapes/types raise errors before calculation. Python converts numeric
inputs to contiguous float64 arrays and types to uint8 (`1` call, `0` put,
`255` invalid). The lower-level binding requires those exact dtypes, contiguous
storage and matching lengths.

Bindings copy inputs into RAII `std::vector` storage while holding the GIL,
allocate NumPy-owned output arrays, then release the GIL only around the native
loop. The loop accesses native storage and private output buffers, with no Python
API calls. Output pointers are never returned as dangling views. Copies are
intentional; conversion, allocation, copying and binding costs are included in
benchmarks. This follows the documented [pybind11 array interface](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html)
and [GIL lifetime rules](https://pybind11.readthedocs.io/en/stable/advanced/misc.html).

## Numerical contract and failure behavior

- Positive spot, strike and maturity; finite rate, dividend and price/volatility.
  Negative market price or volatility and unknown option type produce NaN.
- Discounted no-arbitrage bounds use continuous dividend yield. Price within
  `1e-8` of the lower bound returns **zero IV**. Zero volatility pricing returns
  that lower bound. Expired options remain invalid, as before.
- The root bracket remains `[1e-6, 5.0]`; `maxiter=200`, `xtol=1e-8` and
  relative tolerance `4 * double epsilon` match the existing
  [SciPy Brent configuration](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html).
  A price inside financial bounds can still require volatility beyond the bracket.
- Batch inversion returns `values`, `status` and `iterations`. Status codes are
  `0 ok`, `1 zero_volatility`, `2 invalid_input`, `3 outside_bounds`,
  `4 unbracketed`, `5 max_iterations`, `6 nonfinite_result`.
- Invalid input, bounds and unbracketed roots retain NaN and the existing
  unavailable-IV/exclusion rules. Exhaustion or nonfinite arithmetic is exposed
  in the result; application calls raise a diagnostic error through
  `checked_values()`. Native exceptions also become explicit errors. They never
  trigger a Python retry. The surface wrapper propagates numerical engine errors.
- Optional solver arguments allow tests/callers to exercise exhaustion. The
  application uses the original limits. No parallel solver or fast-math flags
  are used. The native target disables floating point contraction.

## Installation and build

The existing installation stays valid with only `requirements.txt` or
`requirements-dev.txt`. No C++ tools are needed for Python operation.

For C++, activate a project virtual environment, install a C++17 compiler
(Visual Studio C++ Build Tools on Windows; GCC/Clang and Python development
headers on Linux), then:

```text
python -m pip install -r requirements-dev.txt -r requirements-native.txt
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --config Release --parallel 2
ctest --test-dir build/native -C Release --output-on-failure
python -c "from src import _vol_native; print(_vol_native.compiler, _vol_native.build_type)"
```

If CMake selects a different Python, pass its absolute path with
`-DPython_EXECUTABLE=...`. On a machine with several Visual Studio versions,
select one explicitly, for example `-G "Visual Studio 17 2022" -A x64` on a
fresh build directory. Reconfigure/rebuild after changing native source or
Python versions. Close Python processes using the extension before rebuilding
on Windows, where loaded DLLs are locked. `VOL_OUTPUT_DIR` can override the
default output directory, `src/`; importing requires the extension in the
`src` package.

The local Codex host exposed both `Path` and `PATH` to MSBuild; local CMake
commands were launched through `subprocess.run(...,
env={k.upper(): v for k, v in os.environ.items()})` to remove that host-specific
duplication. A fresh `build/native-release` directory verified Release flags
`/O2 /Ob2 /DNDEBUG` and `/fp:strict`; the initial failed configure directory was
not used for measurements. Normal developer shells should not need this workaround.

## Backend selection and application use

`VOL_NUMERICAL_BACKEND=python|cpp|auto` applies to Flask, the CLI, history tools
and standalone numerical calls. Default is `auto`. A request or explicit batch
argument takes precedence over the environment:

```python
from src.surface_config import SurfaceRequest
from src.surface_service import process_surface_inputs
request = SurfaceRequest("SPY", numerical_backend="cpp")
# Explicit saved raw inputs, original spot and original valuation timestamp:
result = process_surface_inputs(request, raw_df, spot, valuation_time_utc)
print(result.diagnostics["numerical_engine"])
```

```text
python main.py SPY --smooth --backend cpp --diagnostics_json demo-output/diagnostics.json
python history.py --store YOUR_LOCAL_STORE process SNAPSHOT_ID --backend cpp
python history.py --store YOUR_LOCAL_STORE replay SNAPSHOT_ID --processing-id PROCESSING_ID
```

The first command performs the existing live fetch; Flask/CLI tests substitute
deterministic synthetic quotes at a fixed clock, and the service parity test and
benchmark use saved real quotes. Neither tests nor benchmark fetch. Do not use the read-only `history_bundle` as a
processing destination. Copy an observation to a local store or use the offline
service API.

`python` never imports the extension. `cpp` fails clearly if it is absent,
before live fetching. `auto` falls back only when the extension is not installed.
A broken binary/DLL or computation failure reports an error. It does not change
backends. Request-local `ContextVar` state keeps concurrent requests independent.
Diagnostics and CLI output identify the backend. Batch diagnostics include status
counts and maximum solver iterations; Python's original row cleaner is outside
those batch counts. These counters describe submitted batch work, including
invalid contracts, rather than quotes used in the final surface.

## History and deployment

Processing identity includes Python source, native source, selected engine,
dependencies and (for C++) compiled source ID, engine version, compiler, build
configuration and binary SHA-256. Changing backend produces a distinct processing
ID for the same immutable raw observation. Historical processing can explicitly
override the collection backend. Replay selects the **artifact's** backend even
if the current environment requests another one, and refuses a missing or changed
engine. The original spot, valuation time, quote policy and raw lineage remain
unchanged. Reading/downloads of saved C++ artifacts do not need the extension.

Comparison ignores the saved backend selector and requires equal actual
processing identities plus all model and quote-policy settings. This lets a
legacy observation without a selector compare with a newer observation after
both are reprocessed by the same engine, including an explicit backend override.
The saved parameters and immutable manifests are not rewritten.

The older committed history bundle remains readable; replaying old processing
requires its original code/dependencies, or an explicit reprocess under this
version. Existing artifacts are not rewritten. Exact native binary identity is
deliberately conservative: rebuilding, changing compiler/platform or replacing a
binary can require explicit reprocessing even if source is equivalent.

Native packages stay outside runtime requirements. `.gitignore` and
`.vercelignore` exclude generated native libraries/build output, so a Windows
binary cannot accidentally become a Linux deployment dependency. Production
continues through Python unless a compatible extension is deliberately installed;
`VOL_NUMERICAL_BACKEND=python` explicitly pins it. A future native deployment
needs a release build for its OS, architecture and Python ABI. No deployment or
AWS resource change is part of this feature.

## Verification and measurement

See [measured results and validation evidence](cpp-engine-results.md) for actual
test counts, platform limits and timing tables. Reproduce the full suites with:

```text
python -m pytest -q
```

Run once with `VOL_NUMERICAL_BACKEND=cpp` and once with `python`. To verify
installation fallback, use an environment without `src/_vol_native*` and set
`auto`; native-specific tests skip, while all reference/application regressions
run. The dedicated missing-extension tests also exercise forced-C++ failure.

```text
python scripts/benchmark_numerics.py --backend python --repeats 7 --profile --output demo-output/python-benchmark.json
python scripts/benchmark_numerics.py --backend cpp --repeats 7 --profile --output demo-output/cpp-benchmark.json
```

Stop tests, compilation and other heavy work before timing. The benchmark uses the
committed real SPY snapshot and its original valuation settings; no network,
temporary dataset or generated quote workload is needed. Each workload warms up
once, then reports all samples, median, min/max and median absolute deviation
(MAD). Calculation measurements include one repricing and three IV inversions per
contract, with conversion/binding overhead. Surface measurements include cleaning,
quality/accounting, validation, projection/optimization and Plotly figure creation.
File reads, startup, compilation, live fetching and browser rendering are excluded.
Subset sizes use deterministic sampling (`random_state=42`); sampled raw subsets
can change forward estimation and optimizer work relative to the whole chain.

Baseline commit `b13c0c9` contains the benchmark harness and fresh evidence with
the application still unchanged from main `34ecc60`. Run the Python benchmark
on that commit with identical requirements to reproduce the pre-upgrade code.
The harness detects the older request schema. Use a separate checkout/environment
rather than resetting a working branch. For calculation measurements, the
pre-upgrade cleaner's arrays were verified exactly equal (including NaNs and
option types) to the canonical saved cleaned arrays now used by both backends.

GitHub Actions in `.github/workflows/numerical-engine.yml` prepares Release
builds, CTest and both forced-backend full suites on Windows/Linux with Python
3.12/3.14, plus an extension-free auto-fallback Linux job. The native jobs assert
that import succeeds before pytest, so a missing extension cannot turn native
coverage into a misleading successful skip. Prepared CI is not evidence of a
successful remote run.
