# C++ engine: measured results and validation

Measured locally on October 15, 2026. This feature starts from main `34ecc60`;
baseline commit `b13c0c9` adds the benchmark harness/evidence with application code
unchanged. The older README's 4.06s-to-2.82s result is not used here.

## Workload and environment

- Real collected SPY observation `6e471b7a5c5d8fe0695afacb`, **2,604 raw contracts**,
  1,627 retained contracts. Valuation time: `2026-10-14T02:45:43.381473+00:00`;
  saved spot: `778.010009765625`. No fetching or new market collection.
- Saved parameters: strikes 93–107% of spot, DTE 1–30, smooth surface, lenient
  quality, rate 0.02, configured dividend yield 0, trade-age limit 120 hours.
  Existing parity-based dividend estimation remains enabled.
- Raw Parquet SHA-256: `c71bc9f90901974be3e73a306c1b7570cdfdd1c4a6fc45de910865dd33324a6b`.
  Canonical calculation input is the original saved cleaned Parquet, SHA-256
  `aa333417a2061eb6b97738a8079e1fbf5e37c70a5a69d17f98c799014843795c`.
  The baseline cleaner's calculation arrays and option types were verified
  exactly equal to those saved arrays, including NaNs. Both current backends
  receive the same canonical input bytes.
- Windows 11 x64 (`10.0.22631`), AMD64 Family 25 Model 117 Stepping 2, 12 logical
  CPUs; CPython **3.14.7**. NumPy **2.5.4**, SciPy **1.18.1**, pandas **2.3.3**,
  Plotly **6.9.0**, PyArrow **26.0.0**. The environment dictionaries in all three
  benchmark reports match exactly.
- Native: C++17, **MSVC 19.44.35211.0**, CMake **4.4.4**, pybind11 **3.1.0**,
  Release `/O2 /Ob2 /DNDEBUG`, `/fp:strict`. No native parallelism or fast-math.
  Source ID: `3c75ebb82cdf393dad15d4c9aa075bb5042695d148425c27924516564c8fc95b`;
  measured binary SHA-256: `698d156ff34e2856d309100d2c672ce9ae2a28d3afb1200a03936ebf514667f8`.

Each workload runs one untimed warm-up and **seven measured repetitions**.
Tables show median ± median absolute deviation (MAD), in **milliseconds**.
Tests and builds were stopped during timing. These are descriptive measurements
on a shared workstation, not dedicated production load testing. Reports retain
every sample, min/max, parameters and profiles.

## Calculation module

One BSM batch repricing and three IV inversions (selected price, bid, ask) per
retained contract, including array conversion, allocations, ownership copies and
binding overhead. Failures/invalid values remain part of the identical workload.
Subsets are deterministic samples of the real saved input, not generated quotes.

| Contracts | Original Python baseline | Current Python reference | C++ | Speedup vs original |
|---:|---:|---:|---:|---:|
| 128 | 18.67 ± 1.75 | 14.73 ± 0.72 | 0.87 ± 0.14 | 21.4× |
| 512 | 76.07 ± 4.88 | 69.67 ± 7.48 | 2.55 ± 0.06 | 29.8× |
| 1,627 | 252.81 ± 15.64 | 226.19 ± 6.81 | 7.66 ± 0.39 | 33.0× |

This is a combined pricing/IV workload. It is not evidence that C++ pricing alone
is 33× faster than NumPy: the existing vectorized price path already uses compiled
code. Removing repeated Python objective callbacks from IV solving is the main
native benefit. The reference batch adapter also converts NumPy scalars to Python
floats before calling the scalar model; its calculation times differ from the
original scalar-loop harness, so both reference columns are reported.

## Complete surface build

Includes filtering/forward estimation, quotes/quality rules, numerical work,
conversion/bindings, exclusion accounting, internal repricing, static-arbitrage
projection/optimization and Plotly figure construction. Excludes file loading,
network fetching, compilation, interpreter startup and browser rendering. Raw
subset sizes use `sample(random_state=42)`; forward estimation and optimizer work
can differ between subsets, so these rows do not imply linear scaling.

| Raw contracts | Original Python baseline | Current Python reference | C++ | Speedup vs original |
|---:|---:|---:|---:|---:|
| 256 | 1,214.3 ± 29.8 | 1,158.0 ± 28.8 | 814.9 ± 5.7 | 1.49× |
| 1,024 | 1,581.9 ± 37.0 | 1,605.4 ± 24.8 | 948.5 ± 29.1 | 1.67× |
| 2,604 | 2,149.4 ± 32.6 | 2,133.3 ± 26.0 | 1,466.2 ± 41.6 | 1.47× |

The full saved chain's median build time fell **31.8%**, from **2.149s to 1.466s**
against the fresh pre-upgrade baseline. The current Python backend remains
approximately at the original full-chain runtime. This does not predict live
request latency, which also includes network fetching and client rendering.

Baseline profiling counted **14,391** scalar IV calls and **202,485** objective
callbacks. Their cumulative time was about 1.07s in a 3.90s instrumented run.
The final native full build submits **227 batches**, covering pricing and
inversion together; quote midpoint/bid/ask inversion is one of these batches.
The final profile still spends substantial time in Pandas metadata copying,
row policy/grouping and SciPy SLSQP: `_project_call_price_slice` takes about 0.81s,
DataFrame apply about 0.86s, and `deepcopy` about 0.65s in a 3.16s instrumented run.
These cumulative times overlap and should not be added. Profiling times include
instrumentation and are not the warm benchmark medians.

Profiling also identified unnecessary Series/metadata copies introduced by an
`iterrows()` quote prepass. The final implementation converts four quote fields
to records once, retaining the same quote-selection function. Remaining costs
are outside the small native engine. Adding native threads would target a small
fraction of total time and is not justified by these measurements.

Raw evidence: [original baseline](evidence/cpp-baseline.json),
[current Python](evidence/cpp-python-benchmark.json),
[final C++](evidence/cpp-benchmark.json).

## Tests actually executed

| Check | Local result | Evidence |
|---|---|---|
| Unmodified application baseline | 142 passed | [baseline output](evidence/cpp-baseline-tests.txt) |
| Release native CTest executable | 1 target passed; multiple pricing, inversion and failure cases | [CTest output](evidence/cpp-native-tests-windows.txt) |
| Complete suite, C++ forced | 172 passed | [C++ output](evidence/cpp-tests-windows.txt) |
| Complete suite, Python forced | 172 passed | [Python output](evidence/cpp-python-tests-windows.txt) |
| Complete suite, extension physically absent, auto selected | 155 passed, 17 native cases skipped | [fallback output](evidence/cpp-fallback-tests-windows.txt) |
| Complete suite after six historical-comparison regression cases, C++ forced | 178 passed | [comparison-fix output](evidence/cpp-comparison-tests-windows.txt) |
| Comparison compatibility and strict-policy checks, Python forced | 7 passed | [focused output](evidence/cpp-comparison-python-windows.txt) |
| Comparison checks with extension physically absent, auto selected | 5 passed, 2 native cases skipped | [focused fallback output](evidence/cpp-comparison-fallback-windows.txt) |

All full Python suites retain four existing NumPy timedelta deprecation warnings.
[Release build output](evidence/cpp-build-windows.txt) records the local build.
The extension was restored after absence testing; generated binaries, build
directories and temporary test stores are ignored by Git.

The new native executable uses active checks even under `NDEBUG`. Python tests
cover 1,500 deterministic generated numerical cases, calls/puts, negative carry,
short maturities, zero volatility, lower/upper bound neighborhoods, bracket
failure, solver exhaustion, invalid/nonfinite inputs, malformed shapes/dtypes,
empty/strided arrays, output ownership, concurrent session isolation and stale
compiled source detection. Generated numerical test cases and history demo quotes
are synthetic; benchmark inputs are collected market observations.

Numerical parity limits are documented in tests: prices `atol=2e-12`,
`rtol=2e-13`; IV `atol=1e-8`, `rtol=0`, with exact zero-boundary behavior. The
real-chain test compares numerical columns, exact flags/confidence/selection and
exact exclusion accounting. Its derived surface coordinates/IV percentage points
use `atol=1e-5`, `rtol=1e-7` to account for optimizer sensitivity to roundoff.
Existing regression assertions, including the `1e-10` golden surface check,
were not loosened or edited.

Tests exercise the Flask route and CLI using the real service/native engine with
network inputs replaced by deterministic synthetic quotes. The real SPY service
test runs both backends. History tests process/save both versions, replay with an
opposite environment preference, prohibit live clock/fetch access, and confirm
that saved C++ files remain readable without the extension. Original valuation
time, spot and raw provenance remain intact. AWS-named tests use mocks/local
fixtures; no AWS resources were modified.

## Portability, deployment and limits

The [hosted CI run](https://github.com/Parsa-Bagheri/vol-surface-explorer/actions/runs/38815460057)
at `fa6f810` passed all five jobs: Release CTest plus **178 tests with C++ forced
and 178 with Python forced** on each Windows/Linux and Python 3.12/3.14 combination;
extension-free Linux passed **159 tests with 19 native cases skipped**. Compilers
were GNU 13.3 and MSVC 19.51. Native import is required before native suites to
prevent misleading skips. [Compact hosted evidence](evidence/cpp-ci-cross-platform.json)
records exact interpreter versions, job links and output. The first fresh-runner
attempt revealed a missing parent for pytest's explicit temporary directory;
the workflow now creates it before testing. No assertions changed. The workflow
also tests the optional web install hook in both compiler-free and native modes;
the PR's current checks record that additional job's result.

The protected C++ website preview (`dpl_ArZv4Xp5r1tAmCoA3f9iGK7HBhCt`) built the
same native source in Release with GNU 11.5 and Python 3.12.15, passed CTest and
24 bounded HTTP checks, including a live SPY surface and all 18 byte-identical
history downloads. Its live diagnostics explicitly reported `cpp`. This was an
engineering preview of the local packaging hook; final release source and
production checks are recorded in [PR #5](https://github.com/Parsa-Bagheri/vol-surface-explorer/pull/5).
The optional build flag compiles on Vercel for its own Linux/Python ABI, then removes
build tools. `VOL_BUILD_NATIVE=0` retains the compiler-free Python installation.
See the guide for staged deployment and selected-backend verification commands.

After renewing the operator's existing AWS session, bounded **read-only** checks
verified both existing S3 archives, the release commit, both Glue partitions,
both analytics projection checksums and **all 18 archive files byte-for-byte
against the web bundle**. The previous Athena execution's status was checked;
SQL was not rerun and no new query was started. [Redacted AWS evidence](evidence/cpp-aws-readonly.json)
contains counts, without account/resource identifiers. The web history reader
uses that same bundle without AWS credentials; deployment HTTP checks separately
verify the website returns its exact bytes. No AWS resources were modified.

Python remains the optional-dependency fallback. A binary replacement changes
processing identity and can require explicit reprocessing.
The European BSM model, quote policy and existing finite-grid arbitrage optimizer
retain their original assumptions. Benchmarks cover one saved SPY chain and its
subsets; they do not establish performance on every instrument or machine.

Build/install, selection, replay and reproduction commands are in
[the engine guide](cpp-engine.md). Baseline commit `b13c0c9` and the raw JSON
reports make the pre-upgrade measurements independently reproducible.

## Resume bullet

Implemented a C++17/pybind11 engine for batched option pricing and bounded implied-volatility solving, preserving Python parity and historical replay provenance; reduced a real 2,604-contract SPY warm surface build by 32% (2.15s to 1.47s) and accelerated the 1,627-contract calculation workload 33× on Windows.
