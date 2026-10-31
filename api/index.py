# Prepare each Vercel worker before it accepts surface requests.
import os
from time import perf_counter, thread_time

# Keep production on its established C++ backend; explicit environment choices win.
os.environ.setdefault("VOL_NUMERICAL_BACKEND", "cpp")
os.environ.setdefault("VOL_REQUEST_TIMING", "1")

_started, _cpu = perf_counter(), thread_time()
from web_app import app, _prepare_surface_runtime
from src.request_timing import log_runtime_ready

_web_wall = (perf_counter() - _started) * 1000
_web_cpu = (thread_time() - _cpu) * 1000
_started, _cpu = perf_counter(), thread_time()
_prepare_surface_runtime()
log_runtime_ready(_web_wall, _web_cpu,
                  preparation_ms=(perf_counter() - _started) * 1000,
                  preparation_thread_cpu_ms=(thread_time() - _cpu) * 1000)
