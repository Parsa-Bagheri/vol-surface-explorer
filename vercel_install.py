"""Install the web runtime; opt into a target-platform native Release build.

Normal pip installations still use requirements.txt without this deployment hook.
VOL_BUILD_NATIVE=0 (default) keeps the existing compiler-free web installation.
An explicitly requested native build fails the deployment if build/checks fail.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    native = os.environ.get("VOL_BUILD_NATIVE", "0")
    if native not in {"0", "1"}:
        raise ValueError("VOL_BUILD_NATIVE must be 0 or 1")

    def run(*args):
        subprocess.run(list(args), check=True)

    uv = ["uv", "pip"]
    run(*uv, "install", "--python", sys.executable, "-r", "requirements.txt")
    if native == "0":
        print("Web installation: Python backend (native build disabled)", flush=True)
        return

    run(*uv, "install", "--python", sys.executable, "-r", "requirements-native.txt")
    import cmake

    with tempfile.TemporaryDirectory(prefix="vol-native-build-") as build:
        run(sys.executable, "-m", "cmake", "-S", "native", "-B", build,
            "-DCMAKE_BUILD_TYPE=Release", f"-DPython_EXECUTABLE={sys.executable}")
        run(sys.executable, "-m", "cmake", "--build", build, "--config", "Release", "--parallel", "2")
        run(str(Path(cmake.CMAKE_BIN_DIR) / "ctest"), "--test-dir", build,
            "-C", "Release", "--output-on-failure")
    run(sys.executable, "-c", "from src import _vol_native; "
        "assert _vol_native.build_type == 'Release'; "
        "print('Web native build:', _vol_native.compiler, _vol_native.build_type, _vol_native.source_sha256)")
    # Build tools are not function runtime dependencies. Keep only the extension.
    run(*uv, "uninstall", "--python", sys.executable, "cmake", "pybind11")


if __name__ == "__main__":
    main()
