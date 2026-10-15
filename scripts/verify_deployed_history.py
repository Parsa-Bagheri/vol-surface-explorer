"""Bounded HTTP checks against an exact deployment and committed history bundle."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.history_reader import ReadOnlySnapshotStore


def verify_deployment(url, bundle, output, vercel_cli=None, node="node", scope=None, include_live=False,
                      expected_backend=None):
    if expected_backend and not include_live:
        raise ValueError("Backend verification requires include_live=True")
    store = ReadOnlySnapshotStore(bundle)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = store.release["observations"]
    checks = [("/", "landing", None), ("/history", "catalog", None)]
    for item in selected:
        snapshot_id, processing_id = item["snapshot_id"], item["processing_id"]
        checks.append((f"/history?left={snapshot_id}&left_processing={processing_id}", "observation", None))
        for filename in ("raw.parquet", "observation.parquet", "manifest.json", "cleaned.parquet", "selected.parquet",
                         "coverage.parquet", "surface.json", "diagnostics.json", "processing-manifest.json"):
            source = ((store.observation_path(snapshot_id) / filename) if filename in {"raw.parquet", "observation.parquet", "manifest.json"}
                      else store.artifact_path(snapshot_id, processing_id) / ("manifest.json" if filename == "processing-manifest.json" else filename))
            checks.append((f"/history/{snapshot_id}/files/{filename}?processing={processing_id}", "download", source))
    if len(selected) == 2:
        checks.append((f"/history?left={selected[0]['snapshot_id']}&right={selected[1]['snapshot_id']}", "comparison", None))
    if include_live:
        checks.append(("/?ticker=SPY&strike_min_pct=93&strike_max_pct=107&dte_min=7&dte_max=30&smooth=1", "live", None))

    def check(item):
        path, kind, expected = item
        with tempfile.TemporaryDirectory(prefix=".http-check-", dir=output.parent) as temporary:
            if vercel_cli:
                body_file, header_file = Path(temporary) / "body", Path(temporary) / "headers"
                command = [node, str(vercel_cli), "curl", path, "--deployment", url]
                if scope:
                    command += ["--scope", scope]
                command += ["--", "--fail", "--silent", "--show-error", "--max-time", "120",
                            "--output", str(body_file), "--dump-header", str(header_file)]
                response = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
                if response.returncode:
                    raise AssertionError(f"HTTP check failed: {path}: {response.stderr[-500:]}")
                body, headers = body_file.read_bytes(), header_file.read_text().lower()
            else:
                with urlopen(url.rstrip("/") + path, timeout=120) as response:
                    assert response.status == 200
                    body, headers = response.read(), str(response.headers).lower()
            if kind == "download":
                assert body == expected.read_bytes(), f"Download checksum mismatch: {path}"
                assert "cache-control: no-store" in headers and "content-disposition: attachment" in headers
            else:
                text = body.decode("utf-8")
                assert "Observation unavailable" not in text and "error-card" not in text, f"App error: {path}"
                if kind == "landing":
                    assert "click here to see saved surfaces" in text
                elif kind == "catalog":
                    assert all(i["snapshot_id"] in text for i in selected)
                elif kind == "observation":
                    assert "Plotly.newPlot" in text and "Spot source:" in text
                elif kind == "comparison":
                    assert "supported comparison points" in text and text.count("Plotly.newPlot") == 3
                elif kind == "live":
                    assert "Plotly.newPlot" in text and "Contracts fetched" in text and "Contracts used" in text
                    if expected_backend:
                        backend = re.search(r"Numerical backend:\s*<strong>(python|cpp)</strong>", text)
                        assert backend and backend.group(1) == expected_backend, "Unexpected live numerical backend"
                if kind in {"catalog", "observation", "comparison", "live"}:
                    assert "cache-control: no-store" in headers
            return {"path": path, "kind": kind, "passed": True, "sha256": hashlib.sha256(body).hexdigest()}

    # Small fixed concurrency; no recurring traffic or unbounded retries.
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(check, checks))
    report = {"url": url, "verified": True, "checks": results,
              "downloads": sum(c["kind"] == "download" for c in results), "live_checked": include_live,
              "verified_live_backend": expected_backend}
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--bundle", default="history_bundle")
    parser.add_argument("--output", required=True)
    parser.add_argument("--vercel-cli", help="Installed CLI JavaScript entrypoint; credentials stay with the CLI")
    parser.add_argument("--node", default="node")
    parser.add_argument("--scope")
    parser.add_argument("--include-live", action="store_true")
    parser.add_argument("--expected-backend", choices=("python", "cpp"),
                        help="Require this backend in the live surface diagnostics (also use --include-live)")
    args = parser.parse_args()
    report = verify_deployment(args.url, args.bundle, args.output, args.vercel_cli, args.node, args.scope,
                               args.include_live, args.expected_backend)
    print(json.dumps({"verified": report["verified"], "checks": len(report["checks"]), "downloads": report["downloads"]}))


if __name__ == "__main__":
    main()
