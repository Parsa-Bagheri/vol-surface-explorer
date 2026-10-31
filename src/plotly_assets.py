"""Read build-time browser asset metadata without importing Plotly on the homepage."""
import json
from pathlib import Path
import re

def plotly_script_url():
    path = Path(__file__).resolve().parents[1] / "static" / "plotly-version.json"
    version = json.loads(path.read_text(encoding="utf-8"))["version"]
    if not isinstance(version, str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("Invalid Plotly asset version; rerun the deployment install hook.")
    return f"https://cdn.plot.ly/plotly-{version}.min.js"
