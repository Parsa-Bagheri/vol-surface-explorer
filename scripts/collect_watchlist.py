"""One bounded collect/process pass for an OS scheduler on persistent storage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.data_fetch import get_options_data, get_price_observation
from src.history_store import LocalObservationStore
from src.surface_config import SurfaceRequest
from src.surface_service import _validated_request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--store", required=True)
    parser.add_argument("--dte-max", type=int, default=30)
    args = parser.parse_args()
    if len(args.tickers) > 10:
        parser.error("Watchlists are limited to 10 tickers per pass.")
    store = LocalObservationStore(args.store)
    results = []
    for ticker in args.tickers:
        start = time.perf_counter()
        snapshot_id = None
        try:
            request = _validated_request(SurfaceRequest(ticker, dte_max=args.dte_max))
            valuation = pd.Timestamp.now(tz="UTC")
            spot = get_price_observation(request.ticker)
            raw = get_options_data(request.ticker, min_dte=request.dte_min, max_dte=request.dte_max, as_of_utc=valuation)
            observation = store.save(request, raw, spot, valuation)
            snapshot_id = observation["snapshot_id"]
            collection_seconds = time.perf_counter() - start
            artifact = store.process(snapshot_id)
            results.append({"ticker": request.ticker, "snapshot_id": snapshot_id, "partial": observation["partial"],
                            "processing_id": artifact["processing_id"], "collection_seconds": collection_seconds,
                            "processing_seconds": time.perf_counter() - start - collection_seconds})
        except Exception as exc:
            results.append({"ticker": ticker, "snapshot_id": snapshot_id, "error": str(exc), "elapsed_seconds": time.perf_counter() - start})
    print(json.dumps({"results": results}, indent=2))
    return int(any("error" in item for item in results))


if __name__ == "__main__":
    raise SystemExit(main())
