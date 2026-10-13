"""Small generated fixtures, explicitly not collected or backfilled market history."""
from __future__ import annotations

import pandas as pd
from src.data_cleaner import _black_scholes_price
from src.history_store import LocalObservationStore
from src.surface_config import SurfaceRequest
from src.time_utils import time_to_expiration_years


def fixture_quotes(as_of: str, spot: float = 100, shift: float = 0.0,
                   expirations: tuple[str, ...] = ("2026-10-16", "2026-10-23", "2026-11-06")) -> pd.DataFrame:
    rows = []
    timestamp = pd.Timestamp(as_of)
    for expiry in expirations:
        years = time_to_expiration_years(expiry, timestamp)
        for strike in range(90, 111, 2):
            sigma = .24 + shift + .5 * (strike / spot - 1) ** 2 + .03 * years
            for kind in ("call", "put"):
                price = _black_scholes_price(kind, spot, strike, years, .02, sigma)
                spread = max(.002, min(.02, price * .008))
                rows.append({"contractSymbol": f"DEMO{expiry}{kind[0]}{strike}", "expirationDate": expiry,
                             "optionType": kind, "strike": float(strike), "bid": max(.0001, price - spread),
                             "ask": price + spread, "lastPrice": price, "lastTradeDate": timestamp,
                             "volume": 150, "openInterest": 800, "impliedVolatility": sigma,
                             "snapshotTimestampUtc": timestamp.isoformat()})
    raw = pd.DataFrame(rows)
    raw.attrs["fetchDiagnostics"] = {"selected_expirations": [{"expiration": e} for e in expirations],
                                     "details": [{"expiration": e, "snapshot_timestamp_utc": as_of} for e in expirations],
                                     "outcomes": [{"expiration": e, "status": "received", "collected_at_utc": as_of} for e in expirations]}
    return raw


def seed_demo(store: LocalObservationStore) -> list[dict]:
    request = SurfaceRequest("DEMO", strike_min_pct=.9, strike_max_pct=1.1, dte_min=0, dte_max=60, smooth=True)
    observations = []
    for name, as_of, spot, shift, expiries in (
        ("synthetic-a", "2026-10-12T18:00:00Z", 100, 0, ("2026-10-16", "2026-10-23", "2026-11-06", "2026-11-13")),
        ("synthetic-b", "2026-10-13T18:00:00Z", 101, .035, ("2026-10-16", "2026-10-23", "2026-11-06")),
    ):
        raw = fixture_quotes(as_of, spot, shift, expiries)
        if name == "synthetic-b":
            raw.attrs["fetchDiagnostics"]["selected_expirations"].append({"expiration": "2026-11-13"})
            raw.attrs["fetchDiagnostics"]["outcomes"].append({"expiration": "2026-11-13", "status": "failed",
                                                            "error": "Simulated interruption", "collected_at_utc": as_of})
        observation = store.save(request, raw,
                                 {"value": spot, "provider": "synthetic_fixture", "method": "configured fixture spot",
                                  "collected_at_utc": as_of, "source_timestamp_utc": as_of},
                                 as_of, source_kind="synthetic_fixture", label=f"Synthetic {name[-1].upper()} — generated BSM quotes",
                                 snapshot_id=name)
        store.process(name)
        observations.append(observation)
    return observations
