# Five-minute historical surface walkthrough

Prepare the local demo with the setup commands in [history.md](history.md). Use a new fixture store when processing source changes.

**0:00–0:45 — Show the product.** Open `/history` and select synthetic A and B. Point out both original valuation times and spots; A has complete requested coverage, B has three of four expirations. Explain that these are generated BSM quotes with a simulated missing expiration, not collected market history.

**0:45–1:30 — Explain raw provenance.** Download A's observation manifest and raw Parquet. Show the stored clock, spot lookup/source timestamp, requested expirations, per-expiration outcomes, saved settings, schema version, file SHA-256 and raw row count. `stableContractKey` joins contracts across observations; `contractId` identifies rows inside a build.

**1:30–2:15 — Replay offline.** Run `history.py --store STORE replay synthetic-a`. It restores original spot 100 and valuation `2026-10-12T18:00:00Z`. The test suite blocks sockets, Yahoo requests, live-price functions and the current clock on this path. Show matching replay and the recorded processing ID; explain that a changed source/dependency fingerprint requires explicit reprocessing and creates a separate immutable artifact.

**2:15–3:00 — Query quality.** Run the coverage SQL. A reconciles 88 = 47 used + 41 excluded; B reconciles 66 = 32 + 34. Query `observations WHERE partial` to identify B. Describe duplicate/malformed quarantine and the existing exclusive exclusion counts. Completely failed captures remain saved and SQL-visible without a fabricated surface.

**3:00–3:45 — Compare support.** Show the two surfaces and gold observed-IV markers, then the B-minus-A delta. In the measured fixture, 1,088 grid points are supported and 64 are excluded; comparison covers DTE 4–24. Blank cells have no jointly supported fitted values. Rates/settings/code must match; each side keeps its original spot and time. Interpolation produces fitted values, not extra observed chains.

**3:45–4:30 — Explain failures and durability.** A manifest is published only after validated files are flushed and a same-filesystem directory rename completes. Immutable IDs prevent overwrite; checksums expose corruption. Interrupted staging directories are excluded. An ID publication lock has a recorded owner; stale-lock cleanup requires verifying the owner is no longer active. Local coordination does not claim cross-host transactions.

**4:30–5:00 — State measured scope.** The 154-row synthetic workload uses 189,204 Parquet bytes. On the recorded Windows/Python environment, warm replay median was 1.58 seconds and coverage SQL median 0.13 seconds. These include integrity verification but exclude market network latency and browser rendering. The methodology is numerical financial modeling, with internal consistency tests; it is not predictive ML or an independent accuracy benchmark. S3/Athena code is prepared and fake-client-tested, not deployed.
