-- All raw observations, including total collection failures without a fitted surface.
SELECT snapshot_id, ticker, valuation_time_utc, partial,
       received_expirations, expected_expirations, raw_rows
FROM observations ORDER BY valuation_time_utc;

SELECT snapshot_id, expected_expirations - received_expirations AS missing,
       failed_expirations, malformed_rows, duplicate_rows
FROM observations WHERE partial OR malformed_rows > 0 OR duplicate_rows > 0;

-- One row per (snapshot_id, processing_id). Filter a processing version when
-- multiple explicit reprocessing versions are saved.
SELECT snapshot_id, processing_id, raw_rows, used_rows, excluded_rows,
       raw_rows = used_rows + excluded_rows AS reconciles
FROM coverage;

-- Stable source contract symbols, not build-local contractId, join across times.
SELECT a.stableContractKey, a.impliedVolatilityFinal AS iv_a,
       b.impliedVolatilityFinal AS iv_b,
       100 * (b.impliedVolatilityFinal - a.impliedVolatilityFinal) AS change_vol_points
FROM cleaned a JOIN cleaned b USING (stableContractKey)
WHERE a.snapshot_id='synthetic-a' AND b.snapshot_id='synthetic-b'
  AND a.includeInSurface AND b.includeInSurface
ORDER BY ABS(change_vol_points) DESC LIMIT 10;

SELECT snapshot_id, stableContractKey, COUNT(*) AS raw_duplicates
FROM raw_quotes GROUP BY snapshot_id, stableContractKey HAVING COUNT(*) > 1;
