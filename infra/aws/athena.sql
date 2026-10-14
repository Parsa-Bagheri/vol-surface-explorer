-- PREPARED ONLY. Replace bucket/prefix and review access/costs before execution.
-- Export coverage.parquet separately into analytics/date=YYYY-MM-DD/ticker=DEMO/.
-- Keep one explicitly selected processing version per snapshot in this projection.
CREATE EXTERNAL TABLE IF NOT EXISTS volatility_coverage (
  snapshot_id string,
  processing_id string,
  schema_version bigint,
  valuation_time_utc timestamp,
  source_kind string,
  partial boolean,
  expected_expirations bigint,
  received_expirations bigint,
  raw_rows bigint,
  cleaned_rows bigint,
  used_rows bigint,
  excluded_rows bigint,
  malformed_rows bigint,
  duplicate_rows bigint
)
PARTITIONED BY (observation_date string, ticker string)
STORED AS PARQUET
LOCATION 's3://REPLACE_BUCKET/analytics/';

-- Explicit registration avoids a crawler and makes the date/ticker scan bounded.
ALTER TABLE volatility_coverage ADD IF NOT EXISTS
PARTITION (observation_date='2026-10-12', ticker='DEMO')
LOCATION 's3://REPLACE_BUCKET/analytics/date=2026-10-12/ticker=DEMO/';

SELECT snapshot_id, partial, raw_rows, used_rows, excluded_rows
FROM volatility_coverage
WHERE observation_date='2026-10-12' AND ticker='DEMO'
LIMIT 20;
