-- Prediction lifecycle columns (PostgreSQL / Supabase).
-- Safe to run multiple times (ADD COLUMN IF NOT EXISTS).
--
-- Identity / time:
--   provider_game_id   — canonical game identity (existing)
--   start_time_utc     — canonical tip-off / kickoff timestamp
--   game_date_pacific  — derived display/filter date (NOT identity)
--
-- Status split:
--   game_status:       SCHEDULED | LIVE | FINAL
--   prediction_status: UPCOMING | ACTIVE | SETTLED | VOID
--     (legacy FINAL values should be migrated to SETTLED)

ALTER TABLE predictions ADD COLUMN IF NOT EXISTS start_time_utc TIMESTAMP;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS game_date_pacific DATE;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS game_status TEXT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS pipeline_run_id TEXT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS model_version TEXT;

-- Backfill Pacific display date from legacy game_date where missing
UPDATE predictions
SET game_date_pacific = game_date
WHERE game_date_pacific IS NULL AND game_date IS NOT NULL;

-- Normalize legacy FINAL prediction_status → SETTLED
UPDATE predictions
SET prediction_status = 'SETTLED'
WHERE UPPER(COALESCE(prediction_status, '')) = 'FINAL';

-- Rows with scores but not yet marked settled
UPDATE predictions
SET game_status = 'FINAL',
    prediction_status = 'SETTLED'
WHERE actual_home_score IS NOT NULL
  AND actual_away_score IS NOT NULL
  AND UPPER(COALESCE(prediction_status, '')) NOT IN ('SETTLED', 'VOID');

CREATE INDEX IF NOT EXISTS idx_predictions_start_time_utc
    ON predictions (start_time_utc);
CREATE INDEX IF NOT EXISTS idx_predictions_game_date_pacific
    ON predictions (game_date_pacific);
CREATE INDEX IF NOT EXISTS idx_predictions_game_status
    ON predictions (game_status);
CREATE INDEX IF NOT EXISTS idx_predictions_prediction_status
    ON predictions (prediction_status);
