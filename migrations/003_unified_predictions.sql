-- Unified predictions + prediction_options schema (PostgreSQL / Supabase).
-- Safe to run multiple times (IF NOT EXISTS / ADD COLUMN IF NOT EXISTS).
--
-- Prefer deploy-time application via:
--   python -m scripts.init_database
-- which also creates ORM tables. This SQL is for manual/ops use when the
-- web service has SCHEMA_AUTO_MIGRATE disabled (production default).

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id SERIAL PRIMARY KEY,
    provider_game_id TEXT,
    game_signature TEXT,
    sport TEXT NOT NULL,
    league TEXT,
    game_date DATE NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    predicted_winner TEXT NOT NULL,
    win_probability DOUBLE PRECISION,
    confidence_level TEXT NOT NULL,
    bet_type TEXT,
    bet_units DOUBLE PRECISION,
    bet_recommendation TEXT,
    feature_snapshot TEXT,
    model_name TEXT,
    prediction_status TEXT,
    actual_home_score INTEGER,
    actual_away_score INTEGER,
    actual_winner TEXT,
    correct INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    data_source TEXT,
    is_fallback BOOLEAN
);

ALTER TABLE predictions ADD COLUMN IF NOT EXISTS provider_game_id TEXT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS game_signature TEXT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS sport TEXT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS league TEXT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS win_probability DOUBLE PRECISION;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS feature_snapshot TEXT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS bet_type TEXT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS bet_units DOUBLE PRECISION;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS bet_recommendation TEXT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS model_name TEXT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS prediction_status TEXT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS actual_home_score INTEGER;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS actual_away_score INTEGER;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS actual_winner TEXT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS correct INTEGER;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS created_at TIMESTAMP;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS data_source TEXT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS is_fallback BOOLEAN;

CREATE TABLE IF NOT EXISTS prediction_options (
    option_id SERIAL PRIMARY KEY,
    prediction_id INTEGER NOT NULL REFERENCES predictions(prediction_id),
    option_name TEXT NOT NULL,
    probability DOUBLE PRECISION NOT NULL,
    rank INTEGER NOT NULL,
    UNIQUE(prediction_id, option_name)
);

CREATE INDEX IF NOT EXISTS idx_predictions_sport_date
    ON predictions (sport, game_date);
CREATE UNIQUE INDEX IF NOT EXISTS uq_predictions_game_signature
    ON predictions (game_signature);
CREATE INDEX IF NOT EXISTS idx_predictions_provider_game_id
    ON predictions (provider_game_id);
CREATE INDEX IF NOT EXISTS idx_options_prediction
    ON prediction_options (prediction_id);
