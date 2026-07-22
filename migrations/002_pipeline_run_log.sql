-- Persist per-sport outcomes from the daily prediction pipeline so admins
-- can confirm which sports ran, when, and why a sport failed (if any).
-- Safe to run multiple times (IF NOT EXISTS).
--
-- SQLite:
CREATE TABLE IF NOT EXISTS pipeline_run_log (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT,
    predictions_count INTEGER NOT NULL DEFAULT 0,
    run_at TEXT NOT NULL
);

-- PostgreSQL equivalent (run instead of the SQLite DDL above):
-- CREATE TABLE IF NOT EXISTS pipeline_run_log (
--     run_id SERIAL PRIMARY KEY,
--     sport TEXT NOT NULL,
--     status TEXT NOT NULL,
--     error_message TEXT,
--     predictions_count INTEGER NOT NULL DEFAULT 0,
--     run_at TEXT NOT NULL
-- );
--
-- The daily cron / UnifiedPredictionService also auto-creates this table
-- via ensure_pipeline_run_log() using the correct dialect.
