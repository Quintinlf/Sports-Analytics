-- Persist per-sport outcomes from the daily prediction pipeline so admins
-- can confirm which sports ran, when, and why a sport failed (if any).
-- Safe to run multiple times (IF NOT EXISTS).
--
-- Target: PostgreSQL / Supabase.
-- Local SQLite is created automatically by ensure_pipeline_run_log() in
-- data/prediction_service.py (uses INTEGER PRIMARY KEY AUTOINCREMENT).

CREATE TABLE IF NOT EXISTS pipeline_run_log (
    run_id SERIAL PRIMARY KEY,
    sport TEXT NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT,
    predictions_count INTEGER NOT NULL DEFAULT 0,
    run_at TEXT NOT NULL
);
