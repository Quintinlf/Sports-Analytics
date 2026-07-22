-- Idempotency log for weekly (and future) reviewer emails.
-- Prevents duplicate sends for the same reviewer + email_type + UTC date
-- when GitHub Actions retries or workflow_dispatch overlaps a cron run.
--
-- Safe to run multiple times (IF NOT EXISTS).
-- Compatible with SQLite and PostgreSQL.

CREATE TABLE IF NOT EXISTS email_send_log (
    reviewer_id TEXT NOT NULL,
    email_type TEXT NOT NULL,
    send_date TEXT NOT NULL,
    email TEXT,
    sent_at TEXT NOT NULL,
    PRIMARY KEY (reviewer_id, email_type, send_date)
);
