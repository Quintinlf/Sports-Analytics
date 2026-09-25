# Dual database boundary

Live product path and training/eval still use different DB accessors.

| Concern | Module | Target |
|---------|--------|--------|
| Cron ingest, API, settlement, smoke | `scripts/db_utils.py` + SQLAlchemy (`backend/db.py`) | `DATABASE_URL` / Supabase / Render Postgres (SQLite locally) |
| Training, ensemble weights, legacy feedback loop | `data/database/database_handler.py` (`SportsAnalyticsDB`) | Local SQLite `sports_analytics.db` |

**Rules**

1. Live features must not write through `SportsAnalyticsDB`.
2. Settlement of production predictions runs via `scripts/settle_live_predictions.py` on the SQLAlchemy DB.
3. Full consolidation is deferred; do not assume training settlement updates the dashboard.
