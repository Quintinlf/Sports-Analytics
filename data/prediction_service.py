"""Unified Prediction Service - orchestrates live data from all sports."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

PIPELINE_RUN_LOG_DDL = """
CREATE TABLE IF NOT EXISTS pipeline_run_log (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT,
    predictions_count INTEGER NOT NULL DEFAULT 0,
    run_at TEXT NOT NULL
)
"""

PIPELINE_RUN_LOG_DDL_PG = """
CREATE TABLE IF NOT EXISTS pipeline_run_log (
    run_id SERIAL PRIMARY KEY,
    sport TEXT NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT,
    predictions_count INTEGER NOT NULL DEFAULT 0,
    run_at TEXT NOT NULL
)
"""


def ensure_pipeline_run_log(engine: Engine) -> None:
    """Create pipeline_run_log if missing (SQLite + Postgres)."""
    ddl = PIPELINE_RUN_LOG_DDL_PG if engine.dialect.name == "postgresql" else PIPELINE_RUN_LOG_DDL
    with engine.begin() as conn:
        conn.execute(text(ddl))


def record_pipeline_run(
    engine: Optional[Engine],
    sport: str,
    status: str,
    predictions_count: int = 0,
    error_message: Optional[str] = None,
) -> None:
    """Persist one sport's outcome for a pipeline run. No-op if engine is None."""
    if engine is None:
        return
    try:
        ensure_pipeline_run_log(engine)
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO pipeline_run_log
                        (sport, status, error_message, predictions_count, run_at)
                    VALUES
                        (:sport, :status, :error_message, :predictions_count, :run_at)
                    """
                ),
                {
                    "sport": sport,
                    "status": status,
                    "error_message": error_message,
                    "predictions_count": int(predictions_count),
                    "run_at": datetime.utcnow().isoformat(),
                },
            )
    except Exception as exc:
        logger.warning("Failed to write pipeline_run_log for %s: %s", sport, exc)


class UnifiedPredictionService:
    """Orchestrates live prediction data collection across all sports."""

    def __init__(self, nba_service: Any, mlb_service: Any, fifa_service: Any):
        """Initialize with sport-specific service instances."""
        self.services = [nba_service, mlb_service, fifa_service]
        # Populated by fetch_all(): {sport_name: "ExceptionClass: message"} for
        # any service that raised instead of returning games. Callers (the
        # cron script) use this to fail loudly and report which sport broke,
        # instead of a silent "0 predictions" reading as a clean run.
        self.last_run_failures: Dict[str, str] = {}
        # Optional engine for persisting pipeline_run_log rows during fetch_all.
        self._run_log_engine: Optional[Engine] = None
        logger.info(f"Initialized UnifiedPredictionService with {len(self.services)} sport services")

    def fetch_all(self, engine: Optional[Engine] = None) -> List[Dict[str, Any]]:
        """Fetch predictions from all services and combine results.

        When ``engine`` is provided, one ``pipeline_run_log`` row is written per
        sport (success or failure) so outcomes survive past process exit.
        """
        combined_predictions: List[Dict[str, Any]] = []
        self.last_run_failures = {}
        self._run_log_engine = engine

        for service in self.services:
            sport_name = getattr(service, "sport_name", "Unknown")
            try:
                logger.info(f"Invoking data collection for: {sport_name}")
                games = service.fetch_upcoming_games()
                count = len(games) if games else 0
                if games:
                    combined_predictions.extend(games)
                    logger.info(f"Collected {count} predictions from {sport_name}")
                else:
                    logger.info(f"No predictions returned from {sport_name}")
                record_pipeline_run(
                    self._run_log_engine,
                    sport_name,
                    status="ok",
                    predictions_count=count,
                )
            except Exception as e:
                logger.error(f"Service execution failure on {type(service).__name__} ({sport_name}): {e}", exc_info=True)
                self.last_run_failures[sport_name] = f"{type(e).__name__}: {e}"
                record_pipeline_run(
                    self._run_log_engine,
                    sport_name,
                    status="error",
                    predictions_count=0,
                    error_message=f"{type(e).__name__}: {e}",
                )
                continue

        logger.info(f"Unified service collected {len(combined_predictions)} total predictions")
        if self.last_run_failures:
            logger.error(
                "Sports that produced NO predictions this run due to an error: %s",
                self.last_run_failures,
            )
        return combined_predictions

    def sync_to_database(
        self, engine: Any, predictions: List[Dict[str, Any]], insertion_callback: Callable
    ) -> bool:
        """Archive-preserving sync that UPSERTs prediction records."""
        try:
            if not predictions:
                logger.warning("No fresh live predictions collected to insert.")
                return False

            inserted_count = 0
            for pred in predictions:
                if isinstance(pred.get("feature_snapshot"), dict):
                    pred["feature_snapshot"] = json.dumps(pred["feature_snapshot"])

                try:
                    insertion_callback(engine, pred)
                    inserted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to insert prediction for {pred.get('home_team')} vs {pred.get('away_team')}: {e}")
                    continue

            logger.info(f"Successfully synchronized {inserted_count} prediction rows via UPSERT.")
            return inserted_count > 0
        except Exception as e:
            logger.error(f"Database synchronization layer failed: {e}", exc_info=True)
            return False
