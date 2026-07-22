"""Prediction engine entry point for CLI scripts and GitHub Actions.

This module imports ML and sports-data dependencies. Do not import it from
the FastAPI web service — use scripts/cron_daily_predictions.py instead.
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.engine import Engine

from scripts.db_utils import insert_prediction

logger = logging.getLogger(__name__)


def create_prediction_service():
    """Build the unified live-prediction orchestrator (lazy ML imports)."""
    from data.fifa_predictions_service import FIFALivePredictionService
    from data.mlb_predictions_service import MLBLivePredictionService
    from data.nba_predictions_service import NBALivePredictionService, OffSeasonStrategy
    from data.prediction_service import UnifiedPredictionService

    return UnifiedPredictionService(
        nba_service=NBALivePredictionService(strategy=OffSeasonStrategy.EMPTY),
        mlb_service=MLBLivePredictionService(),
        fifa_service=FIFALivePredictionService(),
    )


def fetch_live_predictions() -> list[dict[str, Any]]:
    """Fetch live predictions from all sport pipelines."""
    service = create_prediction_service()
    logger.info("Polling live sports registries for current game schedules...")
    return service.fetch_all()


def sync_predictions_to_database(engine: Engine, predictions: list[dict[str, Any]]) -> bool:
    """Persist fetched predictions to the database."""
    service = create_prediction_service()
    logger.info("Synchronizing %s prediction(s) to database...", len(predictions))
    return service.sync_to_database(engine, predictions, insert_prediction)


def run_live_prediction_pipeline(engine: Engine) -> bool:
    """Fetch live predictions and sync them to the database.

    Returns False if ANY sport failed with an error (e.g. its model
    couldn't load) — even when other sports succeeded — so the cron job
    exits non-zero and the failure is visible in GitHub Actions instead of
    reading as a clean run with a quietly-missing sport.
    """
    service = create_prediction_service()
    logger.info("Polling live sports registries for current game schedules...")
    predictions = service.fetch_all(engine=engine)
    failures = dict(service.last_run_failures)

    if not predictions:
        logger.warning("Live sports streams returned no predictions.")
        if failures:
            logger.error("No predictions were produced, and these sports errored: %s", failures)
        return False

    logger.info("Synchronizing %s prediction(s) to database...", len(predictions))
    synced = service.sync_to_database(engine, predictions, insert_prediction)

    if failures:
        logger.error(
            "Run produced predictions for some sports, but these sports errored "
            "and contributed NOTHING this run: %s",
            failures,
        )
        return False

    return synced
