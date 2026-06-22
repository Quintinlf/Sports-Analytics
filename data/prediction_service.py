"""Unified Prediction Service - orchestrates live data from all sports."""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class UnifiedPredictionService:
    """Orchestrates live prediction data collection across all sports."""

    def __init__(self, nba_service: Any, mlb_service: Any, fifa_service: Any):
        """Initialize with sport-specific service instances."""
        self.services = [nba_service, mlb_service, fifa_service]
        logger.info(f"Initialized UnifiedPredictionService with {len(self.services)} sport services")

    def fetch_all(self) -> List[Dict[str, Any]]:
        """Fetch predictions from all services and combine results."""
        combined_predictions: List[Dict[str, Any]] = []

        for service in self.services:
            sport_name = getattr(service, "sport_name", "Unknown")
            try:
                logger.info(f"Invoking data collection for: {sport_name}")
                games = service.fetch_upcoming_games()
                if games:
                    combined_predictions.extend(games)
                    logger.info(f"Collected {len(games)} predictions from {sport_name}")
                else:
                    logger.info(f"No predictions returned from {sport_name}")
            except Exception as e:
                logger.error(f"Service execution failure on {type(service).__name__} ({sport_name}): {e}", exc_info=True)
                continue

        logger.info(f"Unified service collected {len(combined_predictions)} total predictions")
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
