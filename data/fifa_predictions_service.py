"""FIFA/Soccer Live Predictions Service - integrates with free soccer data sources.

Discovery covers club + international competitions via data.competitions.
Only fixtures with available national-squad model inputs receive AI scores;
others are stored as schedule-only rows (no fabricated predictions).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from data.nba_predictions_service import OffSeasonStrategy
from data.explanation_engine import build_snapshot, explain_fifa_prediction
from data.demo_data import demo_predictions_enabled
from data.prediction_errors import ModelUnavailableError
from data.competitions.soccer_schedule import SoccerScheduleProvider
from data.competitions.types import PredictPolicy

SCHEDULE_ONLY_NOTE = (
    "Fixture discovered, but model unavailable for this competition."
)

logger = logging.getLogger(__name__)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests package not found. FIFA live data will not be available.")

_FIFA_MODEL_PATH = os.path.join("machine_learning", "models", "fifa_ensemble.pkl")
_fifa_model_cache: Dict[str, Any] = {}


def _load_fifa_model():
    """Lazy-load the trained FIFA ensemble bundle.

    Raises ModelUnavailableError (never returns None) if the artifact is
    missing/corrupt — callers must let this propagate so the pipeline fails
    loudly instead of silently falling back to a guess.

    joblib/sklearn are imported here rather than at module scope so the
    FastAPI web service doesn't need ML dependencies at startup, matching
    data/mlb_predictions_service.py's _load_mlb_model() pattern.
    """
    if "bundle" in _fifa_model_cache:
        return _fifa_model_cache["bundle"]

    if not os.path.exists(_FIFA_MODEL_PATH):
        raise ModelUnavailableError(
            f"No trained FIFA model found at {_FIFA_MODEL_PATH}. "
            "Run training/fifa_trainer.py to train one."
        )

    try:
        import joblib

        bundle = joblib.load(_FIFA_MODEL_PATH)
        logger.info("Loaded trained FIFA ensemble version %s", bundle.get("version"))
    except Exception as exc:
        raise ModelUnavailableError(f"Failed to load FIFA model: {exc}") from exc

    _fifa_model_cache["bundle"] = bundle
    return bundle


def _confidence_label(probability: float) -> str:
    if probability >= 0.55:
        return "HIGH"
    if probability >= 0.40:
        return "MEDIUM"
    return "LOW"


class FIFALivePredictionService:
    """Fetches live FIFA/Soccer games and formats for prediction pipeline."""

    def __init__(self, strategy: OffSeasonStrategy = OffSeasonStrategy.EMPTY):
        self.strategy = strategy
        self.sport_name = "FIFA"
        self.api_base = "https://www.thesportsdb.com/api/v1/json/3"

    def fetch_upcoming_games(self) -> List[Dict[str, Any]]:
        """Fetch live soccer fixtures via free API or fallback.

        Model-load failures (ModelUnavailableError) are intentionally NOT
        caught here — they propagate to UnifiedPredictionService.fetch_all(),
        which logs them loudly and skips FIFA for this run rather than
        inserting hardcoded-confidence predictions.
        """
        if not REQUESTS_AVAILABLE:
            logger.warning("requests not available. Returning empty predictions.")
            return self.handle_off_season()

        # Fail fast and loud if the model can't load, before spending time
        # on network calls — this is a deployment problem, not "no games".
        model_bundle = _load_fifa_model()

        try:
            logger.info("Fetching live FIFA/Soccer games...")

            fixtures = self._fetch_free_fixtures()

            if not fixtures:
                logger.info("No upcoming soccer fixtures found")
                return self.handle_off_season()

            return self.build_prediction_rows(fixtures, model_bundle)
        except ModelUnavailableError:
            raise
        except Exception as e:
            logger.error(f"Failed to fetch live FIFA/Soccer schedule: {e}", exc_info=True)
            return self.handle_off_season()

    def _fetch_free_fixtures(self) -> List[Dict[str, Any]]:
        """Discover fixtures via competition catalog (club + international)."""
        try:
            provider = SoccerScheduleProvider(api_base=self.api_base)
            return provider.fetch_as_fifa_dicts()
        except Exception as e:
            logger.warning("Error fetching soccer fixtures: %s", e)
            return []

    def build_prediction_rows(
        self, fixtures: List[Dict[str, Any]], model_bundle: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Transform fixtures into DB rows.

        full_model competitions with squad profiles → AI prediction.
        schedule_only competitions, or missing model inputs → schedule-only
        row (is_fallback=True). Never fabricates an AI winner.
        """
        prediction_rows = []
        model = model_bundle["model"]
        squad_profiles = model_bundle["squad_profiles"]
        for fixture in fixtures:
            try:
                home_team = fixture.get("home_team", "Unknown")
                away_team = fixture.get("away_team", "Unknown")
                policy = self._fixture_policy(fixture)

                if policy == PredictPolicy.SCHEDULE_ONLY:
                    prediction_rows.append(
                        self._schedule_only_row(fixture, home_team, away_team)
                    )
                    continue

                proba: Optional[Dict[str, float]] = model.predict_match(
                    squad_profiles, home_team, away_team
                )
                if proba is None:
                    logger.info(
                        "Schedule-only for %s vs %s: model unavailable for this competition.",
                        home_team,
                        away_team,
                    )
                    prediction_rows.append(
                        self._schedule_only_row(fixture, home_team, away_team)
                    )
                    continue

                top_outcome = max(proba, key=proba.get)
                top_prob = proba[top_outcome]
                predicted_winner = {
                    "HOME_WIN": home_team,
                    "AWAY_WIN": away_team,
                    "DRAW": "Draw",
                }[top_outcome]
                confidence_level = _confidence_label(top_prob)
                metrics = {
                    "model_home_win_probability": round(proba["HOME_WIN"], 4),
                    "model_draw_probability": round(proba["DRAW"], 4),
                    "model_away_win_probability": round(proba["AWAY_WIN"], 4),
                    "schedule_only": False,
                }
                squad_maps = model.squad_metric_maps(squad_profiles, home_team, away_team)
                home_metrics, away_metrics = squad_maps if squad_maps else ({}, {})
                explanation = explain_fifa_prediction(
                    home_metrics=home_metrics,
                    away_metrics=away_metrics,
                    predicted_winner=predicted_winner,
                    home_team=home_team,
                    away_team=away_team,
                    win_probability=top_prob,
                    confidence_level=confidence_level,
                    outcome_probabilities=proba,
                )

                feature_snapshot = build_snapshot(
                    sport="FIFA",
                    data_source="thesportsdb",
                    is_fallback=False,
                    confidence_score=top_prob,
                    explanations=explanation["explanations"],
                    metrics=metrics,
                    why_factors=explanation["why_factors"],
                    risk_factors=explanation["risk_factors"],
                )

                row = {
                    "sport": "SOCCER",
                    "league": fixture.get("league", "International"),
                    "provider_game_id": str(fixture.get("id") or ""),
                    "game_date": str(
                        fixture.get("utc_date", datetime.today().strftime("%Y-%m-%d"))
                    ).split("T")[0],
                    "home_team": home_team,
                    "away_team": away_team,
                    "predicted_winner": predicted_winner,
                    "win_probability": top_prob,
                    "confidence_level": confidence_level,
                    "bet_type": "Moneyline",
                    "bet_units": 0.5,
                    "bet_recommendation": f"Lean {predicted_winner}",
                    "feature_snapshot": json.dumps(feature_snapshot),
                    "model_name": "FIFA-PCA-Ensemble-v1",
                    "data_source": "thesportsdb",
                    "is_fallback": False,
                    "prediction_status": "UPCOMING",
                    "actual_home_score": None,
                    "actual_away_score": None,
                    "actual_winner": None,
                    "correct": None,
                    "created_at": datetime.utcnow().isoformat(),
                }
                prediction_rows.append(row)
            except Exception as e:
                logger.warning(f"Skipped FIFA fixture due to parse error: {e}")
                continue

        logger.info(f"Built {len(prediction_rows)} FIFA/Soccer prediction rows")
        return prediction_rows

    @staticmethod
    def _fixture_policy(fixture: Dict[str, Any]) -> PredictPolicy:
        raw = fixture.get("predict_policy")
        if isinstance(raw, PredictPolicy):
            return raw
        if isinstance(raw, str):
            try:
                return PredictPolicy(raw)
            except ValueError:
                pass
        return PredictPolicy.FULL_MODEL

    def _schedule_only_row(
        self,
        fixture: Dict[str, Any],
        home_team: str,
        away_team: str,
    ) -> Dict[str, Any]:
        """Persist a discovered fixture without fabricating an AI prediction."""
        feature_snapshot = build_snapshot(
            sport="FIFA",
            data_source="thesportsdb",
            is_fallback=True,
            confidence_score=0.0,
            explanations=[],
            metrics={
                "schedule_only": True,
                "offseason_notice": SCHEDULE_ONLY_NOTE,
                "discovery_note": SCHEDULE_ONLY_NOTE,
            },
            why_factors=[],
            risk_factors=[],
        )
        return {
            "sport": "SOCCER",
            "league": fixture.get("league", "International"),
            "provider_game_id": str(fixture.get("id") or ""),
            "game_date": str(
                fixture.get("utc_date", datetime.today().strftime("%Y-%m-%d"))
            ).split("T")[0],
            "home_team": home_team,
            "away_team": away_team,
            "predicted_winner": "Scheduled",
            "win_probability": 0.0,
            "confidence_level": "N/A",
            "bet_type": "Moneyline",
            "bet_units": 0.0,
            "bet_recommendation": "Scheduled fixture — no AI prediction",
            "feature_snapshot": json.dumps(feature_snapshot),
            "model_name": None,
            "data_source": "thesportsdb",
            "is_fallback": True,
            "prediction_status": "UPCOMING",
            "actual_home_score": None,
            "actual_away_score": None,
            "actual_winner": None,
            "correct": None,
            "created_at": datetime.utcnow().isoformat(),
        }

    def handle_off_season(self) -> List[Dict[str, Any]]:
        """Return empty when no fixtures available.

        Fabricated "final" fixtures are only ever returned when
        ENABLE_DEMO_PREDICTIONS is explicitly set — production must not
        insert invented scores into the live predictions table.
        """
        logger.info(f"FIFA/Soccer off-season or no fixtures. Applying strategy: {self.strategy.value}")
        if not demo_predictions_enabled():
            return []

        # Dev-only: keep the FIFA tab populated with clearly-fabricated demo
        # samples so the dashboard has something to render locally.
        today = datetime.utcnow().strftime("%Y-%m-%d")
        return [
            {
                "sport": "SOCCER",
                "league": "FIFA World Cup",
                "game_date": today,
                "home_team": "France",
                "away_team": "Argentina",
                "predicted_winner": "Argentina",
                "win_probability": 0.58,
                "confidence_level": "MEDIUM",
                "bet_type": "Moneyline",
                "bet_units": 0.25,
                "bet_recommendation": "Review historical edge: Man City",
                "feature_snapshot": json.dumps(
                    build_snapshot(
                        sport="FIFA",
                        data_source="soccer_offseason_fallback",
                        is_fallback=True,
                        confidence_score=0.58,
                        explanations=[
                            {"label": "Recent Form", "weight": 0.3, "value": "Fallback"},
                            {"label": "Home Advantage", "weight": 0.25, "value": "Historical"},
                            {"label": "Goals For / Against", "weight": 0.25, "value": "1.9 / 1.1"},
                            {"label": "Injury Status", "weight": 0.2, "value": "Unavailable"},
                        ],
                        metrics={
                            "offseason_notice": "Limited live FIFA fixtures available. Showing fallback completed games for review.",
                            "xg_available": False,
                        },
                    )
                ),
                "model_name": "FIFA-Fallback-v1",
                "data_source": "soccer_offseason_fallback",
                "is_fallback": True,
                "prediction_status": "FINAL",
                "actual_home_score": 1,
                "actual_away_score": 2,
                "actual_winner": "Argentina",
                "correct": 1,
                "created_at": datetime.utcnow().isoformat(),
            },
            {
                "sport": "SOCCER",
                "league": "UEFA European Championships",
                "game_date": today,
                "home_team": "England",
                "away_team": "Spain",
                "predicted_winner": "Spain",
                "win_probability": 0.54,
                "confidence_level": "LOW",
                "bet_type": "Moneyline",
                "bet_units": 0.25,
                "bet_recommendation": "Small edge Spain",
                "feature_snapshot": json.dumps(
                    build_snapshot(
                        sport="FIFA",
                        data_source="soccer_offseason_fallback",
                        is_fallback=True,
                        confidence_score=0.54,
                        explanations=[
                            {"label": "Recent Form", "weight": 0.28, "value": "Fallback"},
                            {"label": "Home Advantage", "weight": 0.26, "value": "Historical"},
                            {"label": "Possession %", "weight": 0.24, "value": "53%"},
                            {"label": "Injury Status", "weight": 0.22, "value": "Unavailable"},
                        ],
                        metrics={
                            "offseason_notice": "Limited live FIFA fixtures available. Showing fallback completed games for review.",
                            "xg_available": False,
                        },
                    )
                ),
                "model_name": "FIFA-Fallback-v1",
                "data_source": "soccer_offseason_fallback",
                "is_fallback": True,
                "prediction_status": "FINAL",
                "actual_home_score": 2,
                "actual_away_score": 2,
                "actual_winner": "Draw",
                "correct": 0,
                "created_at": datetime.utcnow().isoformat(),
            },
        ]
