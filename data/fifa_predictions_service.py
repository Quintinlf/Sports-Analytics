"""FIFA/Soccer Live Predictions Service - integrates with free soccer data sources.

Live fixtures are scoped to major international tournaments (FIFA World Cup,
UEFA European Championships) rather than domestic club leagues, because
machine_learning/models/fifa_ensemble.pkl (see training/fifa_trainer.py) is
trained on national-squad profiles from those exact competitions. A club
fixture like "Arsenal vs Man City" has no corresponding squad profile and
would always fall back anyway.
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
        """Fetch fixtures using a free/fallback source.

        Scoped to the two international competitions fifa_ensemble.pkl was
        trained on (see training/fifa_trainer.py COMPETITIONS) — thesportsdb
        league IDs confirmed directly against the live API.
        """
        try:
            league_ids = [
                ("FIFA World Cup", "4429"),
                ("UEFA European Championships", "4502"),
            ]
            fixtures: List[Dict[str, Any]] = []
            for league_name, league_id in league_ids:
                url = f"{self.api_base}/eventsnextleague.php?id={league_id}"
                resp = requests.get(url, timeout=15)
                if resp.status_code != 200:
                    continue
                payload = resp.json()
                events = payload.get("events") or []
                for ev in events[:6]:
                    fixtures.append(
                        {
                            "id": ev.get("idEvent"),
                            "league": league_name,
                            "home_team": ev.get("strHomeTeam"),
                            "away_team": ev.get("strAwayTeam"),
                            "utc_date": ev.get("dateEvent"),
                        }
                    )
            return fixtures
        except Exception as e:
            logger.warning(f"Error fetching fixtures: {e}")
            return []

    def build_prediction_rows(
        self, fixtures: List[Dict[str, Any]], model_bundle: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Transform fixture data into prediction DB shape using the real
        trained model. Fixtures where either squad has no profile on file
        (e.g. didn't qualify for a tracked tournament) are skipped with a
        logged warning rather than filled with a guessed prediction.
        """
        prediction_rows = []
        model = model_bundle["model"]
        squad_profiles = model_bundle["squad_profiles"]
        for fixture in fixtures:
            try:
                home_team = fixture.get("home_team", "Unknown")
                away_team = fixture.get("away_team", "Unknown")

                proba: Optional[Dict[str, float]] = model.predict_match(
                    squad_profiles, home_team, away_team
                )
                if proba is None:
                    logger.warning(
                        "Skipping %s vs %s: no squad profile on file for one or both teams.",
                        home_team, away_team,
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
                    "game_date": str(fixture.get("utc_date", datetime.today().strftime("%Y-%m-%d")).split("T")[0]),
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
