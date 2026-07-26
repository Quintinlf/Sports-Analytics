"""MLB Live Predictions Service - integrates with statsapi for real MLB schedules."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from data.nba_predictions_service import OffSeasonStrategy
from data.explanation_engine import build_snapshot, explain_mlb_prediction
from data.mlb_context import build_mlb_context
from data.mlb_live_features import build_mlb_live_features
from data.prediction_errors import ModelUnavailableError

logger = logging.getLogger(__name__)

try:
    import statsapi
    STATSAPI_AVAILABLE = True
except ImportError:
    STATSAPI_AVAILABLE = False
    logger.warning("statsapi package not found. MLB live data will not be available.")

_MLB_MODEL_POINTER_PATH = os.path.join("machine_learning", "models", "mlb_latest.json")
_MLB_MODEL_DIR = os.path.dirname(_MLB_MODEL_POINTER_PATH)
_mlb_model_cache: Dict[str, Any] = {}


def _load_mlb_model():
    """Lazy-load the latest trained MLB win predictor.

    Raises ModelUnavailableError (never returns None) if the pointer file or
    model artifact is missing/corrupt — callers must let this propagate so
    the pipeline fails loudly instead of silently falling back to a guess.

    ML dependencies (lightgbm) are imported here rather than at module scope so
    the FastAPI web service doesn't need them at startup — matching how
    scripts/prediction_runner.py lazily imports the *_predictions_service modules.
    """
    if "model" in _mlb_model_cache:
        return _mlb_model_cache["model"]

    if not os.path.exists(_MLB_MODEL_POINTER_PATH):
        raise ModelUnavailableError(
            f"No trained MLB model pointer at {_MLB_MODEL_POINTER_PATH}. "
            "Run training/mlb_trainer.py to train one."
        )

    try:
        with open(_MLB_MODEL_POINTER_PATH) as fh:
            pointer = json.load(fh)
        from machine_learning.lightgbm_models import LGBMWinPredictor

        # Pointer stores a basename only (cross-platform); resolve it against
        # this process's own models directory rather than trusting an
        # embedded path, which may carry the training host's separator.
        model_path = os.path.join(_MLB_MODEL_DIR, os.path.basename(pointer["lgbm_win_path"]))
        model = LGBMWinPredictor.load(model_path)
        logger.info("Loaded trained MLB win model version %s", pointer.get("version"))
    except Exception as exc:
        raise ModelUnavailableError(f"Failed to load MLB model: {exc}") from exc

    _mlb_model_cache["model"] = model
    return model


class MLBLivePredictionService:
    """Fetches live MLB games via statsapi and formats for prediction pipeline."""

    def __init__(self, strategy: OffSeasonStrategy = OffSeasonStrategy.EMPTY):
        self.strategy = strategy
        self.sport_name = "MLB"

    def fetch_upcoming_games(self) -> List[Dict[str, Any]]:
        """Fetch live MLB schedule for today and nearby dates.

        Model-load failures (ModelUnavailableError) are intentionally NOT
        caught here — they propagate to UnifiedPredictionService.fetch_all(),
        which logs them loudly and skips MLB for this run rather than
        inserting hardcoded-confidence predictions.
        """
        if not STATSAPI_AVAILABLE:
            logger.warning("statsapi not available. Returning empty predictions.")
            return self.handle_off_season()

        # Fail fast and loud if the model can't load, before spending time
        # on network calls — this is a deployment problem, not "no games".
        model = _load_mlb_model()

        try:
            logger.info("Fetching live MLB games via statsapi...")

            # Get today and next few days of games
            today_str = datetime.today().strftime("%Y-%m-%d")
            schedule = statsapi.schedule(date=today_str)

            if not schedule:
                logger.info("No MLB games found for today")
                return self.handle_off_season()

            return self.build_prediction_rows(schedule, model)
        except ModelUnavailableError:
            raise
        except Exception as e:
            logger.error(f"Failed to query live MLB schedule: {e}", exc_info=True)
            return self.handle_off_season()

    def build_prediction_rows(self, games: List[Dict[str, Any]], model) -> List[Dict[str, Any]]:
        """Transform statsapi schedule into prediction DB shape.

        Games the model can't score (insufficient live history — e.g. very
        early season) are skipped with a logged warning rather than filled
        with a guessed prediction.
        """
        prediction_rows = []
        for game in games:
            try:
                home_team = game.get("home_name", "Unknown")
                away_team = game.get("away_name", "Unknown")
                game_date = str(
                    game.get("game_datetime", datetime.today().strftime("%Y-%m-%d")).split("T")[0]
                )

                mlb_ctx = build_mlb_context(game)

                live_features = build_mlb_live_features(
                    home_team, away_team, as_of_date=game_date, feature_cols=model.feature_names
                )
                if live_features is None:
                    logger.warning(
                        "Skipping %s @ %s: insufficient live history for a model prediction "
                        "(e.g. early season).",
                        away_team, home_team,
                    )
                    continue

                pred = model.predict_win_probability(live_features)
                win_prob = float(pred["win_prob"][0])
                predicted_winner = home_team if win_prob >= 0.5 else away_team
                confidence_level = str(pred["confidence_label"][0])

                metrics = {k: v for k, v in mlb_ctx.get("metrics", {}).items() if v is not None}
                metrics["model_win_probability"] = round(win_prob, 4)
                metrics["model_point_diff"] = round(float(pred["point_diff"][0]), 2)

                explanation = explain_mlb_prediction(
                    features=live_features,
                    predicted_winner=predicted_winner,
                    home_team=home_team,
                    away_team=away_team,
                    win_probability=win_prob,
                    confidence_level=confidence_level,
                    missing_data_warnings=mlb_ctx.get("missing_data_warnings", []),
                    pitcher_explanations=mlb_ctx.get("explanations", []),
                )
                feature_snapshot = build_snapshot(
                    sport="MLB",
                    data_source="mlb_statsapi",
                    is_fallback=False,
                    confidence_score=win_prob,
                    explanations=explanation["explanations"],
                    metrics=metrics,
                    why_factors=explanation["why_factors"],
                    risk_factors=explanation["risk_factors"],
                )
                feature_snapshot["starting_pitchers"] = mlb_ctx.get("starting_pitchers", {})
                feature_snapshot["bullpen"] = mlb_ctx.get("bullpen", {})
                feature_snapshot["lineups"] = mlb_ctx.get("lineups", {})
                feature_snapshot["missing_data_warnings"] = mlb_ctx.get("missing_data_warnings", [])

                row = {
                    "sport": "MLB",
                    "league": game.get("league", "MLB"),
                    "provider_game_id": str(game.get("game_id") or ""),
                    "game_date": game_date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "predicted_winner": predicted_winner,
                    "win_probability": win_prob,
                    "confidence_level": confidence_level,
                    "bet_type": "Moneyline",
                    "bet_units": 1.0,
                    "bet_recommendation": f"Lean {predicted_winner}",
                    "feature_snapshot": json.dumps(feature_snapshot),
                    "model_name": "MLB-LightGBM-v1",
                    "data_source": "mlb_statsapi",
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
                logger.warning(f"Skipped MLB game due to parse/inference error: {e}")
                continue

        logger.info(f"Built {len(prediction_rows)} MLB prediction rows")
        return prediction_rows

    def handle_off_season(self) -> List[Dict[str, Any]]:
        """Return empty when MLB is off-season."""
        logger.info(f"MLB off-season detected. Applying strategy: {self.strategy.value}")
        return []
