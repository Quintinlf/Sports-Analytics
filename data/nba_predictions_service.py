"""NBA Live Predictions Service - wraps existing NBA loader infrastructure."""
from __future__ import annotations

import json
import logging
import os
from enum import Enum
from datetime import datetime
from typing import Any, Dict, List, Optional

from data.nba_live_features import build_nba_live_features
from data.explanation_engine import build_snapshot, explain_nba_prediction
from data.prediction_errors import ModelUnavailableError
from data.competitions.nba_schedule import NbaScheduleProvider

logger = logging.getLogger(__name__)

_NBA_MODEL_POINTER_PATH = os.path.join("machine_learning", "models", "nba_latest.json")
_NBA_MODEL_DIR = os.path.dirname(_NBA_MODEL_POINTER_PATH)
_nba_model_cache: Dict[str, Any] = {}


def _load_nba_model():
    """Lazy-load the production NBA ensemble (GP + LightGBM + Elo).

    Raises ModelUnavailableError (never returns None) if the pointer file or
    any component artifact is missing/corrupt — callers must let this
    propagate so the pipeline fails loudly instead of silently falling back
    to a guess.

    ML dependencies are imported here rather than at module scope so the
    FastAPI web service doesn't need them at startup — matching the lazy
    load pattern in data/mlb_predictions_service.py and
    data/fifa_predictions_service.py.
    """
    if "ensemble" in _nba_model_cache:
        return _nba_model_cache["ensemble"]

    if not os.path.exists(_NBA_MODEL_POINTER_PATH):
        raise ModelUnavailableError(
            f"No trained NBA model pointer at {_NBA_MODEL_POINTER_PATH}. "
            "Run training/trainer.py (ModelTrainer.full_retrain) to train one."
        )

    try:
        with open(_NBA_MODEL_POINTER_PATH) as fh:
            pointer = json.load(fh)
        from ensemble.ensemble_predictor import EnsemblePredictor

        # Pointer stores basenames only (cross-platform); resolve against
        # this process's own models directory rather than trusting an
        # embedded path, which may carry the training host's separator.
        def _resolve(key: str) -> str:
            return os.path.join(_NBA_MODEL_DIR, os.path.basename(pointer[key]))

        ensemble = EnsemblePredictor(model_dir=_NBA_MODEL_DIR, db=None)
        ensemble.load_models(
            gp_path=_resolve("gp_path"),
            lgbm_win_path=_resolve("lgbm_win_path"),
            lgbm_quantile_path=_resolve("lgbm_quantile_path"),
            elo_path=_resolve("elo_path"),
        )
        logger.info("Loaded trained NBA ensemble version %s", pointer.get("version"))
    except Exception as exc:
        raise ModelUnavailableError(f"Failed to load NBA ensemble: {exc}") from exc

    _nba_model_cache["ensemble"] = ensemble
    return ensemble


def _to_native_float(value: Any) -> float:
    """Coerce a scalar or length-1 array-like (numpy) into a plain Python float."""
    try:
        import numpy as np
        arr = np.asarray(value, dtype=float).reshape(-1)
        return float(arr[0])
    except Exception:
        return float(value)


class OffSeasonStrategy(Enum):
    """Handling strategy when a sport is off-season."""
    EMPTY = "EMPTY"
    SUMMER_LEAGUES = "SUMMER_LEAGUES"
    HISTORICAL = "HISTORICAL"


class NBALivePredictionService:
    """Fetches live NBA games and formats them for the prediction pipeline."""

    def __init__(self, strategy: OffSeasonStrategy = OffSeasonStrategy.EMPTY):
        self.strategy = strategy
        self.sport_name = "NBA"

    def fetch_upcoming_games(self) -> List[Dict[str, Any]]:
        """Fetch live NBA schedule and return prediction rows.

        Model-load failures (ModelUnavailableError) are intentionally NOT
        caught here — they propagate to UnifiedPredictionService.fetch_all(),
        which logs them loudly and skips NBA for this run rather than
        inserting hardcoded-confidence predictions.
        """
        # Fail fast and loud if the model can't load, before spending time
        # on network calls — this is a deployment problem, not "no games".
        ensemble = _load_nba_model()

        try:
            logger.info("Fetching live NBA games via nba_api...")
            raw_games = NbaScheduleProvider().fetch_as_nba_dicts()

            if not raw_games:
                logger.info("No upcoming NBA games found (offseason or empty scoreboard)")
                return self.handle_off_season()

            return self.build_prediction_rows(raw_games, ensemble)
        except ModelUnavailableError:
            raise
        except Exception as e:
            logger.error(f"Error fetching live NBA data: {e}", exc_info=True)
            return self.handle_off_season()

    def build_prediction_rows(self, raw_games: List[Dict[str, Any]], ensemble) -> List[Dict[str, Any]]:
        """Transform NBA loader output into prediction DB shape using the
        real ensemble model. Games the model can't score (insufficient live
        history — e.g. season just started) are skipped with a logged
        warning rather than filled with a guessed prediction.
        """
        prediction_rows = []
        for game in raw_games:
            try:
                home_team = game.get("home_team", "Unknown")
                away_team = game.get("away_team", "Unknown")
                game_date = str(game.get("game_date", datetime.today().strftime("%Y-%m-%d")))

                row = self._predict_one(
                    ensemble=ensemble,
                    home_team=home_team,
                    away_team=away_team,
                    game_date=game_date,
                    provider_game_id=str(game.get("GAME_ID") or ""),
                    prediction_status="UPCOMING",
                )
                if row is None:
                    logger.warning(
                        "Skipping %s @ %s: insufficient live history for a model prediction "
                        "(e.g. early season).",
                        away_team, home_team,
                    )
                    continue
                prediction_rows.append(row)
            except Exception as e:
                logger.warning(f"Skipped game due to parse error: {e}")
                continue

        logger.info(f"Built {len(prediction_rows)} NBA prediction rows")
        return prediction_rows

    def handle_off_season(self) -> List[Dict[str, Any]]:
        """Return empty when scoreboard has no upcoming games.

        Historical FINAL games are never exposed as upcoming predictions.
        """
        logger.info(
            "NBA off-season / empty upcoming schedule. Strategy=%s (no FINAL fallback)",
            self.strategy.value,
        )
        return []

    def _predict_one(
        self,
        ensemble,
        home_team: str,
        away_team: str,
        game_date: str,
        provider_game_id: str,
        prediction_status: str,
        actual_home_score: Optional[int] = None,
        actual_away_score: Optional[int] = None,
        actual_winner: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Run the ensemble model for one matchup and build a prediction row.

        Returns None if there isn't enough live history to build features —
        callers should skip the game rather than insert a guess.
        """
        live = build_nba_live_features(
            home_team, away_team, as_of_date=game_date, feature_cols=ensemble.lgbm_win.feature_names
        )
        if live is None:
            return None
        features_df, home_team_id, away_team_id = live

        result = ensemble.predict(home_team_id, away_team_id, features_df)
        win_prob = float(result["win_prob_calibrated"])
        predicted_winner = home_team if win_prob >= 0.5 else away_team
        confidence_level = result["confidence"]

        contributions = result.get("model_contributions", {})
        metrics: Dict[str, Any] = {
            "spread": result.get("spread"),
            "q10": result.get("q10"),
            "q90": result.get("q90"),
            "uncertainty": result.get("uncertainty"),
            "win_prob_raw": result.get("win_prob"),
        }
        for key, contrib in contributions.items():
            contrib_win_prob = _to_native_float(contrib.get("win_prob", 0.5))
            metrics[f"{key}_win_prob"] = round(contrib_win_prob, 4)

        explanation = explain_nba_prediction(
            features=features_df,
            predicted_winner=predicted_winner,
            home_team=home_team,
            away_team=away_team,
            win_probability=win_prob,
            confidence_level=confidence_level,
        )
        # Keep ensemble blend weights in metrics for debugging; UI uses why_factors.
        metrics["ensemble_weights"] = {
            key: _to_native_float(contrib.get("weight", 0.0))
            for key, contrib in contributions.items()
        }

        feature_snapshot = build_snapshot(
            sport="NBA",
            data_source="nba_api",
            is_fallback=False,
            confidence_score=win_prob,
            explanations=explanation["explanations"],
            metrics=metrics,
            why_factors=explanation["why_factors"],
            risk_factors=explanation["risk_factors"],
        )

        return {
            "sport": "NBA",
            "league": "NBA",
            "game_date": game_date,
            "home_team": home_team,
            "away_team": away_team,
            "provider_game_id": provider_game_id,
            "predicted_winner": predicted_winner,
            "win_probability": win_prob,
            "confidence_level": confidence_level,
            "bet_type": "Moneyline",
            "bet_units": 0.5,
            "bet_recommendation": f"Lean {predicted_winner}",
            "feature_snapshot": json.dumps(feature_snapshot),
            "model_name": "NBA-Ensemble-v1",
            "data_source": "nba_api",
            "is_fallback": False,
            "prediction_status": prediction_status,
            "actual_home_score": actual_home_score,
            "actual_away_score": actual_away_score,
            "actual_winner": actual_winner,
            "correct": (
                int(predicted_winner == actual_winner) if actual_winner is not None else None
            ),
            "created_at": datetime.utcnow().isoformat(),
        }
