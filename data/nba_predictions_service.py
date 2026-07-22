"""NBA Live Predictions Service - wraps existing NBA loader infrastructure."""
from __future__ import annotations

import json
import logging
import os
from enum import Enum
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

from data.nba_loader import fetch_upcoming_games as fetch_nba_raw_games
from data.nba_live_features import build_nba_live_features
from data.explanation_engine import build_snapshot
from data.prediction_errors import ModelUnavailableError

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
            raw_games = fetch_nba_raw_games()

            if not raw_games:
                logger.info("No upcoming NBA games found")
                return self._fetch_recent_final_games(ensemble, limit=8) or self.handle_off_season()

            return self.build_prediction_rows(raw_games, ensemble)
        except ModelUnavailableError:
            raise
        except Exception as e:
            logger.error(f"Error fetching live NBA data: {e}", exc_info=True)
            return self._fetch_recent_final_games(ensemble, limit=8) or self.handle_off_season()

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
        """Return empty or alternative data based on off-season strategy."""
        logger.info(f"NBA off-season detected. Applying strategy: {self.strategy.value}")
        if self.strategy == OffSeasonStrategy.HISTORICAL:
            logger.info("Would fetch historical data (not yet implemented)")
            return []
        return []

    def _fetch_recent_final_games(self, ensemble, limit: int = 8) -> List[Dict[str, Any]]:
        """Fallback to recent completed NBA games when no upcoming games exist.

        Runs the real ensemble model against each team's pre-game state (as
        of that game's date, using only strictly-earlier history) so the
        displayed prediction is a genuine forecast rather than derived from
        the already-known result. actual_* / correct columns are filled from
        the real final score for review purposes.
        """
        season_candidates = []
        year = datetime.utcnow().year
        season_candidates.append(f"{year-1}-{str(year)[-2:]}")
        season_candidates.append(f"{year-2}-{str(year-1)[-2:]}")

        games_df: Optional[pd.DataFrame] = None
        for season in season_candidates:
            try:
                finder = leaguegamefinder.LeagueGameFinder(
                    season_nullable=season,
                    season_type_nullable="Regular Season",
                    league_id_nullable="00",
                    timeout=30,
                )
                df = finder.get_data_frames()[0]
                if df is not None and not df.empty:
                    games_df = df
                    break
            except Exception as e:
                logger.warning(f"Historical NBA fetch failed for season {season}: {e}")

        if games_df is None or games_df.empty:
            return []

        df = games_df.copy()
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        df = df.dropna(subset=["GAME_DATE"]).sort_values("GAME_DATE", ascending=False)

        rows: List[Dict[str, Any]] = []
        seen = set()
        for game_id, group in df.groupby("GAME_ID", sort=False):
            if game_id in seen or len(rows) >= limit:
                continue

            home = group[group["MATCHUP"].astype(str).str.contains("vs\\.", regex=True)]
            away = group[group["MATCHUP"].astype(str).str.contains("@", regex=False)]
            if home.empty or away.empty:
                continue

            home_row = home.iloc[0]
            away_row = away.iloc[0]
            home_score = int(home_row.get("PTS", 0) or 0)
            away_score = int(away_row.get("PTS", 0) or 0)
            actual_winner = str(home_row["TEAM_NAME"]) if home_score >= away_score else str(away_row["TEAM_NAME"])
            game_date = home_row["GAME_DATE"].strftime("%Y-%m-%d")

            row = self._predict_one(
                ensemble=ensemble,
                home_team=str(home_row["TEAM_NAME"]),
                away_team=str(away_row["TEAM_NAME"]),
                game_date=game_date,
                provider_game_id=str(game_id),
                prediction_status="FINAL",
                actual_home_score=home_score,
                actual_away_score=away_score,
                actual_winner=actual_winner,
            )
            if row is not None:
                rows.append(row)
            seen.add(game_id)

        return rows

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
        explanations = []
        metrics: Dict[str, Any] = {
            "spread": result.get("spread"),
            "q10": result.get("q10"),
            "q90": result.get("q90"),
            "uncertainty": result.get("uncertainty"),
            "win_prob_raw": result.get("win_prob"),
        }
        label_map = {
            "gp": "Gaussian Process (spread)",
            "lgbm_win": "LightGBM Win Model",
            "lgbm_quantile": "LightGBM Quantile Spread",
            "elo": "Elo Rating",
        }
        for key, contrib in contributions.items():
            weight = _to_native_float(contrib.get("weight", 0.0))
            contrib_win_prob = _to_native_float(contrib.get("win_prob", 0.5))
            explanations.append({
                "label": label_map.get(key, key),
                "weight": round(weight, 4),
                "value": f"{contrib_win_prob:.1%} home win",
            })
            metrics[f"{key}_win_prob"] = round(contrib_win_prob, 4)

        feature_snapshot = build_snapshot(
            sport="NBA",
            data_source="nba_api",
            is_fallback=False,
            confidence_score=win_prob,
            explanations=explanations,
            metrics=metrics,
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
