"""
Ensemble Predictor

Composes GP, LGBMWin, LGBMQuantile, and Elo models into a single
weighted ensemble.  Weights are managed by WeightManager and persisted
to the retraining_metadata table.
"""

import json
import os
import pickle
from typing import Dict, Optional

import numpy as np
import pandas as pd

from machine_learning.gp_model import GaussianProcessPredictor
from machine_learning.lightgbm_models import LGBMWinPredictor, LGBMQuantilePredictor
from machine_learning.elo_model import EloModel
from ensemble.ensemble_weights import WeightManager, default_weights


class EnsemblePredictor:
    """
    Weighted ensemble of four component models:
        * GP (point-differential with uncertainty)
        * LGBMWin (calibrated win probability)
        * LGBMQuantile (Q10 / Q50 / Q90 spread)
        * Elo (rating-based win probability)

    Usage
    -----
        ep = EnsemblePredictor()
        ep.load_models(gp_path, lgbm_win_path, lgbm_quantile_path, elo_path)
        result = ep.predict(home_team_id, away_team_id, features_df)
    """

    def __init__(self, model_dir: str = 'machine_learning/models', db=None):
        self.model_dir = model_dir
        self._db = db
        self._weight_manager = WeightManager(db)
        self.weights: Dict[str, float] = self._weight_manager.get_weights()

        self.gp: Optional[GaussianProcessPredictor] = None
        self.lgbm_win: Optional[LGBMWinPredictor] = None
        self.lgbm_quantile: Optional[LGBMQuantilePredictor] = None
        self.elo: Optional[EloModel] = None

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def load_models(
        self,
        gp_path: str,
        lgbm_win_path: str,
        lgbm_quantile_path: str,
        elo_path: str,
    ) -> None:
        """Load all four component models from disk."""
        self.gp = GaussianProcessPredictor.load(gp_path)
        self.lgbm_win = LGBMWinPredictor.load(lgbm_win_path)
        self.lgbm_quantile = LGBMQuantilePredictor.load(lgbm_quantile_path)
        self.elo = EloModel.load(elo_path)
        # Refresh weights every time models are (re)loaded
        self.weights = self._weight_manager.get_weights()

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        home_team_id: int,
        away_team_id: int,
        features: pd.DataFrame,
    ) -> Dict:
        """
        Produce a blended prediction for a single matchup.

        Parameters
        ----------
        home_team_id : int
            NBA team ID for the home side.
        away_team_id : int
            NBA team ID for the away side.
        features : pd.DataFrame
            One-row DataFrame of matchup features (same schema used to
            train LGBMWin and LGBMQuantile).

        Returns
        -------
        dict with keys:
            win_prob          float  – blended home win probability [0, 1]
            spread            float  – predicted home point differential
            q10               float  – 10th-percentile spread
            q90               float  – 90th-percentile spread
            uncertainty       float  – standard deviation estimate
            confidence        str    – "HIGH" / "MEDIUM" / "LOW"
            model_contributions  dict  – per-model win_prob and weight
        """
        contributions: Dict[str, Dict] = {}

        # ---- GP ---------------------------------------------------
        gp_win_prob, gp_spread = self._gp_predict(features)
        contributions['gp'] = {
            'win_prob': gp_win_prob,
            'spread': gp_spread,
            'weight': self.weights.get('gp', 0.30),
        }

        # ---- LGBM Win --------------------------------------------
        lgbm_win_out = self._lgbm_win_predict(features)
        contributions['lgbm_win'] = {
            'win_prob': lgbm_win_out.get('win_prob', 0.5),
            'spread': lgbm_win_out.get('point_diff', 0.0),
            'weight': self.weights.get('lgbm_win', 0.30),
        }

        # ---- LGBM Quantile ---------------------------------------
        lgbm_q_out = self._lgbm_quantile_predict(features)
        contributions['lgbm_quantile'] = {
            'win_prob': lgbm_q_out.get('win_prob', 0.5),
            'spread': lgbm_q_out.get('spread', 0.0),
            'q10': lgbm_q_out.get('q10', 0.0),
            'q90': lgbm_q_out.get('q90', 0.0),
            'weight': self.weights.get('lgbm_quantile', 0.25),
        }

        # ---- Elo --------------------------------------------------
        elo_win_prob = self._elo_predict(home_team_id, away_team_id)
        elo_spread = self.elo.predict_spread(home_team_id, away_team_id) if self.elo else 0.0
        contributions['elo'] = {
            'win_prob': elo_win_prob,
            'spread': elo_spread,
            'weight': self.weights.get('elo', 0.15),
        }

        # ---- Blend -----------------------------------------------
        blended_win_prob = sum(
            c['win_prob'] * c['weight'] for c in contributions.values()
        )
        blended_spread = sum(
            c['spread'] * c['weight'] for c in contributions.values()
        )

        # Normalise weights in case they don't sum to 1.0
        total_weight = sum(c['weight'] for c in contributions.values())
        if total_weight > 0:
            blended_win_prob /= total_weight
            blended_spread /= total_weight

        blended_win_prob = float(np.clip(blended_win_prob, 0.0, 1.0))

        # Uncertainty from GP std + quantile width
        q10 = contributions['lgbm_quantile'].get('q10', blended_spread - 6)
        q90 = contributions['lgbm_quantile'].get('q90', blended_spread + 6)
        quantile_width = abs(float(q90) - float(q10))
        uncertainty = quantile_width / 4.0  # rough std estimate

        confidence = _confidence_label(uncertainty)

        return {
            'win_prob': round(blended_win_prob, 4),
            'spread': round(float(blended_spread), 2),
            'q10': round(float(q10), 2),
            'q90': round(float(q90), 2),
            'uncertainty': round(uncertainty, 4),
            'confidence': confidence,
            'model_contributions': contributions,
        }

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def update_weights(self, performance: Dict[str, float]) -> None:
        """
        Re-compute and persist weights given per-model performance metrics.

        Parameters
        ----------
        performance : dict
            model_key -> recent performance score (e.g. R², accuracy).
        """
        self.weights = self._weight_manager.update_weights(performance)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Pickle the entire ensemble (including component models)."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> 'EnsemblePredictor':
        """Load a previously saved ensemble."""
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _gp_predict(self, features: pd.DataFrame):
        """Return (win_prob, spread) from GP model, or fallback."""
        if self.gp is None:
            return 0.5, 0.0
        try:
            X = features.values if hasattr(features, 'values') else features
            pred_out = self.gp.predict(X, return_std=True)
            if isinstance(pred_out, tuple):
                spread_arr = pred_out[0]
                std_arr = pred_out[1] if len(pred_out) > 1 else np.array([1.0])
            else:
                spread_arr = pred_out
                std_arr = np.array([1.0])

            spread_val = float(spread_arr[0])
            std_val = float(std_arr[0])
            win_prob = float(1.0 / (1.0 + np.exp(-spread_val / max(std_val, 1.0))))
            return win_prob, spread_val
        except Exception:
            return 0.5, 0.0

    def _lgbm_win_predict(self, features: pd.DataFrame) -> Dict:
        """Return prediction dict from LGBMWinPredictor, or fallback."""
        if self.lgbm_win is None:
            return {'win_prob': 0.5, 'point_diff': 0.0}
        try:
            return self.lgbm_win.predict_win_probability(features)
        except Exception:
            return {'win_prob': 0.5, 'point_diff': 0.0}

    def _lgbm_quantile_predict(self, features: pd.DataFrame) -> Dict:
        """Return Q10/Q50/Q90 dict from LGBMQuantilePredictor, or fallback."""
        if self.lgbm_quantile is None:
            return {'win_prob': 0.5, 'spread': 0.0, 'q10': -6.0, 'q90': 6.0}
        try:
            preds = self.lgbm_quantile.predict(features)
            q50 = float(preds.get('q50', [0.0])[0])
            q10 = float(preds.get('q10', [q50 - 6])[0])
            q90 = float(preds.get('q90', [q50 + 6])[0])
            from machine_learning.lightgbm_models import point_diff_to_win_prob
            win_prob = float(point_diff_to_win_prob(np.array([q50]))[0])
            return {'win_prob': win_prob, 'spread': q50, 'q10': q10, 'q90': q90}
        except Exception:
            return {'win_prob': 0.5, 'spread': 0.0, 'q10': -6.0, 'q90': 6.0}

    def _elo_predict(self, home_team_id: int, away_team_id: int) -> float:
        """Return Elo win probability, or fallback."""
        if self.elo is None:
            return 0.5
        try:
            return float(self.elo.predict_win_probability(home_team_id, away_team_id))
        except Exception:
            return 0.5


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _confidence_label(uncertainty: float) -> str:
    if uncertainty < 6.0:
        return 'HIGH'
    if uncertainty < 12.0:
        return 'MEDIUM'
    return 'LOW'
