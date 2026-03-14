"""
Experimental iterative predictor.

Provides a confidence-gated prediction loop on top of EnsemblePredictor.
"""

from typing import Any, Dict, Optional

import pandas as pd

from data.database.database_handler import SportsAnalyticsDB
from ensemble.ensemble_predictor import EnsemblePredictor
from evaluators.prediction_logger import PredictionLogger


class IterativePredictor:
    """
    Iterative prediction pipeline with optional weight adaptation between
    attempts when confidence is below threshold.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        max_iterations: int = 5,
        db_path: str = 'sports_analytics.db',
        verbose: bool = True,
    ):
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.verbose = verbose

        self.db = SportsAnalyticsDB(db_path)
        self.logger = PredictionLogger(self.db)
        self.ensemble = EnsemblePredictor(db=self.db)

        self.stats = {
            'total_predictions': 0,
            'retraining_triggered': 0,
            'avg_iterations': 0.0,
            'avg_confidence': 0.0,
        }

    def load_models(
        self,
        gp_path: str,
        lgbm_win_path: str,
        lgbm_quantile_path: str,
        elo_path: str,
    ) -> None:
        self.ensemble.load_models(gp_path, lgbm_win_path, lgbm_quantile_path, elo_path)

    def predict_with_retraining(
        self,
        home_team: str,
        away_team: str,
        game_date: str,
        home_team_id: int,
        away_team_id: int,
        features: pd.DataFrame,
        game_id: Optional[str] = None,
        performance_feedback: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Iterate prediction attempts until confidence threshold is met.

        Parameters
        ----------
        features : pd.DataFrame
            One-row feature DataFrame expected by the ensemble components.
        performance_feedback : dict, optional
            If provided and confidence is low, used to update ensemble weights.
        """
        if not isinstance(features, pd.DataFrame):
            raise TypeError('features must be a pandas DataFrame')
        if len(features) == 0:
            raise ValueError('features cannot be empty')

        iteration = 0
        retraining_triggered = False
        history = []
        final = None

        while iteration < self.max_iterations:
            iteration += 1
            pred = self.ensemble.predict(home_team_id, away_team_id, features.iloc[[0]])

            confidence_score = self._confidence_to_score(pred.get('confidence', 'LOW'))
            pred['confidence_score'] = confidence_score
            history.append({'iteration': iteration, 'prediction': pred.copy()})

            if self.verbose:
                print(
                    f"Iteration {iteration}: win_prob={float(pred['win_prob']):.3f}, "
                    f"spread={float(pred['spread']):.2f}, conf={pred['confidence']}"
                )

            final = pred
            if confidence_score >= self.confidence_threshold:
                break

            if iteration < self.max_iterations:
                retraining_triggered = True
                self.stats['retraining_triggered'] += 1
                if performance_feedback:
                    self.ensemble.update_weights(performance_feedback)

        result = {
            'game_id': game_id,
            'game_date': game_date,
            'home_team': home_team,
            'away_team': away_team,
            'predicted_spread': float(final.get('spread', 0.0)) if final else 0.0,
            'predicted_winner': home_team if (final and float(final.get('spread', 0.0)) >= 0) else away_team,
            'win_probability': float(final.get('win_prob', 0.5)) if final else 0.5,
            'confidence_score': float(final.get('confidence_score', 0.45)) if final else 0.45,
            'confidence_level': final.get('confidence', 'LOW') if final else 'LOW',
            'pred_std': float(final.get('uncertainty', 0.0)) if final else 0.0,
            'ci_lower': float(final.get('q10', 0.0)) if final else 0.0,
            'ci_upper': float(final.get('q90', 0.0)) if final else 0.0,
            'model_versions': final.get('model_contributions', {}) if final else {},
            'iteration_count': iteration,
            'retraining_triggered': retraining_triggered,
            'iteration_history': history,
            'ensemble_output': final or {},
        }

        self._update_stats(iteration, result['confidence_score'])
        return result

    def save_prediction_to_db(self, prediction: Dict[str, Any], features: Optional[Dict] = None) -> int:
        payload = {
            'game_id': prediction.get('game_id'),
            'game_date': prediction.get('game_date'),
            'home_team': prediction.get('home_team'),
            'away_team': prediction.get('away_team'),
            'spread': prediction.get('predicted_spread', 0.0),
            'win_prob': prediction.get('win_probability', 0.5),
            'q10': prediction.get('ci_lower'),
            'q90': prediction.get('ci_upper'),
            'uncertainty': prediction.get('pred_std'),
            'confidence': prediction.get('confidence_level', 'LOW'),
            'model_contributions': prediction.get('model_versions', {}),
        }
        return self.logger.log_prediction(payload, features=features)

    def get_statistics(self) -> Dict[str, Any]:
        return dict(self.stats)

    def close(self) -> None:
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _update_stats(self, iterations: int, confidence: float) -> None:
        n_prev = self.stats['total_predictions']
        n_new = n_prev + 1
        self.stats['total_predictions'] = n_new
        self.stats['avg_iterations'] = (self.stats['avg_iterations'] * n_prev + iterations) / n_new
        self.stats['avg_confidence'] = (self.stats['avg_confidence'] * n_prev + confidence) / n_new

    @staticmethod
    def _confidence_to_score(level: str) -> float:
        level = (level or 'LOW').upper()
        return {'HIGH': 0.85, 'MEDIUM': 0.65, 'LOW': 0.45}.get(level, 0.45)
