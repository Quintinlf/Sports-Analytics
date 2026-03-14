"""
Prediction Logger

Handles inserting predictions + feature snapshots to the DB and
matching them to actual game results once scores are known.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from data.database.database_handler import SportsAnalyticsDB


# ---------------------------------------------------------------------------
# Team-name normalisation (ported from PredictionMatcher in model_trainer.py)
# ---------------------------------------------------------------------------

_TEAM_ALIASES: Dict[str, str] = {
    'LA Clippers': 'Los Angeles Clippers',
    'LA Lakers': 'Los Angeles Lakers',
    'L.A. Clippers': 'Los Angeles Clippers',
    'L.A. Lakers': 'Los Angeles Lakers',
}


def normalize_team_name(name: str) -> str:
    """Canonical team name used for DB matching."""
    name = name.strip()
    return _TEAM_ALIASES.get(name, name)


# ---------------------------------------------------------------------------


class PredictionLogger:
    """
    Write predictions to the database and later resolve them to results.

    Parameters
    ----------
    db : SportsAnalyticsDB
        Open database connection.
    model_version : str, optional
        Version tag to stamp on every prediction row.
    """

    def __init__(self, db: SportsAnalyticsDB, model_version: Optional[str] = None):
        self._db = db
        self._model_version = model_version or ''

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_prediction(
        self,
        prediction: Dict,
        features: Optional[Dict] = None,
    ) -> int:
        """
        Insert one prediction row and, optionally, its feature snapshot.

        Parameters
        ----------
        prediction : dict
            Output from EnsemblePredictor.predict() merged with game context.
            Expected keys: game_date, home_team, away_team, and the standard
            ensemble output keys (win_prob, spread, q10, q90, uncertainty,
            confidence).  Optional: game_id, predicted_home_score,
            predicted_away_score, model_contributions.
        features : dict, optional
            Raw feature values to serialise into prediction_features.

        Returns
        -------
        int  prediction_id of the newly inserted row.
        """
        home_team = normalize_team_name(prediction.get('home_team', ''))
        away_team = normalize_team_name(prediction.get('away_team', ''))

        spread = float(prediction.get('spread', 0.0))
        win_prob = float(prediction.get('win_prob', 0.5))
        confidence = prediction.get('confidence', 'MEDIUM')
        # Map confidence label to numeric score
        _conf_score = {'HIGH': 0.85, 'MEDIUM': 0.65, 'LOW': 0.45}
        confidence_score = _conf_score.get(confidence, 0.65)

        predicted_winner = home_team if spread >= 0 else away_team

        prediction_data = {
            'game_id': prediction.get('game_id'),
            'game_date': prediction.get('game_date', datetime.now().strftime('%Y-%m-%d')),
            'home_team': home_team,
            'away_team': away_team,
            'predicted_spread': spread,
            'predicted_home_score': prediction.get('predicted_home_score'),
            'predicted_away_score': prediction.get('predicted_away_score'),
            'predicted_winner': predicted_winner,
            'win_probability': win_prob,
            'confidence_score': confidence_score,
            'confidence_level': confidence,
            'pred_std': prediction.get('uncertainty'),
            'ci_lower': prediction.get('q10'),
            'ci_upper': prediction.get('q90'),
            'model_versions': prediction.get('model_contributions', {}),
            'model_version': self._model_version,
            'notes': prediction.get('notes'),
        }

        prediction_id = self._db.insert_prediction(prediction_data)

        if features:
            self._db.log_prediction_features(prediction_id, features)

        return prediction_id

    # ------------------------------------------------------------------
    # Result matching
    # ------------------------------------------------------------------

    def match_result(
        self,
        game_date: str,
        home_team: str,
        away_team: str,
        actual_home_score: int,
        actual_away_score: int,
        date_tolerance_days: int = 2,
    ) -> List[int]:
        """
        Find unmatched predictions for the given game and write results.

        Parameters
        ----------
        game_date : str  YYYY-MM-DD
        home_team : str
        away_team : str
        actual_home_score : int
        actual_away_score : int
        date_tolerance_days : int
            Search window around game_date.

        Returns
        -------
        list[int]  prediction_ids that were updated.
        """
        home_team = normalize_team_name(home_team)
        away_team = normalize_team_name(away_team)

        date_dt = datetime.strptime(game_date, '%Y-%m-%d')
        start = (date_dt - timedelta(days=date_tolerance_days)).strftime('%Y-%m-%d')
        end = (date_dt + timedelta(days=date_tolerance_days)).strftime('%Y-%m-%d')

        candidates = self._db.get_predictions_by_date(start, end)

        actual_spread = float(actual_home_score - actual_away_score)
        actual_winner = home_team if actual_spread >= 0 else away_team

        updated_ids: List[int] = []
        for row in candidates:
            if (
                normalize_team_name(row.get('home_team', '')) == home_team
                and normalize_team_name(row.get('away_team', '')) == away_team
            ):
                pred_id = row['prediction_id']
                pred_spread = float(row.get('predicted_spread', 0))
                ci_lower = row.get('ci_lower') or (pred_spread - 6)
                ci_upper = row.get('ci_upper') or (pred_spread + 6)

                result_data = {
                    'actual_home_score': actual_home_score,
                    'actual_away_score': actual_away_score,
                    'actual_spread': actual_spread,
                    'actual_winner': actual_winner,
                    'prediction_error': abs(pred_spread - actual_spread),
                    'correct_winner': (pred_spread >= 0) == (actual_spread >= 0),
                    'within_ci': float(ci_lower) <= actual_spread <= float(ci_upper),
                }
                self._db.insert_result(pred_id, result_data)
                updated_ids.append(pred_id)

        return updated_ids

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_unmatched_predictions(self, days: int = 7) -> List[Dict]:
        """
        Return predictions from the past *days* days that have no result row yet.
        """
        if self._db.conn is None:
            return []

        cursor = self._db.conn.cursor()
        cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        cursor.execute(
            """
            SELECT p.*
            FROM predictions p
            LEFT JOIN prediction_results r ON p.prediction_id = r.prediction_id
            WHERE r.result_id IS NULL
              AND p.game_date >= ?
            ORDER BY p.game_date
            """,
            (cutoff,),
        )

        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
