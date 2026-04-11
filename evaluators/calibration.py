"""
Calibration Model

Fits isotonic regression on historical win probabilities vs. outcomes
and applies the mapping to raw model outputs.
"""

import os
import pickle
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression

from data.database.database_handler import SportsAnalyticsDB


class CalibrationModel:
    """
    Post-hoc probability calibrator using isotonic regression.

    Usage
    -----
        cal = CalibrationModel()
        cal.fit(db)
        calibrated = cal.calibrate(raw_probabilities)
        cal.save('machine_learning/models/calibration_model.pkl')
    """

    _SAVE_PATH = os.path.join('machine_learning', 'models', 'calibration_model.pkl')

    def __init__(self):
        self._iso: Optional[IsotonicRegression] = None
        self._fitted = False

    # ------------------------------------------------------------------

    def fit_from_arrays(
        self,
        probs: np.ndarray,
        outcomes: np.ndarray,
        min_samples: int = 10,
    ) -> 'CalibrationModel':
        """Fit isotonic model from in-memory probability/outcome arrays."""
        p = np.asarray(probs, dtype=float)
        y = np.asarray(outcomes, dtype=float)

        if len(p) < int(min_samples):
            self._iso = None
            self._fitted = False
            return self

        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(p, y)

        self._iso = iso
        self._fitted = True
        return self

    def fit_recent(
        self,
        db: SportsAnalyticsDB,
        limit: int = 200,
        min_samples: int = 10,
    ) -> 'CalibrationModel':
        """Fit from the most recent settled predictions with known correctness."""
        if db.conn is None:
            self._iso = None
            self._fitted = False
            return self

        cursor = db.conn.cursor()
        cursor.execute(
            """
            SELECT
                p.win_probability,
                COALESCE(p.correct, r.correct_winner) AS correct_winner
            FROM predictions p
            LEFT JOIN prediction_results r ON p.prediction_id = r.prediction_id
            WHERE p.win_probability IS NOT NULL
              AND COALESCE(p.correct, r.correct_winner) IS NOT NULL
            ORDER BY p.prediction_timestamp DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        rows = cursor.fetchall()

        if len(rows) < int(min_samples):
            self._iso = None
            self._fitted = False
            return self

        probs = np.array([float(r[0]) for r in rows], dtype=float)
        outcomes = np.array([int(r[1]) for r in rows], dtype=float)
        return self.fit_from_arrays(probs=probs, outcomes=outcomes, min_samples=min_samples)

    def fit(self, db: SportsAnalyticsDB) -> 'CalibrationModel':
        """
        Build the isotonic mapping from logged prediction history.

        Queries prediction_results JOIN predictions for rows where
        both win_probability and correct_winner are available.

        Parameters
        ----------
        db : SportsAnalyticsDB
            Open database connection with historical data.

        Returns
        -------
        self
        """
        return self.fit_recent(db=db, limit=1000000, min_samples=10)

    def calibrate(self, win_probs: np.ndarray) -> np.ndarray:
        """
        Apply isotonic calibration to raw win probabilities.

        Parameters
        ----------
        win_probs : array-like, shape (n,)

        Returns
        -------
        np.ndarray  calibrated probabilities in [0, 1]
        """
        arr = np.asarray(win_probs, dtype=float)
        if not self._fitted or self._iso is None:
            return np.clip(arr, 0.0, 1.0)
        return np.clip(self._iso.predict(arr), 0.0, 1.0)

    # ------------------------------------------------------------------

    def save(self, filepath: Optional[str] = None) -> str:
        filepath = filepath or self._SAVE_PATH
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        return filepath

    @classmethod
    def load(cls, filepath: Optional[str] = None) -> 'CalibrationModel':
        filepath = filepath or cls._SAVE_PATH
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected CalibrationModel, got {type(obj).__name__}")
        return obj

    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._fitted
