"""
Ensemble Weights Manager

Loads and persists model weights from/to the retraining_metadata table.
Default weights: GP=0.30, LGBM-Win=0.30, LGBM-Quantile=0.25, Elo=0.15
"""

import json
from typing import Dict, Optional

_DEFAULT_WEIGHTS: Dict[str, float] = {
    'gp': 0.30,
    'lgbm_win': 0.30,
    'lgbm_quantile': 0.25,
    'elo': 0.15,
}


class WeightManager:
    """
    Manage ensemble model weights stored in the database so they survive
    across process restarts and are updated after each retraining cycle.
    """

    def __init__(self, db=None):
        """
        Parameters
        ----------
        db : SportsAnalyticsDB or None
            If None, only default weights are used (no persistence).
        """
        self._db = db

    # ------------------------------------------------------------------
    def get_weights(self) -> Dict[str, float]:
        """
        Return current weights.

        Reads from the most recent retraining_metadata row if available;
        falls back to defaults.
        """
        if self._db is None:
            return dict(_DEFAULT_WEIGHTS)

        try:
            state = self._db.get_retraining_state()
            raw = state.get('ensemble_weights')
            if raw:
                loaded = json.loads(raw) if isinstance(raw, str) else raw
                # Fill any missing keys with defaults
                weights = dict(_DEFAULT_WEIGHTS)
                weights.update(loaded)
                return weights
        except Exception:
            pass

        return dict(_DEFAULT_WEIGHTS)

    def update_weights(self, performance: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust weights based on recent per-model performance metrics.

        Parameters
        ----------
        performance : dict
            Mapping model_key -> recent_r2 (or accuracy, etc.).
            Keys should be a subset of {'gp', 'lgbm_win', 'lgbm_quantile', 'elo'}.

        Returns
        -------
        dict  New normalised weights.
        """
        import numpy as np

        current = self.get_weights()

        if not performance:
            return current

        # Use softmax of performance scores to compute new weights
        keys = list(current.keys())
        scores = [max(0.0, performance.get(k, 0.5)) for k in keys]
        exp_scores = [float(x) for x in np.exp(scores)]
        total = sum(exp_scores)
        new_weights = {k: exp_scores[i] / total for i, k in enumerate(keys)}

        if self._db is not None:
            state = self._db.get_retraining_state()
            self._db.update_retraining_state(
                incremental_count=state.get('incremental_count', 0),
                model_version=state.get('model_version'),
                ensemble_weights=new_weights,
            )

        return new_weights


def default_weights() -> Dict[str, float]:
    """Return a copy of the default ensemble weights."""
    return dict(_DEFAULT_WEIGHTS)
