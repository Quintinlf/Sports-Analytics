"""
Incremental Update Pipeline

Lightweight update step intended to run daily between full retrains.
Updates retraining metadata and triggers weekly retrain when threshold
is reached.
"""

from datetime import datetime
from typing import Dict, Optional

from data.database.database_handler import SportsAnalyticsDB
from ensemble.ensemble_weights import WeightManager
from training.weekly_retrain import WeeklyRetrain


class IncrementalUpdater:
    """
    Manages incremental update state and adaptive ensemble weighting.
    """

    def __init__(self, db_path: str = 'sports_analytics.db', retrain_threshold: int = 7):
        self.db_path = db_path
        self.retrain_threshold = retrain_threshold

    def update(
        self,
        performance: Optional[Dict[str, float]] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Apply one incremental update step.

        Parameters
        ----------
        performance : dict, optional
            Per-model recent score, e.g. {'gp': 0.58, 'lgbm_win': 0.62, ...}
            If provided, ensemble weights are adapted and persisted.

        Returns
        -------
        dict with update details and retrain status.
        """
        with SportsAnalyticsDB(self.db_path) as db:
            state = db.get_retraining_state()
            current_count = int(state.get('incremental_count') or 0)
            model_version = state.get('model_version')

            # Optional adaptive weight update
            wm = WeightManager(db)
            if performance:
                weights = wm.update_weights(performance)
            else:
                weights = wm.get_weights()

            new_count = current_count + 1
            db.update_retraining_state(
                incremental_count=new_count,
                model_version=model_version,
                full_retrain=False,
                ensemble_weights=weights,
            )

        retrain = WeeklyRetrain(self.db_path, threshold=self.retrain_threshold)
        retrain_result = retrain.check_and_retrain(force=False, verbose=verbose)

        out = {
            'timestamp': datetime.now().isoformat(),
            'incremental_count_before': current_count,
            'incremental_count_after': new_count,
            'weights': weights,
            'retrain_triggered': bool(retrain_result.get('triggered', False)),
            'retrain_result': retrain_result,
        }

        if verbose:
            print(f"Incremental update complete: {current_count} -> {new_count}")
            if out['retrain_triggered']:
                print('Weekly retrain triggered.')

        return out
