"""
Weekly Retrain Trigger

Checks incremental retraining counters and runs full retraining when
threshold is reached.
"""

from datetime import datetime
from typing import Dict

from data.database.database_handler import SportsAnalyticsDB
from training.trainer import ModelTrainer


class WeeklyRetrain:
    """Policy object for deciding when to run full retraining."""

    def __init__(self, db_path: str = 'sports_analytics.db', threshold: int = 7):
        self.db_path = db_path
        self.threshold = threshold

    def check_and_retrain(self, force: bool = False, verbose: bool = True) -> Dict:
        """
        Run full retrain when incremental_count >= threshold, unless force=False.

        Returns
        -------
        dict with action metadata.
        """
        with SportsAnalyticsDB(self.db_path) as db:
            state = db.get_retraining_state()
            count = int(state.get('incremental_count') or 0)

        should_retrain = force or count >= self.threshold
        if not should_retrain:
            return {
                'triggered': False,
                'reason': f'incremental_count={count} < threshold={self.threshold}',
                'incremental_count': count,
            }

        trainer = ModelTrainer(db_path=self.db_path)
        retrain_result = trainer.full_retrain(verbose=verbose)
        return {
            'triggered': True,
            'reason': 'forced' if force else f'incremental_count reached {count}',
            'timestamp': datetime.now().isoformat(),
            'retrain_result': retrain_result,
        }
