"""
Training Orchestration

Full retraining pipeline for production models:
- Gaussian Process spread model
- LightGBM quantile spread model
- LightGBM calibrated win model
- Elo model
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from data.database.database_handler import SportsAnalyticsDB
from data.extended_data_loader import get_extended_training_dataset
from ensemble.ensemble_weights import default_weights
from machine_learning.elo_model import EloModel
from machine_learning.gp_model import GaussianProcessPredictor
from machine_learning.lightgbm_models import (
    LGBMQuantilePredictor,
    LGBMWinPredictor,
    prepare_features_and_target,
)


class ModelTrainer:
    """Central entrypoint for full model retraining."""

    def __init__(self, db_path: str = 'sports_analytics.db'):
        self.db_path = db_path
        db_root = Path(db_path).expanduser().resolve().parent
        self.model_dir = str(db_root / 'machine_learning' / 'models')
        os.makedirs(self.model_dir, exist_ok=True)

    def full_retrain(self, verbose: bool = True) -> Dict:
        """
        Train all production models and persist artifacts.

        Returns
        -------
        dict with `model_version`, `model_paths`, and summary metrics.
        """
        if verbose:
            print('\n' + '=' * 70)
            print('FULL RETRAIN STARTED')
            print('=' * 70)

        version = datetime.now().strftime('v%Y%m%d_%H%M%S')

        # ------------------------------------------------------------------
        # 1) Load dataset
        # ------------------------------------------------------------------
        data = get_extended_training_dataset(db_path=self.db_path, verbose=verbose)
        games_df: pd.DataFrame = data['games_df']
        matchup_df: pd.DataFrame = data['matchup_df']

        X_df, y_diff, y_win, feature_cols = prepare_features_and_target(matchup_df)
        y_diff = np.asarray(y_diff, dtype=float)
        y_win = np.asarray(y_win, dtype=int)

        if len(X_df) < 100:
            raise ValueError(f'Not enough samples for retraining: {len(X_df)}')

        # Chronological split if GAME_DATE available
        if 'GAME_DATE' in matchup_df.columns:
            order = np.argsort(pd.to_datetime(matchup_df['GAME_DATE']).to_numpy())
            X_df = X_df.iloc[order].reset_index(drop=True)
            y_diff = y_diff[order]
            y_win = y_win[order]

        split_idx = int(len(X_df) * 0.8)
        X_train = X_df.iloc[:split_idx]
        X_val = X_df.iloc[split_idx:]
        y_diff_train = y_diff[:split_idx]
        y_diff_val = y_diff[split_idx:]
        y_win_train = y_win[:split_idx]
        y_win_val = y_win[split_idx:]

        model_paths: Dict[str, str] = {}

        # ------------------------------------------------------------------
        # 2) GP model
        # ------------------------------------------------------------------
        gp = GaussianProcessPredictor(kernel_type='combined')
        gp.fit(X_train.values, y_diff_train, verbose=verbose, auto_save=False)
        gp_path = os.path.join(self.model_dir, f'gp_{version}.pkl')
        gp.save(gp_path)
        model_paths['gp'] = gp_path

        # ------------------------------------------------------------------
        # 3) LightGBM quantile model
        # ------------------------------------------------------------------
        lgbm_q = LGBMQuantilePredictor()
        lgbm_q.train(
            X_train.values,
            y_diff_train,
            X_val.values,
            y_diff_val,
            quantiles=(0.1, 0.5, 0.9),
            num_boost_round=500,
            early_stopping_rounds=50,
        )
        lgbm_q_path = os.path.join(self.model_dir, f'lgbm_quantile_{version}.pkl')
        lgbm_q.save(lgbm_q_path)
        model_paths['lgbm_quantile'] = lgbm_q_path

        # ------------------------------------------------------------------
        # 4) LightGBM calibrated win model
        # ------------------------------------------------------------------
        lgbm_win = LGBMWinPredictor()
        lgbm_metrics = lgbm_win.train(
            X_train,
            y_diff_train,
            y_win_train,
            X_val,
            y_diff_val,
            y_win_val,
        )
        lgbm_win_path = os.path.join(self.model_dir, f'lgbm_win_{version}.pkl')
        lgbm_win.save(lgbm_win_path)
        model_paths['lgbm_win'] = lgbm_win_path

        # ------------------------------------------------------------------
        # 5) Elo model
        # ------------------------------------------------------------------
        elo = EloModel()
        elo.fit(games_df)
        elo_path = os.path.join(self.model_dir, f'elo_{version}.pkl')
        elo.save(elo_path)
        model_paths['elo'] = elo_path

        # ------------------------------------------------------------------
        # 6) Persist retraining metadata
        # ------------------------------------------------------------------
        with SportsAnalyticsDB(self.db_path) as db:
            db.update_retraining_state(
                incremental_count=0,
                model_version=version,
                full_retrain=True,
                ensemble_weights=default_weights(),
            )

        if verbose:
            print('=' * 70)
            print('FULL RETRAIN COMPLETE')
            print(f'Model version: {version}')
            print('=' * 70 + '\n')

        return {
            'model_version': version,
            'model_paths': model_paths,
            'metrics': {
                'n_samples': len(X_df),
                'n_features': len(feature_cols),
                'lgbm_win': lgbm_metrics,
            },
        }
