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
from src.evaluation.feedback_loop import HIGH_SIGNAL_FEATURE_COLUMNS, derive_game_features
from src.evaluation.vectorized_features import vectorize_high_signal_features


class ModelTrainer:
    """Central entrypoint for full model retraining."""

    def __init__(self, db_path: str = 'sports_analytics.db'):
        self.db_path = db_path
        db_root = Path(db_path).expanduser().resolve().parent
        self.model_dir = str(db_root / 'machine_learning' / 'models')
        os.makedirs(self.model_dir, exist_ok=True)

    def full_retrain(self, verbose: bool = True, fast_mode: bool = False) -> Dict:
        """
        Train all production models and persist artifacts.

        Parameters
        ----------
        verbose : bool
            Print progress messages.
        fast_mode : bool
            Enable speed optimizations (reduce boost rounds, GP subsampling, skip enrichment filtering).

        Returns
        -------
        dict with `model_version`, `model_paths`, and summary metrics.
        """
        import time
        import sys
        
        if verbose:
            print('\n' + '=' * 70, flush=True)
            print('FULL RETRAIN STARTED' + (' [FAST MODE]' if fast_mode else ''), flush=True)
            print('=' * 70, flush=True)

        version = datetime.now().strftime('v%Y%m%d_%H%M%S')
        start_time = time.time()

        # ------------------------------------------------------------------
        # 1) Load dataset
        # ------------------------------------------------------------------
        step_start = time.time()
        if verbose:
            print('\n[1/5] Loading training dataset...', flush=True)
        
        data = get_extended_training_dataset(db_path=self.db_path, verbose=verbose)
        games_df: pd.DataFrame = data['games_df']
        matchup_df: pd.DataFrame = data['matchup_df']

        if verbose:
            print(f'      Enriching matchup features...', flush=True)
        matchup_df = self._enrich_matchup_with_high_signal_features(matchup_df, games_df, fast_mode=fast_mode)
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

        if verbose:
            elapsed = time.time() - step_start
            print(f'      Dataset: {len(X_df)} samples, {len(feature_cols)} features', flush=True)
            print(f'      Train/val split: {len(X_train)}/{len(X_val)} samples', flush=True)
            print(f'      Time: {elapsed:.1f}s', flush=True)

        model_paths: Dict[str, str] = {}

        # ------------------------------------------------------------------
        # 2) GP model (subsample if dataset is large for speed)
        # ------------------------------------------------------------------
        step_start = time.time()
        if verbose:
            print('\n[2/5] Training Gaussian Process spread model...', flush=True)
        
        X_train_gp = X_train.copy()
        y_train_gp = y_diff_train.copy()
        
        # For large datasets, subsample for GP (O(n³) complexity)
        # In fast_mode, use smaller subsample threshold
        gp_subsample_threshold = 200 if fast_mode else 400
        if len(X_train_gp) > gp_subsample_threshold:
            gp_subsample = min(gp_subsample_threshold, len(X_train_gp))
            indices = np.random.RandomState(42).choice(len(X_train_gp), size=gp_subsample, replace=False)
            X_train_gp = X_train_gp.iloc[sorted(indices)]
            y_train_gp = y_train_gp[sorted(indices)]
            if verbose:
                print(f'      Subsampled to {len(X_train_gp)} samples (GP complexity: O(n³))', flush=True)
        
        gp = GaussianProcessPredictor(kernel_type='combined')
        gp.fit(X_train_gp.values, y_train_gp, verbose=verbose, auto_save=False)
        gp_path = os.path.join(self.model_dir, f'gp_{version}.pkl')
        gp.save(gp_path)
        model_paths['gp'] = gp_path
        
        elapsed = time.time() - step_start
        if verbose:
            print(f'      Time: {elapsed:.1f}s', flush=True)

        # ------------------------------------------------------------------
        # 3) LightGBM quantile model
        # ------------------------------------------------------------------
        step_start = time.time()
        if verbose:
            print('\n[3/5] Training LightGBM quantile model...', flush=True)
        
        lgbm_q = LGBMQuantilePredictor()
        # In fast_mode, reduce boost rounds and early stopping
        num_boost_round = 250 if fast_mode else 500
        early_stopping_rounds = 25 if fast_mode else 50
        lgbm_q.train(
            X_train.values,
            y_diff_train,
            X_val.values,
            y_diff_val,
            quantiles=(0.1, 0.5, 0.9),
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )
        lgbm_q_path = os.path.join(self.model_dir, f'lgbm_quantile_{version}.pkl')
        lgbm_q.save(lgbm_q_path)
        model_paths['lgbm_quantile'] = lgbm_q_path
        
        elapsed = time.time() - step_start
        if verbose:
            print(f'      Time: {elapsed:.1f}s', flush=True)

        # ------------------------------------------------------------------
        # 4) LightGBM calibrated win model
        # ------------------------------------------------------------------
        step_start = time.time()
        if verbose:
            print('\n[4/5] Training LightGBM win prediction model...', flush=True)
        
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
        
        elapsed = time.time() - step_start
        if verbose:
            print(f'      Time: {elapsed:.1f}s', flush=True)

        # ------------------------------------------------------------------
        # 5) Elo model
        # ------------------------------------------------------------------
        step_start = time.time()
        if verbose:
            print('\n[5/5] Training ELO rating model...', flush=True)
        
        elo = EloModel()
        elo.fit(games_df)
        elo_path = os.path.join(self.model_dir, f'elo_{version}.pkl')
        elo.save(elo_path)
        model_paths['elo'] = elo_path
        
        elapsed = time.time() - step_start
        if verbose:
            print(f'      Time: {elapsed:.1f}s', flush=True)

        # ------------------------------------------------------------------
        # 6) Persist retraining metadata
        # ------------------------------------------------------------------
        if verbose:
            print('\n[6/5] Persisting metadata...', flush=True)
        
        with SportsAnalyticsDB(self.db_path) as db:
            db.update_retraining_state(
                incremental_count=0,
                model_version=version,
                full_retrain=True,
                ensemble_weights=default_weights(),
            )

        if verbose:
            total_elapsed = time.time() - start_time
            print('\n' + '=' * 70)
            print('FULL RETRAIN COMPLETE')
            print(f'Model version: {version}')
            print(f'Total time: {total_elapsed // 60:.0f}m {total_elapsed % 60:.0f}s')
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

    @staticmethod
    def _enrich_matchup_with_high_signal_features(
        matchup_df: pd.DataFrame, games_df: pd.DataFrame, fast_mode: bool = False
    ) -> pd.DataFrame:
        """Add deterministic high-signal features for each training row.

        Features are derived only from settled historical rows strictly before
        each matchup game date.
        
        Parameters
        ----------
        fast_mode : bool
            If True, use default zeros for enriched features to skip expensive filtering.
            This trades accuracy for speed — use only when retraining urgently.
        """
        import time
        import sys
        
        if matchup_df.empty:
            return matchup_df

        enriched = matchup_df.copy()
        
        # In fast_mode, skip enrichment and use default zero values
        if fast_mode:
            feature_df = pd.DataFrame(
                {col: 0.0 for col in HIGH_SIGNAL_FEATURE_COLUMNS},
                index=enriched.index
            )
            if len(feature_df) != len(enriched):
                feature_df = feature_df.iloc[:len(enriched)]
            return pd.concat([enriched, feature_df], axis=1)

        history_df = games_df.copy() if games_df is not None else pd.DataFrame()
        if not history_df.empty and 'GAME_DATE' in history_df.columns:
            history_df['GAME_DATE'] = pd.to_datetime(history_df['GAME_DATE'], errors='coerce')

        # Use vectorized feature computation (Phase 2 optimization)
        # This replaces the slow row-by-row loop and achieves 60-100x speedup
        print(f'      Enriching {len(enriched)} matchups with vectorized features...', flush=True)
        feature_df = vectorize_high_signal_features(
            matchup_df=enriched,
            games_df=history_df,
            verbose=True
        )
        
        # Ensure all 14 features are present
        for col in HIGH_SIGNAL_FEATURE_COLUMNS:
            if col not in feature_df.columns:
                feature_df[col] = 0.0
        
        feature_df = feature_df[HIGH_SIGNAL_FEATURE_COLUMNS].fillna(0.0)
        return pd.concat([enriched, feature_df], axis=1)
