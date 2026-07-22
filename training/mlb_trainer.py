"""
MLB Training Orchestration

Trimmed retraining pipeline for MLB production models (team-level v1):
- LightGBM quantile run-differential model
- LightGBM calibrated win model

No GP or Elo component for MLB v1 — team-level rolling run differential only,
no pitcher features yet (see data/mlb_feature_engineering.py).
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data.mlb_loader import fetch_mlb_games
from data.mlb_feature_engineering import calculate_mlb_rolling_stats, prepare_mlb_training_data
from machine_learning.lightgbm_models import (
    LGBMQuantilePredictor,
    LGBMWinPredictor,
    prepare_features_and_target,
)


class MLBModelTrainer:
    """Central entrypoint for MLB model retraining (team-level v1)."""

    def __init__(self, model_dir: str = 'machine_learning/models'):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def full_retrain(
        self,
        seasons: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Train MLB LightGBM quantile + win models and persist artifacts.

        Returns
        -------
        dict with `model_version`, `model_paths`, `pointer_path`, and summary metrics.
        """
        if verbose:
            print('\n' + '=' * 70, flush=True)
            print('MLB FULL RETRAIN STARTED', flush=True)
            print('=' * 70, flush=True)

        version = datetime.now().strftime('v%Y%m%d_%H%M%S')

        # ------------------------------------------------------------------
        # 1) Load dataset
        # ------------------------------------------------------------------
        if verbose:
            print('\n[1/3] Loading MLB training dataset...', flush=True)

        games_df = fetch_mlb_games(seasons=seasons, verbose=verbose)
        games_with_stats = calculate_mlb_rolling_stats(games_df, window=10)
        matchup_df, _, _ = prepare_mlb_training_data(games_with_stats, verbose=verbose)

        X_df, y_diff, y_win, feature_cols = prepare_features_and_target(matchup_df)
        y_diff = np.asarray(y_diff, dtype=float)
        y_win = np.asarray(y_win, dtype=int)

        if len(X_df) < 100:
            raise ValueError(f'Not enough samples for MLB retraining: {len(X_df)}')

        # Chronological split
        order = np.argsort(pd.to_datetime(matchup_df['GAME_DATE']).to_numpy())
        X_df = X_df.iloc[order].reset_index(drop=True)
        y_diff = y_diff[order]
        y_win = y_win[order]

        split_idx = int(len(X_df) * 0.8)
        X_train, X_val = X_df.iloc[:split_idx], X_df.iloc[split_idx:]
        y_diff_train, y_diff_val = y_diff[:split_idx], y_diff[split_idx:]
        y_win_train, y_win_val = y_win[:split_idx], y_win[split_idx:]

        if verbose:
            print(f'      Dataset: {len(X_df)} samples, {len(feature_cols)} features', flush=True)
            print(f'      Train/val split: {len(X_train)}/{len(X_val)} samples', flush=True)

        model_paths: Dict[str, str] = {}

        # ------------------------------------------------------------------
        # 2) LightGBM quantile model (run differential)
        # ------------------------------------------------------------------
        if verbose:
            print('\n[2/3] Training LightGBM quantile model (run differential)...', flush=True)

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
        lgbm_q_path = os.path.join(self.model_dir, f'mlb_lgbm_quantile_{version}.pkl')
        lgbm_q.save(lgbm_q_path)
        model_paths['lgbm_quantile'] = lgbm_q_path

        # ------------------------------------------------------------------
        # 3) LightGBM calibrated win model
        # ------------------------------------------------------------------
        if verbose:
            print('\n[3/3] Training LightGBM win prediction model...', flush=True)

        lgbm_win = LGBMWinPredictor()
        lgbm_metrics = lgbm_win.train(
            X_train,
            y_diff_train,
            y_win_train,
            X_val,
            y_diff_val,
            y_win_val,
        )
        lgbm_win_path = os.path.join(self.model_dir, f'mlb_lgbm_win_{version}.pkl')
        lgbm_win.save(lgbm_win_path)
        model_paths['lgbm_win'] = lgbm_win_path

        # ------------------------------------------------------------------
        # 4) Persist a lightweight per-sport pointer (kept separate from NBA's
        #    retraining_metadata/ensemble-weight state so per-sport model
        #    versions don't get conflated).
        #
        #    Only basenames are stored (not full paths) so the pointer is
        #    portable between OSes: os.path.join() embeds the host's native
        #    separator, and a Windows-trained pointer with a literal "\\" in
        #    it will fail to resolve on the Linux runners this trains for
        #    (GitHub Actions / Render). The loader re-joins the basename with
        #    its own os.path.join(models_dir, ...) at read time.
        # ------------------------------------------------------------------
        pointer = {
            'version': version,
            'lgbm_quantile_path': os.path.basename(lgbm_q_path),
            'lgbm_win_path': os.path.basename(lgbm_win_path),
            'feature_cols': feature_cols,
            'trained_at': datetime.now().isoformat(),
            'metrics': lgbm_metrics,
        }
        pointer_path = os.path.join(self.model_dir, 'mlb_latest.json')
        with open(pointer_path, 'w') as fh:
            json.dump(pointer, fh, indent=2)

        if verbose:
            print('\n' + '=' * 70)
            print('MLB FULL RETRAIN COMPLETE')
            print(f'Model version: {version}')
            print(f"Accuracy: {lgbm_metrics.get('accuracy', 0.0):.1%}  "
                  f"Brier: {lgbm_metrics.get('brier_score', 0.0):.4f}")
            print('=' * 70 + '\n')

        return {
            'model_version': version,
            'model_paths': model_paths,
            'pointer_path': pointer_path,
            'metrics': {
                'n_samples': len(X_df),
                'n_features': len(feature_cols),
                'lgbm_win': lgbm_metrics,
            },
        }


if __name__ == '__main__':
    trainer = MLBModelTrainer()
    trainer.full_retrain()
