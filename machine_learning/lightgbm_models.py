"""
LightGBM Models Module

Contains:
- LGBMQuantilePredictor  — Q10/Q50/Q90 quantile regression for point differential
- LGBMWinPredictor       — Calibrated win-probability predictor built on top of quantile model
- prepare_features_and_target() helper
- point_diff_to_win_prob() helper
"""

import os
import pickle
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def prepare_features_and_target(matchup_df: pd.DataFrame):
    """
    Split a matchup DataFrame into (X, y_diff, y_win, feature_names).

    Feature columns are any HOME_* / AWAY_* columns that contain
    _ROLL, REST_DAYS, WIN_STREAK, IS_BACK_TO_BACK, or WIN_RATE_10.
    """
    feature_cols = [
        col for col in matchup_df.columns
        if ('HOME_' in col or 'AWAY_' in col)
        and ('_ROLL' in col or 'REST_DAYS' in col or 'WIN_STREAK' in col
             or 'IS_BACK_TO_BACK' in col or 'WIN_RATE_10' in col)
    ]
    X = matchup_df[feature_cols].copy().fillna(matchup_df[feature_cols].mean())
    y_diff = matchup_df['POINT_DIFF'].values
    y_win = matchup_df['HOME_WIN'].values
    return X, y_diff, y_win, feature_cols


def point_diff_to_win_prob(point_diff: np.ndarray, scale: float = 14.0) -> np.ndarray:
    """
    Logistic conversion from point differential to home-team win probability.

    scale=14 is empirically reasonable for NBA spreads.
    """
    return 1.0 / (1.0 + np.exp(-point_diff / scale))


# ---------------------------------------------------------------------------
# LGBMQuantilePredictor
# ---------------------------------------------------------------------------

class LGBMQuantilePredictor:
    """
    LightGBM quantile-regression ensemble: Q10 / Q50 / Q90.

    Use predict() for a dict of arrays, or predict_with_intervals() for a
    ready-to-use DataFrame.
    """

    def __init__(self, params: dict = None, regularize_streak: bool = True):
        self.regularize_streak = regularize_streak
        self.base_params = {
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'max_depth': 7,
            'min_data_in_leaf': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1,
        }
        if params:
            self.base_params.update(params)

        self.models: dict = {}
        self.feature_names = None
        self.is_fitted = False

    # ------------------------------------------------------------------
    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        quantiles: tuple = (0.1, 0.5, 0.9),
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
    ) -> dict:
        """Train one LightGBM model per quantile."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm required: pip install lightgbm")

        print(f"  Training LightGBM quantile models {quantiles} "
              f"on {X_train.shape[0]} samples / {X_train.shape[1]} features")

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = []
        if X_val is not None and y_val is not None:
            valid_sets = [lgb.Dataset(X_val, label=y_val, reference=train_data)]

        callbacks = [lgb.log_evaluation(period=0)]
        if valid_sets and early_stopping_rounds:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

        for q in quantiles:
            q_key = f'q{int(q * 100)}'
            p = {**self.base_params, 'objective': 'quantile', 'alpha': q, 'metric': 'quantile'}
            model = lgb.train(
                p,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=valid_sets or None,
                callbacks=callbacks,
            )
            self.models[q_key] = model
            print(f"    {q_key.upper()}: {model.num_trees()} trees")

        self.is_fitted = True
        return self.models

    def predict(self, X) -> dict:
        """Return {'q10': array, 'q50': array, 'q90': array}."""
        if not self.is_fitted:
            raise ValueError("Not fitted. Call train() first.")
        return {key: model.predict(X) for key, model in self.models.items()}

    def predict_with_intervals(self, X) -> pd.DataFrame:
        """Return DataFrame with point_estimate, lower, upper, uncertainty."""
        p = self.predict(X)
        return pd.DataFrame({
            'point_estimate': p['q50'],
            'lower': p['q10'],
            'upper': p['q90'],
            'uncertainty': (p['q90'] - p['q10']) / 2.0,
        })

    def feature_importance(self, feature_names: list = None, top_n: int = 20) -> pd.DataFrame:
        """Feature importance from the Q50 model (gain-based)."""
        if 'q50' not in self.models:
            raise ValueError("Q50 model not available.")
        importance = self.models['q50'].feature_importance(importance_type='gain')
        names = feature_names or self.feature_names or [f'f{i}' for i in range(len(importance))]
        df = pd.DataFrame({'feature': names[:len(importance)], 'importance': importance})
        df = df.sort_values('importance', ascending=False)
        return df.head(top_n).reset_index(drop=True)

    def recalibrate(self, X_calib, y_calib) -> dict:
        """Report current 80% interval coverage on a calibration set."""
        if not self.is_fitted:
            raise ValueError("Not fitted. Call train() first.")
        p = self.predict(X_calib)
        in_interval = (y_calib >= p['q10']) & (y_calib <= p['q90'])
        coverage = float(in_interval.mean())
        print(f"  Calibration coverage: {coverage:.1%}  (target 80%)")
        return {'coverage': coverage, 'n_samples': len(y_calib)}

    # ------------------------------------------------------------------
    def save(self, filepath: str) -> None:
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, 'wb') as fh:
            pickle.dump({
                'models': {k: v.model_to_string() for k, v in self.models.items()},
                'feature_names': self.feature_names,
                'base_params': self.base_params,
                'regularize_streak': self.regularize_streak,
            }, fh)
        print(f"  Saved -> {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'LGBMQuantilePredictor':
        import lightgbm as lgb
        with open(filepath, 'rb') as fh:
            data = pickle.load(fh)
        inst = cls(params=data['base_params'], regularize_streak=data.get('regularize_streak', True))
        inst.feature_names = data['feature_names']
        inst.models = {k: lgb.Booster(model_str=v) for k, v in data['models'].items()}
        inst.is_fitted = True
        return inst


# ---------------------------------------------------------------------------
# LGBMWinPredictor
# ---------------------------------------------------------------------------

class LGBMWinPredictor:
    """
    Calibrated home-team win probability predictor.

    Internally uses LGBMQuantilePredictor for point-differential estimation
    and IsotonicRegression for probability calibration.
    """

    def __init__(self):
        self.quantile_model: LGBMQuantilePredictor = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        self.calibrator: IsotonicRegression = None
        self.interval_scale: float = None

    # ------------------------------------------------------------------
    def train(
        self,
        X_train: pd.DataFrame,
        y_diff_train: np.ndarray,
        y_win_train: np.ndarray,
        X_val: pd.DataFrame,
        y_diff_val: np.ndarray,
        y_win_val: np.ndarray,
    ) -> dict:
        """
        Train quantile model + isotonic calibrator.

        Parameters
        ----------
        X_train / X_val : pd.DataFrame  Feature matrices.
        y_diff_*        : np.ndarray    Point-differential targets.
        y_win_*         : np.ndarray    Binary home-win (1/0) targets.

        Returns
        -------
        dict  Training metrics.
        """
        self.feature_names = X_train.columns.tolist()

        X_tr = pd.DataFrame(self.scaler.fit_transform(X_train), columns=X_train.columns)
        X_vl = pd.DataFrame(self.scaler.transform(X_val), columns=X_val.columns)

        print("\n  Training LightGBM Win Predictor")

        # 1. Quantile model
        self.quantile_model = LGBMQuantilePredictor(regularize_streak=True)
        self.quantile_model.train(
            X_tr.values, y_diff_train,
            X_vl.values, y_diff_val,
            quantiles=(0.1, 0.5, 0.9),
            num_boost_round=500,
            early_stopping_rounds=50,
        )

        # 2. Calibrator on validation
        val_preds = self.quantile_model.predict(X_vl.values)
        win_prob_raw = point_diff_to_win_prob(val_preds['q50'])

        try:
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(win_prob_raw, y_win_val)
            self.calibrator = iso
            win_prob_cal = iso.predict(win_prob_raw)
        except Exception as exc:
            print(f"  Warning: calibrator failed ({exc}); using raw probabilities")
            win_prob_cal = win_prob_raw

        # 3. Interval scale (mean half-width on validation)
        try:
            half_width = (val_preds['q90'] - val_preds['q10']) / 2.0
            self.interval_scale = float(np.nanmean(half_width))
        except Exception:
            self.interval_scale = None

        win_pred = (win_prob_cal > 0.5).astype(int)
        accuracy = accuracy_score(y_win_val, win_pred)
        brier = brier_score_loss(y_win_val, win_prob_cal)

        print(f"  Accuracy: {accuracy:.1%}  Brier: {brier:.4f}")
        self.is_fitted = True

        return {'accuracy': accuracy, 'brier_score': brier}

    # ------------------------------------------------------------------
    def predict_win_probability(self, X) -> dict:
        """
        Predict home-team win probability.

        Returns
        -------
        dict
            win_prob, point_diff, lower (Q10), upper (Q90),
            uncertainty (half-width), confidence_score, confidence_label
        """
        if not self.is_fitted:
            raise ValueError("Not fitted. Call train() first.")

        if isinstance(X, pd.DataFrame):
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        preds = self.quantile_model.predict(X_scaled)

        win_prob_raw = point_diff_to_win_prob(preds['q50'])
        if self.calibrator is not None:
            try:
                win_prob = self.calibrator.predict(win_prob_raw)
            except Exception:
                win_prob = win_prob_raw
        else:
            win_prob = win_prob_raw

        half_width = (preds['q90'] - preds['q10']) / 2.0
        if self.interval_scale and self.interval_scale > 0:
            norm_uncert = 1.0 - np.minimum(1.0, half_width / (2.0 * self.interval_scale))
        else:
            norm_uncert = 1.0 - np.tanh(np.nanmean(half_width))

        prob_margin = np.abs(win_prob - 0.5) * 2.0
        confidence_score = 0.7 * prob_margin + 0.3 * norm_uncert

        try:
            confidence_label = np.full_like(confidence_score, 'LOW', dtype=object)
            confidence_label[confidence_score >= 0.65] = 'HIGH'
            mask_med = (confidence_score >= 0.40) & (confidence_score < 0.65)
            confidence_label[mask_med] = 'MEDIUM'
        except Exception:
            if confidence_score >= 0.65:
                confidence_label = 'HIGH'
            elif confidence_score >= 0.40:
                confidence_label = 'MEDIUM'
            else:
                confidence_label = 'LOW'

        return {
            'win_prob': win_prob,
            'point_diff': preds['q50'],
            'lower': preds['q10'],
            'upper': preds['q90'],
            'uncertainty': half_width,
            'confidence_score': confidence_score,
            'confidence_label': confidence_label,
        }

    # ------------------------------------------------------------------
    def save(self, filepath: str) -> None:
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        joblib.dump({
            'quantile_model': self.quantile_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'calibrator': self.calibrator,
            'interval_scale': self.interval_scale,
        }, filepath)
        print(f"  Saved -> {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'LGBMWinPredictor':
        data = joblib.load(filepath)
        inst = cls()
        inst.quantile_model = data['quantile_model']
        inst.scaler = data['scaler']
        inst.feature_names = data['feature_names']
        inst.calibrator = data.get('calibrator')
        inst.interval_scale = data.get('interval_scale')
        inst.is_fitted = True
        return inst
