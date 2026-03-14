"""
Gaussian Process Model Module

Provides GaussianProcessPredictor with RBF / Matérn / RQ / combined kernels
and the train_gp_models() comparison utility.
"""

import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, ConstantKernel as C, Matern, RationalQuadratic, WhiteKernel,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')


class GaussianProcessPredictor:
    """
    Gaussian Process regressor for NBA point-differential prediction.

    Supports kernel_type = 'rbf' | 'matern' | 'rq' | 'combined'.
    Predictions return (mean, std); confidence intervals are available via
    get_confidence_intervals().
    """

    def __init__(
        self,
        kernel_type: str = 'matern',
        length_scale: float = 1.0,
        noise_level: float = 0.1,
        random_state: int = 42,
    ):
        self.kernel_type = kernel_type
        self.random_state = random_state

        if kernel_type == 'rbf':
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))
        elif kernel_type == 'matern':
            kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale, nu=2.5, length_scale_bounds=(1e-2, 1e2))
        elif kernel_type == 'rq':
            kernel = C(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale, alpha=1.0)
        elif kernel_type == 'combined':
            kernel = (
                C(1.0, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))
                + C(1.0, (1e-3, 1e3)) * Matern(length_scale, nu=1.5, length_scale_bounds=(1e-2, 1e2))
            )
        else:
            raise ValueError(f"Unknown kernel_type '{kernel_type}'")

        kernel = kernel + WhiteKernel(noise_level, noise_level_bounds=(1e-5, 1e1))

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            alpha=1e-10,
            random_state=random_state,
            normalize_y=True,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None

    # ------------------------------------------------------------------
    def fit(self, X, y, verbose: bool = True, auto_save: bool = True):
        if verbose:
            print(f"  Training GP ({self.kernel_type})...")

        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
            X = X.values

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        if verbose:
            lml = self.model.log_marginal_likelihood(self.model.kernel_.theta)
            print(f"    Kernel: {self.model.kernel_}")
            print(f"    Log-marginal-likelihood: {lml:.2f}")

        if auto_save:
            os.makedirs(_MODEL_DIR, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(_MODEL_DIR, f'gp_{self.kernel_type}_{ts}.pkl')
            self.save(path)
            if verbose:
                print(f"    Auto-saved -> {os.path.basename(path)}")

        return self

    def predict(self, X, return_std: bool = True):
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if hasattr(X, 'values'):
            X = X.values
        X_scaled = self.scaler.transform(X)
        if return_std:
            return self.model.predict(X_scaled, return_std=True)
        return self.model.predict(X_scaled)

    def get_confidence_intervals(self, X, confidence: float = 0.95):
        """Return (mean, lower, upper) arrays at the given confidence level."""
        mean, std = self.predict(X, return_std=True)
        z = stats.norm.ppf((1 + confidence) / 2)
        return mean, mean - z * std, mean + z * std

    def score(self, X, y) -> float:
        return r2_score(y, self.predict(X, return_std=False))

    # ------------------------------------------------------------------
    def save(self, filepath: str) -> None:
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, 'wb') as fh:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'kernel_type': self.kernel_type,
                'is_fitted': self.is_fitted,
                'feature_names': self.feature_names,
                'saved_at': datetime.now().isoformat(),
            }, fh)

    @classmethod
    def load(cls, filepath: str) -> 'GaussianProcessPredictor':
        with open(filepath, 'rb') as fh:
            data = pickle.load(fh)
        inst = cls(kernel_type=data['kernel_type'])
        inst.model = data['model']
        inst.scaler = data['scaler']
        inst.is_fitted = data['is_fitted']
        inst.feature_names = data.get('feature_names')
        return inst


# ---------------------------------------------------------------------------

def train_gp_models(
    X_train,
    y_train,
    X_test,
    y_test,
    kernel_types: list = None,
    verbose: bool = True,
) -> dict:
    """
    Train and compare GP models with different kernels.

    Returns
    -------
    dict with keys:
        models      : {kernel: GaussianProcessPredictor}
        predictions : {kernel: {'mean': array, 'std': array}}
        metrics     : list of dicts (kernel, r2, rmse, mae, coverage)
        best_model  : GaussianProcessPredictor with highest R²
        best_kernel : str
    """
    if kernel_types is None:
        kernel_types = ['rbf', 'matern', 'rq']

    results = {'models': {}, 'predictions': {}, 'metrics': []}

    for kernel in kernel_types:
        if verbose:
            print(f"\n{'='*50}\nKernel: {kernel.upper()}")

        gp = GaussianProcessPredictor(kernel_type=kernel)
        gp.fit(X_train, y_train, verbose=verbose)

        y_pred, y_std = gp.predict(X_test, return_std=True)
        r2 = r2_score(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))

        _, lower, upper = gp.get_confidence_intervals(X_test, confidence=0.95)
        coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))

        results['models'][kernel] = gp
        results['predictions'][kernel] = {'mean': y_pred, 'std': y_std}
        results['metrics'].append({
            'kernel': kernel, 'r2': r2, 'rmse': rmse, 'mae': mae, 'coverage': coverage,
        })

        if verbose:
            print(f"  R²={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}  95%-CI coverage={coverage:.2%}")

    best_idx = int(np.argmax([m['r2'] for m in results['metrics']]))
    best_kernel = results['metrics'][best_idx]['kernel']
    results['best_model'] = results['models'][best_kernel]
    results['best_kernel'] = best_kernel

    if verbose:
        print(f"\nBest kernel: {best_kernel.upper()}  R²={results['metrics'][best_idx]['r2']:.4f}")

    return results
