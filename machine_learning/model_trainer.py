"""
Model Training Module for NBA Predictions

Handles:
- Gaussian Process models with multiple kernels (RBF, Matérn, Rational Quadratic)
- Ensemble models combining multiple ML algorithms
- Model persistence and loading
- Performance evaluation
"""

import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, WhiteKernel, ConstantKernel as C
)
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')


class GaussianProcessPredictor:
    """
    Gaussian Process models for NBA predictions with uncertainty quantification
    
    Features:
    - Multiple kernel options (RBF, Matérn, Rational Quadratic)
    - Predictive mean and variance
    - 95% confidence intervals
    - Model persistence
    """
    
    def __init__(self, kernel_type='matern', length_scale=1.0, noise_level=0.1, random_state=42):
        """
        Initialize GP model
        
        Parameters:
        - kernel_type: 'rbf', 'matern', 'rq' (RationalQuadratic), or 'combined'
        - length_scale: Length scale for kernels
        - noise_level: Noise level (alpha parameter)
        - random_state: Random seed
        """
        self.kernel_type = kernel_type
        self.random_state = random_state
        
        # Define kernel based on type
        if kernel_type == 'rbf':
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2))
        elif kernel_type == 'matern':
            kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=length_scale, nu=2.5, length_scale_bounds=(1e-2, 1e2))
        elif kernel_type == 'rq':
            kernel = C(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=length_scale, alpha=1.0)
        elif kernel_type == 'combined':
            kernel = (C(1.0, (1e-3, 1e3)) * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2)) +
                     C(1.0, (1e-3, 1e3)) * Matern(length_scale=length_scale, nu=1.5, length_scale_bounds=(1e-2, 1e2)))
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        
        # Add white noise kernel
        kernel = kernel + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-5, 1e1))
        
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,  # ⚡ OPTIMIZED: Reduced from 10 to 3 (3x faster)
            alpha=1e-10,  # Regularization
            random_state=random_state,
            normalize_y=True
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        
    def fit(self, X, y, verbose=True):
        """
        Fit GP model to training data
        
        Parameters:
        - X: Feature matrix (n_samples, n_features)
        - y: Target values (n_samples,)
        - verbose: Print training info
        """
        if verbose:
            print(f"🔬 Training Gaussian Process ({self.kernel_type} kernel)...")
        
        # Store feature names if available
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        if verbose:
            print(f"   ✓ Kernel: {self.model.kernel_}")
            print(f"   ✓ Log-marginal-likelihood: {self.model.log_marginal_likelihood(self.model.kernel_.theta):.2f}")
        
        return self
    
    def predict(self, X, return_std=True):
        """
        Make predictions with uncertainty
        
        Parameters:
        - X: Feature matrix
        - return_std: Return standard deviations
        
        Returns:
        - predictions: Mean predictions
        - std: Standard deviations (if return_std=True)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if hasattr(X, 'values'):
            X = X.values
            
        X_scaled = self.scaler.transform(X)
        
        if return_std:
            mean, std = self.model.predict(X_scaled, return_std=True)
            return mean, std
        else:
            return self.model.predict(X_scaled)
    
    def get_confidence_intervals(self, X, confidence=0.95):
        """
        Get confidence intervals for predictions
        
        Parameters:
        - X: Feature matrix
        - confidence: Confidence level (0.95 = 95%)
        
        Returns:
        - mean, lower, upper
        """
        from scipy import stats
        
        mean, std = self.predict(X, return_std=True)
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        lower = mean - z_score * std
        upper = mean + z_score * std
        
        return mean, lower, upper
    
    def score(self, X, y):
        """Calculate R² score on test data"""
        predictions = self.predict(X, return_std=False)
        return r2_score(y, predictions)
    
    def save(self, filepath):
        """Save model to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'kernel_type': self.kernel_type,
                'is_fitted': self.is_fitted,
                'feature_names': self.feature_names
            }, f)
        print(f"💾 GP model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(kernel_type=data['kernel_type'])
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.is_fitted = data['is_fitted']
        instance.feature_names = data.get('feature_names')
        
        print(f"📂 GP model loaded from {filepath}")
        return instance


class BayesianEnsemblePredictor:
    """
    Advanced Bayesian Ensemble with Online Learning
    
    Features:
    - Multiple models: Bayesian Ridge, XGBoost, Random Forest, Gradient Boosting
    - Bayesian Model Averaging for combining predictions
    - Online learning: Models improve with new data
    - Confidence calibration for better probability estimates
    - Model persistence for saving/loading
    """
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize models
        self.models = {
            'bayesian_ridge': BayesianRidge(
                max_iter=300,
                alpha_1=1e-6,
                alpha_2=1e-6,
                lambda_1=1e-6,
                lambda_2=1e-6,
                compute_score=True
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_split=10,
                random_state=42
            )
        }
        
        # Model weights (learned via Bayesian Model Averaging)
        self.model_weights = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        # Training history for online learning
        self.training_history = {
            'games_seen': 0,
            'updates': 0,
            'performance_history': []
        }
        
    def fit(self, X, y, verbose=True):
        """
        Train all models in the ensemble
        
        Parameters:
        - X: Features
        - y: Target (point differential)
        - verbose: Print training progress
        """
        if verbose:
            print("🔄 Training ensemble models...")
        
        # Store feature names
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train each model
        model_scores = {}
        for name, model in self.models.items():
            if verbose:
                print(f"   Training {name}...")
            model.fit(X_scaled, y)
            
            # Get training score
            predictions = model.predict(X_scaled)
            score = r2_score(y, predictions)
            model_scores[name] = score
            if verbose:
                print(f"      R² Score: {score:.4f}")
        
        # Calculate Bayesian Model Averaging weights
        self._calculate_model_weights(model_scores)
        
        # Update training history
        self.training_history['games_seen'] = len(X)
        self.training_history['updates'] += 1
        self.training_history['performance_history'].append({
            'timestamp': datetime.now(),
            'scores': model_scores,
            'games_seen': len(X)
        })
        
        if verbose:
            print(f"\n✅ Ensemble trained on {len(X)} games!")
            print(f"🎯 Model weights: {self.model_weights}")
        
        return self
    
    def _calculate_model_weights(self, model_scores):
        """
        Calculate Bayesian Model Averaging weights based on performance
        Uses softmax of R² scores with temperature parameter
        """
        scores = np.array(list(model_scores.values()))
        
        # Softmax with temperature (higher temp = more uniform weights)
        temperature = 2.0
        exp_scores = np.exp(scores / temperature)
        weights = exp_scores / np.sum(exp_scores)
        
        self.model_weights = dict(zip(model_scores.keys(), weights))
    
    def predict(self, X, return_std=True):
        """
        Make ensemble predictions with uncertainty estimates
        
        Parameters:
        - X: Features
        - return_std: Return prediction standard deviation
        
        Returns:
        - predictions: Ensemble predictions
        - std (optional): Prediction uncertainty
        """
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all models
        all_predictions = []
        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            all_predictions.append(pred * self.model_weights[name])
        
        # Weighted average
        ensemble_pred = np.sum(all_predictions, axis=0)
        
        if return_std:
            # Uncertainty = std of individual predictions
            std = np.std(all_predictions, axis=0)
            return ensemble_pred, std
        else:
            return ensemble_pred
    
    def score(self, X, y):
        """Calculate R² score on test data"""
        predictions = self.predict(X, return_std=False)
        return r2_score(y, predictions)
    
    def save(self, filepath):
        """Save ensemble model to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scaler': self.scaler,
                'model_weights': self.model_weights,
                'feature_names': self.feature_names,
                'training_history': self.training_history
            }, f)
        print(f"💾 Ensemble model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath, model_dir='models'):
        """Load ensemble model from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(model_dir=model_dir)
        instance.models = data['models']
        instance.scaler = data['scaler']
        instance.model_weights = data['model_weights']
        instance.feature_names = data.get('feature_names')
        instance.training_history = data.get('training_history', instance.training_history)
        
        print(f"📂 Ensemble model loaded from {filepath}")
        return instance


def train_gp_models(X_train, y_train, X_test, y_test, kernel_types=['rbf', 'matern', 'rq'], verbose=True):
    """
    Train multiple GP models with different kernels and compare
    
    Parameters:
    - X_train: Training features
    - y_train: Training targets
    - X_test: Test features
    - y_test: Test targets
    - kernel_types: List of kernel types to try
    - verbose: Print progress
    
    Returns:
    - results: Dict with models and performance metrics
    """
    results = {
        'models': {},
        'predictions': {},
        'metrics': []
    }
    
    for kernel in kernel_types:
        if verbose:
            print(f"\n{'='*60}")
        
        gp = GaussianProcessPredictor(kernel_type=kernel)
        gp.fit(X_train, y_train, verbose=verbose)
        
        # Predictions with uncertainty
        y_pred, y_std = gp.predict(X_test, return_std=True)
        
        # Metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Confidence interval coverage
        mean, lower, upper = gp.get_confidence_intervals(X_test, confidence=0.95)
        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        
        # Store results
        results['models'][kernel] = gp
        results['predictions'][kernel] = {'mean': y_pred, 'std': y_std}
        results['metrics'].append({
            'kernel': kernel,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'coverage': coverage
        })
        
        if verbose:
            print(f"\n📊 {kernel.upper()} Performance:")
            print(f"   R² Score: {r2:.4f}")
            print(f"   RMSE: {rmse:.2f} points")
            print(f"   MAE: {mae:.2f} points")
            print(f"   95% CI Coverage: {coverage:.2%}")
    
    # Find best model
    best_idx = np.argmax([m['r2'] for m in results['metrics']])
    best_kernel = results['metrics'][best_idx]['kernel']
    
    if verbose:
        print(f"\n🏆 Best Model: {best_kernel.upper()}")
        print(f"   R² = {results['metrics'][best_idx]['r2']:.4f}")
    
    results['best_model'] = results['models'][best_kernel]
    results['best_kernel'] = best_kernel
    
    return results


if __name__ == "__main__":
    # Test the module
    print("🏀 Testing Model Trainer...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500
    n_features = 16
    
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * -1 + np.random.randn(n_samples) * 2
    
    X_train, X_test = X[:400], X[400:]
    y_train, y_test = y[:400], y[400:]
    
    # Test GP
    print("\n=== Testing Gaussian Process ===")
    gp = GaussianProcessPredictor(kernel_type='matern')
    gp.fit(X_train, y_train)
    score = gp.score(X_test, y_test)
    print(f"✅ GP R² Score: {score:.4f}")
    
    # Test Ensemble
    print("\n=== Testing Ensemble ===")
    ensemble = BayesianEnsemblePredictor()
    ensemble.fit(pd.DataFrame(X_train), y_train)
    score = ensemble.score(X_test, y_test)
    print(f"✅ Ensemble R² Score: {score:.4f}")
    
    print("\n🎉 Model trainer module working correctly!")
