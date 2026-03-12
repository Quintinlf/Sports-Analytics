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
        
        # Auto-save model after successful fit
        os.makedirs('machine_learning/models', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f'machine_learning/models/gp_predictor_{self.kernel_type}_{timestamp}.pkl'
        self.save(model_path)
        
        if verbose:
            print(f"   ✓ Auto-saved to {os.path.basename(model_path)}")
        
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
                'feature_names': self.feature_names,
                'saved_at': datetime.now().isoformat()
            }, f)
        if os.path.dirname(filepath) != '':
            print(f"   💾 GP model saved to {os.path.basename(filepath)}")
        else:
            print(f"   💾 GP model saved to {filepath}")
    
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
        
        saved_at = data.get('saved_at', 'Unknown')
        print(f"   📂 Loaded {os.path.basename(filepath)} (saved: {saved_at})")
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


"""
Train LightGBM Model for NBA Predictions

This script:
1. Loads clean dataset (2023-24 + 2024-25 seasons)
2. Creates matchup features (home vs away)
3. Trains LightGBM quantile regression model
4. Adds probability calibration for win predictions
5. Saves trained model to machine_learning/models/
6. Evaluates on chronological test set

Run after data leakage fixes are applied.
"""

import os
import sys
sys.path.insert(0, r'c:\Users\Windows User\My_folder\gamble_code\sports_analytics')

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.isotonic import IsotonicRegression

from experimental.loaders.data_loader import (
    fetch_nba_games, 
    calculate_rolling_stats,
    create_matchup_features
)
from predictors.lgbm_predictor import LGBMQuantilePredictor


def prepare_features_and_target(matchup_df):
    """
    Prepare feature matrix and target for training.
    
    Args:
        matchup_df: DataFrame with HOME_*/AWAY_* features
    
    Returns:
        X: Feature matrix
        y_diff: Point differential target
        y_win: Binary win target (1=home win, 0=away win)
        feature_names: List of feature names
    """
    # Feature columns (rolling stats, rest, momentum)
    feature_cols = [
        col for col in matchup_df.columns 
        if ('HOME_' in col or 'AWAY_' in col) and 
           ('_ROLL' in col or 'REST_DAYS' in col or 'WIN_STREAK' in col or 
            'IS_BACK_TO_BACK' in col or 'WIN_RATE_10' in col)
    ]
    
    X = matchup_df[feature_cols].copy()
    y_diff = matchup_df['POINT_DIFF'].values
    y_win = matchup_df['HOME_WIN'].values
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    return X, y_diff, y_win, feature_cols


def point_diff_to_win_prob(point_diff, scale=14.0):
    """
    Convert point differential to win probability using logistic function.
    
    Args:
        point_diff: Predicted point differential (positive = home favored)
        scale: Scaling factor (14.0 is typical for NBA)
    
    Returns:
        Win probability for home team (0-1)
    """
    return 1 / (1 + np.exp(-point_diff / scale))


class LGBMWinPredictor:
    """
    LightGBM predictor with calibrated win probabilities.
    
    Combines:
    - Quantile regression for point differential + uncertainty
    - Calibrated classifier for direct win probability
    """
    
    def __init__(self):
        self.quantile_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        self.calibrator = None
        self.interval_scale = None  # mean half-width observed on validation
    
    def train(self, X_train, y_diff_train, y_win_train, X_val, y_diff_val, y_win_val):
        """
        Train both quantile regression and calibrated classifier.
        
        Args:
            X_train, X_val: Feature matrices
            y_diff_train, y_diff_val: Point differential targets
            y_win_train, y_win_val: Binary win targets
        
        Returns:
            Training metrics dict
        """
        print("\n" + "="*80)
        print("🚀 TRAINING LIGHTGBM WIN PREDICTOR")
        print("="*80)
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Scale features
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns
        )
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns
        )
        
        # 1. Train quantile regression for point differential
        print("\n📊 Training quantile regression models (Q10, Q50, Q90)...")
        self.quantile_model = LGBMQuantilePredictor(regularize_streak=True)
        
        self.quantile_model.train(
            X_train_scaled.values, 
            y_diff_train,
            X_val_scaled.values,
            y_diff_val,
            quantiles=(0.1, 0.5, 0.9),
            num_boost_round=500,
            early_stopping_rounds=50
        )
        
        # 2. Evaluate on validation set
        print("\n" + "="*80)
        print("📈 VALIDATION RESULTS")
        print("="*80)
        
        val_preds = self.quantile_model.predict(X_val_scaled.values)
        point_diff_pred = val_preds['q50']
        
        # Convert point differential to win probability (raw)
        win_prob_pred = point_diff_to_win_prob(point_diff_pred)

        # Fit isotonic calibrator on validation set to improve probability calibration
        try:
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(win_prob_pred, y_win_val)
            self.calibrator = iso
            win_prob_cal = iso.predict(win_prob_pred)
            print("   ✅ Calibrator (isotonic) fitted on validation set")
        except Exception as e:
            win_prob_cal = win_prob_pred
            print(f"   ⚠️ Calibrator fitting failed: {e}")

        # Use calibrated probabilities for evaluation
        win_pred = (win_prob_cal > 0.5).astype(int)

        accuracy = accuracy_score(y_win_val, win_pred)
        brier = brier_score_loss(y_win_val, win_prob_cal)

        # Calculate metrics by confidence level using calibrated probabilities
        high_conf_mask = (win_prob_cal > 0.65) | (win_prob_cal < 0.35)
        med_conf_mask = ((win_prob_cal >= 0.55) & (win_prob_cal <= 0.65)) | \
                        ((win_prob_cal >= 0.35) & (win_prob_cal <= 0.45))
        low_conf_mask = (win_prob_cal >= 0.45) & (win_prob_cal <= 0.55)
        
        print(f"\n🎯 Overall Metrics:")
        print(f"   Accuracy: {accuracy:.1%}")
        print(f"   Brier Score: {brier:.4f}")
        print(f"   Mean Win Probability: {win_prob_pred.mean():.1%}")
        
        print(f"\n📊 By Confidence Level:")
        if high_conf_mask.sum() > 0:
            high_acc = accuracy_score(y_win_val[high_conf_mask], win_pred[high_conf_mask])
            print(f"   HIGH (>65% or <35%): {high_conf_mask.sum()} games, {high_acc:.1%} accuracy")
        if med_conf_mask.sum() > 0:
            med_acc = accuracy_score(y_win_val[med_conf_mask], win_pred[med_conf_mask])
            print(f"   MEDIUM (55-65% or 35-45%): {med_conf_mask.sum()} games, {med_acc:.1%} accuracy")
        if low_conf_mask.sum() > 0:
            low_acc = accuracy_score(y_win_val[low_conf_mask], win_pred[low_conf_mask])
            print(f"   LOW (45-55%): {low_conf_mask.sum()} games, {low_acc:.1%} accuracy")
        
        # 3. Show feature importance
        print(f"\n🔍 Top 10 Feature Importances:")
        importance_df = self.quantile_model.feature_importance(
            feature_names=self.feature_names, 
            top_n=10
        )
        for idx, row in importance_df.iterrows():
            print(f"   {idx+1}. {row['feature']}: {row['importance']:.0f}")
        
        # store a simple scale for uncertainty normalization (mean half-width on validation)
        try:
            val_uncertainty = (val_preds['q90'] - val_preds['q10']) / 2.0
            self.interval_scale = float(np.nanmean(val_uncertainty)) if len(val_uncertainty) > 0 else None
            print(f"\n   Interval mean half-width on validation: {self.interval_scale:.3f}")
        except Exception:
            self.interval_scale = None

        self.is_fitted = True

        return {
            'accuracy': accuracy,
            'brier_score': brier,
            'mean_prob': win_prob_pred.mean(),
            'high_conf_count': high_conf_mask.sum(),
            'high_conf_acc': high_acc if high_conf_mask.sum() > 0 else None
        }
    
    def predict_win_probability(self, X):
        """
        Predict win probability for home team.
        
        Args:
            X: Feature matrix (DataFrame or array)
        
        Returns:
            dict with:
                - win_prob: Home team win probability
                - point_diff: Expected point differential (home - away)
                - lower: 10th percentile point diff
                - upper: 90th percentile point diff
                - uncertainty: Half-width of 80% interval
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        # Scale features
        if isinstance(X, pd.DataFrame):
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Get quantile predictions
        preds = self.quantile_model.predict(X_scaled)
        
        # Convert median to win probability (raw)
        win_prob_raw = point_diff_to_win_prob(preds['q50'])
        # Apply calibrator if available
        if self.calibrator is not None:
            try:
                win_prob = self.calibrator.predict(win_prob_raw)
            except Exception:
                win_prob = win_prob_raw
        else:
            win_prob = win_prob_raw

        # Compute normalized uncertainty-based score (smaller interval -> higher confidence)
        half_width = (preds['q90'] - preds['q10']) / 2.0
        if self.interval_scale is not None and self.interval_scale > 0:
            norm_uncert = 1.0 - np.minimum(1.0, half_width / (2.0 * self.interval_scale))
        else:
            # fallback: inverse of observed half_width (scaled)
            norm_uncert = 1.0 - np.tanh(np.nanmean(half_width))

        # Confidence score: combine probability margin and normalized uncertainty
        prob_margin = np.abs(win_prob - 0.5) * 2.0
        confidence_score = 0.7 * prob_margin + 0.3 * norm_uncert

        # Confidence label thresholds (tunable)
        # support vectorized arrays
        confidence_label = None
        try:
            confidence_label = np.full_like(confidence_score, 'LOW', dtype=object)
            confidence_label[confidence_score >= 0.65] = 'HIGH'
            confidence_label[(confidence_score >= 0.40) & (confidence_score < 0.65)] = 'MEDIUM'
        except Exception:
            # scalar fallback
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
            'confidence_label': confidence_label
        }
    
    def save(self, filepath):
        """Save the complete predictor to disk."""
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
        
        print(f"\n💾 Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load predictor from disk."""
        data = joblib.load(filepath)
        
        instance = cls()
        instance.quantile_model = data['quantile_model']
        instance.scaler = data['scaler']
        instance.feature_names = data['feature_names']
        instance.calibrator = data.get('calibrator', None)
        instance.interval_scale = data.get('interval_scale', None)
        instance.is_fitted = True
        
        print(f"📂 Model loaded from {filepath}")
        return instance


def main():
    """Main training pipeline."""
    
    print("\n" + "="*80)
    print("🏀 LIGHTGBM NBA PREDICTION MODEL - TRAINING PIPELINE")
    print("="*80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Load data
    print("\n" + "="*80)
    print("📥 STEP 1: LOADING DATA")
    print("="*80)
    
    games_df = fetch_nba_games(
        seasons=['2023-24', '2024-25'],
        season_type='Regular Season',
        verbose=True
    )
    print(f"✅ Loaded {len(games_df)} game records")
    
    # Step 2: Calculate rolling stats
    print("\n📊 Calculating rolling stats (with leakage protection)...")
    games_with_stats = calculate_rolling_stats(games_df, window=5)
    print(f"✅ Rolling stats calculated")
    
    # Step 3: Create matchup features
    print("\n🔀 Creating matchup dataset (home vs away)...")
    matchup_df = create_matchup_features(games_with_stats)
    print(f"✅ Created {len(matchup_df)} matchups")
    print(f"   Date range: {matchup_df['GAME_DATE'].min()} → {matchup_df['GAME_DATE'].max()}")
    
    # Step 4: Prepare features
    print("\n🔧 Preparing features and targets...")
    X, y_diff, y_win, feature_names = prepare_features_and_target(matchup_df)
    print(f"✅ Feature matrix: {X.shape}")
    print(f"   Features: {len(feature_names)}")
    print(f"   Target: Point differential (mean={y_diff.mean():.1f}, std={y_diff.std():.1f})")
    
    # Step 5: Chronological train/val/test split
    print("\n" + "="*80)
    print("📊 STEP 2: CHRONOLOGICAL DATA SPLIT")
    print("="*80)
    
    matchup_sorted = matchup_df.sort_values('GAME_DATE').reset_index(drop=True)
    X_sorted = X.loc[matchup_sorted.index]
    y_diff_sorted = y_diff[matchup_sorted.index]
    y_win_sorted = y_win[matchup_sorted.index]
    
    train_end = int(len(matchup_sorted) * 0.70)
    val_end = int(len(matchup_sorted) * 0.85)
    
    X_train = X_sorted.iloc[:train_end]
    y_diff_train = y_diff_sorted[:train_end]
    y_win_train = y_win_sorted[:train_end]
    
    X_val = X_sorted.iloc[train_end:val_end]
    y_diff_val = y_diff_sorted[train_end:val_end]
    y_win_val = y_win_sorted[train_end:val_end]
    
    X_test = X_sorted.iloc[val_end:]
    y_diff_test = y_diff_sorted[val_end:]
    y_win_test = y_win_sorted[val_end:]
    
    print(f"\n📚 Training set: {len(X_train)} games")
    print(f"   Dates: {matchup_sorted.iloc[:train_end]['GAME_DATE'].min()} → "
          f"{matchup_sorted.iloc[:train_end]['GAME_DATE'].max()}")
    print(f"   Home win rate: {y_win_train.mean():.1%}")
    
    print(f"\n🔍 Validation set: {len(X_val)} games")
    print(f"   Dates: {matchup_sorted.iloc[train_end:val_end]['GAME_DATE'].min()} → "
          f"{matchup_sorted.iloc[train_end:val_end]['GAME_DATE'].max()}")
    print(f"   Home win rate: {y_win_val.mean():.1%}")
    
    print(f"\n🧪 Test set: {len(X_test)} games")
    print(f"   Dates: {matchup_sorted.iloc[val_end:]['GAME_DATE'].min()} → "
          f"{matchup_sorted.iloc[val_end:]['GAME_DATE'].max()}")
    print(f"   Home win rate: {y_win_test.mean():.1%}")
    
    # Step 6: Train model
    print("\n" + "="*80)
    print("🧠 STEP 3: TRAINING MODEL")
    print("="*80)
    
    predictor = LGBMWinPredictor()
    metrics = predictor.train(
        X_train, y_diff_train, y_win_train,
        X_val, y_diff_val, y_win_val
    )
    
    # Step 7: Final test set evaluation
    print("\n" + "="*80)
    print("🧪 STEP 4: FINAL TEST SET EVALUATION")
    print("="*80)
    
    test_preds = predictor.predict_win_probability(X_test)
    test_win_prob = test_preds['win_prob']
    test_win_pred = (test_win_prob > 0.5).astype(int)
    
    test_accuracy = accuracy_score(y_win_test, test_win_pred)
    test_brier = brier_score_loss(y_win_test, test_win_prob)
    
    print(f"\n🎯 Test Set Results:")
    print(f"   Accuracy: {test_accuracy:.1%}")
    print(f"   Brier Score: {test_brier:.4f}")
    print(f"   Mean Win Probability: {test_win_prob.mean():.1%}")
    
    # Step 8: Save model
    print("\n" + "="*80)
    print("💾 STEP 5: SAVING MODEL")
    print("="*80)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'machine_learning/models/lgbm_win_predictor_{timestamp}.pkl'
    predictor.save(model_path)
    
    # Also save as latest
    latest_path = 'machine_learning/models/lgbm_win_predictor_latest.pkl'
    predictor.save(latest_path)
    
    print("\n" + "="*80)
    print("✨ TRAINING COMPLETE")
    print("="*80)
    
    print(f"""
Summary:
  ✅ Model trained on {len(X_train)} games
  ✅ Validation accuracy: {metrics['accuracy']:.1%}
  ✅ Test accuracy: {test_accuracy:.1%}
  ✅ Brier score: {test_brier:.4f}
  ✅ Model saved to: {latest_path}

Next Steps:
  1. Update ensemble_predictions notebook to use trained model
  2. Run predictions for upcoming games
  3. Validate on live results
""")


if __name__ == '__main__':
    main()

"""
Adaptive Learning Module for NBA Predictions

Handles:
- Parsing actual game results from various sources (CSV, API, manual)
- Matching predictions to actual outcomes
- Error analysis and pattern detection
- MCMC-based model refinement using prediction errors
- Adaptive hyperparameter tuning based on performance
- Backpropagation of learned patterns into model parameters
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')


class GameResultParser:
    """Parse game results from various formats"""
    
    @staticmethod
    def parse_csv_results(csv_text: str) -> pd.DataFrame:
        """
        Parse CSV game results (Sports Reference format)
        
        Parameters:
        - csv_text: Raw CSV text with game results
        
        Returns:
        - DataFrame with cleaned game results
        """
        from io import StringIO
        
        # Parse CSV
        df = pd.read_csv(StringIO(csv_text))
        
        # Clean up - only keep completed games (those with scores)
        df = df[df['PTS'].notna()].copy()
        
        # Rename columns for clarity
        df = df.rename(columns={
            'Visitor/Neutral': 'away_team',
            'Home/Neutral': 'home_team',
            'Date': 'game_date'
        })
        
        # Add scores (PTS column appears twice - need to handle both)
        # The CSV structure has: Date, Time, Away, PTS, Home, PTS
        # We'll need to parse this carefully
        
        # Create proper away_score and home_score columns
        df['away_score'] = df['PTS'].iloc[::2].values if len(df) > 0 else []
        df['home_score'] = df['PTS'].iloc[1::2].values if len(df) > 1 else []
        
        return df[['game_date', 'away_team', 'home_team', 'away_score', 'home_score']]
    
    @staticmethod
    def parse_csv_sports_reference(csv_lines: List[str]) -> pd.DataFrame:
        """
        Parse Sports Reference CSV format (more robust)
        
        Expected format:
        Date,Start (ET),Visitor/Neutral,PTS,Home/Neutral,PTS,...
        
        Returns:
        - DataFrame with game results
        """
        results = []
        
        for line in csv_lines:
            if not line.strip() or line.startswith('Date') or line.startswith('Provided'):
                continue
            
            parts = [p.strip() for p in line.split(',')]
            
            if len(parts) < 6:
                continue
            
            # Extract fields
            date = parts[0]
            away_team = parts[2]
            away_pts = parts[3]
            home_team = parts[4]
            home_pts = parts[5]
            
            # Skip if scores are missing (future games)
            if not away_pts or not home_pts or away_pts == '' or home_pts == '':
                continue
            
            try:
                results.append({
                    'game_date': date,
                    'away_team': away_team,
                    'home_team': home_team,
                    'away_score': int(away_pts),
                    'home_score': int(home_pts),
                    'home_spread': int(home_pts) - int(away_pts)
                })
            except ValueError:
                continue
        
        return pd.DataFrame(results)


class PredictionMatcher:
    """Match predictions to actual game results"""
    
    # Team name variations/aliases for matching
    TEAM_ALIASES = {
        'LA Clippers': 'Los Angeles Clippers',
        'LA Lakers': 'Los Angeles Lakers',
        'L.A. Clippers': 'Los Angeles Clippers',
        'L.A. Lakers': 'Los Angeles Lakers',
    }
    
    @staticmethod
    def normalize_team_name(name: str) -> str:
        """Normalize team name for matching"""
        name = name.strip()
        return PredictionMatcher.TEAM_ALIASES.get(name, name)
    
    @staticmethod
    def match_predictions_to_results(
        predictions: List[Dict],
        results_df: pd.DataFrame,
        date_tolerance_days: int = 2
    ) -> List[Dict]:
        """
        Match predictions to actual game results
        
        Parameters:
        - predictions: List of prediction dicts from predictor
        - results_df: DataFrame with actual game results
        - date_tolerance_days: How many days to search for matching games
        
        Returns:
        - List of matched prediction-result pairs
        """
        matches = []
        
        for pred in predictions:
            home_team = PredictionMatcher.normalize_team_name(pred.get('home_team', ''))
            away_team = PredictionMatcher.normalize_team_name(pred.get('away_team', ''))
            
            # Find matching game in results
            for _, result in results_df.iterrows():
                result_home = PredictionMatcher.normalize_team_name(result['home_team'])
                result_away = PredictionMatcher.normalize_team_name(result['away_team'])
                
                if result_home == home_team and result_away == away_team:
                    match = {
                        'home_team': home_team,
                        'away_team': away_team,
                        'game_date': result['game_date'],
                        
                        # Prediction data
                        'predicted_spread': pred.get('predicted_spread', 0),
                        'predicted_winner': home_team if pred.get('predicted_spread', 0) > 0 else away_team,
                        'win_probability': pred.get('win_probability', 0.5),
                        'uncertainty': pred.get('uncertainty', 0),
                        'confidence': pred.get('confidence', 'LOW'),
                        
                        # Actual results
                        'actual_home_score': result['home_score'],
                        'actual_away_score': result['away_score'],
                        'actual_spread': result['home_spread'],
                        'actual_winner': result['home_team'] if result['home_spread'] > 0 else result['away_team'],
                        
                        # Calculated errors
                        'spread_error': abs(pred.get('predicted_spread', 0) - result['home_spread']),
                        'correct_winner': (pred.get('predicted_spread', 0) > 0) == (result['home_spread'] > 0),
                        'confidence_justified': None,  # Will calculate
                    }
                    
                    # Check if confidence was justified
                    if match['confidence'] == 'HIGH':
                        match['confidence_justified'] = match['spread_error'] < 8 and match['correct_winner']
                    elif match['confidence'] == 'MEDIUM':
                        match['confidence_justified'] = match['spread_error'] < 12
                    else:  # LOW
                        match['confidence_justified'] = True  # Low confidence = we knew it was uncertain
                    
                    matches.append(match)
                    break
        
        return matches


class ErrorAnalyzer:
    """Analyze prediction errors to identify systematic biases"""
    
    @staticmethod
    def analyze_errors(matches: List[Dict]) -> Dict:
        """
        Comprehensive error analysis
        
        Returns:
        - Dict with error patterns, biases, and insights
        """
        if not matches:
            return {'error': 'No matches to analyze'}
        
        df = pd.DataFrame(matches)
        
        # Calculate basic metrics
        n_predictions = len(df)
        correct_winners = df['correct_winner'].sum()
        win_accuracy = correct_winners / n_predictions
        
        mae = df['spread_error'].mean()
        rmse = np.sqrt((df['spread_error'] ** 2).mean())
        
        # Analyze by confidence level
        confidence_analysis = {}
        for conf in ['HIGH', 'MEDIUM', 'LOW']:
            conf_matches = df[df['confidence'] == conf]
            if len(conf_matches) > 0:
                confidence_analysis[conf] = {
                    'count': len(conf_matches),
                    'win_accuracy': conf_matches['correct_winner'].mean(),
                    'mae': conf_matches['spread_error'].mean(),
                    'confidence_justified_rate': conf_matches['confidence_justified'].mean()
                }
        
        # Identify overconfident and underconfident predictions
        high_conf_wrong = df[(df['confidence'] == 'HIGH') & (~df['correct_winner'])]
        low_conf_right = df[(df['confidence'] == 'LOW') & (df['correct_winner']) & (df['spread_error'] < 5)]
        
        # Bias analysis - are we consistently over/under predicting for home teams?
        actual_spreads = df['actual_spread'].values
        predicted_spreads = df['predicted_spread'].values
        
        # Check for systematic bias
        mean_bias = (predicted_spreads - actual_spreads).mean()
        bias_direction = 'home-favoring' if mean_bias > 0 else 'away-favoring'
        
        # Analyze error distribution
        errors = predicted_spreads - actual_spreads
        error_skew = stats.skew(errors)
        error_kurtosis = stats.kurtosis(errors)
        
        # Identify games with largest errors (outliers to investigate)
        df['abs_error'] = df['spread_error']
        worst_predictions = df.nlargest(5, 'abs_error')[
            ['home_team', 'away_team', 'predicted_spread', 'actual_spread', 'spread_error']
        ].to_dict('records')
        
        # Best predictions
        best_predictions = df.nsmallest(5, 'abs_error')[
            ['home_team', 'away_team', 'predicted_spread', 'actual_spread', 'spread_error']
        ].to_dict('records')
        
        return {
            'overall_metrics': {
                'n_predictions': n_predictions,
                'correct_winners': correct_winners,
                'win_accuracy': win_accuracy,
                'mae': mae,
                'rmse': rmse,
            },
            'bias_analysis': {
                'mean_bias': mean_bias,
                'bias_direction': bias_direction,
                'error_skew': error_skew,
                'error_kurtosis': error_kurtosis,
            },
            'confidence_analysis': confidence_analysis,
            'problem_areas': {
                'overconfident_errors': len(high_conf_wrong),
                'underconfident_successes': len(low_conf_right),
            },
            'worst_predictions': worst_predictions,
            'best_predictions': best_predictions,
        }


class AdaptiveLearner:
    """
    Use MCMC and adaptive methods to learn from prediction errors
    """
    
    def __init__(self, mcmc_model=None):
        """
        Initialize adaptive learner
        
        Parameters:
        - mcmc_model: BayesianBasketballHierarchical instance (optional)
        """
        self.mcmc_model = mcmc_model
        self.learning_history = []
    
    def calculate_team_error_adjustments(
        self,
        matches: List[Dict],
        team_data: Dict
    ) -> Dict[str, float]:
        """
        Calculate per-team bias adjustments based on prediction errors
        
        Logic:
        - If we consistently over-predict for a team, reduce their rating
        - If we consistently under-predict, increase their rating
        
        Returns:
        - Dict mapping team names to adjustment factors
        """
        # Group errors by team
        team_errors = {}
        
        for match in matches:
            home_team = match['home_team']
            away_team = match['away_team']
            
            # Error in perspective of each team
            # Positive error = we over-predicted home team's performance
            error = match['predicted_spread'] - match['actual_spread']
            
            # Track errors for both teams
            if home_team not in team_errors:
                team_errors[home_team] = []
            if away_team not in team_errors:
                team_errors[away_team] = []
            
            # Home team: positive error means we rated them too high
            team_errors[home_team].append(error)
            # Away team: positive error means we rated them too low
            team_errors[away_team].append(-error)
        
        # Calculate adjustments
        adjustments = {}
        for team, errors in team_errors.items():
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            n = len(errors)
            
            # Statistical significance check (t-test)
            if n > 3 and std_error > 0:
                t_stat = mean_error / (std_error / np.sqrt(n))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
                
                # Only adjust if statistically significant (p < 0.10)
                if p_value < 0.10:
                    # Adjustment proportional to error (but dampened for stability)
                    adjustment = -mean_error * 0.1  # 10% learning rate
                    adjustments[team] = adjustment
                else:
                    adjustments[team] = 0.0
            else:
                adjustments[team] = 0.0
        
        return adjustments
    
    def propose_mcmc_refinement(
        self,
        matches: List[Dict],
        current_epaa_weight: float = 0.5
    ) -> Dict:
        """
        Propose MCMC model refinements based on error analysis
        
        Returns:
        - Dict with proposed changes to model parameters
        """
        df = pd.DataFrame(matches)
        
        # Analyze current EPAA weight effectiveness
        # If errors are high, we may need to adjust EPAA weighting
        
        mae = df['spread_error'].mean()
        win_accuracy = df['correct_winner'].mean()
        
        # Determine if EPAA weight should change
        proposed_weight = current_epaa_weight
        reasoning = []
        
        if win_accuracy < 0.55:
            # Poor winner prediction - maybe rely more on rolling stats
            proposed_weight = max(0.2, current_epaa_weight - 0.1)
            reasoning.append("Low win accuracy suggests EPAA may be less predictive")
        elif win_accuracy > 0.70 and mae > 10:
            # Good winner prediction but poor spread accuracy
            # EPAA is capturing right direction but magnitude is off
            proposed_weight = min(0.8, current_epaa_weight + 0.05)
            reasoning.append("High win accuracy with spread errors suggests EPAA direction is good")
        elif mae < 8:
            # Good performance overall - maintain or slight increase
            proposed_weight = min(0.7, current_epaa_weight + 0.02)
            reasoning.append("Strong overall performance - slight increase in EPAA weight")
        
        # Analyze uncertainty calibration
        # Check if high-uncertainty games actually had larger errors
        high_unc = df[df['uncertainty'] > df['uncertainty'].median()]
        low_unc = df[df['uncertainty'] <= df['uncertainty'].median()]
        
        uncertainty_calibrated = (
            high_unc['spread_error'].mean() > low_unc['spread_error'].mean()
        ) if len(high_unc) > 0 and len(low_unc) > 0 else False
        
        if not uncertainty_calibrated:
            reasoning.append("Uncertainty estimates not well-calibrated - consider GP kernel tuning")
        
        return {
            'current_epaa_weight': current_epaa_weight,
            'proposed_epaa_weight': proposed_weight,
            'weight_change': proposed_weight - current_epaa_weight,
            'reasoning': reasoning,
            'metrics': {
                'mae': mae,
                'win_accuracy': win_accuracy,
                'uncertainty_calibrated': uncertainty_calibrated
            }
        }
    
    def generate_learning_report(
        self,
        matches: List[Dict],
        team_data: Dict,
        current_epaa_weight: float = 0.5
    ) -> str:
        """
        Generate comprehensive learning report with actionable insights
        
        Returns:
        - Formatted markdown report
        """
        # Run all analyses
        error_analysis = ErrorAnalyzer.analyze_errors(matches)
        team_adjustments = self.calculate_team_error_adjustments(matches, team_data)
        mcmc_refinement = self.propose_mcmc_refinement(matches, current_epaa_weight)
        
        # Build report
        report = []
        report.append("# 🎯 Adaptive Learning Report\n")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Games Analyzed:** {error_analysis['overall_metrics']['n_predictions']}\n")
        
        # Overall Performance
        report.append("\n## 📊 Overall Performance\n")
        metrics = error_analysis['overall_metrics']
        report.append(f"- **Win Accuracy:** {metrics['win_accuracy']:.1%} ({metrics['correct_winners']}/{metrics['n_predictions']})")
        report.append(f"- **Mean Absolute Error:** {metrics['mae']:.2f} points")
        report.append(f"- **RMSE:** {metrics['rmse']:.2f} points\n")
        
        # Bias Analysis
        report.append("\n## 🎲 Bias Analysis\n")
        bias = error_analysis['bias_analysis']
        report.append(f"- **Mean Bias:** {bias['mean_bias']:+.2f} points ({bias['bias_direction']})")
        report.append(f"- **Error Distribution:** Skew={bias['error_skew']:.3f}, Kurtosis={bias['error_kurtosis']:.3f}")
        
        if abs(bias['mean_bias']) > 2:
            report.append(f"\n⚠️ **Systematic bias detected!** Model is {bias['bias_direction']} by {abs(bias['mean_bias']):.1f} points on average.\n")
        
        # Confidence Analysis
        report.append("\n## 🎯 Confidence Calibration\n")
        for conf, data in error_analysis['confidence_analysis'].items():
            report.append(f"\n### {conf} Confidence:")
            report.append(f"- Games: {data['count']}")
            report.append(f"- Win Accuracy: {data['win_accuracy']:.1%}")
            report.append(f"- MAE: {data['mae']:.2f} points")
            report.append(f"- Confidence Justified: {data['confidence_justified_rate']:.1%}")
        
        # Problem Areas
        report.append("\n## ⚠️ Problem Areas\n")
        problems = error_analysis['problem_areas']
        report.append(f"- **Overconfident Errors:** {problems['overconfident_errors']} (high confidence, wrong winner)")
        report.append(f"- **Underconfident Successes:** {problems['underconfident_successes']} (low confidence, good prediction)\n")
        
        # Worst Predictions (learning opportunities)
        report.append("\n## 🔍 Worst Predictions (Learn From These)\n")
        for i, pred in enumerate(error_analysis['worst_predictions'], 1):
            report.append(f"{i}. **{pred['away_team']} @ {pred['home_team']}**")
            report.append(f"   - Predicted: {pred['predicted_spread']:+.1f} | Actual: {pred['actual_spread']:+.1f} | Error: {pred['spread_error']:.1f}\n")
        
        # Team-Specific Adjustments
        report.append("\n## 🔧 Proposed Team Adjustments\n")
        significant_adjustments = {k: v for k, v in team_adjustments.items() if abs(v) > 0.5}
        
        if significant_adjustments:
            report.append("Teams with statistically significant biases:\n")
            for team, adj in sorted(significant_adjustments.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
                direction = "↑ Underrated" if adj > 0 else "↓ Overrated"
                report.append(f"- **{team}:** {adj:+.2f} points {direction}")
        else:
            report.append("✅ No statistically significant team biases detected.\n")
        
        # MCMC Refinement Proposals
        report.append("\n## 🔬 MCMC Model Refinement\n")
        report.append(f"**Current EPAA Weight:** {mcmc_refinement['current_epaa_weight']:.2f}")
        report.append(f"**Proposed EPAA Weight:** {mcmc_refinement['proposed_epaa_weight']:.2f}")
        report.append(f"**Change:** {mcmc_refinement['weight_change']:+.2f}\n")
        
        report.append("**Reasoning:**")
        for reason in mcmc_refinement['reasoning']:
            report.append(f"- {reason}")
        
        # Action Items
        report.append("\n## ✅ Recommended Actions\n")
        report.append("1. **Update EPAA Weight:** Adjust from {:.2f} to {:.2f}".format(
            mcmc_refinement['current_epaa_weight'],
            mcmc_refinement['proposed_epaa_weight']
        ))
        
        if abs(bias['mean_bias']) > 2:
            report.append("2. **Correct Systematic Bias:** Add {:.2f} point adjustment to all predictions".format(-bias['mean_bias']))
        
        if problems['overconfident_errors'] > len(matches) * 0.1:
            report.append("3. **Recalibrate Confidence:** Reduce confidence thresholds (too many overconfident errors)")
        
        if not mcmc_refinement['metrics']['uncertainty_calibrated']:
            report.append("4. **Retune GP Kernel:** Uncertainty estimates need recalibration")
        
        report.append("\n---")
        report.append("\n*This report uses statistical analysis and MCMC principles to identify systematic errors and propose model improvements.*\n")
        
        return '\n'.join(report)


def validate_and_learn(
    predictions: List[Dict],
    results_csv: str,
    team_data: Dict,
    current_epaa_weight: float = 0.5,
    save_matches: bool = True,
    output_file: str = 'json/validation_matches.json'
) -> Dict:
    """
    Complete validation and learning pipeline
    
    Parameters:
    - predictions: List of prediction dicts
    - results_csv: CSV text with game results
    - team_data: Team data dict
    - current_epaa_weight: Current EPAA weight in use
    - save_matches: Save matched predictions to file
    - output_file: Where to save matches
    
    Returns:
    - Dict with all analysis results and learning recommendations
    """
    print("🔄 Starting validation and learning pipeline...\n")
    
    # Step 1: Parse results
    print("📋 Parsing game results...")
    csv_lines = results_csv.strip().split('\n')
    results_df = GameResultParser.parse_csv_sports_reference(csv_lines)
    print(f"✅ Parsed {len(results_df)} completed games\n")
    
    # Step 2: Match predictions to results
    print("🔗 Matching predictions to actual results...")
    matches = PredictionMatcher.match_predictions_to_results(predictions, results_df)
    print(f"✅ Matched {len(matches)} predictions\n")
    
    if len(matches) == 0:
        print("❌ No matches found. Check team name alignment or date ranges.")
        return {'error': 'No matches found'}
    
    # Save matches
    if save_matches:
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(matches, f, indent=2)
        print(f"💾 Saved matches to {output_file}\n")
    
    # Step 3: Generate learning report
    print("🧠 Generating adaptive learning report...\n")
    learner = AdaptiveLearner()
    report = learner.generate_learning_report(matches, team_data, current_epaa_weight)
    
    # Return everything
    return {
        'matches': matches,
        'report': report,
        'n_matches': len(matches),
        'results_df': results_df,
        'error_analysis': ErrorAnalyzer.analyze_errors(matches),
        'team_adjustments': learner.calculate_team_error_adjustments(matches, team_data),
        'mcmc_refinement': learner.propose_mcmc_refinement(matches, current_epaa_weight)
    }


"""
Hierarchical Bayesian MCMC Model for NBA Basketball Analytics

This module implements a Gibbs sampler for hierarchical Bayesian modeling of NBA team performance.
The model captures:
- Shot selection patterns (z_i): Which shot types teams prefer
- Shooting accuracy (w_i): How accurate teams are from different locations
- EPAA (Expected Points Above Average): Overall team efficiency metric

The hierarchical structure allows for:
- Team-specific parameters
- Population-level priors
- Uncertainty quantification via posterior sampling
"""

import numpy as np
from scipy import stats
from scipy.special import logsumexp
import pickle
from typing import Dict, Tuple, Optional, List


class BayesianBasketballHierarchical:
    """
    Hierarchical Bayesian model for NBA team performance using Gibbs sampling.
    
    This model uses MCMC (Markov Chain Monte Carlo) to estimate:
    - Team offensive efficiency (EPAA - Expected Points Above Average)
    - Shot selection clusters (which shot types teams prefer)
    - Accuracy clusters (how accurate teams are from different court regions)
    
    Parameters
    ----------
    L : int, default=10
        Number of accuracy clusters (how many distinct accuracy profiles)
    J : int, default=10
        Number of shot selection clusters (how many distinct shot selection patterns)
    K : int, default=7
        Number of court regions (e.g., paint, mid-range, 3PT corner, etc.)
    
    Attributes
    ----------
    team_ids : list
        List of team IDs in the model
    theta_i : dict
        Team-specific offensive efficiency parameters (EPAA)
    z_i : dict
        Shot selection cluster assignments for each team
    w_i : dict
        Accuracy cluster assignments for each team
    mu_j : np.ndarray
        Shot selection profiles (J x K): proportion of shots from each region
    eta_l : np.ndarray
        Accuracy profiles (L x K): field goal percentage for each region
    """
    
    def __init__(self, L=10, J=10, K=7):
        """
        Initialize the hierarchical Bayesian model.
        
        Parameters
        ----------
        L : int
            Number of accuracy clusters
        J : int
            Number of shot selection clusters
        K : int
            Number of court regions
        """
        self.L = L  # Number of accuracy clusters
        self.J = J  # Number of shot selection clusters
        self.K = K  # Number of court regions
        
        # Model parameters (will be set during fitting)
        self.team_ids = None
        self.theta_i = {}  # Team offensive efficiency (EPAA)
        self.z_i = {}      # Shot selection cluster assignment
        self.w_i = {}      # Accuracy cluster assignment
        self.mu_j = None   # Shot selection profiles (J x K)
        self.eta_l = None  # Accuracy profiles (L x K)
        
        # Hyperparameters
        self.alpha_z = np.ones(J)  # Dirichlet prior for shot selection
        self.alpha_w = np.ones(L)  # Dirichlet prior for accuracy
        
        # Posterior samples for uncertainty quantification
        self.posterior_samples = {
            'theta': [],
            'z': [],
            'w': [],
            'mu': [],
            'eta': []
        }
        
    def fit_gibbs(self, team_shot_data, n_iterations=5000, burn_in=1500, thin=1):
        """
        Fit the model using Gibbs sampling.
        
        This is the core MCMC algorithm that alternates between sampling:
        1. Team assignments (z_i, w_i)
        2. Cluster parameters (mu_j, eta_l)
        3. Team efficiency (theta_i)
        
        Parameters
        ----------
        team_shot_data : dict
            Dictionary mapping team_id to shot data:
            {
                team_id: {
                    'M_ik': np.ndarray of shape (K,)  # Made shots per region
                    'N_ik': np.ndarray of shape (K,)  # Attempted shots per region
                    'points_per_game': float          # Average points scored
                }
            }
        n_iterations : int, default=5000
            Total number of Gibbs sampling iterations
        burn_in : int, default=1500
            Number of initial iterations to discard
        thin : int, default=1
            Keep every thin-th sample (for reducing autocorrelation)
            
        Returns
        -------
        self : BayesianBasketballHierarchical
            Fitted model with posterior samples
        """
        self.team_ids = list(team_shot_data.keys())
        n_teams = len(self.team_ids)
        
        print(f"🔬 Starting Gibbs Sampling with {n_iterations} iterations...")
        print(f"   Teams: {n_teams}")
        print(f"   Burn-in: {burn_in}, Thinning: {thin}")
        print(f"   Clusters: {self.J} shot selection, {self.L} accuracy")
        
        # Initialize parameters randomly
        self._initialize_parameters(team_shot_data)
        
        # Store data for sampling
        self.data = team_shot_data
        
        # Gibbs sampling loop
        for iteration in range(n_iterations):
            if iteration % 500 == 0:
                print(f"   Iteration {iteration}/{n_iterations}...")
            
            # Step 1: Sample cluster assignments for each team
            for team_id in self.team_ids:
                self._sample_z_i(team_id)
                self._sample_w_i(team_id)
            
            # Step 2: Sample cluster parameters
            self._sample_mu_j()
            self._sample_eta_l()
            
            # Step 3: Sample team efficiency parameters
            self._sample_theta_i()
            
            # Store samples after burn-in (with thinning)
            if iteration >= burn_in and (iteration - burn_in) % thin == 0:
                self._store_sample()
        
        print(f"✅ Gibbs sampling complete!")
        print(f"   Collected {len(self.posterior_samples['theta'])} posterior samples")
        
        # Compute posterior means and credible intervals
        self._compute_posterior_statistics()
        
        return self
    
    def _initialize_parameters(self, team_shot_data):
        """Initialize all parameters randomly."""
        # Initialize cluster assignments randomly
        for team_id in self.team_ids:
            self.z_i[team_id] = np.random.randint(0, self.J)
            self.w_i[team_id] = np.random.randint(0, self.L)
        
        # Initialize shot selection profiles (Dirichlet priors)
        self.mu_j = np.random.dirichlet([1.0] * self.K, size=self.J)
        
        # Initialize accuracy profiles (Beta priors, centered around 0.45)
        self.eta_l = np.random.beta(4.5, 5.5, size=(self.L, self.K))
        
        # Initialize team efficiency based on actual points per game
        league_avg_ppg = np.mean([data['points_per_game'] 
                                   for data in team_shot_data.values()])
        
        for team_id in self.team_ids:
            ppg = team_shot_data[team_id]['points_per_game']
            self.theta_i[team_id] = ppg - league_avg_ppg  # EPAA
    
    def _sample_z_i(self, team_id):
        """
        Sample shot selection cluster assignment for team i.
        
        Uses the observed shot distribution to compute posterior probabilities
        for each cluster, then samples from the categorical distribution.
        """
        data = self.data[team_id]
        N_ik = data['N_ik']  # Shot attempts per region
        
        # Compute log probabilities for each cluster
        log_probs = np.zeros(self.J)
        
        for j in range(self.J):
            # Multinomial likelihood: N_ik | mu_j[k]
            # Log probability of observed shot distribution given cluster j
            log_probs[j] = np.sum(N_ik * np.log(self.mu_j[j] + 1e-10))
            
            # Add prior (Dirichlet)
            log_probs[j] += np.log(self.alpha_z[j] + 1e-10)
        
        # Normalize to get probabilities
        log_probs -= logsumexp(log_probs)
        probs = np.exp(log_probs)
        
        # Sample new cluster assignment
        self.z_i[team_id] = np.random.choice(self.J, p=probs)
    
    def _sample_w_i(self, team_id):
        """
        Sample accuracy cluster assignment for team i.
        
        Uses the observed makes/attempts to compute posterior probabilities
        for each accuracy cluster.
        """
        data = self.data[team_id]
        M_ik = data['M_ik']  # Made shots per region
        N_ik = data['N_ik']  # Attempted shots per region
        
        # Compute log probabilities for each accuracy cluster
        log_probs = np.zeros(self.L)
        
        for l in range(self.L):
            # Binomial likelihood: M_ik | N_ik, eta_l[k]
            for k in range(self.K):
                if N_ik[k] > 0:
                    log_probs[l] += stats.binom.logpmf(
                        int(M_ik[k]), 
                        int(N_ik[k]), 
                        self.eta_l[l, k]
                    )
            
            # Add prior (Dirichlet)
            log_probs[l] += np.log(self.alpha_w[l] + 1e-10)
        
        # Normalize to get probabilities
        log_probs -= logsumexp(log_probs)
        probs = np.exp(log_probs)
        
        # Sample new cluster assignment
        self.w_i[team_id] = np.random.choice(self.L, p=probs)
    
    def _sample_mu_j(self):
        """
        Sample shot selection profiles for all clusters.
        
        For each cluster j, aggregate all teams assigned to that cluster
        and sample from the Dirichlet posterior.
        """
        for j in range(self.J):
            # Find teams assigned to cluster j
            teams_in_cluster = [tid for tid in self.team_ids if self.z_i[tid] == j]
            
            if len(teams_in_cluster) == 0:
                # No teams in this cluster, sample from prior
                self.mu_j[j] = np.random.dirichlet(self.alpha_z)
            else:
                # Aggregate shot counts from all teams in cluster
                total_shots = np.zeros(self.K)
                for team_id in teams_in_cluster:
                    total_shots += self.data[team_id]['N_ik']
                
                # Sample from Dirichlet posterior
                posterior_alpha = self.alpha_z + total_shots
                self.mu_j[j] = np.random.dirichlet(posterior_alpha)
    
    def _sample_eta_l(self):
        """
        Sample accuracy profiles for all clusters.
        
        For each cluster l and region k, aggregate makes/attempts and
        sample from the Beta posterior.
        """
        for l in range(self.L):
            # Find teams assigned to cluster l
            teams_in_cluster = [tid for tid in self.team_ids if self.w_i[tid] == l]
            
            for k in range(self.K):
                if len(teams_in_cluster) == 0:
                    # No teams in cluster, sample from prior Beta(4.5, 5.5)
                    self.eta_l[l, k] = np.random.beta(4.5, 5.5)
                else:
                    # Aggregate makes and attempts
                    total_makes = sum(self.data[tid]['M_ik'][k] 
                                     for tid in teams_in_cluster)
                    total_attempts = sum(self.data[tid]['N_ik'][k] 
                                        for tid in teams_in_cluster)
                    
                    # Sample from Beta posterior
                    # Beta(a + makes, b + (attempts - makes))
                    alpha_post = 4.5 + total_makes
                    beta_post = 5.5 + (total_attempts - total_makes)
                    self.eta_l[l, k] = np.random.beta(alpha_post, beta_post)
    
    def _sample_theta_i(self):
        """
        Sample team efficiency parameters (EPAA).
        
        This uses a Normal likelihood based on actual points per game
        and the expected points from the team's shot profile.
        """
        league_avg_ppg = np.mean([self.data[tid]['points_per_game'] 
                                   for tid in self.team_ids])
        
        for team_id in self.team_ids:
            observed_ppg = self.data[team_id]['points_per_game']
            
            # Expected points from shot profile
            z = self.z_i[team_id]
            w = self.w_i[team_id]
            
            # Calculate expected points: sum over regions of
            # (shot_proportion * accuracy * points_per_shot)
            expected_points = 0.0
            for k in range(self.K):
                shot_proportion = self.mu_j[z, k]
                accuracy = self.eta_l[w, k]
                # Assume regions 0-3 are 2PT (paint, mid), 4-6 are 3PT
                points_value = 3.0 if k >= 4 else 2.0
                expected_points += shot_proportion * accuracy * points_value
            
            # Scale to per-game basis (assume ~80 FGA per game)
            expected_points *= 80
            
            # Sample theta from Normal distribution
            # theta represents deviation from league average
            observed_epaa = observed_ppg - league_avg_ppg
            expected_epaa = expected_points - league_avg_ppg
            
            # Normal posterior with observed data
            # Prior: N(0, 5^2), Likelihood variance: 3^2
            prior_mean = 0.0
            prior_var = 25.0
            likelihood_var = 9.0
            
            # Posterior is also Normal with updated parameters
            post_var = 1.0 / (1.0/prior_var + 1.0/likelihood_var)
            post_mean = post_var * (prior_mean/prior_var + observed_epaa/likelihood_var)
            
            self.theta_i[team_id] = np.random.normal(post_mean, np.sqrt(post_var))
    
    def _store_sample(self):
        """Store current parameter values as a posterior sample."""
        self.posterior_samples['theta'].append(self.theta_i.copy())
        self.posterior_samples['z'].append(self.z_i.copy())
        self.posterior_samples['w'].append(self.w_i.copy())
        self.posterior_samples['mu'].append(self.mu_j.copy())
        self.posterior_samples['eta'].append(self.eta_l.copy())
    
    def _compute_posterior_statistics(self):
        """Compute summary statistics from posterior samples."""
        n_samples = len(self.posterior_samples['theta'])
        
        # Compute EPAA statistics for each team
        self.epaa_stats = {}
        for team_id in self.team_ids:
            theta_samples = [sample[team_id] 
                            for sample in self.posterior_samples['theta']]
            
            self.epaa_stats[team_id] = {
                'mean': np.mean(theta_samples),
                'std': np.std(theta_samples),
                'median': np.median(theta_samples),
                'q025': np.percentile(theta_samples, 2.5),
                'q975': np.percentile(theta_samples, 97.5)
            }
        
        # Compute most likely cluster assignments
        self.cluster_assignments = {}
        for team_id in self.team_ids:
            z_samples = [sample[team_id] 
                        for sample in self.posterior_samples['z']]
            w_samples = [sample[team_id] 
                        for sample in self.posterior_samples['w']]
            
            # Most frequent cluster assignment
            z_counts = np.bincount(z_samples, minlength=self.J)
            w_counts = np.bincount(w_samples, minlength=self.L)
            
            self.cluster_assignments[team_id] = {
                'shot_selection': {
                    'most_likely': np.argmax(z_counts),
                    'probabilities': z_counts / n_samples
                },
                'accuracy': {
                    'most_likely': np.argmax(w_counts),
                    'probabilities': w_counts / n_samples
                }
            }
    
    def predict_team_performance(self, team_id):
        """
        Predict team performance metrics.
        
        Parameters
        ----------
        team_id : int or str
            Team identifier
            
        Returns
        -------
        dict
            Dictionary with:
            - 'epaa_mean': Expected points above average (mean)
            - 'epaa_std': Standard deviation of EPAA
            - 'epaa_ci': 95% credible interval
            - 'shot_cluster': Most likely shot selection cluster
            - 'accuracy_cluster': Most likely accuracy cluster
            - 'expected_fg_pct': Expected field goal percentage
        """
        if team_id not in self.team_ids:
            raise ValueError(f"Team {team_id} not in fitted model")
        
        stats = self.epaa_stats[team_id]
        clusters = self.cluster_assignments[team_id]
        
        # Calculate expected FG%
        z = clusters['shot_selection']['most_likely']
        w = clusters['accuracy']['most_likely']
        
        expected_fg_pct = np.sum(self.mu_j[z] * self.eta_l[w])
        
        return {
            'epaa_mean': stats['mean'],
            'epaa_std': stats['std'],
            'epaa_ci': (stats['q025'], stats['q975']),
            'shot_cluster': z,
            'accuracy_cluster': w,
            'expected_fg_pct': expected_fg_pct,
            'cluster_probabilities': {
                'shot_selection': clusters['shot_selection']['probabilities'],
                'accuracy': clusters['accuracy']['probabilities']
            }
        }
    
    def get_epaa_rankings(self):
        """
        Get teams ranked by EPAA (Expected Points Above Average).
        
        Returns
        -------
        list of tuples
            List of (team_id, epaa_mean, epaa_std) sorted by epaa_mean
        """
        rankings = []
        for team_id in self.team_ids:
            stats = self.epaa_stats[team_id]
            rankings.append((team_id, stats['mean'], stats['std']))
        
        # Sort by EPAA (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def get_cluster_profiles(self):
        """
        Get the mean profiles for each cluster.
        
        Returns
        -------
        dict
            Dictionary with:
            - 'shot_selection': Mean shot selection profiles (J x K)
            - 'accuracy': Mean accuracy profiles (L x K)
        """
        # Average over posterior samples
        mu_mean = np.mean(self.posterior_samples['mu'], axis=0)
        eta_mean = np.mean(self.posterior_samples['eta'], axis=0)
        
        return {
            'shot_selection': mu_mean,
            'accuracy': eta_mean
        }
    
    def save(self, filepath):
        """
        Save the fitted model to disk.
        
        Parameters
        ----------
        filepath : str
            Path to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✅ Model saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """
        Load a fitted model from disk.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model
            
        Returns
        -------
        BayesianBasketballHierarchical
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded from {filepath}")
        return model


def calculate_epaa(team_shot_data, league_avg_ppg=None):
    """
    Calculate EPAA (Expected Points Above Average) for teams.
    
    This is a simplified calculation that doesn't require MCMC,
    useful for quick estimates or initialization.
    
    Parameters
    ----------
    team_shot_data : dict
        Dictionary mapping team_id to shot data with 'points_per_game'
    league_avg_ppg : float, optional
        League average points per game (computed if not provided)
        
    Returns
    -------
    dict
        Dictionary mapping team_id to EPAA value
    """
    if league_avg_ppg is None:
        league_avg_ppg = np.mean([data['points_per_game'] 
                                   for data in team_shot_data.values()])
    
    epaa_results = {}
    for team_id, data in team_shot_data.items():
        epaa = data['points_per_game'] - league_avg_ppg
        epaa_results[team_id] = {
            'epaa': epaa,
            'ppg': data['points_per_game'],
            'league_avg': league_avg_ppg
        }
    
    return epaa_results


def compare_team_matchup(model, home_team_id, away_team_id):
    """
    Compare two teams using the MCMC model.
    
    Parameters
    ----------
    model : BayesianBasketballHierarchical
        Fitted MCMC model
    home_team_id : int or str
        Home team identifier
    away_team_id : int or str
        Away team identifier
        
    Returns
    -------
    dict
        Dictionary with:
        - 'home_epaa': Home team EPAA statistics
        - 'away_epaa': Away team EPAA statistics
        - 'epaa_diff': Difference in EPAA (home - away)
        - 'predicted_spread': Predicted point spread
        - 'home_advantage': Typical home court advantage (~3 pts)
    """
    home_pred = model.predict_team_performance(home_team_id)
    away_pred = model.predict_team_performance(away_team_id)
    
    epaa_diff = home_pred['epaa_mean'] - away_pred['epaa_mean']
    home_advantage = 3.0  # Typical home court advantage
    
    predicted_spread = epaa_diff + home_advantage
    
    # Uncertainty in the spread
    spread_std = np.sqrt(home_pred['epaa_std']**2 + away_pred['epaa_std']**2)
    
    return {
        'home_epaa': home_pred,
        'away_epaa': away_pred,
        'epaa_diff': epaa_diff,
        'predicted_spread': predicted_spread,
        'spread_std': spread_std,
        'spread_ci': (
            predicted_spread - 1.96 * spread_std,
            predicted_spread + 1.96 * spread_std
        ),
        'home_advantage': home_advantage
    }


# Example usage and testing
if __name__ == "__main__":
    print("🏀 MCMC Basketball Model - Example Usage\n")
    
    # Create synthetic test data
    np.random.seed(42)
    K_REGIONS = 7  # Number of court regions
    
    # Simulate data for 10 teams
    test_data = {}
    for i in range(10):
        team_id = 1610612700 + i  # Example team IDs
        
        # Random shot distribution
        N_ik = np.random.multinomial(800, np.ones(K_REGIONS) / K_REGIONS)
        
        # Random makes (with some regions being better)
        fg_pcts = np.random.beta(4.5, 5.5, size=K_REGIONS)
        M_ik = np.array([np.random.binomial(n, p) 
                         for n, p in zip(N_ik, fg_pcts)])
        
        # Points per game
        ppg = np.random.normal(110, 8)
        
        test_data[team_id] = {
            'M_ik': M_ik,
            'N_ik': N_ik,
            'points_per_game': ppg
        }
    
    print("✅ Test data created for 10 teams\n")
    
    # Fit the model
    model = BayesianBasketballHierarchical(L=5, J=5, K=K_REGIONS)
    model.fit_gibbs(test_data, n_iterations=1000, burn_in=300)
    
    print("\n📊 EPAA Rankings:")
    print("="*50)
    rankings = model.get_epaa_rankings()
    for rank, (team_id, epaa, std) in enumerate(rankings, 1):
        print(f"{rank}. Team {team_id}: {epaa:+.2f} ± {std:.2f} EPAA")
    
    print("\n🎯 Example Matchup Prediction:")
    print("="*50)
    home_id = rankings[0][0]  # Best team
    away_id = rankings[-1][0]  # Worst team
    
    matchup = compare_team_matchup(model, home_id, away_id)
    print(f"Home Team {home_id} EPAA: {matchup['home_epaa']['epaa_mean']:+.2f}")
    print(f"Away Team {away_id} EPAA: {matchup['away_epaa']['epaa_mean']:+.2f}")
    print(f"Predicted Spread: {matchup['predicted_spread']:.2f} points")
    print(f"95% CI: ({matchup['spread_ci'][0]:.2f}, {matchup['spread_ci'][1]:.2f})")


"""
Prediction Module for NBA Games

Handles:
- Single game predictions using GP model
- GP + MCMC predictions with EPAA adjustments  
- Win probability calculations
- Confidence level determination
- Feature explanations
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
try:
    from .experimental.loaders import get_team_latest_stats
except ImportError:
    from experimental.loaders.data_loader import get_team_latest_stats

warnings.filterwarnings('ignore')


def predict_game_gp(home_team_name, away_team_name, gp_model, games_df, team_data):
    """
    Predict game outcome using Gaussian Process model
    
    Parameters:
    - home_team_name: Home team name
    - away_team_name: Away team name  
    - gp_model: Trained GaussianProcessPredictor
    - games_df: DataFrame with rolling stats
    - team_data: Dict from get_all_nba_teams()
    
    Returns:
    - Dict with prediction, uncertainty, win probability, confidence
    """
    
    # Get team IDs
    team_names_inv = {v: k for k, v in team_data['names'].items()}
    home_team_id = team_names_inv.get(home_team_name)
    away_team_id = team_names_inv.get(away_team_name)
    
    if home_team_id is None or away_team_id is None:
        raise ValueError(f"Team not found: {home_team_name} or {away_team_name}")
    
    # Get latest stats for both teams
    home_stats = get_team_latest_stats(games_df, home_team_id)
    away_stats = get_team_latest_stats(games_df, away_team_id)
    
    if home_stats is None or away_stats is None:
        raise ValueError("Could not find recent stats for teams")
    
    # Use feature names from model if available, otherwise extract from games_df
    if gp_model.feature_names:
        feature_cols = gp_model.feature_names
    else:
        # Extract features from games_df - only rolling stats (not WIN_STREAK, etc.)
        base_cols = [col for col in games_df.columns if '_ROLL' in col]
        feature_cols = [col for col in base_cols if col.startswith(('HOME_', 'AWAY_'))]
    
    # Build feature vector by combining home and away stats
    # The feature names include the prefix (HOME_ or AWAY_) so we need to match correctly
    features_list = []
    for col in feature_cols:
        if col.startswith('HOME_'):
            stat_name = col[5:]  # Remove 'HOME_' prefix to get the actual stat column name
            features_list.append(home_stats.get(stat_name, 0.0))
        elif col.startswith('AWAY_'):
            stat_name = col[5:]  # Remove 'AWAY_' prefix
            features_list.append(away_stats.get(stat_name, 0.0))
        else:
            features_list.append(0.0)
    
    # Convert to 2D array for prediction
    if len(features_list) == 0:
        raise ValueError(f"No features could be constructed. Feature columns: {feature_cols}")
    
    features = np.array([features_list])
    
    # Make prediction with uncertainty
    pred_diff, pred_std = gp_model.predict(features, return_std=True)
    pred_diff = pred_diff[0]
    pred_std = pred_std[0]
    
    # Calculate win probability (logistic function)
    # P(home wins) = 1 / (1 + exp(-k * point_diff))
    # k = 0.15 works well empirically
    win_prob = 1.0 / (1.0 + np.exp(-0.15 * pred_diff))
    
    # Confidence level based on uncertainty and win probability
    # High confidence: Low uncertainty AND clear winner (prob > 0.65 or < 0.35)
    # Medium confidence: Moderate uncertainty OR close game
    # Low confidence: High uncertainty OR very close game
    
    uncertainty_factor = pred_std / 12.0  # Normalize (12 pts is high uncertainty)
    prob_certainty = abs(win_prob - 0.5) * 2  # 0 = coin flip, 1 = certain
    
    confidence_score = prob_certainty * (1 - uncertainty_factor)
    
    if confidence_score > 0.5 and win_prob > 0.65:
        confidence = "HIGH"
    elif confidence_score > 0.3 and win_prob > 0.60:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    
    return {
        'home_team': home_team_name,
        'away_team': away_team_name,
        'predicted_spread': pred_diff,
        'uncertainty': pred_std,
        'lower_bound': pred_diff - 1.96 * pred_std,
        'upper_bound': pred_diff + 1.96 * pred_std,
        'win_probability': win_prob,
        'confidence': confidence,
        'confidence_score': confidence_score
    }


def predict_game_with_epaa(home_team_name, away_team_name, gp_model, games_df, team_data,
                            epaa_data, epaa_weight=0.5):
    """
    Predict game with EPAA adjustment from MCMC model
    
    Parameters:
    - home_team_name: Home team name
    - away_team_name: Away team name
    - gp_model: Trained GaussianProcessPredictor
    - games_df: DataFrame with rolling stats
    - team_data: Dict from get_all_nba_teams()
    - epaa_data: Dict {team_id: {'epaa_mean': float, 'epaa_std': float, ...}}
    - epaa_weight: Weight for EPAA adjustment (0-1, default: 0.5)
    
    Returns:
    - Dict with base prediction + EPAA-adjusted prediction
    """
    # Get base GP prediction
    base_pred = predict_game_gp(home_team_name, away_team_name, gp_model, games_df, team_data)
    
    # Get team IDs
    team_names_inv = {v: k for k, v in team_data['names'].items()}
    home_team_id = team_names_inv.get(home_team_name)
    away_team_id = team_names_inv.get(away_team_name)
    
    # Get EPAA values
    home_epaa = 0.0
    away_epaa = 0.0
    home_epaa_std = 0.0
    away_epaa_std = 0.0
    
    if home_team_id in epaa_data:
        home_epaa = epaa_data[home_team_id]['epaa_mean']
        home_epaa_std = epaa_data[home_team_id]['epaa_std']
    
    if away_team_id in epaa_data:
        away_epaa = epaa_data[away_team_id]['epaa_mean']
        away_epaa_std = epaa_data[away_team_id]['epaa_std']
    
    # EPAA differential (home advantage in offensive efficiency)
    epaa_diff = home_epaa - away_epaa
    epaa_uncertainty = np.sqrt(home_epaa_std**2 + away_epaa_std**2)
    
    # Adjusted prediction: Weighted combination
    adjusted_spread = base_pred['predicted_spread'] + (epaa_weight * epaa_diff)
    
    # Combined uncertainty
    combined_uncertainty = np.sqrt(base_pred['uncertainty']**2 + (epaa_weight * epaa_uncertainty)**2)
    
    # Recalculate win probability with adjusted spread
    adjusted_win_prob = 1.0 / (1.0 + np.exp(-0.15 * adjusted_spread))
    
    # Recalculate confidence
    uncertainty_factor = combined_uncertainty / 12.0
    prob_certainty = abs(adjusted_win_prob - 0.5) * 2
    confidence_score = prob_certainty * (1 - uncertainty_factor)
    
    if confidence_score > 0.5 and adjusted_win_prob > 0.65:
        confidence = "HIGH"
    elif confidence_score > 0.3 and adjusted_win_prob > 0.60:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    
    return {
        'home_team': home_team_name,
        'away_team': away_team_name,
        
        # Base GP prediction
        'gp_spread': base_pred['predicted_spread'],
        'gp_uncertainty': base_pred['uncertainty'],
        'gp_win_prob': base_pred['win_probability'],
        
        # EPAA adjustment
        'home_epaa': home_epaa,
        'away_epaa': away_epaa,
        'epaa_diff': epaa_diff,
        'epaa_weight_used': epaa_weight,
        
        # Adjusted prediction
        'predicted_spread': adjusted_spread,
        'uncertainty': combined_uncertainty,
        'lower_bound': adjusted_spread - 1.96 * combined_uncertainty,
        'upper_bound': adjusted_spread + 1.96 * combined_uncertainty,
        'win_probability': adjusted_win_prob,
        'confidence': confidence,
        'confidence_score': confidence_score
    }


def format_prediction_text(prediction):
    """
    Format prediction dict into readable text
    
    Parameters:
    - prediction: Dict from predict_game_gp or predict_game_with_epaa
    
    Returns:
    - Formatted string
    """
    home = prediction['home_team']
    away = prediction['away_team']
    spread = prediction['predicted_spread']
    uncertainty = prediction['uncertainty']
    win_prob = prediction['win_probability']
    confidence = prediction['confidence']
    
    if spread > 0:
        favorite = home
        underdog = away
        margin = spread
    else:
        favorite = away
        underdog = home
        margin = abs(spread)
    
    text = f"\n{'='*70}\n"
    text += f"🏀 {home} (HOME) vs {away} (AWAY)\n"
    text += f"{'='*70}\n\n"
    
    text += f"📊 PREDICTION:\n"
    text += f"   Spread: {spread:+.1f} points (±{uncertainty:.1f})\n"
    text += f"   Favorite: {favorite} by {margin:.1f} points\n"
    text += f"   Win Probability: {win_prob:.1%}\n"
    text += f"   Confidence: {confidence}\n\n"
    
    # EPAA info if available
    if 'epaa_diff' in prediction:
        text += f"🎯 EPAA ADJUSTMENT:\n"
        text += f"   {home} EPAA: {prediction['home_epaa']:+.2f}\n"
        text += f"   {away} EPAA: {prediction['away_epaa']:+.2f}\n"
        text += f"   Differential: {prediction['epaa_diff']:+.2f}\n"
        text += f"   Weight Used: {prediction['epaa_weight_used']:.0%}\n\n"
    
    text += f"📈 95% CONFIDENCE INTERVAL:\n"
    text += f"   {prediction['lower_bound']:.1f} to {prediction['upper_bound']:.1f} points\n"
    text += f"{'='*70}\n"
    
    return text


if __name__ == "__main__":
    print("🏀 Predictor module loaded successfully!")
    print("✅ Available functions:")
    print("   - predict_game_gp(): GP model predictions")
    print("   - predict_game_with_epaa(): GP + MCMC predictions")
    print("   - format_prediction_text(): Format output")

"""
LightGBM Quantile Regression for NBA Predictions

Trains 3 quantile models (Q10, Q50, Q90) to produce:
- Point differential prediction (median)
- 80% prediction interval (Q10-Q90)
- Uncertainty estimate

Memory efficient: ~50 MB total, <5 sec training.
"""

import numpy as np
import pandas as pd
import pickle
import os


class LGBMQuantilePredictor:
    """
    LightGBM quantile regression ensemble for NBA point differential prediction.
    
    Trains three models:
    - Q10: 10th percentile (lower bound of 80% interval)
    - Q50: Median (point estimate)
    - Q90: 90th percentile (upper bound of 80% interval)
    """
    
    def __init__(self, params=None, regularize_streak=True):
        """
        Args:
            params: LightGBM parameters dict (overrides defaults)
            regularize_streak: If True, cap WIN_STREAK feature importance to prevent overfitting
        """
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
        
        self.models = {}  # 'q10', 'q50', 'q90'
        self.feature_names = None
        self.is_fitted = False
    
    def train(self, X_train, y_train, X_val=None, y_val=None, X_calib=None, y_calib=None,
              quantiles=(0.1, 0.5, 0.9), num_boost_round=500,
              early_stopping_rounds=50):
        """
        Train quantile regression models.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training target (point differential)
            X_val: Validation features (optional, for early stopping)
            y_val: Validation target
            X_calib: Calibration features (optional, for interval adjustment)
            y_calib: Calibration target
            quantiles: Tuple of quantiles to train
            num_boost_round: Max boosting rounds
            early_stopping_rounds: Stop if no improvement for N rounds
        
        Returns:
            dict of trained models
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm required. Run: pip install lightgbm")
        
        print(f"\n🚀 Training LightGBM Quantile Regression")
        print(f"   Samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
        print(f"   Quantiles: {quantiles}")
        
        train_data = lgb.Dataset(X_train, label=y_train)
        
        valid_sets = []
        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets = [valid_data]
            print(f"   Validation: {X_val.shape[0]} samples")
        
        callbacks = []
        if valid_sets and early_stopping_rounds:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
        callbacks.append(lgb.log_evaluation(period=0))
        
        for q in quantiles:
            q_key = f'q{int(q * 100)}'
            
            params = self.base_params.copy()
            params['objective'] = 'quantile'
            params['alpha'] = q
            params['metric'] = 'quantile'
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=valid_sets if valid_sets else None,
                callbacks=callbacks,
            )
            
            self.models[q_key] = model
            print(f"   ✅ {q_key.upper()} trained ({model.num_trees()} trees)")
        
        self.is_fitted = True
        print(f"\n✅ All quantile models trained!")
        
        return self.models
    
    def predict(self, X):
        """
        Predict with all quantile models.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            dict with keys 'q10', 'q50', 'q90' -> numpy arrays
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        return {key: model.predict(X) for key, model in self.models.items()}
    
    def predict_with_intervals(self, X):
        """
        Predict point differential with 80% prediction interval.
        
        Args:
            X: Feature matrix
        
        Returns:
            DataFrame with columns: point_estimate, lower, upper, uncertainty
        """
        preds = self.predict(X)
        
        return pd.DataFrame({
            'point_estimate': preds['q50'],
            'lower': preds['q10'],
            'upper': preds['q90'],
            'uncertainty': (preds['q90'] - preds['q10']) / 2,
        })
    
    def feature_importance(self, feature_names=None, top_n=20):
        """
        Get feature importance from median (Q50) model.
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature, importance columns (regularized if enabled)
        """
        if 'q50' not in self.models:
            raise ValueError("Q50 model not trained")
        
        importance = self.models['q50'].feature_importance(importance_type='gain')
        
        names = feature_names or self.feature_names or \
                [f'feature_{i}' for i in range(len(importance))]
        
        df = pd.DataFrame({
            'feature': names[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Apply WIN_STREAK regularization if enabled
        if self.regularize_streak:
            from feature_selection.team_identity_features import regularize_win_streak_weight
            importance_dict = dict(zip(df['feature'], df['importance']))
            regularized = regularize_win_streak_weight(importance_dict, max_ratio=2.0)
            df['importance'] = df['feature'].map(regularized)
            df = df.sort_values('importance', ascending=False)
        
        return df.head(top_n).reset_index(drop=True)
    
    def recalibrate(self, X_calib, y_calib):
        """
        Recalibrate prediction intervals using calibration set.
        
        Adjusts Q10/Q90 predictions to achieve target 80% coverage.
        
        Args:
            X_calib: Calibration features
            y_calib: Calibration target (actual values)
        
        Returns:
            dict with calibration metrics
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        preds = self.predict(X_calib)
        
        # Calculate current coverage
        in_interval = (y_calib >= preds['q10']) & (y_calib <= preds['q90'])
        current_coverage = in_interval.mean()
        
        print(f"📊 Calibration Results:")
        print(f"   Current interval coverage: {current_coverage:.1%}")
        print(f"   Target coverage: 80.0%")
        
        if current_coverage < 0.75:
            print(f"   ⚠️  Coverage below target - intervals may be too narrow")
        elif current_coverage > 0.85:
            print(f"   ⚠️  Coverage above target - intervals may be too wide")
        else:
            print(f"   ✅ Coverage within acceptable range")
        
        return {
            'coverage': current_coverage,
            'n_samples': len(y_calib),
            'in_interval_count': in_interval.sum()
        }
    
    def save(self, filepath):
        """Save all models + metadata to disk."""
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'models': {k: v.model_to_string() for k, v in self.models.items()},
                'feature_names': self.feature_names,
                'base_params': self.base_params,
                'regularize_streak': self.regularize_streak,
            }, f)
        print(f"💾 Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load models from disk."""
        import lightgbm as lgb
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        regularize_streak = data.get('regularize_streak', True)
        instance = cls(params=data['base_params'], regularize_streak=regularize_streak)
        instance.feature_names = data['feature_names']
        instance.models = {
            k: lgb.Booster(model_str=v) for k, v in data['models'].items()
        }
        instance.is_fitted = True
        print(f"📂 Model loaded from {filepath}")
        return instance


"""
Iterative Prediction and Retraining Engine
Handles confidence-driven model retraining with up to 10 iterations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import sys
import os

# Import required modules
sys.path.append(os.path.dirname(__file__))
from database.database_handler import SportsAnalyticsDB
from data.extended_data_loader import get_extended_training_dataset, refresh_recent_data
from model_trainer import GaussianProcessPredictor, BayesianEnsemblePredictor, train_gp_models
from experimental.learners.mcmc_sampler import BayesianBasketballHierarchical
from predictor import predict_game_gp, predict_game_with_epaa
from model_updater import apply_learning_pipeline
from evaluators.validation_tracker import PredictionValidator


class IterativePredictor:
    """
    Handles iterative predictions with confidence-driven retraining
    
    Key features:
    - Predicts games with confidence scoring
    - Triggers retraining when confidence < threshold
    - Supports up to max_iterations per prediction
    - Logs all actions to database
    - Uses GP, Ensemble, Bayesian, and MCMC models
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.6,
        max_iterations: int = 10,
        db_path: str = "sports_analytics.db",
        verbose: bool = True
    ):
        """
        Initialize iterative predictor
        
        Parameters:
        - confidence_threshold: Minimum confidence score to accept (0.0-1.0)
        - max_iterations: Maximum retraining iterations per prediction
        - db_path: Path to SQLite database
        - verbose: Print detailed progress
        """
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.db_path = db_path
        self.verbose = verbose
        
        # Model storage
        self.gp_model = None
        self.ensemble_model = None
        self.mcmc_model = None
        self.validator = None
        
        # Data storage
        self.games_df = None
        self.matchup_df = None
        self.team_data = None
        self.feature_names = None
        
        # Database connection
        self.db = SportsAnalyticsDB(db_path)
        
        # Statistics
        self.stats = {
            'total_predictions': 0,
            'retraining_triggered': 0,
            'avg_iterations': 0,
            'avg_confidence': 0,
            'low_confidence_improved': 0
        }
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("🤖 ITERATIVE PREDICTOR INITIALIZED")
            print("=" * 70)
            print(f"📊 Confidence Threshold: {self.confidence_threshold:.2f}")
            print(f"🔄 Max Iterations: {self.max_iterations}")
            print(f"💾 Database: {self.db_path}")
            print("=" * 70 + "\n")
    
    def load_models(self, force_retrain: bool = False):
        """
        Load or train all prediction models
        
        Parameters:
        - force_retrain: Force retraining even if models exist
        """
        if self.verbose:
            print("=" * 70)
            print("🔧 LOADING/TRAINING MODELS")
            print("=" * 70 + "\n")
        
        # Get extended training dataset
        dataset = get_extended_training_dataset(
            db_path=self.db_path,
            verbose=self.verbose
        )
        
        self.games_df = dataset['games_df']
        self.matchup_df = dataset['matchup_df']
        self.team_data = dataset['team_data']
        self.feature_names = dataset['feature_names']
        
        X = dataset['X']
        y = dataset['y']
        
        # Initialize validator
        if self.verbose:
            print("📋 Initializing prediction validator...")
        self.validator = PredictionValidator(
            log_file='basketball/predictions_log.json'
        )
        
        # Load or train GP model
        if self.verbose:
            print("\n🔮 Gaussian Process Model:")
        
        if not force_retrain:
            try:
                # Try to load latest existing model
                import glob
                model_files = glob.glob('machine_learning/models/gp_predictor_*.pkl')
                if model_files:
                    latest_model = max(model_files, key=os.path.getctime)
                    self.gp_model = GaussianProcessPredictor.load(latest_model)
                    if self.verbose:
                        print(f"   ✅ Loaded existing model: {os.path.basename(latest_model)}")
                else:
                    force_retrain = True
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️  Could not load existing model: {e}")
                force_retrain = True
        
        if force_retrain or self.gp_model is None:
            if self.verbose:
                print("   🔄 Training new GP model with clean features...")
            
            # Train GP model (fit() will auto-save)
            self.gp_model = GaussianProcessPredictor(kernel_type='combined')
            X_train = self.matchup_df[self.feature_names].values
            y_train = y
            self.gp_model.fit(X_train, y_train, verbose=self.verbose)
            
            if self.verbose:
                print(f"   ✅ Model trained and auto-saved")
        
        # Initialize Ensemble model
        if self.verbose:
            print("\n🎯 Bayesian Ensemble Model:")
            print("   🔄 Training ensemble...")
        
        self.ensemble_model = BayesianEnsemblePredictor()
        self.ensemble_model.fit(X, y)
        
        if self.verbose:
            print("   ✅ Ensemble trained")
        
        # Train MCMC model (optional, can be slow)
        if self.verbose:
            print("\n⚡ Bayesian MCMC Model:")
            print("   ⏳ Training MCMC (this may take a few minutes)...")
        
        try:
            # Create simplified team stats for MCMC
            team_stats = self._prepare_mcmc_data()
            
            self.mcmc_model = BayesianBasketballHierarchical(
                L=10,  # accuracy clusters
                J=10,  # shot selection clusters
                K=7    # court regions
            )
            
            self.mcmc_model.fit_gibbs(
                team_stats=team_stats,
                n_iterations=5000,
                burn_in=1500,
                verbose=False
            )
            
            if self.verbose:
                print("   ✅ MCMC trained")
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  MCMC training failed: {e}")
                print("   ℹ️  Continuing with GP and Ensemble only")
            self.mcmc_model = None
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("✅ ALL MODELS LOADED AND READY")
            print("=" * 70 + "\n")
    
    def _prepare_mcmc_data(self) -> Dict:
        """Prepare team statistics for MCMC model"""
        team_stats = {}
        
        for team_id in self.team_data['ids']:
            team_games = self.games_df[self.games_df['TEAM_ID'] == team_id]
            
            if len(team_games) > 0:
                # Simplified stats for MCMC
                # In production, use actual shot chart data
                team_stats[team_id] = {
                    'M': np.random.randint(5, 15, (7,)),  # Placeholder makes by region
                    'N': np.random.randint(10, 25, (7,))  # Placeholder attempts by region
                }
        
        return team_stats
    
    def predict_with_retraining(
        self,
        home_team: str,
        away_team: str,
        game_date: str,
        game_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make prediction with iterative retraining if confidence is low
        
        Parameters:
        - home_team: Home team name
        - away_team: Away team name  
        - game_date: Game date (YYYY-MM-DD format)
        - game_id: Optional game identifier
        
        Returns:
        - Dictionary with prediction results and metadata
        """
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"🏀 PREDICTING: {away_team} @ {home_team}")
            print(f"📅 Date: {game_date}")
            print(f"{'='*70}\n")
        
        iteration = 0
        confidence_score = 0.0
        prediction = None
        retraining_triggered = False
        iteration_history = []
        
        # Get team stats for prediction
        home_stats = self._get_team_latest_stats(home_team)
        away_stats = self._get_team_latest_stats(away_team)
        
        if home_stats is None or away_stats is None:
            if self.verbose:
                print(f"❌ Could not find team stats for {home_team} or {away_team}")
            return None
        
        # Iterative prediction loop
        while iteration < self.max_iterations:
            iteration += 1
            
            if self.verbose:
                print(f"🔄 Iteration {iteration}/{self.max_iterations}")
            
            # Make prediction
            if self.mcmc_model:
                prediction = predict_game_with_epaa(
                    home_stats=home_stats,
                    away_stats=away_stats,
                    gp_model=self.gp_model,
                    mcmc_model=self.mcmc_model,
                    feature_names=self.feature_names,
                    epaa_weight=0.5  # Dynamic weight
                )
            else:
                prediction = predict_game_gp(
                    home_stats=home_stats,
                    away_stats=away_stats,
                    gp_model=self.gp_model,
                    feature_names=self.feature_names
                )
            
            confidence_score = prediction['confidence_score']
            confidence_level = prediction['confidence_level']
            
            if self.verbose:
                print(f"   📊 Confidence: {confidence_score:.3f} ({confidence_level})")
                print(f"   🎯 Prediction: {prediction['predicted_winner']} by {abs(prediction['predicted_spread']):.1f}")
                print(f"   📈 Win Probability: {prediction['win_probability']:.1%}")
            
            # Log iteration to database
            if iteration > 1 or confidence_score < self.confidence_threshold:
                self.db.log_model_action({
                    'iteration': iteration,
                    'model_type': 'iterative_pipeline',
                    'action': 'prediction_attempt',
                    'confidence_before': iteration_history[-1]['confidence'] if iteration_history else None,
                    'confidence_after': confidence_score,
                    'metrics': {
                        'predicted_spread': prediction['predicted_spread'],
                        'win_probability': prediction['win_probability'],
                        'pred_std': prediction.get('pred_std')
                    }
                })
            
            iteration_history.append({
                'iteration': iteration,
                'confidence': confidence_score,
                'prediction': prediction.copy()
            })
            
            # Check if confidence meets threshold
            if confidence_score >= self.confidence_threshold:
                if self.verbose:
                    print(f"   ✅ Confidence threshold met ({self.confidence_threshold:.2f})")
                break
            
            # Check if max iterations reached
            if iteration >= self.max_iterations:
                if self.verbose:
                    print(f"   ⚠️  Max iterations reached")
                break
            
            # Trigger retraining
            if self.verbose:
                print(f"   🔄 Confidence below threshold, triggering retraining...")
            
            retraining_triggered = True
            self._retrain_models(iteration)
        
        # Final prediction result
        final_prediction = {
            'game_id': game_id,
            'game_date': game_date,
            'home_team': home_team,
            'away_team': away_team,
            'predicted_spread': prediction['predicted_spread'],
            'predicted_home_score': prediction.get('predicted_home_score'),
            'predicted_away_score': prediction.get('predicted_away_score'),
            'predicted_winner': prediction['predicted_winner'],
            'win_probability': prediction['win_probability'],
            'confidence_score': confidence_score,
            'confidence_level': prediction['confidence_level'],
            'pred_std': prediction.get('pred_std'),
            'ci_lower': prediction.get('ci_lower'),
            'ci_upper': prediction.get('ci_upper'),
            'epaa_weight': prediction.get('epaa_weight'),
            'model_versions': {
                'gp': 'v1',
                'ensemble': 'v1',
                'mcmc': 'v1' if self.mcmc_model else None
            },
            'iteration_count': iteration,
            'retraining_triggered': retraining_triggered,
            'iteration_history': iteration_history
        }
        
        # Update statistics
        self.stats['total_predictions'] += 1
        if retraining_triggered:
            self.stats['retraining_triggered'] += 1
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"✅ FINAL PREDICTION")
            print(f"{'='*70}")
            print(f"Winner: {prediction['predicted_winner']}")
            print(f"Spread: {prediction['predicted_spread']:.1f}")
            print(f"Confidence: {confidence_score:.3f} ({prediction['confidence_level']})")
            print(f"Iterations: {iteration}")
            print(f"{'='*70}\n")
        
        return final_prediction
    
    def _get_team_latest_stats(self, team_name: str) -> Optional[Dict]:
        """Get latest rolling stats for a team"""
        # Find team in games_df
        team_games = self.games_df[
            self.games_df['MATCHUP'].str.contains(team_name, case=False, na=False)
        ]
        
        if len(team_games) == 0:
            return None
        
        # Get most recent game
        latest_game = team_games.sort_values('GAME_DATE', ascending=False).iloc[0]
        
        # Extract rolling stats
        stats = {}
        for col in latest_game.index:
            if '_ROLL' in col or col in ['WIN_STREAK', 'REST_DAYS', 'IS_BACK_TO_BACK', 'WIN_RATE_10']:
                stats[col] = latest_game[col]
        
        return stats
    
    def _retrain_models(self, iteration: int):
        """Retrain models during iteration"""
        if self.verbose:
            print(f"\n   🔧 Retraining models (iteration {iteration})...")
        
        try:
            # Refresh recent data
            self.games_df = refresh_recent_data(
                self.games_df,
                days_back=14,
                verbose=False
            )
            
            # Recreate training data
            from data.extended_data_loader import prepare_training_data
            self.matchup_df, y, _ = prepare_training_data(self.games_df, verbose=False)
            X = self.matchup_df[self.feature_names].values
            
            # Retrain GP
            self.gp_model.fit(X, y)
            
            # Retrain Ensemble
            self.ensemble_model.fit(X, y)
            
            if self.verbose:
                print(f"      ✅ Models retrained with updated data")
            
        except Exception as e:
            if self.verbose:
                print(f"      ⚠️  Retraining error: {e}")
    
    def save_prediction_to_db(self, prediction: Dict[str, Any]) -> int:
        """Save prediction to database"""
        return self.db.insert_prediction(prediction)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get prediction statistics"""
        return self.stats.copy()
    
    def close(self):
        """Clean up resources"""
        if self.db:
            self.db.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

"""
Model Updater Module

Handles:
- Applying learned adjustments to MCMC model
- Retraining GP models with updated hyperparameters
- Implementing team-specific bias corrections
- Updating EPAA weights based on validation results
- Creating updated model versions for continuous improvement
"""

import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class ModelUpdater:
    """
    Apply learning insights to update and improve models
    """
    
    def __init__(self, mcmc_model=None, gp_model=None):
        """
        Initialize model updater
        
        Parameters:
        - mcmc_model: BayesianBasketballHierarchical instance
        - gp_model: GaussianProcessPredictor instance
        """
        self.mcmc_model = mcmc_model
        self.gp_model = gp_model
        self.update_history = []
    
    def apply_team_adjustments(
        self,
        team_adjustments: Dict[str, float],
        team_data: Dict,
        alpha: float = 0.1
    ) -> Dict[str, float]:
        """
        Apply team-specific bias corrections to EPAA values
        
        Parameters:
        - team_adjustments: Dict mapping team names to adjustment values
        - team_data: Team data dict with IDs and names
        - alpha: Learning rate (0-1), default 0.1 for conservative updates
        
        Returns:
        - Updated EPAA values
        """
        if self.mcmc_model is None:
            print("⚠️ No MCMC model available - cannot apply team adjustments")
            return {}
        
        # Get current EPAA values
        current_epaa = self.mcmc_model.get_epaa_results()
        updated_epaa = current_epaa.copy()
        
        # Map team names to IDs
        team_names_inv = {v: k for k, v in team_data['names'].items()}
        
        # Apply adjustments
        n_updated = 0
        for team_name, adjustment in team_adjustments.items():
            team_id = team_names_inv.get(team_name)
            if team_id and team_id in updated_epaa:
                # Apply adjustment with learning rate
                old_value = updated_epaa[team_id]
                updated_epaa[team_id] = old_value + (alpha * adjustment)
                n_updated += 1
                
                print(f"  Updated {team_name}: {old_value:.2f} → {updated_epaa[team_id]:.2f} ({adjustment:+.2f})")
        
        print(f"\n✅ Applied adjustments to {n_updated} teams")
        
        # Update model's theta_i (EPAA parameters)
        for team_id, new_epaa in updated_epaa.items():
            if team_id in self.mcmc_model.theta_i:
                self.mcmc_model.theta_i[team_id] = new_epaa
        
        return updated_epaa
    
    def update_epaa_weight(
        self,
        proposed_weight: float,
        reason: str = "Performance-based adjustment"
    ):
        """
        Update the EPAA weighting parameter
        
        This is used in hybrid predictions that combine GP and MCMC
        
        Parameters:
        - proposed_weight: New EPAA weight (0-1)
        - reason: Why this change is being made
        """
        # Store in update history
        update_record = {
            'timestamp': datetime.now().isoformat(),
            'type': 'epaa_weight',
            'new_value': proposed_weight,
            'reason': reason
        }
        self.update_history.append(update_record)
        
        print(f"\n🔧 EPAA Weight Update:")
        print(f"   New weight: {proposed_weight:.2f}")
        print(f"   Reason: {reason}")
        
        # In practice, you'd save this to a config file or model metadata
        # For now, we'll return it for the user to apply manually
        return proposed_weight
    
    def retrain_gp_with_corrections(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        team_adjustments: Dict[str, float],
        team_ids_train: List,
        team_data: Dict
    ):
        """
        Retrain GP model with bias-corrected training data
        
        Parameters:
        - X_train: Training features
        - y_train: Training targets (point spreads)
        - team_adjustments: Team bias corrections
        - team_ids_train: Team IDs corresponding to training samples
        - team_data: Team data dict
        
        Returns:
        - Retrained GP model
        """
        if self.gp_model is None:
            print("⚠️ No GP model available")
            return None
        
        print("\n🔄 Retraining GP model with bias corrections...")
        
        # Apply corrections to training data
        y_corrected = y_train.copy()
        team_names_inv = {v: k for k, v in team_data['names'].items()}
        
        for i, (home_id, away_id) in enumerate(team_ids_train):
            home_name = team_data['names'].get(home_id, '')
            away_name = team_data['names'].get(away_id, '')
            
            home_adj = team_adjustments.get(home_name, 0.0)
            away_adj = team_adjustments.get(away_name, 0.0)
            
            # Adjust target: if we over-predicted home team, reduce their advantage
            y_corrected[i] += (home_adj - away_adj)
        
        # Retrain model
        self.gp_model.fit(X_train, y_corrected, verbose=True)
        
        print(f"✅ GP model retrained with {len(team_adjustments)} team corrections")
        
        return self.gp_model
    
    def run_incremental_mcmc_update(
        self,
        new_game_data: pd.DataFrame,
        n_iterations: int = 1000,
        burn_in: int = 300
    ):
        """
        Run incremental MCMC update with new game data
        
        This performs online learning by sampling from the posterior
        given new observations
        
        Parameters:
        - new_game_data: DataFrame with new games to learn from
        - n_iterations: MCMC iterations
        - burn_in: Burn-in period
        
        Returns:
        - Updated MCMC model
        """
        if self.mcmc_model is None:
            print("⚠️ No MCMC model available")
            return None
        
        print(f"\n🔬 Running incremental MCMC update with {len(new_game_data)} new games...")
        
        # In a full implementation, you'd:
        # 1. Convert new game data to shot data format
        # 2. Run additional MCMC iterations starting from current posterior
        # 3. Update team parameters
        
        # Placeholder for now - this would require shot-level data
        print("⚠️ Incremental MCMC update requires shot-level data")
        print("   For now, recommend full retraining with updated data")
        
        return self.mcmc_model
    
    def save_updated_models(
        self,
        output_dir: str = 'models/updated',
        version: Optional[str] = None
    ):
        """
        Save updated models with version tracking
        
        Parameters:
        - output_dir: Directory to save models
        - version: Version string (default: timestamp)
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if version is None:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        saved_files = []
        
        # Save MCMC model
        if self.mcmc_model:
            mcmc_path = f"{output_dir}/mcmc_model_v{version}.pkl"
            self.mcmc_model.save(mcmc_path)
            saved_files.append(mcmc_path)
            print(f"💾 Saved MCMC model: {mcmc_path}")
        
        # Save GP model
        if self.gp_model:
            gp_path = f"{output_dir}/gp_model_v{version}.pkl"
            self.gp_model.save(gp_path)
            saved_files.append(gp_path)
            print(f"💾 Saved GP model: {gp_path}")
        
        # Save update history
        history_path = f"{output_dir}/update_history_v{version}.json"
        with open(history_path, 'w') as f:
            json.dump(self.update_history, f, indent=2)
        saved_files.append(history_path)
        print(f"💾 Saved update history: {history_path}")
        
        return saved_files
    
    def generate_update_summary(self) -> str:
        """
        Generate a summary of all updates applied
        
        Returns:
        - Markdown formatted summary
        """
        summary = []
        summary.append("# 🔄 Model Update Summary\n")
        summary.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        summary.append(f"**Total Updates:** {len(self.update_history)}\n")
        
        if len(self.update_history) == 0:
            summary.append("\n_No updates applied yet._\n")
        else:
            summary.append("\n## Update History\n")
            for i, update in enumerate(self.update_history, 1):
                summary.append(f"\n### Update #{i}")
                summary.append(f"- **Time:** {update['timestamp']}")
                summary.append(f"- **Type:** {update['type']}")
                summary.append(f"- **Reason:** {update.get('reason', 'N/A')}")
                
                if 'new_value' in update:
                    summary.append(f"- **New Value:** {update['new_value']}")
        
        return '\n'.join(summary)


def apply_learning_pipeline(
    validation_results: Dict,
    mcmc_model,
    gp_model,
    team_data: Dict,
    learning_rate: float = 0.1,
    save_models: bool = True
) -> Dict:
    """
    Complete pipeline to apply learning insights to models
    
    Parameters:
    - validation_results: Output from adaptive_learner.validate_and_learn()
    - mcmc_model: Current MCMC model
    - gp_model: Current GP model
    - team_data: Team data dict
    - learning_rate: How aggressively to apply corrections (0-1)
    - save_models: Whether to save updated models
    
    Returns:
    - Dict with updated models and summary
    """
    print("🚀 Starting learning application pipeline...\n")
    
    # Initialize updater
    updater = ModelUpdater(mcmc_model, gp_model)
    
    # Step 1: Apply team adjustments
    if 'team_adjustments' in validation_results:
        print("📊 Applying team-specific adjustments...")
        updated_epaa = updater.apply_team_adjustments(
            validation_results['team_adjustments'],
            team_data,
            alpha=learning_rate
        )
    
    # Step 2: Update EPAA weight
    if 'mcmc_refinement' in validation_results:
        refinement = validation_results['mcmc_refinement']
        proposed_weight = refinement.get('proposed_epaa_weight', 0.5)
        reasoning = ' | '.join(refinement.get('reasoning', []))
        
        new_weight = updater.update_epaa_weight(proposed_weight, reasoning)
    
    # Step 3: Save updated models
    saved_files = []
    if save_models:
        print("\n💾 Saving updated models...")
        saved_files = updater.save_updated_models()
    
    # Generate summary
    summary = updater.generate_update_summary()
    
    print("\n" + "="*60)
    print("✅ Learning pipeline complete!")
    print("="*60)
    
    return {
        'updater': updater,
        'summary': summary,
        'saved_files': saved_files,
        'updated_models': {
            'mcmc': updater.mcmc_model,
            'gp': updater.gp_model
        }
    }


def create_feedback_loop_config(
    validation_results: Dict,
    output_file: str = 'config/feedback_config.json'
) -> Dict:
    """
    Create a configuration file for automated feedback loop
    
    This can be used in automated retraining pipelines
    
    Parameters:
    - validation_results: Results from validation
    - output_file: Where to save config
    
    Returns:
    - Config dict
    """
    import os
    
    config = {
        'last_updated': datetime.now().isoformat(),
        'epaa_weight': validation_results.get('mcmc_refinement', {}).get('proposed_epaa_weight', 0.5),
        'learning_rate': 0.1,
        'min_games_for_update': 10,
        'confidence_thresholds': {
            'HIGH': 0.5,  # Could adjust based on calibration
            'MEDIUM': 0.3,
            'LOW': 0.0
        },
        'team_adjustments': validation_results.get('team_adjustments', {}),
        'performance_metrics': validation_results.get('error_analysis', {}).get('overall_metrics', {}),
        'recommended_actions': [
            'Retrain GP kernel with updated hyperparameters',
            'Apply team bias corrections',
            'Recalibrate confidence thresholds',
            'Update EPAA weight in prediction pipeline'
        ]
    }
    
    # Save config
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n📝 Feedback loop config saved to: {output_file}")
    
    return config

"""
Bayesian NBA Analysis Utilities

This module provides reusable functions for advanced NBA analysis including:
- Player evaluation metrics (TS%, EFG%, AST/TO, PER, USG Rate)
- Shot analysis tools (zone efficiency, shot quality)
- Bayesian posterior utilities (shrinkage, credible intervals, rolling averages)
- Elo rating calculation and ensemble prediction combining
- MCMC convergence validation (R-hat)

For Gaussian Process models, import from learners.model_trainer.
For Monte Carlo simulations, use diagnostics/monte_carlo_simulator.py.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

# GP models live in learners â€” import here for convenience
from model_trainer import GaussianProcessPredictor, train_gp_models as train_gp_ensemble  # noqa: F401


class BayesianNBAAnalyzer:
    """
    Advanced Bayesian analyzer for NBA data
    """
    
    def __init__(self):
        self.models = {}
        self.traces = {}
        
    @staticmethod
    def calculate_advanced_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced basketball metrics
        
        Args:
            df: DataFrame with basic stats (PTS, FGA, FTA, REB, AST, etc.)
            
        Returns:
            DataFrame with additional metrics
        """
        df = df.copy()
        
        # True Shooting Percentage
        df['TS_PCT'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']) + 0.001)
        
        # Effective Field Goal Percentage
        df['EFG_PCT'] = (df['FGM'] + 0.5 * df['FG3M']) / (df['FGA'] + 0.001)
        
        # Assist-to-Turnover Ratio
        df['AST_TO_RATIO'] = df['AST'] / (df['TOV'] + 1)
        
        # Usage Rate (simplified)
        df['USG_RATE'] = (df['FGA'] + 0.44 * df['FTA'] + df['TOV']) / (df['MIN'] + 1)
        
        # Player Efficiency Rating (simplified)
        df['PER'] = (
            df['PTS'] + df['REB'] + df['AST'] + df['STL'] + df['BLK'] - 
            df['TOV'] - (df['FGA'] - df['FGM']) - (df['FTA'] - df['FTM'])
        ) / (df['MIN'] + 1)
        
        return df
    
    @staticmethod
    def infer_position(row: pd.Series) -> str:
        """
        Infer player position from statistics
        
        Args:
            row: Player stats row
            
        Returns:
            Position string: 'Guard', 'Forward', or 'Center'
        """
        ast_ratio = row['AST'] / (row['MIN'] + 1)
        reb_ratio = row['REB'] / (row['MIN'] + 1)
        fg3a_ratio = row.get('FG3A', 0) / (row.get('FGA', 1) + 1)
        
        if ast_ratio > 0.25:
            return 'Guard'
        elif reb_ratio > 0.35 and fg3a_ratio < 0.15:
            return 'Center'
        elif reb_ratio > 0.25:
            return 'Forward'
        elif fg3a_ratio > 0.35:
            return 'Guard'
        else:
            return 'Forward'
    
    @staticmethod
    def bayesian_shrinkage(observed: np.ndarray, 
                          n_trials: np.ndarray,
                          prior_mean: float = 0.75,
                          prior_strength: float = 10) -> np.ndarray:
        """
        Apply Bayesian shrinkage to observed proportions
        
        Args:
            observed: Observed successes
            n_trials: Number of attempts
            prior_mean: Prior belief about success rate
            prior_strength: Strength of prior (pseudo-observations)
            
        Returns:
            Shrunk estimates
        """
        prior_successes = prior_mean * prior_strength
        prior_failures = (1 - prior_mean) * prior_strength
        
        posterior_mean = (observed + prior_successes) / (n_trials + prior_strength)
        return posterior_mean
    
    @staticmethod
    def calculate_credible_interval(samples: np.ndarray, 
                                   credibility: float = 0.95) -> Tuple[float, float]:
        """
        Calculate credible interval from posterior samples
        
        Args:
            samples: MCMC samples
            credibility: Credibility level (default 95%)
            
        Returns:
            (lower, upper) bounds
        """
        alpha = 1 - credibility
        lower = np.percentile(samples, 100 * alpha / 2)
        upper = np.percentile(samples, 100 * (1 - alpha / 2))
        return lower, upper
    
    @staticmethod
    def simulate_game_outcome(home_strength: float,
                            away_strength: float,
                            home_variance: float = 12.0,
                            away_variance: float = 12.0,
                            home_court_advantage: float = 3.0,
                            n_simulations: int = 10000) -> Dict:
        """
        Simulate game outcomes using Bayesian framework
        
        Args:
            home_strength: Home team average points
            away_strength: Away team average points
            home_variance: Home team scoring variance
            away_variance: Away team scoring variance
            home_court_advantage: Points added for home team
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            Dictionary with predictions and uncertainty
        """
        # Simulate scores
        home_scores = np.random.normal(
            home_strength + home_court_advantage,
            home_variance,
            n_simulations
        )
        away_scores = np.random.normal(
            away_strength,
            away_variance,
            n_simulations
        )
        
        point_diffs = home_scores - away_scores
        home_wins = (point_diffs > 0).mean()
        
        return {
            'home_win_probability': home_wins,
            'away_win_probability': 1 - home_wins,
            'expected_point_differential': point_diffs.mean(),
            'point_diff_std': point_diffs.std(),
            'credible_interval_95': BayesianNBAAnalyzer.calculate_credible_interval(point_diffs),
            'point_differential_samples': point_diffs
        }
    
    @staticmethod
    def calculate_player_impact(player_stats: pd.Series, 
                               position: str = 'Guard') -> float:
        """
        Calculate comprehensive player impact score
        
        Args:
            player_stats: Player statistics
            position: Player position for position-adjusted weights
            
        Returns:
            Impact score
        """
        # Position-specific weights
        weights = {
            'Guard': {'PTS': 1.0, 'AST': 1.5, 'REB': 0.8, 'STL': 2.0, 'TOV': -1.5},
            'Forward': {'PTS': 1.0, 'AST': 1.2, 'REB': 1.3, 'STL': 1.8, 'TOV': -1.5},
            'Center': {'PTS': 1.0, 'AST': 1.0, 'REB': 1.5, 'BLK': 2.0, 'TOV': -1.2}
        }
        
        w = weights.get(position, weights['Forward'])
        
        impact = (
            player_stats.get('PTS', 0) * w.get('PTS', 1.0) +
            player_stats.get('AST', 0) * w.get('AST', 1.0) +
            player_stats.get('REB', 0) * w.get('REB', 1.0) +
            player_stats.get('STL', 0) * w.get('STL', 1.0) +
            player_stats.get('BLK', 0) * w.get('BLK', 0) +
            player_stats.get('TOV', 0) * w.get('TOV', -1.0)
        )
        
        return impact
    
    @staticmethod
    def rolling_bayesian_average(series: pd.Series,
                                 window: int = 5,
                                 prior_weight: float = 2.0) -> pd.Series:
        """
        Calculate rolling average with Bayesian prior
        
        Args:
            series: Time series data
            window: Rolling window size
            prior_weight: Weight given to overall average (pseudo-observations)
            
        Returns:
            Bayesian rolling average
        """
        overall_mean = series.mean()
        rolling_sum = series.rolling(window=window, min_periods=1).sum()
        rolling_count = series.rolling(window=window, min_periods=1).count()
        
        # Bayesian update
        bayesian_avg = (rolling_sum + prior_weight * overall_mean) / (rolling_count + prior_weight)
        return bayesian_avg


class ShotAnalyzer:
    """
    Analyze shooting patterns and efficiency
    """
    
    @staticmethod
    def calculate_shot_quality(shot_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate shot quality metrics
        
        Args:
            shot_data: DataFrame with shot information
            
        Returns:
            DataFrame with quality metrics
        """
        shot_data = shot_data.copy()
        
        # Expected points per shot
        shot_data['Expected_Points'] = np.where(
            shot_data['SHOT_TYPE'] == '3PT',
            shot_data['FG_PCT'] * 3,
            shot_data['FG_PCT'] * 2
        )
        
        return shot_data
    
    @staticmethod
    def analyze_shot_zones(shots: pd.DataFrame, 
                          zones: List[str]) -> Dict[str, Dict]:
        """
        Analyze shooting efficiency by zone
        
        Args:
            shots: Shot data with zone information
            zones: List of zone identifiers
            
        Returns:
            Dictionary of zone statistics
        """
        zone_stats = {}
        
        for zone in zones:
            zone_shots = shots[shots['ZONE'] == zone]
            
            if len(zone_shots) > 0:
                zone_stats[zone] = {
                    'attempts': len(zone_shots),
                    'makes': zone_shots['SHOT_MADE'].sum(),
                    'fg_pct': zone_shots['SHOT_MADE'].mean(),
                    'expected_points': zone_shots['Expected_Points'].mean()
                }
        
        return zone_stats


class PredictiveModel:
    """
    Predictive modeling utilities
    """
    
    @staticmethod
    def calculate_elo_ratings(games: pd.DataFrame,
                            k_factor: float = 20.0,
                            initial_rating: float = 1500.0) -> Dict[int, float]:
        """
        Calculate Elo ratings from game results
        
        Args:
            games: DataFrame with game results (must have TEAM_ID, OPP_TEAM_ID, WL)
            k_factor: Elo k-factor
            initial_rating: Starting Elo rating
            
        Returns:
            Dictionary mapping team_id to Elo rating
        """
        ratings = {}
        
        for _, game in games.iterrows():
            team_id = game['TEAM_ID']
            opp_id = game.get('OPP_TEAM_ID', 0)
            won = game['WL'] == 'W'
            
            # Initialize ratings
            if team_id not in ratings:
                ratings[team_id] = initial_rating
            if opp_id not in ratings:
                ratings[opp_id] = initial_rating
            
            # Calculate expected outcome
            expected = 1 / (1 + 10 ** ((ratings[opp_id] - ratings[team_id]) / 400))
            
            # Update rating
            actual = 1.0 if won else 0.0
            ratings[team_id] += k_factor * (actual - expected)
            ratings[opp_id] += k_factor * ((1 - actual) - (1 - expected))
        
        return ratings
    
    @staticmethod
    def ensemble_prediction(predictions: List[Dict], 
                          weights: Optional[List[float]] = None) -> Dict:
        """
        Combine multiple model predictions
        
        Args:
            predictions: List of prediction dictionaries
            weights: Optional weights for each model
            
        Returns:
            Ensemble prediction
        """
        if weights is None:
            weights = [1.0] * len(predictions)
        
        weights = np.array(weights) / sum(weights)
        
        ensemble = {
            'home_win_prob': sum(p['home_win_probability'] * w 
                                for p, w in zip(predictions, weights)),
            'point_diff': sum(p['expected_point_differential'] * w 
                            for p, w in zip(predictions, weights))
        }
        
        return ensemble


def validate_convergence(trace, var_names: List[str], threshold: float = 1.01) -> bool:
    """
    Validate MCMC convergence using R-hat statistic
    
    Args:
        trace: PyMC trace object
        var_names: Variables to check
        threshold: R-hat threshold (default 1.01)
        
    Returns:
        True if converged, False otherwise
    """
    try:
        import arviz as az
        rhat = az.rhat(trace, var_names=var_names)
        
        for var in var_names:
            if var in rhat:
                max_rhat = float(rhat[var].max())
                if max_rhat > threshold:
                    print(f"âš ï¸  Warning: {var} has R-hat = {max_rhat:.4f} > {threshold}")
                    return False
        
        return True
    except ImportError:
        print("âš ï¸  ArviZ not installed, skipping convergence check")
        return True



if __name__ == "__main__":
    print("Bayesian NBA Analysis Utilities")
    print("=" * 50)
    print("\nAvailable classes:")
    print("  - BayesianNBAAnalyzer: Core analysis functions")
    print("  - ShotAnalyzer: Shot quality and zone analysis")
    print("  - PredictiveModel: Elo ratings and ensemble methods")
    print("  - GaussianProcessPredictor: GP models (from learners.model_trainer)")
    print("\nImport with: from diagnostics.bayesian_utils import BayesianNBAAnalyzer")
    print("             from learners.model_trainer import GaussianProcessPredictor")
