"""
Bayesian NBA Analysis Utilities

This module provides reusable functions for advanced NBA analysis including:
- Hierarchical Bayesian models
- MCMC sampling helpers
- Player evaluation metrics
- Shot analysis tools
- Uncertainty quantification
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


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
                    print(f"⚠️  Warning: {var} has R-hat = {max_rhat:.4f} > {threshold}")
                    return False
        
        return True
    except ImportError:
        print("⚠️  ArviZ not installed, skipping convergence check")
        return True


class GaussianProcessPredictor:
    """
    Gaussian Process models for NBA predictions with uncertainty quantification
    
    Provides flexible, non-parametric regression with:
    - Multiple kernel options (RBF, Matérn, RationalQuadratic)
    - Predictive mean and variance
    - Confidence intervals
    - Model persistence
    """
    
    def __init__(self, kernel_type='rbf', length_scale=1.0, noise_level=0.1, random_state=42):
        """
        Initialize GP model
        
        Args:
            kernel_type: 'rbf', 'matern', 'rq' (RationalQuadratic), or 'combined'
            length_scale: Length scale for kernels
            noise_level: Noise level (alpha parameter)
            random_state: Random seed
        """
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            RBF, Matern, RationalQuadratic, WhiteKernel, ConstantKernel as C
        )
        
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
            n_restarts_optimizer=10,
            alpha=1e-10,  # Regularization
            random_state=random_state,
            normalize_y=True
        )
        
        self.scaler = None
        self.is_fitted = False
        
    def fit(self, X, y):
        """
        Fit GP model to training data
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
        """
        from sklearn.preprocessing import StandardScaler
        
        print(f"🔬 Training Gaussian Process ({self.kernel_type} kernel)...")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        print(f"   ✓ Kernel: {self.model.kernel_}")
        print(f"   ✓ Log-marginal-likelihood: {self.model.log_marginal_likelihood(self.model.kernel_.theta):.2f}")
        
        return self
    
    def predict(self, X, return_std=True, return_cov=False):
        """
        Make predictions with uncertainty
        
        Args:
            X: Feature matrix
            return_std: Return standard deviations
            return_cov: Return full covariance matrix
            
        Returns:
            predictions: Mean predictions
            std or cov: Standard deviations or covariance (if requested)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        if return_cov:
            mean, cov = self.model.predict(X_scaled, return_cov=True)
            return mean, cov
        elif return_std:
            mean, std = self.model.predict(X_scaled, return_std=True)
            return mean, std
        else:
            return self.model.predict(X_scaled)
    
    def get_confidence_intervals(self, X, confidence=0.95):
        """
        Get confidence intervals for predictions
        
        Args:
            X: Feature matrix
            confidence: Confidence level (0.95 = 95%)
            
        Returns:
            mean, lower, upper
        """
        from scipy import stats
        
        mean, std = self.predict(X, return_std=True)
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        lower = mean - z_score * std
        upper = mean + z_score * std
        
        return mean, lower, upper
    
    def score(self, X, y):
        """
        Calculate R² score on test data
        
        Args:
            X: Test features
            y: True values
            
        Returns:
            R² score
        """
        from sklearn.metrics import r2_score
        predictions = self.predict(X, return_std=False)
        return r2_score(y, predictions)
    
    def save(self, filepath):
        """Save model to disk"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'kernel_type': self.kernel_type,
                'is_fitted': self.is_fitted
            }, f)
        print(f"💾 GP model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load model from disk"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(kernel_type=data['kernel_type'])
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.is_fitted = data['is_fitted']
        
        print(f"📂 GP model loaded from {filepath}")
        return instance


def train_gp_ensemble(X_train, y_train, X_test, y_test, kernel_types=['rbf', 'matern', 'rq']):
    """
    Train multiple GP models with different kernels and compare
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        kernel_types: List of kernel types to try
        
    Returns:
        results: Dict with models and performance metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    results = {
        'models': {},
        'predictions': {},
        'metrics': []
    }
    
    for kernel in kernel_types:
        print(f"\n{'='*60}")
        gp = GaussianProcessPredictor(kernel_type=kernel)
        gp.fit(X_train, y_train)
        
        # Predictions with uncertainty
        y_pred, y_std = gp.predict(X_test, return_std=True)
        
        # Metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Negative log-likelihood (NLL)
        nll = -np.mean(
            -0.5 * np.log(2 * np.pi * y_std**2) - 
            0.5 * ((y_test - y_pred)**2) / (y_std**2)
        )
        
        # Store results
        results['models'][kernel] = gp
        results['predictions'][kernel] = {'mean': y_pred, 'std': y_std}
        results['metrics'].append({
            'Kernel': kernel,
            'R²': r2,
            'RMSE': rmse,
            'MAE': mae,
            'NLL': nll,
            'Mean Uncertainty': np.mean(y_std)
        })
        
        print(f"   R² Score: {r2:.4f}")
        print(f"   RMSE: {rmse:.2f}")
        print(f"   MAE: {mae:.2f}")
        print(f"   NLL: {nll:.4f}")
        print(f"   Avg Uncertainty (σ): {np.mean(y_std):.2f}")
    
    return results


if __name__ == "__main__":
    print("Bayesian NBA Analysis Utilities")
    print("=" * 50)
    print("\nAvailable classes:")
    print("  - BayesianNBAAnalyzer: Core analysis functions")
    print("  - ShotAnalyzer: Shot quality and zone analysis")
    print("  - PredictiveModel: Elo ratings and ensemble methods")
    print("  - GaussianProcessPredictor: GP models with uncertainty")
    print("\nImport with: from bayesian_utils import BayesianNBAAnalyzer, GaussianProcessPredictor")
