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

"""
Monte Carlo Simulator for NBA Game Predictions
Runs thousands of simulations per game using model quantile outputs.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats as scipy_stats

# Import model and data loader
from learners.train_lgbm_model import LGBMWinPredictor
from loaders.data_loader import (
    fetch_nba_games,
    calculate_rolling_stats,
    get_all_nba_teams
)


def triangular_sample(q10, q50, q90, n_samples=1):
    """
    Sample from asymmetric triangular distribution using three quantiles.
    
    Parameters:
    - q10, q50, q90: The 10th, 50th, and 90th percentiles
    - n_samples: Number of samples to draw
    
    Returns:
    - Array of sampled values
    """
    # Use scipy triangular distribution
    # Mode parameter c = (mode - left) / (right - left)
    left = q10
    right = q90
    mode = q50
    
    if right <= left:
        # Degenerate case: return mode
        return np.full(n_samples, mode)
    
    c = (mode - left) / (right - left)
    c = np.clip(c, 0.01, 0.99)  # Ensure valid triangular distribution
    
    # Sample from triangular
    samples = scipy_stats.triang.rvs(c, loc=left, scale=right - left, size=n_samples)
    return samples


def get_team_latest_features(team_name, team_name_to_id, games_df_with_stats):
    """
    Get the most recent rolling stats for a team.
    
    Args:
        team_name: Full team name (e.g., 'Philadelphia 76ers')
        team_name_to_id: Dict mapping team names to IDs
        games_df_with_stats: DataFrame with rolling stats
    
    Returns:
        dict of feature_name: value
    """
    import pandas as pd
    
    team_id = team_name_to_id.get(team_name)
    if not team_id:
        return {}
    
    # Get team's games, sorted by date
    team_games = games_df_with_stats[games_df_with_stats['TEAM_ID'] == team_id].sort_values('GAME_DATE')
    
    if len(team_games) == 0:
        return {}
    
    # Get most recent game's stats
    latest = team_games.iloc[-1]
    
    # Extract rolling features
    features = {}
    for col in games_df_with_stats.columns:
        if '_ROLL' in col or col in ['WIN_STREAK', 'REST_DAYS', 'IS_BACK_TO_BACK', 'WIN_RATE_10']:
            features[col] = latest[col] if pd.notna(latest[col]) else 0.0
    
    return features


def run_monte_carlo_for_game(model, home_team, away_team, team_name_to_id, games_df_with_stats, n_simulations=10000):
    """
    Run Monte Carlo simulation for a single game.
    
    Parameters:
    - model: LGBMWinPredictor instance
    - home_team: Home team name
    - away_team: Away team name
    - team_name_to_id: Dict mapping team names to IDs
    - games_df_with_stats: DataFrame with rolling stats
    - n_simulations: Number of simulations to run
    
    Returns:
    - Dictionary with simulation results
    """
    import pandas as pd
    
    # Get latest features for both teams
    home_features = get_team_latest_features(home_team, team_name_to_id, games_df_with_stats)
    away_features = get_team_latest_features(away_team, team_name_to_id, games_df_with_stats)
    
    if not home_features or not away_features:
        raise ValueError(f"Missing features for {home_team} or {away_team}")
    
    # Create matchup feature vector in expected order
    feature_row = {}
    for col, value in home_features.items():
        feature_row[f'HOME_{col}'] = value
    for col, value in away_features.items():
        feature_row[f'AWAY_{col}'] = value
    
    # Convert to DataFrame with correct feature order
    X = pd.DataFrame([feature_row])
    
    # Align with model's feature names (fill missing with 0)
    for feat in model.feature_names:
        if feat not in X.columns:
            X[feat] = 0.0
    X = X[model.feature_names]
    
    # Get quantile predictions from model
    X_scaled = model.scaler.transform(X)
    quantiles = model.quantile_model.predict(X_scaled)
    
    q10 = quantiles['q10'][0]
    q50 = quantiles['q50'][0]
    q90 = quantiles['q90'][0]
    
    # Run simulations
    point_diffs = triangular_sample(q10, q50, q90, n_simulations)
    
    # Convert to win outcomes (home team wins if point_diff > 0)
    home_wins = (point_diffs > 0).astype(int)
    
    # Calculate statistics
    mc_win_prob = home_wins.mean()
    mc_median_spread = np.median(point_diffs)
    mc_mean_spread = np.mean(point_diffs)
    mc_std_spread = np.std(point_diffs)
    
    # Confidence intervals
    ci_95_low, ci_95_high = np.percentile(point_diffs, [2.5, 97.5])
    ci_90_low, ci_90_high = np.percentile(point_diffs, [5, 95])
    ci_80_low, ci_80_high = np.percentile(point_diffs, [10, 90])
    
    # Model's calibrated prediction for comparison
    model_pred = model.predict_win_probability(X)
    
    return {
        'mc_win_prob': mc_win_prob,
        'mc_median_spread': mc_median_spread,
        'mc_mean_spread': mc_mean_spread,
        'mc_std_spread': mc_std_spread,
        'mc_ci_95': (ci_95_low, ci_95_high),
        'mc_ci_90': (ci_90_low, ci_90_high),
        'mc_ci_80': (ci_80_low, ci_80_high),
        'model_win_prob': model_pred['win_prob'][0],
        'model_confidence': model_pred['confidence_label'][0],
        'model_confidence_score': model_pred['confidence_score'][0],
        'quantiles': {'q10': q10, 'q50': q50, 'q90': q90},
        'point_diff_samples': point_diffs,
    }


def main():
    parser = argparse.ArgumentParser(description='Run Monte Carlo simulations for NBA games')
    parser.add_argument('--model', type=str, default='machine_learning/models/lgbm_win_predictor_latest.pkl',
                        help='Path to trained LGBMWinPredictor model')
    parser.add_argument('--date', type=str, default='2026-02-19',
                        help='Date for predictions (YYYY-MM-DD)')
    parser.add_argument('--n', type=int, default=10000,
                        help='Number of Monte Carlo simulations per game')
    parser.add_argument('--out', type=str, default='diagnostics/mc_simulations.csv',
                        help='Output CSV file path')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"🎲 MONTE CARLO SIMULATOR")
    print(f"{'='*80}\n")
    print(f"📅 Target Date: {args.date}")
    print(f"🔢 Simulations per game: {args.n:,}")
    print(f"📦 Model: {args.model}\n")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = LGBMWinPredictor.load(args.model)
    print("✅ Model loaded\n")
    
    # Load latest team stats
    print("Loading latest team statistics...")
    games_df = fetch_nba_games(seasons=['2024-25'], season_type='Regular Season', verbose=False)
    games_with_stats = calculate_rolling_stats(games_df, window=5)
    print(f"✅ Loaded {len(games_with_stats)} games with rolling stats")
    print(f"   Date range: {games_with_stats['GAME_DATE'].min()} → {games_with_stats['GAME_DATE'].max()}")
    
    # Get team mappings
    team_data = get_all_nba_teams()
    team_name_to_id = {team['full_name']: team['id'] for team in team_data['teams']}
    print(f"✅ Team mappings ready for {len(team_name_to_id)} teams\n")
    
    # Define games for Feb 19, 2026 (from the notebook output)
    games = [
        {'away': 'Atlanta Hawks', 'home': 'Philadelphia 76ers', 'time': '7:00p'},
        {'away': 'Indiana Pacers', 'home': 'Washington Wizards', 'time': '7:00p'},
        {'away': 'Detroit Pistons', 'home': 'New York Knicks', 'time': '7:30p'},
        {'away': 'Toronto Raptors', 'home': 'Chicago Bulls', 'time': '8:00p'},
        {'away': 'Phoenix Suns', 'home': 'San Antonio Spurs', 'time': '8:30p'},
        {'away': 'Boston Celtics', 'home': 'Golden State Warriors', 'time': '10:00p'},
        {'away': 'Orlando Magic', 'home': 'Sacramento Kings', 'time': '10:00p'},
        {'away': 'Denver Nuggets', 'home': 'Los Angeles Clippers', 'time': '10:30p'},
    ]
    
    print(f"Running Monte Carlo simulations for {len(games)} games...\n")
    
    # Run simulations for each game
    results = []
    
    for i, game in enumerate(games, 1):
        away_team = game['away']
        home_team = game['home']
        
        print(f"{'─'*80}")
        print(f"GAME {i}/{len(games)}: {away_team} @ {home_team}")
        print(f"{'─'*80}")
        
        # Check if teams exist
        if away_team not in team_name_to_id or home_team not in team_name_to_id:
            print(f"⚠️  Missing team mapping for {away_team} or {home_team}, skipping...")
            continue
        
        # Run Monte Carlo
        print(f"Running {args.n:,} simulations...")
        try:
            mc_result = run_monte_carlo_for_game(
                model, home_team, away_team, team_name_to_id, games_with_stats, args.n
            )
        except Exception as e:
            print(f"⚠️  Error running simulation: {e}, skipping...")
            continue
        
        # Determine predicted winner
        if mc_result['mc_win_prob'] > 0.5:
            winner = home_team
            pred_winner_prob = mc_result['mc_win_prob']
        else:
            winner = away_team
            pred_winner_prob = 1 - mc_result['mc_win_prob']
        
        print(f"\n📊 SIMULATION RESULTS:")
        print(f"  🏆 Predicted Winner:    {winner}")
        print(f"  🎯 MC Win Probability:  {mc_result['mc_win_prob']:.1%} (home)")
        print(f"  📈 Winner Probability:  {pred_winner_prob:.1%}")
        print(f"  📊 Expected Spread:     {mc_result['mc_median_spread']:+.1f} points (median)")
        print(f"  📊 Mean Spread:         {mc_result['mc_mean_spread']:+.1f} ± {mc_result['mc_std_spread']:.1f}")
        print(f"  📉 95% CI:              [{mc_result['mc_ci_95'][0]:+.1f}, {mc_result['mc_ci_95'][1]:+.1f}]")
        print(f"  📉 90% CI:              [{mc_result['mc_ci_90'][0]:+.1f}, {mc_result['mc_ci_90'][1]:+.1f}]")
        print(f"  📉 80% CI:              [{mc_result['mc_ci_80'][0]:+.1f}, {mc_result['mc_ci_80'][1]:+.1f}]")
        print(f"  🔮 Model Quantiles:     Q10={mc_result['quantiles']['q10']:+.1f}, "
              f"Q50={mc_result['quantiles']['q50']:+.1f}, Q90={mc_result['quantiles']['q90']:+.1f}")
        print(f"  💪 Model Confidence:    {mc_result['model_confidence']} "
              f"(score={mc_result['model_confidence_score']:.3f})")
        print(f"  🎲 Simulations:         {args.n:,}")
        print()
        
        # Store result
        results.append({
            'game_num': i,
            'away_team': away_team,
            'home_team': home_team,
            'time': game['time'],
            'predicted_winner': winner,
            'mc_win_prob_home': mc_result['mc_win_prob'],
            'mc_win_prob_winner': pred_winner_prob,
            'mc_median_spread': mc_result['mc_median_spread'],
            'mc_mean_spread': mc_result['mc_mean_spread'],
            'mc_std_spread': mc_result['mc_std_spread'],
            'ci_95_low': mc_result['mc_ci_95'][0],
            'ci_95_high': mc_result['mc_ci_95'][1],
            'ci_90_low': mc_result['mc_ci_90'][0],
            'ci_90_high': mc_result['mc_ci_90'][1],
            'ci_80_low': mc_result['mc_ci_80'][0],
            'ci_80_high': mc_result['mc_ci_80'][1],
            'q10': mc_result['quantiles']['q10'],
            'q50': mc_result['quantiles']['q50'],
            'q90': mc_result['quantiles']['q90'],
            'model_win_prob': mc_result['model_win_prob'],
            'model_confidence': mc_result['model_confidence'],
            'model_confidence_score': mc_result['model_confidence_score'],
            'n_simulations': args.n,
        })
    
    # Save results
    df_results = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df_results.to_csv(args.out, index=False)
    
    print(f"\n{'='*80}")
    print(f"✅ SIMULATIONS COMPLETE")
    print(f"{'='*80}\n")
    print(f"📁 Results saved to: {args.out}")
    print(f"📊 Total games simulated: {len(results)}")
    print(f"🎲 Total simulations run: {len(results) * args.n:,}")
    
    # Summary statistics
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"  Average winner probability: {df_results['mc_win_prob_winner'].mean():.1%}")
    print(f"  Highest confidence game:    {df_results.loc[df_results['mc_win_prob_winner'].idxmax(), 'predicted_winner']} "
          f"({df_results['mc_win_prob_winner'].max():.1%})")
    print(f"  Lowest confidence game:     {df_results.loc[df_results['mc_win_prob_winner'].idxmin(), 'predicted_winner']} "
          f"({df_results['mc_win_prob_winner'].min():.1%})")
    print(f"  Average spread magnitude:   {df_results['mc_median_spread'].abs().mean():.1f} points")
    print(f"  Model confidence counts:")
    print(f"    HIGH:   {(df_results['model_confidence'] == 'HIGH').sum()} games")
    print(f"    MEDIUM: {(df_results['model_confidence'] == 'MEDIUM').sum()} games")
    print(f"    LOW:    {(df_results['model_confidence'] == 'LOW').sum()} games")
    print()


if __name__ == '__main__':
    main()
