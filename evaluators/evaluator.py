"""
Model Evaluation for NBA Predictions

Metrics:
- RMSE, MAE, R-squared for point differential
- Win accuracy (correct winner %)
- Brier score (probability calibration)
- Log loss
- Interval coverage (% within 80% interval)
- Calibration curve
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelEvaluator:
    """Comprehensive evaluation metrics for NBA prediction models"""
    
    @staticmethod
    def evaluate(y_true, y_pred, y_pred_lower=None, y_pred_upper=None, y_pred_prob=None):
        """
        Calculate all evaluation metrics.
        
        Args:
            y_true: Actual point differentials
            y_pred: Predicted point differentials (median/mean)
            y_pred_lower: Lower bound of prediction interval (optional)
            y_pred_upper: Upper bound of prediction interval (optional)
            y_pred_prob: Predicted home win probabilities (optional)
        
        Returns:
            dict with all metrics
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        
        # Point differential metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'median_abs_error': float(np.median(np.abs(y_true - y_pred))),
        }
        
        # Win accuracy
        actual_home_wins = y_true > 0
        predicted_home_wins = y_pred > 0
        metrics['win_accuracy'] = float((actual_home_wins == predicted_home_wins).mean())
        
        # Close games percentage
        close_games = np.abs(y_pred) < 1.0
        metrics['close_games_pct'] = float(close_games.mean())
        
        # Interval coverage
        if y_pred_lower is not None and y_pred_upper is not None:
            y_pred_lower = np.asarray(y_pred_lower, dtype=float)
            y_pred_upper = np.asarray(y_pred_upper, dtype=float)
            within = (y_true >= y_pred_lower) & (y_true <= y_pred_upper)
            metrics['interval_coverage'] = float(within.mean())
            metrics['avg_interval_width'] = float(np.mean(y_pred_upper - y_pred_lower))
        
        # Brier score + log loss
        if y_pred_prob is not None:
            y_pred_prob = np.asarray(y_pred_prob, dtype=float)
            y_true_binary = (y_true > 0).astype(float)
            metrics['brier_score'] = float(np.mean((y_pred_prob - y_true_binary) ** 2))
            
            # Log loss
            eps = 1e-10
            p = np.clip(y_pred_prob, eps, 1 - eps)
            metrics['log_loss'] = float(-np.mean(
                y_true_binary * np.log(p) + (1 - y_true_binary) * np.log(1 - p)
            ))
        
        return metrics
    
    @staticmethod
    def calibration_curve(y_true, y_pred_prob, n_bins=10):
        """
        Compute calibration curve for win probability predictions.
        
        Args:
            y_true: Actual point differentials
            y_pred_prob: Predicted home win probabilities
            n_bins: Number of probability bins
        
        Returns:
            DataFrame with pred_prob, actual_win_rate, count per bin
        """
        y_true_bin = (np.asarray(y_true) > 0).astype(float)
        probs = np.asarray(y_pred_prob)
        
        bins = np.linspace(0, 1, n_bins + 1)
        calibration = []
        
        for i in range(n_bins):
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
            if mask.sum() > 0:
                calibration.append({
                    'bin_center': (bins[i] + bins[i + 1]) / 2,
                    'mean_predicted_prob': float(probs[mask].mean()),
                    'actual_win_rate': float(y_true_bin[mask].mean()),
                    'count': int(mask.sum()),
                })
        
        return pd.DataFrame(calibration)
    
    @staticmethod
    def print_report(metrics):
        """
        Pretty-print evaluation results.
        
        Args:
            metrics: dict from evaluate()
        """
        print("\n" + "=" * 70)
        print("📊 MODEL EVALUATION REPORT")
        print("=" * 70)
        
        print(f"\n🎯 Point Differential:")
        print(f"   RMSE:              {metrics['rmse']:.2f} points")
        print(f"   MAE:               {metrics['mae']:.2f} points")
        print(f"   Median Abs Error:  {metrics['median_abs_error']:.2f} points")
        print(f"   R²:                {metrics['r2']:.4f}")
        
        print(f"\n🏆 Win Prediction:")
        print(f"   Accuracy:          {metrics['win_accuracy']:.1%}")
        
        if 'interval_coverage' in metrics:
            print(f"\n📦 80% Prediction Interval:")
            print(f"   Coverage:          {metrics['interval_coverage']:.1%} (target: 80%)")
            print(f"   Avg Width:         {metrics['avg_interval_width']:.1f} points")
        
        if 'brier_score' in metrics:
            print(f"\n📈 Probabilistic Calibration:")
            print(f"   Brier Score:       {metrics['brier_score']:.4f} (lower = better)")
            print(f"   Log Loss:          {metrics.get('log_loss', 'N/A'):.4f}")
        
        print("\n" + "=" * 70)


"""
Diagnostics for LGBMWinPredictor
- Loads latest model
- Recreates dataset and chronological splits
- Evaluates on validation and test sets
- Outputs metrics and plots to diagnostics/
"""
import os
import sys
sys.path.insert(0, r'c:\Users\Windows User\My_folder\gamble_code\sports_analytics')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score, log_loss
from sklearn.calibration import calibration_curve

from learners.train_lgbm_model import (
    fetch_nba_games, calculate_rolling_stats, create_matchup_features,
    prepare_features_and_target, LGBMWinPredictor
)

OUT_DIR = 'diagnostics'
os.makedirs(OUT_DIR, exist_ok=True)

print('\n📊 RUNNING LGBM DIAGNOSTICS')

# Load model
model = LGBMWinPredictor.load('machine_learning/models/lgbm_win_predictor_latest.pkl')

# Recreate dataset
games_df = fetch_nba_games(seasons=['2023-24','2024-25'], season_type='Regular Season', verbose=False)
print(f"Loaded {len(games_df)} raw game rows")
games_with_stats = calculate_rolling_stats(games_df, window=5)
matchup_df = create_matchup_features(games_with_stats)

X, y_diff, y_win, feature_names = prepare_features_and_target(matchup_df)

# Chronological split
matchup_sorted = matchup_df.sort_values('GAME_DATE').reset_index(drop=True)
X_sorted = X.loc[matchup_sorted.index]
y_diff_sorted = y_diff[matchup_sorted.index]
y_win_sorted = y_win[matchup_sorted.index]

train_end = int(len(matchup_sorted) * 0.70)
val_end = int(len(matchup_sorted) * 0.85)

X_val = X_sorted.iloc[train_end:val_end]
y_val = y_win_sorted[train_end:val_end]

X_test = X_sorted.iloc[val_end:]
y_test = y_win_sorted[val_end:]

print(f"Data split: train {train_end}, val {len(X_val)}, test {len(X_test)}")

# Predict on validation and test
val_preds = model.predict_win_probability(X_val)
test_preds = model.predict_win_probability(X_test)

# Extract arrays
val_prob = np.asarray(val_preds['win_prob'])
val_lower = np.asarray(val_preds['lower'])
val_upper = np.asarray(val_preds['upper'])
val_unc = np.asarray(val_preds['uncertainty'])
val_conf_score = np.asarray(val_preds.get('confidence_score'))
val_conf_label = val_preds.get('confidence_label')

test_prob = np.asarray(test_preds['win_prob'])
test_lower = np.asarray(test_preds['lower'])
test_upper = np.asarray(test_preds['upper'])
test_unc = np.asarray(test_preds['uncertainty'])
test_conf_score = np.asarray(test_preds.get('confidence_score'))

# Metrics
val_pred_labels = (val_prob > 0.5).astype(int)
val_acc = accuracy_score(y_val, val_pred_labels)
val_brier = brier_score_loss(y_val, val_prob)
try:
    val_auc = roc_auc_score(y_val, val_prob)
except Exception:
    val_auc = None

print('\nValidation Metrics:')
print(f'  Accuracy: {val_acc:.3f}')
print(f'  Brier: {val_brier:.4f}')
if val_auc is not None:
    print(f'  AUC: {val_auc:.3f}')

# Test metrics
test_pred_labels = (test_prob > 0.5).astype(int)
test_acc = accuracy_score(y_test, test_pred_labels)
test_brier = brier_score_loss(y_test, test_prob)
try:
    test_auc = roc_auc_score(y_test, test_prob)
except Exception:
    test_auc = None

print('\nTest Metrics:')
print(f'  Accuracy: {test_acc:.3f}')
print(f'  Brier: {test_brier:.4f}')
if test_auc is not None:
    print(f'  AUC: {test_auc:.3f}')

# Calibration curve (validation)
prob_true, prob_pred = calibration_curve(y_val, val_prob, n_bins=10)
plt.figure()
plt.plot(prob_pred, prob_true, marker='o', label='Model')
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel('Predicted probability')
plt.ylabel('Observed frequency')
plt.title('Calibration Curve (validation)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUT_DIR,'calibration_curve_val.png'), dpi=150)
print('Saved calibration_curve_val.png')

# Probability histogram
plt.figure()
plt.hist(val_prob, bins=20, alpha=0.7)
plt.title('Predicted probability distribution (validation)')
plt.xlabel('Predicted win probability')
plt.ylabel('Count')
plt.savefig(os.path.join(OUT_DIR,'prob_dist_val.png'), dpi=150)
print('Saved prob_dist_val.png')

# Interval width histogram
plt.figure()
plt.hist(val_unc, bins=30, alpha=0.7)
plt.title('Prediction interval half-width (validation)')
plt.xlabel('Half-width (points)')
plt.ylabel('Count')
plt.savefig(os.path.join(OUT_DIR,'interval_width_val.png'), dpi=150)
print('Saved interval_width_val.png')

# Feature importance (from Q50 model)
try:
    fi = model.quantile_model.feature_importance(feature_names=model.feature_names, top_n=30)
    plt.figure(figsize=(8,6))
    sns.barplot(x='importance', y='feature', data=fi)
    plt.title('Feature importance (Q50)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,'feature_importance_q50.png'), dpi=150)
    print('Saved feature_importance_q50.png')
except Exception as e:
    print('Could not plot feature importance:', e)

# Scatter: prob vs interval width
plt.figure(figsize=(6,4))
plt.scatter(val_prob, val_unc, alpha=0.5)
plt.xlabel('Predicted prob (val)')
plt.ylabel('Interval half-width')
plt.title('Prob vs Interval Width (validation)')
plt.grid(True)
plt.savefig(os.path.join(OUT_DIR,'prob_vs_uncertainty_val.png'), dpi=150)
print('Saved prob_vs_uncertainty_val.png')

print('\nDiagnostics complete. Plots and metrics saved in', OUT_DIR)


"""
Verification Script: Check that Rolling Stats Leakage is Fixed

Run this to confirm that rolling features only use PRIOR games, not current game.
"""

import sys
sys.path.insert(0, r'c:\Users\Windows User\My_folder\gamble_code\sports_analytics')

import pandas as pd
from experimental.loaders.data_loader import fetch_nba_games, calculate_rolling_stats

print("="*80)
print("🔍 VERIFICATION: Rolling Stats Leakage Check")
print("="*80)

# Fetch a small sample to inspect manually
print("\n📥 Fetching 2024-25 season data...")
games_df = fetch_nba_games(seasons=['2024-25'], season_type='Regular Season', verbose=False)
games_with_stats = calculate_rolling_stats(games_df, window=5)

# Focus on ONE team's first 15 games
sample_team = games_with_stats['TEAM_ID'].iloc[0]
team_games = games_with_stats[games_with_stats['TEAM_ID'] == sample_team].sort_values('GAME_DATE').head(15).copy()

print(f"\n📊 Sample Team: {team_games['TEAM_ABBREVIATION'].iloc[0]}")
print(f"   First 15 games of season\n")

# Show key columns
display_cols = ['GAME_DATE', 'TEAM_ABBREVIATION', 'WL', 'PTS', 'PTS_ROLL', 'WIN_RATE_10', 'WIN_STREAK']
print(team_games[display_cols].to_string())

# Manual verification for Game 10
print("\n" + "="*80)
print("🧮 MANUAL VERIFICATION: Game 10")
print("="*80)

if len(team_games) >= 10:
    game_10 = team_games.iloc[9]  # 0-indexed
    games_1_to_9 = team_games.iloc[0:9]
    
    # Calculate what PTS_ROLL SHOULD be (average of Games 1-9, excluding NaN)
    expected_pts_roll = games_1_to_9['PTS'].mean()
    actual_pts_roll = game_10['PTS_ROLL']
    
    # Calculate what WIN_RATE_10 SHOULD be (win % of Games 1-9)
    expected_win_rate = (games_1_to_9['WL'] == 'W').mean()
    actual_win_rate = game_10['WIN_RATE_10']
    
    print(f"\nGame 10 PTS: {game_10['PTS']:.1f}")
    print(f"\nExpected PTS_ROLL (avg of Games 1-9): {expected_pts_roll:.2f}")
    print(f"Actual PTS_ROLL:                       {actual_pts_roll:.2f}")
    
    if abs(expected_pts_roll - actual_pts_roll) < 0.01:
        print("✅ PASS: PTS_ROLL does NOT include Game 10")
    else:
        print("❌ FAIL: PTS_ROLL includes Game 10's data (LEAKAGE DETECTED)")
    
    print(f"\nExpected WIN_RATE_10 (Games 1-9): {expected_win_rate:.3f}")
    print(f"Actual WIN_RATE_10:                {actual_win_rate:.3f}")
    
    if abs(expected_win_rate - actual_win_rate) < 0.01:
        print("✅ PASS: WIN_RATE_10 does NOT include Game 10")
    else:
        print("❌ FAIL: WIN_RATE_10 includes Game 10's data (LEAKAGE DETECTED)")

print("\n" + "="*80)
print("✅ WHAT TO CHECK:")
print("="*80)
print("""
Game 1:
  ✓ PTS_ROLL should be NaN (no prior games)
  ✓ WIN_RATE_10 should be NaN (no prior games)
  ✓ WIN_STREAK should be 0.0 (no prior games)

Game 2:
  ✓ PTS_ROLL should equal Game 1's PTS (only 1 prior game)
  ✓ WIN_RATE_10 should be 1.0 or 0.0 (based on Game 1's result)

Game 10:
  ✓ PTS_ROLL should average Games 1-9 ONLY (not include Game 10's PTS)
  ✓ WIN_RATE_10 should be win % of Games 1-9 (not include Game 10's result)
  ✓ WIN_STREAK should reflect Game 9's result (not Game 10)

If ALL checks pass → CLEAN ✅
If ANY fail → LEAKAGE STILL EXISTS ❌
""")

print("\n" + "="*80)
print("🎯 NEXT STEPS")
print("="*80)
print("""
1. If verification PASSES:
   → Rebuild full dataset with fixed features
   → Retrain models from scratch
   → Re-run backtest (expect 55-65% accuracy, NOT 94%)

2. If verification FAILS:
   → Check data_loader.py for correct .shift(1) placement
   → Verify no caching issues
   → Re-run this script

3. After retraining:
   → Compare new backtest accuracy to live 60-62%
   → If they match → leakage is fixed ✅
   → If backtest still inflated → investigate further
""")

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

# GP models are implemented in machine_learning.gp_model
from machine_learning.gp_model import GaussianProcessPredictor, train_gp_models as train_gp_ensemble  # noqa: F401


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

"""
Monte Carlo Simulator for NBA Game Predictions
Runs thousands of simulations per game using model quantile outputs.

Usage:
    python diagnostics/monte_carlo_simulator.py --date 2026-03-09 --n 10000
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats as scipy_stats

from learners.train_lgbm_model import LGBMWinPredictor
from experimental.loaders.data_loader import (
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
    left = q10
    right = q90
    mode = q50

    if right <= left:
        return np.full(n_samples, mode)

    c = (mode - left) / (right - left)
    c = np.clip(c, 0.01, 0.99)

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
    team_id = team_name_to_id.get(team_name)
    if not team_id:
        return {}

    team_games = games_df_with_stats[games_df_with_stats['TEAM_ID'] == team_id].sort_values('GAME_DATE')

    if len(team_games) == 0:
        return {}

    latest = team_games.iloc[-1]

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
    home_features = get_team_latest_features(home_team, team_name_to_id, games_df_with_stats)
    away_features = get_team_latest_features(away_team, team_name_to_id, games_df_with_stats)

    if not home_features or not away_features:
        raise ValueError(f"Missing features for {home_team} or {away_team}")

    feature_row = {}
    for col, value in home_features.items():
        feature_row[f'HOME_{col}'] = value
    for col, value in away_features.items():
        feature_row[f'AWAY_{col}'] = value

    X = pd.DataFrame([feature_row])

    for feat in model.feature_names:
        if feat not in X.columns:
            X[feat] = 0.0
    X = X[model.feature_names]

    X_scaled = model.scaler.transform(X)
    quantiles = model.quantile_model.predict(X_scaled)

    q10 = quantiles['q10'][0]
    q50 = quantiles['q50'][0]
    q90 = quantiles['q90'][0]

    point_diffs = triangular_sample(q10, q50, q90, n_simulations)
    home_wins = (point_diffs > 0).astype(int)

    mc_win_prob = home_wins.mean()
    mc_median_spread = np.median(point_diffs)
    mc_mean_spread = np.mean(point_diffs)
    mc_std_spread = np.std(point_diffs)

    ci_95_low, ci_95_high = np.percentile(point_diffs, [2.5, 97.5])
    ci_90_low, ci_90_high = np.percentile(point_diffs, [5, 95])
    ci_80_low, ci_80_high = np.percentile(point_diffs, [10, 90])

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
    parser.add_argument('--date', type=str, default='2026-03-09',
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

    print(f"Loading model from {args.model}...")
    model = LGBMWinPredictor.load(args.model)
    print("✅ Model loaded\n")

    print("Loading latest team statistics...")
    games_df = fetch_nba_games(seasons=['2024-25'], season_type='Regular Season', verbose=False)
    games_with_stats = calculate_rolling_stats(games_df, window=5)
    print(f"✅ Loaded {len(games_with_stats)} games with rolling stats")
    print(f"   Date range: {games_with_stats['GAME_DATE'].min()} → {games_with_stats['GAME_DATE'].max()}")

    team_data = get_all_nba_teams()
    team_name_to_id = {team['full_name']: team['id'] for team in team_data['teams']}
    print(f"✅ Team mappings ready for {len(team_name_to_id)} teams\n")

    # Define target games (update this list for each run date)
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

    results = []

    for i, game in enumerate(games, 1):
        away_team = game['away']
        home_team = game['home']

        print(f"{'─'*80}")
        print(f"GAME {i}/{len(games)}: {away_team} @ {home_team}")
        print(f"{'─'*80}")

        if away_team not in team_name_to_id or home_team not in team_name_to_id:
            print(f"⚠️  Missing team mapping for {away_team} or {home_team}, skipping...")
            continue

        print(f"Running {args.n:,} simulations...")
        try:
            mc_result = run_monte_carlo_for_game(
                model, home_team, away_team, team_name_to_id, games_with_stats, args.n
            )
        except Exception as e:
            print(f"⚠️  Error running simulation: {e}, skipping...")
            continue

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

    df_results = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df_results.to_csv(args.out, index=False)

    print(f"\n{'='*80}")
    print(f"✅ SIMULATIONS COMPLETE")
    print(f"{'='*80}\n")
    print(f"📁 Results saved to: {args.out}")
    print(f"📊 Total games simulated: {len(results)}")
    print(f"🎲 Total simulations run: {len(results) * args.n:,}")

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
