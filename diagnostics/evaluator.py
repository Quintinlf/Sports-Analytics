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
from loaders.data_loader import fetch_nba_games, calculate_rolling_stats

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
