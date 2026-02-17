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

from train_lgbm_model import (
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
