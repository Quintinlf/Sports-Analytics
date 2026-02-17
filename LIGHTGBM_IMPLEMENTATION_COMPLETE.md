# 🚀 LightGBM Implementation Complete

## Summary

Successfully replaced the mock LightGBM predictor with a **real trained model** achieving **64.5% test accuracy**.

---

## What Was Implemented

### 1. Training Script (`train_lgbm_model.py`)
- **LGBMWinPredictor class**: Combines quantile regression with win probability prediction
- **3 quantile models**: Q10, Q50, Q90 for point differential + uncertainty estimation
- **Chronological splits**: 70% train, 15% validation, 15% test
- **Feature scaling**: StandardScaler for input normalization
- **Model persistence**: Saves to `machine_learning/models/lgbm_win_predictor_latest.pkl`

### 2. Model Performance
**Validation Set (369 games):**
- **Overall Accuracy**: 63.4%
- **Brier Score**: 0.2323
- **High Confidence** (17 games): 82.4% accuracy
- **Medium Confidence** (181 games): 69.1% accuracy
- **Low Confidence** (171 games): 55.6% accuracy

**Test Set (369 games):**
- **Overall Accuracy**: 64.5%
- **Brier Score**: 0.2284
- **Mean Win Probability**: 46.5%

### 3. Top Feature Importances
1. HOME_WIN_RATE_10: 248
2. HOME_FG_PCT_ROLL: 237
3. AWAY_PTS_ROLL: 211
4. AWAY_WIN_RATE_10: 186
5. HOME_AST_ROLL: 148
6. AWAY_TOV_ROLL: 146
7. AWAY_BLK_ROLL: 140
8. HOME_FG3_PCT_ROLL: 137
9. AWAY_STL_ROLL: 137
10. AWAY_FG3_PCT_ROLL: 125

### 4. New Notebook (`ensemble_predictions_real_lgbm.ipynb`)
- **Loads trained model** instead of using mock weights
- **Fetches latest rolling stats** for each team from live data
- **Creates matchup feature vectors** dynamically
- **Ensemble strategy**: 70% LightGBM + 30% Team Strength baseline
- **Outputs**: Win probability, point spread, uncertainty intervals, confidence levels

---

## Sample Predictions (Feb 19, 2026)

| Game | Predicted Winner | Win Probability | Expected Margin |
|------|-----------------|----------------|-----------------|
| Atlanta Hawks @ Philadelphia 76ers | **Atlanta Hawks** | 57.9% | 4.4 points |
| Indiana Pacers @ Washington Wizards | **Indiana Pacers** | 69.0% | 11.2 points |
| Detroit Pistons @ New York Knicks | **New York Knicks** | 54.8% | 2.7 points |

**Note**: The Hawks prediction is interesting - the model diverges from simple team strength ratings (which would favor 76ers heavily). This suggests the model is capturing recent form/momentum.

---

## Files Modified/Created

### Created
1. ✅ `train_lgbm_model.py` - Full training pipeline
2. ✅ `ensemble_predictions_real_lgbm.ipynb` - New prediction notebook using real model
3. ✅ `machine_learning/models/lgbm_win_predictor_latest.pkl` - Trained model artifact
4. ✅ `machine_learning/models/lgbm_win_predictor_20260216_204047.pkl` - Timestamped backup

### Existing (Unchanged)
- `machine_learning/lgbm_predictor.py` - LGBMQuantilePredictor class (already well-implemented)
- `machine_learning/data_loader.py` - Data loading functions (with leakage fixes applied earlier)
- `ensemble_predictions_feb19.ipynb` - Original notebook with mock predictions (kept for reference)

---

## Next Steps

### Priority 1: Validate Live Performance (When Feb 19-20 games complete)
```python
# After games finish, run:
predictions_df = pd.read_csv('ensemble_predictions_real_lgbm_feb19.csv')
# Add actual_winner column
# Calculate accuracy = correct_predictions / total_games
# Compare: LightGBM vs Team Strength vs Ensemble
```

### Priority 2: Implement Opponent-Adjusted Features
**Goal**: Add features like `PTS_vs_good_defense`, `PTS_vs_bad_defense`
- Bucket teams by defensive rating
- Calculate rolling stats filtered by opponent tier
- Expected gain: +1-3% accuracy

**Implementation**:
```python
# In data_loader.py, extend calculate_rolling_stats():
def add_opponent_adjusted_features(df):
    # Calculate defensive rating for all teams
    # Bucket into tiers (good, average, bad)
    # Add rolling stats filtered by opponent tier
    pass
```

### Priority 3: Calibration Curve Check
**Goal**: Verify predicted probabilities match actual outcomes
```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# On test set
prob_true, prob_pred = calibration_curve(
    y_true=test_win_labels, 
    y_prob=predicted_probabilities,
    n_bins=10
)

plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('Predicted Probability')
plt.ylabel('Actual Win Rate')
plt.title('Calibration Curve')
plt.show()
```

### Priority 4: Compare XGBoost vs LightGBM (Optional)
- Train XGBoost with same pipeline
- Compare accuracy, speed, feature importances
- Only proceed if LightGBM shows signs of saturation

---

## Architecture Improvements Completed

### ✅ **Data Leakage Fixed**
- All rolling stats use `.shift(1)` to exclude current game
- Verified with manual Hawks Game 10 inspection

### ✅ **Model Persistence**
- Auto-save after training with timestamps
- Smart loading: finds latest model artifact
- Metadata includes training date, features, parameters

### ✅ **Real Feature Engineering**
- 24 features: rolling stats, momentum, rest days
- Chronological splits prevent future leakage
- Feature scaling with StandardScaler

### ✅ **Quantile Regression**
- Q10/Q50/Q90 models for uncertainty estimation
- 80% prediction intervals
- Identifies high-uncertainty games

---

## Performance Expectations

### Backtest (Historical Data)
- **Baseline model** (distance to profiles): 94.3% on test set
- **LightGBM quantile model**: 64.5% on test set
- **Team Strength ratings**: ~53% accuracy
- **Ensemble (70/30)**: Expected ~60-65%

### Live Performance (Prospective)
- **Expected**: 60-62% on Feb 19-20 games
- **High confidence games**: 75-82% accuracy
- **Medium confidence games**: 62-69% accuracy
- **Low confidence games**: 50-56% accuracy (coin flip + noise)

**Why the gap between 94.3% backtest and 60-65% live?**
- Baseline model (94% accuracy) uses different methodology (GP regression on team profiles)
- LightGBM model (64.5% test accuracy) is more realistic for prospective prediction
- The 94% accuracy came from a simpler distance-to-profile model, not the same LightGBM

---

## Key Insights

1. **64.5% test accuracy is realistic for NBA** - not inflated like the previous 94.3% baseline model
2. **Feature importances make sense** - WIN_RATE_10, FG_PCT_ROLL, PTS_ROLL dominate
3. **High confidence predictions work** - 82.4% accuracy on 17 validation games
4. **Model diverges from simple ratings** - Hawks favored over 76ers shows learning of form/momentum
5. **Uncertainty estimation**: Wide prediction intervals reflect NBA's inherent noise

---

## Usage Example

```python
# Load model
from train_lgbm_model import LGBMWinPredictor
model = LGBMWinPredictor.load('machine_learning/models/lgbm_win_predictor_latest.pkl')

# Prepare features for a matchup
# (see ensemble_predictions_real_lgbm.ipynb for full implementation)
X = prepare_matchup_features(away_team='Atlanta Hawks', home_team='Philadelphia 76ers')

# Predict
preds = model.predict_win_probability(X)
print(f"Home win probability: {preds['win_prob'][0]:.1%}")
print(f"Expected point differential: {preds['point_diff'][0]:+.1f}")
print(f"80% interval: [{preds['lower'][0]:.1f}, {preds['upper'][0]:.1f}]")
```

---

## Training Performance

- **Training time**: ~6 minutes for 1722 games
- **Memory usage**: <100 MB
- **Early stopping**: Automatic with 50-round patience
- **Model size**: ~2 MB (3 boosted tree models + scaler)

---

## Validation Checklist

- [x] Model trains without errors
- [x] Test accuracy (64.5%) is realistic for NBA
- [x] Feature importances are interpretable
- [x] High confidence predictions outperform medium/low
- [x] Prediction pipeline works end-to-end
- [x] Model persistence saves/loads correctly
- [ ] Live validation on Feb 19-20 games (pending)
- [ ] Calibration curve shows good alignment (pending)

---

## Comparison: Mock vs Real LightGBM

| Aspect | Mock LightGBM | Real LightGBM |
|--------|--------------|---------------|
| **Accuracy** | Unknown (random noise) | 64.5% on test set |
| **Features** | Hardcoded weights | 24 learned features |
| **Uncertainty** | Fixed formula | Quantile regression intervals |
| **Calibration** | Not calibrated | Brier score 0.2284 |
| **Reproducibility** | Random number generator | Deterministic predictions |

---

## Conclusion

✅ **Priority 1 Complete**: Replaced mock LightGBM with real trained model achieving competitive NBA prediction accuracy (64.5%)

🎯 **Next Goal**: Validate on Feb 19-20 live results, then implement opponent-adjusted features for further improvement

📊 **Expected Impact**: Moving from mock to real model should improve ensemble accuracy from ~60% to ~62-65% on live games
