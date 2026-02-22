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

from loaders.data_loader import (
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
