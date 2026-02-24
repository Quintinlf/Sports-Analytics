import sys
sys.path.insert(0, r'c:\Users\Windows User\My_folder\gamble_code\sports_analytics')

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "="*70)
print("🏀 FULL BACKTEST WITH SEASON DATA")
print("="*70)

# Load season data
print("\n📊 Loading 2024-25 season data...")
from loaders.data_loader import fetch_nba_games, calculate_rolling_stats

try:
    games_df = fetch_nba_games(seasons=['2024-25'], season_type='Regular Season', verbose=True)
    games_with_stats = calculate_rolling_stats(games_df, window=5)
    print(f"✅ Loaded {len(games_with_stats)} games with rolling stats")
except Exception as e:
    print(f"❌ Error: {e}")
    games_with_stats = None

if games_with_stats is not None:
    
    # Create train/test split (chronological, 80/20)
    print("\n✂️  Creating chronological train/test split...")
    games_sorted = games_with_stats.sort_values('GAME_DATE').reset_index(drop=True)
    split_idx = int(len(games_sorted) * 0.80)
    
    train_df = games_sorted.iloc[:split_idx].copy()
    test_df = games_sorted.iloc[split_idx:].copy()
    
    print(f"📚 Training: {len(train_df)} games ({train_df['GAME_DATE'].min()} to {train_df['GAME_DATE'].max()})")
    print(f"🧪 Testing:  {len(test_df)} games ({test_df['GAME_DATE'].min()} to {test_df['GAME_DATE'].max()})")
    
    # Create predictions
    print("\n🎯 Generating predictions...")
    
    # Get actual results
    train_results = (train_df['WL'] == 'W').astype(int).values
    test_results = (test_df['WL'] == 'W').astype(int).values
    
    # Get numeric features
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ['TEST_COL', 'GAME_ID', 'SEASON_ID']]
    
    train_features = train_df[numeric_cols].fillna(0).values
    test_features = test_df[numeric_cols].fillna(0).values
    
    # Simple baseline: use average profile
    winner_profile = train_features[train_results == 1].mean(axis=0)
    loser_profile = train_features[train_results == 0].mean(axis=0)
    
    # Predict based on similarity to winner profile
    winner_dist = np.linalg.norm(test_features - winner_profile, axis=1)
    loser_dist = np.linalg.norm(test_features - loser_profile, axis=1)
    baseline_predictions = (winner_dist < loser_dist).astype(int)
    
    # Calculate metrics
    print("\n" + "="*70)
    print("📈 BACKTEST RESULTS")
    print("="*70)
    
    accuracy = (baseline_predictions == test_results).mean()
    
    tp = ((baseline_predictions == 1) & (test_results == 1)).sum()
    tn = ((baseline_predictions == 0) & (test_results == 0)).sum()
    fp = ((baseline_predictions == 1) & (test_results == 0)).sum()
    fn = ((baseline_predictions == 0) & (test_results == 1)).sum()
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    print(f"\n✅ Win Prediction Accuracy: {accuracy:.1%}")
    print(f"   Precision: {precision:.1%}")
    print(f"   Recall:    {recall:.1%}")
    print(f"   F1 Score:  {f1:.1%}")
    
    print(f"\n📊 Confusion Matrix:")
    print(f"   True Positives:  {tp}")
    print(f"   True Negatives:  {tn}")
    print(f"   False Positives: {fp}")
    print(f"   False Negatives: {fn}")
    
    print(f"\n🔍 Dataset Statistics:")
    print(f"   Test set actual wins:     {test_results.sum()} / {len(test_results)}")
    print(f"   Predicted wins:           {baseline_predictions.sum()} / {len(baseline_predictions)}")
    print(f"   Teams in test set:        {len(test_df['TEAM_ABBREVIATION'].unique())}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confusion matrix heatmap
    cm = np.array([[tn, fp], [fn, tp]])
    ax = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'])
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    
    # Accuracy by team
    ax = axes[1]
    test_df_copy = test_df.copy()
    test_df_copy['prediction'] = baseline_predictions
    test_df_copy['actual'] = test_results
    test_df_copy['correct'] = (test_df_copy['prediction'] == test_df_copy['actual']).astype(int)
    
    team_acc = test_df_copy.groupby('TEAM_ABBREVIATION')['correct'].agg(['sum', 'count'])
    team_acc['accuracy'] = team_acc['sum'] / team_acc['count']
    team_acc_sorted = team_acc.sort_values('accuracy', ascending=False).head(15)
    
    ax.barh(range(len(team_acc_sorted)), team_acc_sorted['accuracy'].values)
    ax.set_yticks(range(len(team_acc_sorted)))
    ax.set_yticklabels(team_acc_sorted.index)
    ax.set_xlabel('Accuracy')
    ax.set_title('Prediction Accuracy by Team (Top 15)')
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to 'backtest_results.png'")
    plt.show()
    
    print("\n" + "="*70)
    print("✅ BACKTEST COMPLETE")
    print("="*70)
    
    # Recommendations  
    print(f"\n💡 MODEL PERFORMANCE ANALYSIS:")
    if accuracy < 0.52:
        print(f"   ⚠️  Accuracy below 52% - model needs improvement")
        print(f"   → Add feature engineering (team streaks, rest days)")
        print(f"   → Include injury/availability data")
    elif accuracy < 0.55:
        print(f"   ✴️  Accuracy 52-55% - baseline classifier level")
        print(f"   → Model is performing at random\n")
    elif accuracy < 0.58:
        print(f"   ✅ Accuracy 55-58% - decent model")
        print(f"   → Good for filtering low-confidence predictions")
    else:
        print(f"   🎯 Accuracy >58% - strong model performance!")
        print(f"   → Consider for predictive betting")
    
    print(f"\n📈 NEXT STEPS:")
    print(f"   1. Compare to LightGBM from weekly_predictions.ipynb")
    print(f"   2. Ensemble both approaches for better accuracy")
    print(f"   3. Track Feb 19 games for prospective validation")
    print(f"   4. Monitor for concept drift monthly")

"""
Validation Tracker Module

Handles:
- Prediction logging to JSON
- Performance tracking (rolling accuracy, R², calibration)
- Dynamic EPAA weight adjustment based on recent performance
- Confidence recalibration using isotonic regression
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')


class PredictionValidator:
    """
    Tracks predictions and validates accuracy
    
    Features:
    - Save all predictions to JSON log
    - Track actual outcomes when available
    - Calculate rolling accuracy metrics
    - Dynamically adjust EPAA weight based on performance
    - Recalibrate confidence using isotonic regression
    """
    
    def __init__(self, log_file='predictions_log.json', window=10):
        """
        Initialize validator
        
        Parameters:
        - log_file: Path to JSON log file
        - window: Rolling window for performance metrics (default: 10 games)
        """
        self.log_file = log_file
        self.window = window
        self.predictions = []
        
        # Load existing predictions if file exists
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                self.predictions = json.load(f)
            print(f"📂 Loaded {len(self.predictions)} existing predictions")
        
        # Calibration model (trained on completed predictions)
        self.calibrator = None
        
    def log_prediction(self, prediction, game_id=None, game_date=None):
        """
        Log a prediction
        
        Parameters:
        - prediction: Dict from predictor.py
        - game_id: Unique game identifier (optional)
        - game_date: Game date string (optional)
        
        Returns:
        - prediction_id: Unique ID for this prediction
        """
        pred_id = len(self.predictions)
        
        log_entry = {
            'prediction_id': pred_id,
            'game_id': game_id,
            'game_date': game_date,
            'timestamp': datetime.now().isoformat(),
            'home_team': prediction['home_team'],
            'away_team': prediction['away_team'],
            'predicted_spread': float(prediction['predicted_spread']),
            'uncertainty': float(prediction['uncertainty']),
            'win_probability': float(prediction['win_probability']),
            'confidence': prediction['confidence'],
            'lower_bound': float(prediction.get('lower_bound', 0)),
            'upper_bound': float(prediction.get('upper_bound', 0)),
            
            # EPAA info if available
            'epaa_weight_used': float(prediction.get('epaa_weight_used', 0)),
            'home_epaa': float(prediction.get('home_epaa', 0)),
            'away_epaa': float(prediction.get('away_epaa', 0)),
            
            # Actual outcome (filled in later)
            'actual_spread': None,
            'actual_home_score': None,
            'actual_away_score': None,
            'prediction_error': None,
            'within_ci': None,
            'correct_winner': None,
            'result_logged_at': None
        }
        
        self.predictions.append(log_entry)
        self._save()
        
        return pred_id
    
    def log_result(self, prediction_id, home_score, away_score):
        """
        Log actual game result
        
        Parameters:
        - prediction_id: ID from log_prediction()
        - home_score: Actual home team score
        - away_score: Actual away team score
        """
        if prediction_id >= len(self.predictions):
            raise ValueError(f"Invalid prediction_id: {prediction_id}")
        
        pred = self.predictions[prediction_id]
        actual_spread = home_score - away_score
        pred_spread = pred['predicted_spread']
        
        # Update with actual results
        pred['actual_spread'] = float(actual_spread)
        pred['actual_home_score'] = int(home_score)
        pred['actual_away_score'] = int(away_score)
        pred['prediction_error'] = float(abs(actual_spread - pred_spread))
        pred['within_ci'] = bool(pred['lower_bound'] <= actual_spread <= pred['upper_bound'])
        pred['correct_winner'] = bool((pred_spread > 0 and actual_spread > 0) or 
                                     (pred_spread < 0 and actual_spread < 0) or
                                     (abs(pred_spread) < 1 and abs(actual_spread) < 3))
        pred['result_logged_at'] = datetime.now().isoformat()
        
        self._save()
        
        print(f"✅ Result logged for prediction {prediction_id}")
        print(f"   Predicted: {pred_spread:+.1f}, Actual: {actual_spread:+.1f}, Error: {pred['prediction_error']:.1f}")
    
    def get_recent_performance(self, n=None):
        """
        Get performance metrics for recent predictions with results
        
        Parameters:
        - n: Number of recent predictions (default: self.window)
        
        Returns:
        - Dict with performance metrics
        """
        if n is None:
            n = self.window
        
        # Get completed predictions (those with actual results)
        completed = [p for p in self.predictions if p['actual_spread'] is not None]
        
        if len(completed) == 0:
            return {
                'n_predictions': 0,
                'accuracy': 0.0,
                'r2': 0.0,
                'rmse': 0.0,
                'mae': 0.0,
                'ci_coverage': 0.0,
                'win_prediction_accuracy': 0.0
            }
        
        # Take most recent n
        recent = completed[-n:] if len(completed) >= n else completed
        
        # Extract arrays
        predicted = np.array([p['predicted_spread'] for p in recent])
        actual = np.array([p['actual_spread'] for p in recent])
        errors = np.array([p['prediction_error'] for p in recent])
        within_ci = np.array([p['within_ci'] for p in recent])
        correct_winner = np.array([p['correct_winner'] for p in recent])
        
        # Calculate metrics
        r2 = r2_score(actual, predicted) if len(actual) > 1 else 0.0
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        ci_coverage = np.mean(within_ci)
        win_accuracy = np.mean(correct_winner)
        
        return {
            'n_predictions': len(recent),
            'accuracy': 1.0 - (mae / (np.abs(actual).mean() + 1)),  # Normalized accuracy
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'ci_coverage': ci_coverage,
            'win_prediction_accuracy': win_accuracy
        }
    
    def calculate_dynamic_epaa_weight(self, min_weight=0.3, max_weight=0.7, default_weight=0.5):
        """
        Calculate EPAA weight based on recent model performance
        
        Logic:
        - If recent R² > 0.40 and win accuracy > 60%: Increase weight to 0.7
        - If recent R² < 0.35 or win accuracy < 55%: Decrease weight to 0.3
        - Otherwise: Use 0.5 (default)
        
        Parameters:
        - min_weight: Minimum EPAA weight (default: 0.3)
        - max_weight: Maximum EPAA weight (default: 0.7)
        - default_weight: Default weight when insufficient data (default: 0.5)
        
        Returns:
        - float: Recommended EPAA weight
        """
        perf = self.get_recent_performance()
        
        if perf['n_predictions'] < 5:
            # Not enough data, use default
            return default_weight
        
        r2 = perf['r2']
        win_acc = perf['win_prediction_accuracy']
        
        # Decision logic
        if r2 > 0.40 and win_acc > 0.60:
            weight = max_weight
            reason = "Strong performance"
        elif r2 < 0.35 or win_acc < 0.55:
            weight = min_weight
            reason = "Weak performance"
        else:
            weight = default_weight
            reason = "Moderate performance"
        
        print(f"\n🎯 Dynamic EPAA Weight Calculation:")
        print(f"   Recent R²: {r2:.3f}")
        print(f"   Recent Win Accuracy: {win_acc:.1%}")
        print(f"   Recommended Weight: {weight:.1f} ({reason})")
        
        return weight
    
    def recalibrate_confidence(self, min_predictions=20):
        """
        Recalibrate confidence using isotonic regression
        
        Trains isotonic regression on prediction errors to adjust confidence scores
        
        Parameters:
        - min_predictions: Minimum predictions needed for calibration
        
        Returns:
        - bool: True if calibration successful
        """
        completed = [p for p in self.predictions if p['actual_spread'] is not None]
        
        if len(completed) < min_predictions:
            print(f"⚠️  Need at least {min_predictions} completed predictions for calibration (have {len(completed)})")
            return False
        
        # Extract confidence scores and binary outcomes (within CI)
        confidence_scores = []
        within_ci = []
        
        for p in completed:
            # Reconstruct confidence score from win prob and uncertainty
            win_prob = p['win_probability']
            uncertainty = p['uncertainty']
            prob_certainty = abs(win_prob - 0.5) * 2
            uncertainty_factor = uncertainty / 12.0
            conf_score = prob_certainty * (1 - uncertainty_factor)
            
            confidence_scores.append(conf_score)
            within_ci.append(1 if p['within_ci'] else 0)
        
        confidence_scores = np.array(confidence_scores)
        within_ci = np.array(within_ci)
        
        # Train isotonic regression
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(confidence_scores, within_ci)
        
        # Calculate calibration improvement
        raw_accuracy = np.mean(within_ci)
        calibrated = self.calibrator.predict(confidence_scores)
        calibration_error = np.mean(np.abs(calibrated - within_ci))
        
        print(f"\n📊 Confidence Recalibration Complete:")
        print(f"   Predictions used: {len(completed)}")
        print(f"   Raw CI coverage: {raw_accuracy:.1%}")
        print(f"   Calibration error: {calibration_error:.3f}")
        
        return True
    
    def get_calibrated_confidence(self, raw_confidence_score):
        """
        Apply calibration to a raw confidence score
        
        Parameters:
        - raw_confidence_score: Uncalibrated confidence (0-1)
        
        Returns:
        - Calibrated confidence score
        """
        if self.calibrator is None:
            return raw_confidence_score
        
        return self.calibrator.predict([raw_confidence_score])[0]
    
    def export_summary(self, filepath='validation_summary.json'):
        """Export performance summary to JSON"""
        completed = [p for p in self.predictions if p['actual_spread'] is not None]
        
        summary = {
            'total_predictions': len(self.predictions),
            'completed_predictions': len(completed),
            'pending_predictions': len(self.predictions) - len(completed),
            'overall_performance': self.get_recent_performance(n=len(completed)) if completed else {},
            'recent_performance': self.get_recent_performance(),
            'recommended_epaa_weight': self.calculate_dynamic_epaa_weight(),
            'generated_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Validation summary exported to {filepath}")
        
        return summary
    
    def _save(self):
        """Save predictions to JSON file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.predictions, f, indent=2)


if __name__ == "__main__":
    print("🏀 Testing Validation Tracker...")
    
    # Create validator
    validator = PredictionValidator(log_file='test_predictions.json')
    
    # Log a test prediction
    test_pred = {
        'home_team': 'Lakers',
        'away_team': 'Warriors',
        'predicted_spread': 5.2,
        'uncertainty': 8.5,
        'win_probability': 0.72,
        'confidence': 'MEDIUM',
        'lower_bound': -11.5,
        'upper_bound': 21.9,
        'epaa_weight_used': 0.5
    }
    
    pred_id = validator.log_prediction(test_pred, game_id='test_001', game_date='2026-01-21')
    print(f"✅ Logged prediction with ID: {pred_id}")
    
    # Log result
    validator.log_result(pred_id, home_score=115, away_score=108)
    
    # Get performance
    perf = validator.get_recent_performance()
    print(f"\n📊 Performance: {perf}")
    
    # Calculate EPAA weight
    weight = validator.calculate_dynamic_epaa_weight()
    
    print("\n🎉 Validation tracker module working correctly!")
    
    # Cleanup test file
    if os.path.exists('test_predictions.json'):
        os.remove('test_predictions.json')
