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
