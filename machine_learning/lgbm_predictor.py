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
    
    def __init__(self, params=None):
        """
        Args:
            params: LightGBM parameters dict (overrides defaults)
        """
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
    
    def train(self, X_train, y_train, X_val=None, y_val=None,
              quantiles=(0.1, 0.5, 0.9), num_boost_round=500,
              early_stopping_rounds=50):
        """
        Train quantile regression models.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training target (point differential)
            X_val: Validation features (optional, for early stopping)
            y_val: Validation target
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
            DataFrame with feature, importance columns
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
        
        return df.head(top_n).reset_index(drop=True)
    
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
            }, f)
        print(f"💾 Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load models from disk."""
        import lightgbm as lgb
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(params=data['base_params'])
        instance.feature_names = data['feature_names']
        instance.models = {
            k: lgb.Booster(model_str=v) for k, v in data['models'].items()
        }
        instance.is_fitted = True
        print(f"📂 Model loaded from {filepath}")
        return instance
