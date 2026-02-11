"""
Iterative Prediction and Retraining Engine
Handles confidence-driven model retraining with up to 10 iterations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import sys
import os

# Import required modules
sys.path.append(os.path.dirname(__file__))
from database_handler import SportsAnalyticsDB
from extended_data_loader import get_extended_training_dataset, refresh_recent_data
from model_trainer import GaussianProcessPredictor, BayesianEnsemblePredictor, train_gp_models
from mcmc_sampler import BayesianBasketballHierarchical
from predictor import predict_game_gp, predict_game_with_epaa
from model_updater import apply_learning_pipeline
from validation_tracker import PredictionValidator


class IterativePredictor:
    """
    Handles iterative predictions with confidence-driven retraining
    
    Key features:
    - Predicts games with confidence scoring
    - Triggers retraining when confidence < threshold
    - Supports up to max_iterations per prediction
    - Logs all actions to database
    - Uses GP, Ensemble, Bayesian, and MCMC models
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.6,
        max_iterations: int = 10,
        db_path: str = "sports_analytics.db",
        verbose: bool = True
    ):
        """
        Initialize iterative predictor
        
        Parameters:
        - confidence_threshold: Minimum confidence score to accept (0.0-1.0)
        - max_iterations: Maximum retraining iterations per prediction
        - db_path: Path to SQLite database
        - verbose: Print detailed progress
        """
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.db_path = db_path
        self.verbose = verbose
        
        # Model storage
        self.gp_model = None
        self.ensemble_model = None
        self.mcmc_model = None
        self.validator = None
        
        # Data storage
        self.games_df = None
        self.matchup_df = None
        self.team_data = None
        self.feature_names = None
        
        # Database connection
        self.db = SportsAnalyticsDB(db_path)
        
        # Statistics
        self.stats = {
            'total_predictions': 0,
            'retraining_triggered': 0,
            'avg_iterations': 0,
            'avg_confidence': 0,
            'low_confidence_improved': 0
        }
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("🤖 ITERATIVE PREDICTOR INITIALIZED")
            print("=" * 70)
            print(f"📊 Confidence Threshold: {self.confidence_threshold:.2f}")
            print(f"🔄 Max Iterations: {self.max_iterations}")
            print(f"💾 Database: {self.db_path}")
            print("=" * 70 + "\n")
    
    def load_models(self, force_retrain: bool = False):
        """
        Load or train all prediction models
        
        Parameters:
        - force_retrain: Force retraining even if models exist
        """
        if self.verbose:
            print("=" * 70)
            print("🔧 LOADING/TRAINING MODELS")
            print("=" * 70 + "\n")
        
        # Get extended training dataset
        dataset = get_extended_training_dataset(
            db_path=self.db_path,
            verbose=self.verbose
        )
        
        self.games_df = dataset['games_df']
        self.matchup_df = dataset['matchup_df']
        self.team_data = dataset['team_data']
        self.feature_names = dataset['feature_names']
        
        X = dataset['X']
        y = dataset['y']
        
        # Initialize validator
        if self.verbose:
            print("📋 Initializing prediction validator...")
        self.validator = PredictionValidator(
            log_path='basketball/predictions_log.json'
        )
        
        # Load or train GP model
        if self.verbose:
            print("\n🔮 Gaussian Process Model:")
        
        if not force_retrain:
            try:
                # Try to load existing model
                import pickle
                import glob
                model_files = glob.glob('machine_learning/models/gp_model_*.pkl')
                if model_files:
                    latest_model = max(model_files, key=os.path.getctime)
                    with open(latest_model, 'rb') as f:
                        self.gp_model = pickle.load(f)
                    if self.verbose:
                        print(f"   ✅ Loaded existing model: {os.path.basename(latest_model)}")
                else:
                    force_retrain = True
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️  Could not load existing model: {e}")
                force_retrain = True
        
        if force_retrain or self.gp_model is None:
            if self.verbose:
                print("   🔄 Training new GP model...")
            
            # Train GP model
            self.gp_model = GaussianProcessPredictor(kernel_type='combined')
            X_train = self.matchup_df[self.feature_names].values
            y_train = y
            self.gp_model.fit(X_train, y_train)
            
            # Save model
            os.makedirs('machine_learning/models', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = f'machine_learning/models/gp_model_{timestamp}.pkl'
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(self.gp_model, f)
            
            if self.verbose:
                print(f"   ✅ Trained and saved: {os.path.basename(model_path)}")
        
        # Initialize Ensemble model
        if self.verbose:
            print("\n🎯 Bayesian Ensemble Model:")
            print("   🔄 Training ensemble...")
        
        self.ensemble_model = BayesianEnsemblePredictor()
        self.ensemble_model.fit(X, y)
        
        if self.verbose:
            print("   ✅ Ensemble trained")
        
        # Train MCMC model (optional, can be slow)
        if self.verbose:
            print("\n⚡ Bayesian MCMC Model:")
            print("   ⏳ Training MCMC (this may take a few minutes)...")
        
        try:
            # Create simplified team stats for MCMC
            team_stats = self._prepare_mcmc_data()
            
            self.mcmc_model = BayesianBasketballHierarchical(
                L=10,  # accuracy clusters
                J=10,  # shot selection clusters
                K=7    # court regions
            )
            
            self.mcmc_model.fit_gibbs(
                team_stats=team_stats,
                n_iterations=5000,
                burn_in=1500,
                verbose=False
            )
            
            if self.verbose:
                print("   ✅ MCMC trained")
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  MCMC training failed: {e}")
                print("   ℹ️  Continuing with GP and Ensemble only")
            self.mcmc_model = None
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("✅ ALL MODELS LOADED AND READY")
            print("=" * 70 + "\n")
    
    def _prepare_mcmc_data(self) -> Dict:
        """Prepare team statistics for MCMC model"""
        team_stats = {}
        
        for team_id in self.team_data['ids']:
            team_games = self.games_df[self.games_df['TEAM_ID'] == team_id]
            
            if len(team_games) > 0:
                # Simplified stats for MCMC
                # In production, use actual shot chart data
                team_stats[team_id] = {
                    'M': np.random.randint(5, 15, (7,)),  # Placeholder makes by region
                    'N': np.random.randint(10, 25, (7,))  # Placeholder attempts by region
                }
        
        return team_stats
    
    def predict_with_retraining(
        self,
        home_team: str,
        away_team: str,
        game_date: str,
        game_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make prediction with iterative retraining if confidence is low
        
        Parameters:
        - home_team: Home team name
        - away_team: Away team name  
        - game_date: Game date (YYYY-MM-DD format)
        - game_id: Optional game identifier
        
        Returns:
        - Dictionary with prediction results and metadata
        """
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"🏀 PREDICTING: {away_team} @ {home_team}")
            print(f"📅 Date: {game_date}")
            print(f"{'='*70}\n")
        
        iteration = 0
        confidence_score = 0.0
        prediction = None
        retraining_triggered = False
        iteration_history = []
        
        # Get team stats for prediction
        home_stats = self._get_team_latest_stats(home_team)
        away_stats = self._get_team_latest_stats(away_team)
        
        if home_stats is None or away_stats is None:
            if self.verbose:
                print(f"❌ Could not find team stats for {home_team} or {away_team}")
            return None
        
        # Iterative prediction loop
        while iteration < self.max_iterations:
            iteration += 1
            
            if self.verbose:
                print(f"🔄 Iteration {iteration}/{self.max_iterations}")
            
            # Make prediction
            if self.mcmc_model:
                prediction = predict_game_with_epaa(
                    home_stats=home_stats,
                    away_stats=away_stats,
                    gp_model=self.gp_model,
                    mcmc_model=self.mcmc_model,
                    feature_names=self.feature_names,
                    epaa_weight=0.5  # Dynamic weight
                )
            else:
                prediction = predict_game_gp(
                    home_stats=home_stats,
                    away_stats=away_stats,
                    gp_model=self.gp_model,
                    feature_names=self.feature_names
                )
            
            confidence_score = prediction['confidence_score']
            confidence_level = prediction['confidence_level']
            
            if self.verbose:
                print(f"   📊 Confidence: {confidence_score:.3f} ({confidence_level})")
                print(f"   🎯 Prediction: {prediction['predicted_winner']} by {abs(prediction['predicted_spread']):.1f}")
                print(f"   📈 Win Probability: {prediction['win_probability']:.1%}")
            
            # Log iteration to database
            if iteration > 1 or confidence_score < self.confidence_threshold:
                self.db.log_model_action({
                    'iteration': iteration,
                    'model_type': 'iterative_pipeline',
                    'action': 'prediction_attempt',
                    'confidence_before': iteration_history[-1]['confidence'] if iteration_history else None,
                    'confidence_after': confidence_score,
                    'metrics': {
                        'predicted_spread': prediction['predicted_spread'],
                        'win_probability': prediction['win_probability'],
                        'pred_std': prediction.get('pred_std')
                    }
                })
            
            iteration_history.append({
                'iteration': iteration,
                'confidence': confidence_score,
                'prediction': prediction.copy()
            })
            
            # Check if confidence meets threshold
            if confidence_score >= self.confidence_threshold:
                if self.verbose:
                    print(f"   ✅ Confidence threshold met ({self.confidence_threshold:.2f})")
                break
            
            # Check if max iterations reached
            if iteration >= self.max_iterations:
                if self.verbose:
                    print(f"   ⚠️  Max iterations reached")
                break
            
            # Trigger retraining
            if self.verbose:
                print(f"   🔄 Confidence below threshold, triggering retraining...")
            
            retraining_triggered = True
            self._retrain_models(iteration)
        
        # Final prediction result
        final_prediction = {
            'game_id': game_id,
            'game_date': game_date,
            'home_team': home_team,
            'away_team': away_team,
            'predicted_spread': prediction['predicted_spread'],
            'predicted_home_score': prediction.get('predicted_home_score'),
            'predicted_away_score': prediction.get('predicted_away_score'),
            'predicted_winner': prediction['predicted_winner'],
            'win_probability': prediction['win_probability'],
            'confidence_score': confidence_score,
            'confidence_level': prediction['confidence_level'],
            'pred_std': prediction.get('pred_std'),
            'ci_lower': prediction.get('ci_lower'),
            'ci_upper': prediction.get('ci_upper'),
            'epaa_weight': prediction.get('epaa_weight'),
            'model_versions': {
                'gp': 'v1',
                'ensemble': 'v1',
                'mcmc': 'v1' if self.mcmc_model else None
            },
            'iteration_count': iteration,
            'retraining_triggered': retraining_triggered,
            'iteration_history': iteration_history
        }
        
        # Update statistics
        self.stats['total_predictions'] += 1
        if retraining_triggered:
            self.stats['retraining_triggered'] += 1
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"✅ FINAL PREDICTION")
            print(f"{'='*70}")
            print(f"Winner: {prediction['predicted_winner']}")
            print(f"Spread: {prediction['predicted_spread']:.1f}")
            print(f"Confidence: {confidence_score:.3f} ({prediction['confidence_level']})")
            print(f"Iterations: {iteration}")
            print(f"{'='*70}\n")
        
        return final_prediction
    
    def _get_team_latest_stats(self, team_name: str) -> Optional[Dict]:
        """Get latest rolling stats for a team"""
        # Find team in games_df
        team_games = self.games_df[
            self.games_df['MATCHUP'].str.contains(team_name, case=False, na=False)
        ]
        
        if len(team_games) == 0:
            return None
        
        # Get most recent game
        latest_game = team_games.sort_values('GAME_DATE', ascending=False).iloc[0]
        
        # Extract rolling stats
        stats = {}
        for col in latest_game.index:
            if '_ROLL' in col or col in ['WIN_STREAK', 'REST_DAYS', 'IS_BACK_TO_BACK', 'WIN_RATE_10']:
                stats[col] = latest_game[col]
        
        return stats
    
    def _retrain_models(self, iteration: int):
        """Retrain models during iteration"""
        if self.verbose:
            print(f"\n   🔧 Retraining models (iteration {iteration})...")
        
        try:
            # Refresh recent data
            self.games_df = refresh_recent_data(
                self.games_df,
                days_back=14,
                verbose=False
            )
            
            # Recreate training data
            from extended_data_loader import prepare_training_data
            self.matchup_df, y, _ = prepare_training_data(self.games_df, verbose=False)
            X = self.matchup_df[self.feature_names].values
            
            # Retrain GP
            self.gp_model.fit(X, y)
            
            # Retrain Ensemble
            self.ensemble_model.fit(X, y)
            
            if self.verbose:
                print(f"      ✅ Models retrained with updated data")
            
        except Exception as e:
            if self.verbose:
                print(f"      ⚠️  Retraining error: {e}")
    
    def save_prediction_to_db(self, prediction: Dict[str, Any]) -> int:
        """Save prediction to database"""
        return self.db.insert_prediction(prediction)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get prediction statistics"""
        return self.stats.copy()
    
    def close(self):
        """Clean up resources"""
        if self.db:
            self.db.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
