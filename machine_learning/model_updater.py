"""
Model Updater Module

Handles:
- Applying learned adjustments to MCMC model
- Retraining GP models with updated hyperparameters
- Implementing team-specific bias corrections
- Updating EPAA weights based on validation results
- Creating updated model versions for continuous improvement
"""

import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class ModelUpdater:
    """
    Apply learning insights to update and improve models
    """
    
    def __init__(self, mcmc_model=None, gp_model=None):
        """
        Initialize model updater
        
        Parameters:
        - mcmc_model: BayesianBasketballHierarchical instance
        - gp_model: GaussianProcessPredictor instance
        """
        self.mcmc_model = mcmc_model
        self.gp_model = gp_model
        self.update_history = []
    
    def apply_team_adjustments(
        self,
        team_adjustments: Dict[str, float],
        team_data: Dict,
        alpha: float = 0.1
    ) -> Dict[str, float]:
        """
        Apply team-specific bias corrections to EPAA values
        
        Parameters:
        - team_adjustments: Dict mapping team names to adjustment values
        - team_data: Team data dict with IDs and names
        - alpha: Learning rate (0-1), default 0.1 for conservative updates
        
        Returns:
        - Updated EPAA values
        """
        if self.mcmc_model is None:
            print("⚠️ No MCMC model available - cannot apply team adjustments")
            return {}
        
        # Get current EPAA values
        current_epaa = self.mcmc_model.get_epaa_results()
        updated_epaa = current_epaa.copy()
        
        # Map team names to IDs
        team_names_inv = {v: k for k, v in team_data['names'].items()}
        
        # Apply adjustments
        n_updated = 0
        for team_name, adjustment in team_adjustments.items():
            team_id = team_names_inv.get(team_name)
            if team_id and team_id in updated_epaa:
                # Apply adjustment with learning rate
                old_value = updated_epaa[team_id]
                updated_epaa[team_id] = old_value + (alpha * adjustment)
                n_updated += 1
                
                print(f"  Updated {team_name}: {old_value:.2f} → {updated_epaa[team_id]:.2f} ({adjustment:+.2f})")
        
        print(f"\n✅ Applied adjustments to {n_updated} teams")
        
        # Update model's theta_i (EPAA parameters)
        for team_id, new_epaa in updated_epaa.items():
            if team_id in self.mcmc_model.theta_i:
                self.mcmc_model.theta_i[team_id] = new_epaa
        
        return updated_epaa
    
    def update_epaa_weight(
        self,
        proposed_weight: float,
        reason: str = "Performance-based adjustment"
    ):
        """
        Update the EPAA weighting parameter
        
        This is used in hybrid predictions that combine GP and MCMC
        
        Parameters:
        - proposed_weight: New EPAA weight (0-1)
        - reason: Why this change is being made
        """
        # Store in update history
        update_record = {
            'timestamp': datetime.now().isoformat(),
            'type': 'epaa_weight',
            'new_value': proposed_weight,
            'reason': reason
        }
        self.update_history.append(update_record)
        
        print(f"\n🔧 EPAA Weight Update:")
        print(f"   New weight: {proposed_weight:.2f}")
        print(f"   Reason: {reason}")
        
        # In practice, you'd save this to a config file or model metadata
        # For now, we'll return it for the user to apply manually
        return proposed_weight
    
    def retrain_gp_with_corrections(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        team_adjustments: Dict[str, float],
        team_ids_train: List,
        team_data: Dict
    ):
        """
        Retrain GP model with bias-corrected training data
        
        Parameters:
        - X_train: Training features
        - y_train: Training targets (point spreads)
        - team_adjustments: Team bias corrections
        - team_ids_train: Team IDs corresponding to training samples
        - team_data: Team data dict
        
        Returns:
        - Retrained GP model
        """
        if self.gp_model is None:
            print("⚠️ No GP model available")
            return None
        
        print("\n🔄 Retraining GP model with bias corrections...")
        
        # Apply corrections to training data
        y_corrected = y_train.copy()
        team_names_inv = {v: k for k, v in team_data['names'].items()}
        
        for i, (home_id, away_id) in enumerate(team_ids_train):
            home_name = team_data['names'].get(home_id, '')
            away_name = team_data['names'].get(away_id, '')
            
            home_adj = team_adjustments.get(home_name, 0.0)
            away_adj = team_adjustments.get(away_name, 0.0)
            
            # Adjust target: if we over-predicted home team, reduce their advantage
            y_corrected[i] += (home_adj - away_adj)
        
        # Retrain model
        self.gp_model.fit(X_train, y_corrected, verbose=True)
        
        print(f"✅ GP model retrained with {len(team_adjustments)} team corrections")
        
        return self.gp_model
    
    def run_incremental_mcmc_update(
        self,
        new_game_data: pd.DataFrame,
        n_iterations: int = 1000,
        burn_in: int = 300
    ):
        """
        Run incremental MCMC update with new game data
        
        This performs online learning by sampling from the posterior
        given new observations
        
        Parameters:
        - new_game_data: DataFrame with new games to learn from
        - n_iterations: MCMC iterations
        - burn_in: Burn-in period
        
        Returns:
        - Updated MCMC model
        """
        if self.mcmc_model is None:
            print("⚠️ No MCMC model available")
            return None
        
        print(f"\n🔬 Running incremental MCMC update with {len(new_game_data)} new games...")
        
        # In a full implementation, you'd:
        # 1. Convert new game data to shot data format
        # 2. Run additional MCMC iterations starting from current posterior
        # 3. Update team parameters
        
        # Placeholder for now - this would require shot-level data
        print("⚠️ Incremental MCMC update requires shot-level data")
        print("   For now, recommend full retraining with updated data")
        
        return self.mcmc_model
    
    def save_updated_models(
        self,
        output_dir: str = 'models/updated',
        version: Optional[str] = None
    ):
        """
        Save updated models with version tracking
        
        Parameters:
        - output_dir: Directory to save models
        - version: Version string (default: timestamp)
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if version is None:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        saved_files = []
        
        # Save MCMC model
        if self.mcmc_model:
            mcmc_path = f"{output_dir}/mcmc_model_v{version}.pkl"
            self.mcmc_model.save(mcmc_path)
            saved_files.append(mcmc_path)
            print(f"💾 Saved MCMC model: {mcmc_path}")
        
        # Save GP model
        if self.gp_model:
            gp_path = f"{output_dir}/gp_model_v{version}.pkl"
            self.gp_model.save(gp_path)
            saved_files.append(gp_path)
            print(f"💾 Saved GP model: {gp_path}")
        
        # Save update history
        history_path = f"{output_dir}/update_history_v{version}.json"
        with open(history_path, 'w') as f:
            json.dump(self.update_history, f, indent=2)
        saved_files.append(history_path)
        print(f"💾 Saved update history: {history_path}")
        
        return saved_files
    
    def generate_update_summary(self) -> str:
        """
        Generate a summary of all updates applied
        
        Returns:
        - Markdown formatted summary
        """
        summary = []
        summary.append("# 🔄 Model Update Summary\n")
        summary.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        summary.append(f"**Total Updates:** {len(self.update_history)}\n")
        
        if len(self.update_history) == 0:
            summary.append("\n_No updates applied yet._\n")
        else:
            summary.append("\n## Update History\n")
            for i, update in enumerate(self.update_history, 1):
                summary.append(f"\n### Update #{i}")
                summary.append(f"- **Time:** {update['timestamp']}")
                summary.append(f"- **Type:** {update['type']}")
                summary.append(f"- **Reason:** {update.get('reason', 'N/A')}")
                
                if 'new_value' in update:
                    summary.append(f"- **New Value:** {update['new_value']}")
        
        return '\n'.join(summary)


def apply_learning_pipeline(
    validation_results: Dict,
    mcmc_model,
    gp_model,
    team_data: Dict,
    learning_rate: float = 0.1,
    save_models: bool = True
) -> Dict:
    """
    Complete pipeline to apply learning insights to models
    
    Parameters:
    - validation_results: Output from adaptive_learner.validate_and_learn()
    - mcmc_model: Current MCMC model
    - gp_model: Current GP model
    - team_data: Team data dict
    - learning_rate: How aggressively to apply corrections (0-1)
    - save_models: Whether to save updated models
    
    Returns:
    - Dict with updated models and summary
    """
    print("🚀 Starting learning application pipeline...\n")
    
    # Initialize updater
    updater = ModelUpdater(mcmc_model, gp_model)
    
    # Step 1: Apply team adjustments
    if 'team_adjustments' in validation_results:
        print("📊 Applying team-specific adjustments...")
        updated_epaa = updater.apply_team_adjustments(
            validation_results['team_adjustments'],
            team_data,
            alpha=learning_rate
        )
    
    # Step 2: Update EPAA weight
    if 'mcmc_refinement' in validation_results:
        refinement = validation_results['mcmc_refinement']
        proposed_weight = refinement.get('proposed_epaa_weight', 0.5)
        reasoning = ' | '.join(refinement.get('reasoning', []))
        
        new_weight = updater.update_epaa_weight(proposed_weight, reasoning)
    
    # Step 3: Save updated models
    saved_files = []
    if save_models:
        print("\n💾 Saving updated models...")
        saved_files = updater.save_updated_models()
    
    # Generate summary
    summary = updater.generate_update_summary()
    
    print("\n" + "="*60)
    print("✅ Learning pipeline complete!")
    print("="*60)
    
    return {
        'updater': updater,
        'summary': summary,
        'saved_files': saved_files,
        'updated_models': {
            'mcmc': updater.mcmc_model,
            'gp': updater.gp_model
        }
    }


def create_feedback_loop_config(
    validation_results: Dict,
    output_file: str = 'config/feedback_config.json'
) -> Dict:
    """
    Create a configuration file for automated feedback loop
    
    This can be used in automated retraining pipelines
    
    Parameters:
    - validation_results: Results from validation
    - output_file: Where to save config
    
    Returns:
    - Config dict
    """
    import os
    
    config = {
        'last_updated': datetime.now().isoformat(),
        'epaa_weight': validation_results.get('mcmc_refinement', {}).get('proposed_epaa_weight', 0.5),
        'learning_rate': 0.1,
        'min_games_for_update': 10,
        'confidence_thresholds': {
            'HIGH': 0.5,  # Could adjust based on calibration
            'MEDIUM': 0.3,
            'LOW': 0.0
        },
        'team_adjustments': validation_results.get('team_adjustments', {}),
        'performance_metrics': validation_results.get('error_analysis', {}).get('overall_metrics', {}),
        'recommended_actions': [
            'Retrain GP kernel with updated hyperparameters',
            'Apply team bias corrections',
            'Recalibrate confidence thresholds',
            'Update EPAA weight in prediction pipeline'
        ]
    }
    
    # Save config
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n📝 Feedback loop config saved to: {output_file}")
    
    return config
