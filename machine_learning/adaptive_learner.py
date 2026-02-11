"""
Adaptive Learning Module for NBA Predictions

Handles:
- Parsing actual game results from various sources (CSV, API, manual)
- Matching predictions to actual outcomes
- Error analysis and pattern detection
- MCMC-based model refinement using prediction errors
- Adaptive hyperparameter tuning based on performance
- Backpropagation of learned patterns into model parameters
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')


class GameResultParser:
    """Parse game results from various formats"""
    
    @staticmethod
    def parse_csv_results(csv_text: str) -> pd.DataFrame:
        """
        Parse CSV game results (Sports Reference format)
        
        Parameters:
        - csv_text: Raw CSV text with game results
        
        Returns:
        - DataFrame with cleaned game results
        """
        from io import StringIO
        
        # Parse CSV
        df = pd.read_csv(StringIO(csv_text))
        
        # Clean up - only keep completed games (those with scores)
        df = df[df['PTS'].notna()].copy()
        
        # Rename columns for clarity
        df = df.rename(columns={
            'Visitor/Neutral': 'away_team',
            'Home/Neutral': 'home_team',
            'Date': 'game_date'
        })
        
        # Add scores (PTS column appears twice - need to handle both)
        # The CSV structure has: Date, Time, Away, PTS, Home, PTS
        # We'll need to parse this carefully
        
        # Create proper away_score and home_score columns
        df['away_score'] = df['PTS'].iloc[::2].values if len(df) > 0 else []
        df['home_score'] = df['PTS'].iloc[1::2].values if len(df) > 1 else []
        
        return df[['game_date', 'away_team', 'home_team', 'away_score', 'home_score']]
    
    @staticmethod
    def parse_csv_sports_reference(csv_lines: List[str]) -> pd.DataFrame:
        """
        Parse Sports Reference CSV format (more robust)
        
        Expected format:
        Date,Start (ET),Visitor/Neutral,PTS,Home/Neutral,PTS,...
        
        Returns:
        - DataFrame with game results
        """
        results = []
        
        for line in csv_lines:
            if not line.strip() or line.startswith('Date') or line.startswith('Provided'):
                continue
            
            parts = [p.strip() for p in line.split(',')]
            
            if len(parts) < 6:
                continue
            
            # Extract fields
            date = parts[0]
            away_team = parts[2]
            away_pts = parts[3]
            home_team = parts[4]
            home_pts = parts[5]
            
            # Skip if scores are missing (future games)
            if not away_pts or not home_pts or away_pts == '' or home_pts == '':
                continue
            
            try:
                results.append({
                    'game_date': date,
                    'away_team': away_team,
                    'home_team': home_team,
                    'away_score': int(away_pts),
                    'home_score': int(home_pts),
                    'home_spread': int(home_pts) - int(away_pts)
                })
            except ValueError:
                continue
        
        return pd.DataFrame(results)


class PredictionMatcher:
    """Match predictions to actual game results"""
    
    # Team name variations/aliases for matching
    TEAM_ALIASES = {
        'LA Clippers': 'Los Angeles Clippers',
        'LA Lakers': 'Los Angeles Lakers',
        'L.A. Clippers': 'Los Angeles Clippers',
        'L.A. Lakers': 'Los Angeles Lakers',
    }
    
    @staticmethod
    def normalize_team_name(name: str) -> str:
        """Normalize team name for matching"""
        name = name.strip()
        return PredictionMatcher.TEAM_ALIASES.get(name, name)
    
    @staticmethod
    def match_predictions_to_results(
        predictions: List[Dict],
        results_df: pd.DataFrame,
        date_tolerance_days: int = 2
    ) -> List[Dict]:
        """
        Match predictions to actual game results
        
        Parameters:
        - predictions: List of prediction dicts from predictor
        - results_df: DataFrame with actual game results
        - date_tolerance_days: How many days to search for matching games
        
        Returns:
        - List of matched prediction-result pairs
        """
        matches = []
        
        for pred in predictions:
            home_team = PredictionMatcher.normalize_team_name(pred.get('home_team', ''))
            away_team = PredictionMatcher.normalize_team_name(pred.get('away_team', ''))
            
            # Find matching game in results
            for _, result in results_df.iterrows():
                result_home = PredictionMatcher.normalize_team_name(result['home_team'])
                result_away = PredictionMatcher.normalize_team_name(result['away_team'])
                
                if result_home == home_team and result_away == away_team:
                    match = {
                        'home_team': home_team,
                        'away_team': away_team,
                        'game_date': result['game_date'],
                        
                        # Prediction data
                        'predicted_spread': pred.get('predicted_spread', 0),
                        'predicted_winner': home_team if pred.get('predicted_spread', 0) > 0 else away_team,
                        'win_probability': pred.get('win_probability', 0.5),
                        'uncertainty': pred.get('uncertainty', 0),
                        'confidence': pred.get('confidence', 'LOW'),
                        
                        # Actual results
                        'actual_home_score': result['home_score'],
                        'actual_away_score': result['away_score'],
                        'actual_spread': result['home_spread'],
                        'actual_winner': result['home_team'] if result['home_spread'] > 0 else result['away_team'],
                        
                        # Calculated errors
                        'spread_error': abs(pred.get('predicted_spread', 0) - result['home_spread']),
                        'correct_winner': (pred.get('predicted_spread', 0) > 0) == (result['home_spread'] > 0),
                        'confidence_justified': None,  # Will calculate
                    }
                    
                    # Check if confidence was justified
                    if match['confidence'] == 'HIGH':
                        match['confidence_justified'] = match['spread_error'] < 8 and match['correct_winner']
                    elif match['confidence'] == 'MEDIUM':
                        match['confidence_justified'] = match['spread_error'] < 12
                    else:  # LOW
                        match['confidence_justified'] = True  # Low confidence = we knew it was uncertain
                    
                    matches.append(match)
                    break
        
        return matches


class ErrorAnalyzer:
    """Analyze prediction errors to identify systematic biases"""
    
    @staticmethod
    def analyze_errors(matches: List[Dict]) -> Dict:
        """
        Comprehensive error analysis
        
        Returns:
        - Dict with error patterns, biases, and insights
        """
        if not matches:
            return {'error': 'No matches to analyze'}
        
        df = pd.DataFrame(matches)
        
        # Calculate basic metrics
        n_predictions = len(df)
        correct_winners = df['correct_winner'].sum()
        win_accuracy = correct_winners / n_predictions
        
        mae = df['spread_error'].mean()
        rmse = np.sqrt((df['spread_error'] ** 2).mean())
        
        # Analyze by confidence level
        confidence_analysis = {}
        for conf in ['HIGH', 'MEDIUM', 'LOW']:
            conf_matches = df[df['confidence'] == conf]
            if len(conf_matches) > 0:
                confidence_analysis[conf] = {
                    'count': len(conf_matches),
                    'win_accuracy': conf_matches['correct_winner'].mean(),
                    'mae': conf_matches['spread_error'].mean(),
                    'confidence_justified_rate': conf_matches['confidence_justified'].mean()
                }
        
        # Identify overconfident and underconfident predictions
        high_conf_wrong = df[(df['confidence'] == 'HIGH') & (~df['correct_winner'])]
        low_conf_right = df[(df['confidence'] == 'LOW') & (df['correct_winner']) & (df['spread_error'] < 5)]
        
        # Bias analysis - are we consistently over/under predicting for home teams?
        actual_spreads = df['actual_spread'].values
        predicted_spreads = df['predicted_spread'].values
        
        # Check for systematic bias
        mean_bias = (predicted_spreads - actual_spreads).mean()
        bias_direction = 'home-favoring' if mean_bias > 0 else 'away-favoring'
        
        # Analyze error distribution
        errors = predicted_spreads - actual_spreads
        error_skew = stats.skew(errors)
        error_kurtosis = stats.kurtosis(errors)
        
        # Identify games with largest errors (outliers to investigate)
        df['abs_error'] = df['spread_error']
        worst_predictions = df.nlargest(5, 'abs_error')[
            ['home_team', 'away_team', 'predicted_spread', 'actual_spread', 'spread_error']
        ].to_dict('records')
        
        # Best predictions
        best_predictions = df.nsmallest(5, 'abs_error')[
            ['home_team', 'away_team', 'predicted_spread', 'actual_spread', 'spread_error']
        ].to_dict('records')
        
        return {
            'overall_metrics': {
                'n_predictions': n_predictions,
                'correct_winners': correct_winners,
                'win_accuracy': win_accuracy,
                'mae': mae,
                'rmse': rmse,
            },
            'bias_analysis': {
                'mean_bias': mean_bias,
                'bias_direction': bias_direction,
                'error_skew': error_skew,
                'error_kurtosis': error_kurtosis,
            },
            'confidence_analysis': confidence_analysis,
            'problem_areas': {
                'overconfident_errors': len(high_conf_wrong),
                'underconfident_successes': len(low_conf_right),
            },
            'worst_predictions': worst_predictions,
            'best_predictions': best_predictions,
        }


class AdaptiveLearner:
    """
    Use MCMC and adaptive methods to learn from prediction errors
    """
    
    def __init__(self, mcmc_model=None):
        """
        Initialize adaptive learner
        
        Parameters:
        - mcmc_model: BayesianBasketballHierarchical instance (optional)
        """
        self.mcmc_model = mcmc_model
        self.learning_history = []
    
    def calculate_team_error_adjustments(
        self,
        matches: List[Dict],
        team_data: Dict
    ) -> Dict[str, float]:
        """
        Calculate per-team bias adjustments based on prediction errors
        
        Logic:
        - If we consistently over-predict for a team, reduce their rating
        - If we consistently under-predict, increase their rating
        
        Returns:
        - Dict mapping team names to adjustment factors
        """
        # Group errors by team
        team_errors = {}
        
        for match in matches:
            home_team = match['home_team']
            away_team = match['away_team']
            
            # Error in perspective of each team
            # Positive error = we over-predicted home team's performance
            error = match['predicted_spread'] - match['actual_spread']
            
            # Track errors for both teams
            if home_team not in team_errors:
                team_errors[home_team] = []
            if away_team not in team_errors:
                team_errors[away_team] = []
            
            # Home team: positive error means we rated them too high
            team_errors[home_team].append(error)
            # Away team: positive error means we rated them too low
            team_errors[away_team].append(-error)
        
        # Calculate adjustments
        adjustments = {}
        for team, errors in team_errors.items():
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            n = len(errors)
            
            # Statistical significance check (t-test)
            if n > 3 and std_error > 0:
                t_stat = mean_error / (std_error / np.sqrt(n))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
                
                # Only adjust if statistically significant (p < 0.10)
                if p_value < 0.10:
                    # Adjustment proportional to error (but dampened for stability)
                    adjustment = -mean_error * 0.1  # 10% learning rate
                    adjustments[team] = adjustment
                else:
                    adjustments[team] = 0.0
            else:
                adjustments[team] = 0.0
        
        return adjustments
    
    def propose_mcmc_refinement(
        self,
        matches: List[Dict],
        current_epaa_weight: float = 0.5
    ) -> Dict:
        """
        Propose MCMC model refinements based on error analysis
        
        Returns:
        - Dict with proposed changes to model parameters
        """
        df = pd.DataFrame(matches)
        
        # Analyze current EPAA weight effectiveness
        # If errors are high, we may need to adjust EPAA weighting
        
        mae = df['spread_error'].mean()
        win_accuracy = df['correct_winner'].mean()
        
        # Determine if EPAA weight should change
        proposed_weight = current_epaa_weight
        reasoning = []
        
        if win_accuracy < 0.55:
            # Poor winner prediction - maybe rely more on rolling stats
            proposed_weight = max(0.2, current_epaa_weight - 0.1)
            reasoning.append("Low win accuracy suggests EPAA may be less predictive")
        elif win_accuracy > 0.70 and mae > 10:
            # Good winner prediction but poor spread accuracy
            # EPAA is capturing right direction but magnitude is off
            proposed_weight = min(0.8, current_epaa_weight + 0.05)
            reasoning.append("High win accuracy with spread errors suggests EPAA direction is good")
        elif mae < 8:
            # Good performance overall - maintain or slight increase
            proposed_weight = min(0.7, current_epaa_weight + 0.02)
            reasoning.append("Strong overall performance - slight increase in EPAA weight")
        
        # Analyze uncertainty calibration
        # Check if high-uncertainty games actually had larger errors
        high_unc = df[df['uncertainty'] > df['uncertainty'].median()]
        low_unc = df[df['uncertainty'] <= df['uncertainty'].median()]
        
        uncertainty_calibrated = (
            high_unc['spread_error'].mean() > low_unc['spread_error'].mean()
        ) if len(high_unc) > 0 and len(low_unc) > 0 else False
        
        if not uncertainty_calibrated:
            reasoning.append("Uncertainty estimates not well-calibrated - consider GP kernel tuning")
        
        return {
            'current_epaa_weight': current_epaa_weight,
            'proposed_epaa_weight': proposed_weight,
            'weight_change': proposed_weight - current_epaa_weight,
            'reasoning': reasoning,
            'metrics': {
                'mae': mae,
                'win_accuracy': win_accuracy,
                'uncertainty_calibrated': uncertainty_calibrated
            }
        }
    
    def generate_learning_report(
        self,
        matches: List[Dict],
        team_data: Dict,
        current_epaa_weight: float = 0.5
    ) -> str:
        """
        Generate comprehensive learning report with actionable insights
        
        Returns:
        - Formatted markdown report
        """
        # Run all analyses
        error_analysis = ErrorAnalyzer.analyze_errors(matches)
        team_adjustments = self.calculate_team_error_adjustments(matches, team_data)
        mcmc_refinement = self.propose_mcmc_refinement(matches, current_epaa_weight)
        
        # Build report
        report = []
        report.append("# 🎯 Adaptive Learning Report\n")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Games Analyzed:** {error_analysis['overall_metrics']['n_predictions']}\n")
        
        # Overall Performance
        report.append("\n## 📊 Overall Performance\n")
        metrics = error_analysis['overall_metrics']
        report.append(f"- **Win Accuracy:** {metrics['win_accuracy']:.1%} ({metrics['correct_winners']}/{metrics['n_predictions']})")
        report.append(f"- **Mean Absolute Error:** {metrics['mae']:.2f} points")
        report.append(f"- **RMSE:** {metrics['rmse']:.2f} points\n")
        
        # Bias Analysis
        report.append("\n## 🎲 Bias Analysis\n")
        bias = error_analysis['bias_analysis']
        report.append(f"- **Mean Bias:** {bias['mean_bias']:+.2f} points ({bias['bias_direction']})")
        report.append(f"- **Error Distribution:** Skew={bias['error_skew']:.3f}, Kurtosis={bias['error_kurtosis']:.3f}")
        
        if abs(bias['mean_bias']) > 2:
            report.append(f"\n⚠️ **Systematic bias detected!** Model is {bias['bias_direction']} by {abs(bias['mean_bias']):.1f} points on average.\n")
        
        # Confidence Analysis
        report.append("\n## 🎯 Confidence Calibration\n")
        for conf, data in error_analysis['confidence_analysis'].items():
            report.append(f"\n### {conf} Confidence:")
            report.append(f"- Games: {data['count']}")
            report.append(f"- Win Accuracy: {data['win_accuracy']:.1%}")
            report.append(f"- MAE: {data['mae']:.2f} points")
            report.append(f"- Confidence Justified: {data['confidence_justified_rate']:.1%}")
        
        # Problem Areas
        report.append("\n## ⚠️ Problem Areas\n")
        problems = error_analysis['problem_areas']
        report.append(f"- **Overconfident Errors:** {problems['overconfident_errors']} (high confidence, wrong winner)")
        report.append(f"- **Underconfident Successes:** {problems['underconfident_successes']} (low confidence, good prediction)\n")
        
        # Worst Predictions (learning opportunities)
        report.append("\n## 🔍 Worst Predictions (Learn From These)\n")
        for i, pred in enumerate(error_analysis['worst_predictions'], 1):
            report.append(f"{i}. **{pred['away_team']} @ {pred['home_team']}**")
            report.append(f"   - Predicted: {pred['predicted_spread']:+.1f} | Actual: {pred['actual_spread']:+.1f} | Error: {pred['spread_error']:.1f}\n")
        
        # Team-Specific Adjustments
        report.append("\n## 🔧 Proposed Team Adjustments\n")
        significant_adjustments = {k: v for k, v in team_adjustments.items() if abs(v) > 0.5}
        
        if significant_adjustments:
            report.append("Teams with statistically significant biases:\n")
            for team, adj in sorted(significant_adjustments.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
                direction = "↑ Underrated" if adj > 0 else "↓ Overrated"
                report.append(f"- **{team}:** {adj:+.2f} points {direction}")
        else:
            report.append("✅ No statistically significant team biases detected.\n")
        
        # MCMC Refinement Proposals
        report.append("\n## 🔬 MCMC Model Refinement\n")
        report.append(f"**Current EPAA Weight:** {mcmc_refinement['current_epaa_weight']:.2f}")
        report.append(f"**Proposed EPAA Weight:** {mcmc_refinement['proposed_epaa_weight']:.2f}")
        report.append(f"**Change:** {mcmc_refinement['weight_change']:+.2f}\n")
        
        report.append("**Reasoning:**")
        for reason in mcmc_refinement['reasoning']:
            report.append(f"- {reason}")
        
        # Action Items
        report.append("\n## ✅ Recommended Actions\n")
        report.append("1. **Update EPAA Weight:** Adjust from {:.2f} to {:.2f}".format(
            mcmc_refinement['current_epaa_weight'],
            mcmc_refinement['proposed_epaa_weight']
        ))
        
        if abs(bias['mean_bias']) > 2:
            report.append("2. **Correct Systematic Bias:** Add {:.2f} point adjustment to all predictions".format(-bias['mean_bias']))
        
        if problems['overconfident_errors'] > len(matches) * 0.1:
            report.append("3. **Recalibrate Confidence:** Reduce confidence thresholds (too many overconfident errors)")
        
        if not mcmc_refinement['metrics']['uncertainty_calibrated']:
            report.append("4. **Retune GP Kernel:** Uncertainty estimates need recalibration")
        
        report.append("\n---")
        report.append("\n*This report uses statistical analysis and MCMC principles to identify systematic errors and propose model improvements.*\n")
        
        return '\n'.join(report)


def validate_and_learn(
    predictions: List[Dict],
    results_csv: str,
    team_data: Dict,
    current_epaa_weight: float = 0.5,
    save_matches: bool = True,
    output_file: str = 'json/validation_matches.json'
) -> Dict:
    """
    Complete validation and learning pipeline
    
    Parameters:
    - predictions: List of prediction dicts
    - results_csv: CSV text with game results
    - team_data: Team data dict
    - current_epaa_weight: Current EPAA weight in use
    - save_matches: Save matched predictions to file
    - output_file: Where to save matches
    
    Returns:
    - Dict with all analysis results and learning recommendations
    """
    print("🔄 Starting validation and learning pipeline...\n")
    
    # Step 1: Parse results
    print("📋 Parsing game results...")
    csv_lines = results_csv.strip().split('\n')
    results_df = GameResultParser.parse_csv_sports_reference(csv_lines)
    print(f"✅ Parsed {len(results_df)} completed games\n")
    
    # Step 2: Match predictions to results
    print("🔗 Matching predictions to actual results...")
    matches = PredictionMatcher.match_predictions_to_results(predictions, results_df)
    print(f"✅ Matched {len(matches)} predictions\n")
    
    if len(matches) == 0:
        print("❌ No matches found. Check team name alignment or date ranges.")
        return {'error': 'No matches found'}
    
    # Save matches
    if save_matches:
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(matches, f, indent=2)
        print(f"💾 Saved matches to {output_file}\n")
    
    # Step 3: Generate learning report
    print("🧠 Generating adaptive learning report...\n")
    learner = AdaptiveLearner()
    report = learner.generate_learning_report(matches, team_data, current_epaa_weight)
    
    # Return everything
    return {
        'matches': matches,
        'report': report,
        'n_matches': len(matches),
        'results_df': results_df,
        'error_analysis': ErrorAnalyzer.analyze_errors(matches),
        'team_adjustments': learner.calculate_team_error_adjustments(matches, team_data),
        'mcmc_refinement': learner.propose_mcmc_refinement(matches, current_epaa_weight)
    }
