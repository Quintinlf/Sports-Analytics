"""
Prediction Module for NBA Games

Handles:
- Single game predictions using GP model
- GP + MCMC predictions with EPAA adjustments  
- Win probability calculations
- Confidence level determination
- Feature explanations
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def predict_game_gp(home_team_name, away_team_name, gp_model, games_df, team_data):
    """
    Predict game outcome using Gaussian Process model
    
    Parameters:
    - home_team_name: Home team name
    - away_team_name: Away team name  
    - gp_model: Trained GaussianProcessPredictor
    - games_df: DataFrame with rolling stats
    - team_data: Dict from get_all_nba_teams()
    
    Returns:
    - Dict with prediction, uncertainty, win probability, confidence
    """
    from data_loader import get_team_latest_stats
    
    # Get team IDs
    team_names_inv = {v: k for k, v in team_data['names'].items()}
    home_team_id = team_names_inv.get(home_team_name)
    away_team_id = team_names_inv.get(away_team_name)
    
    if home_team_id is None or away_team_id is None:
        raise ValueError(f"Team not found: {home_team_name} or {away_team_name}")
    
    # Get latest stats for both teams
    home_stats = get_team_latest_stats(games_df, home_team_id)
    away_stats = get_team_latest_stats(games_df, away_team_id)
    
    if home_stats is None or away_stats is None:
        raise ValueError("Could not find recent stats for teams")
    
    # Use feature names from model if available, otherwise extract from games_df
    if gp_model.feature_names:
        feature_cols = gp_model.feature_names
    else:
        # Extract features from games_df - only rolling stats (not WIN_STREAK, etc.)
        base_cols = [col for col in games_df.columns if '_ROLL' in col]
        feature_cols = [col for col in base_cols if col.startswith(('HOME_', 'AWAY_'))]
    
    # Build feature vector by combining home and away stats
    # The feature names include the prefix (HOME_ or AWAY_) so we need to match correctly
    features_list = []
    for col in feature_cols:
        if col.startswith('HOME_'):
            stat_name = col[5:]  # Remove 'HOME_' prefix to get the actual stat column name
            features_list.append(home_stats.get(stat_name, 0.0))
        elif col.startswith('AWAY_'):
            stat_name = col[5:]  # Remove 'AWAY_' prefix
            features_list.append(away_stats.get(stat_name, 0.0))
        else:
            features_list.append(0.0)
    
    # Convert to 2D array for prediction
    if len(features_list) == 0:
        raise ValueError(f"No features could be constructed. Feature columns: {feature_cols}")
    
    features = np.array([features_list])
    
    # Make prediction with uncertainty
    pred_diff, pred_std = gp_model.predict(features, return_std=True)
    pred_diff = pred_diff[0]
    pred_std = pred_std[0]
    
    # Calculate win probability (logistic function)
    # P(home wins) = 1 / (1 + exp(-k * point_diff))
    # k = 0.15 works well empirically
    win_prob = 1.0 / (1.0 + np.exp(-0.15 * pred_diff))
    
    # Confidence level based on uncertainty and win probability
    # High confidence: Low uncertainty AND clear winner (prob > 0.65 or < 0.35)
    # Medium confidence: Moderate uncertainty OR close game
    # Low confidence: High uncertainty OR very close game
    
    uncertainty_factor = pred_std / 12.0  # Normalize (12 pts is high uncertainty)
    prob_certainty = abs(win_prob - 0.5) * 2  # 0 = coin flip, 1 = certain
    
    confidence_score = prob_certainty * (1 - uncertainty_factor)
    
    if confidence_score > 0.5 and win_prob > 0.65:
        confidence = "HIGH"
    elif confidence_score > 0.3 and win_prob > 0.60:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    
    return {
        'home_team': home_team_name,
        'away_team': away_team_name,
        'predicted_spread': pred_diff,
        'uncertainty': pred_std,
        'lower_bound': pred_diff - 1.96 * pred_std,
        'upper_bound': pred_diff + 1.96 * pred_std,
        'win_probability': win_prob,
        'confidence': confidence,
        'confidence_score': confidence_score
    }


def predict_game_with_epaa(home_team_name, away_team_name, gp_model, games_df, team_data,
                            epaa_data, epaa_weight=0.5):
    """
    Predict game with EPAA adjustment from MCMC model
    
    Parameters:
    - home_team_name: Home team name
    - away_team_name: Away team name
    - gp_model: Trained GaussianProcessPredictor
    - games_df: DataFrame with rolling stats
    - team_data: Dict from get_all_nba_teams()
    - epaa_data: Dict {team_id: {'epaa_mean': float, 'epaa_std': float, ...}}
    - epaa_weight: Weight for EPAA adjustment (0-1, default: 0.5)
    
    Returns:
    - Dict with base prediction + EPAA-adjusted prediction
    """
    # Get base GP prediction
    base_pred = predict_game_gp(home_team_name, away_team_name, gp_model, games_df, team_data)
    
    # Get team IDs
    team_names_inv = {v: k for k, v in team_data['names'].items()}
    home_team_id = team_names_inv.get(home_team_name)
    away_team_id = team_names_inv.get(away_team_name)
    
    # Get EPAA values
    home_epaa = 0.0
    away_epaa = 0.0
    home_epaa_std = 0.0
    away_epaa_std = 0.0
    
    if home_team_id in epaa_data:
        home_epaa = epaa_data[home_team_id]['epaa_mean']
        home_epaa_std = epaa_data[home_team_id]['epaa_std']
    
    if away_team_id in epaa_data:
        away_epaa = epaa_data[away_team_id]['epaa_mean']
        away_epaa_std = epaa_data[away_team_id]['epaa_std']
    
    # EPAA differential (home advantage in offensive efficiency)
    epaa_diff = home_epaa - away_epaa
    epaa_uncertainty = np.sqrt(home_epaa_std**2 + away_epaa_std**2)
    
    # Adjusted prediction: Weighted combination
    adjusted_spread = base_pred['predicted_spread'] + (epaa_weight * epaa_diff)
    
    # Combined uncertainty
    combined_uncertainty = np.sqrt(base_pred['uncertainty']**2 + (epaa_weight * epaa_uncertainty)**2)
    
    # Recalculate win probability with adjusted spread
    adjusted_win_prob = 1.0 / (1.0 + np.exp(-0.15 * adjusted_spread))
    
    # Recalculate confidence
    uncertainty_factor = combined_uncertainty / 12.0
    prob_certainty = abs(adjusted_win_prob - 0.5) * 2
    confidence_score = prob_certainty * (1 - uncertainty_factor)
    
    if confidence_score > 0.5 and adjusted_win_prob > 0.65:
        confidence = "HIGH"
    elif confidence_score > 0.3 and adjusted_win_prob > 0.60:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    
    return {
        'home_team': home_team_name,
        'away_team': away_team_name,
        
        # Base GP prediction
        'gp_spread': base_pred['predicted_spread'],
        'gp_uncertainty': base_pred['uncertainty'],
        'gp_win_prob': base_pred['win_probability'],
        
        # EPAA adjustment
        'home_epaa': home_epaa,
        'away_epaa': away_epaa,
        'epaa_diff': epaa_diff,
        'epaa_weight_used': epaa_weight,
        
        # Adjusted prediction
        'predicted_spread': adjusted_spread,
        'uncertainty': combined_uncertainty,
        'lower_bound': adjusted_spread - 1.96 * combined_uncertainty,
        'upper_bound': adjusted_spread + 1.96 * combined_uncertainty,
        'win_probability': adjusted_win_prob,
        'confidence': confidence,
        'confidence_score': confidence_score
    }


def format_prediction_text(prediction):
    """
    Format prediction dict into readable text
    
    Parameters:
    - prediction: Dict from predict_game_gp or predict_game_with_epaa
    
    Returns:
    - Formatted string
    """
    home = prediction['home_team']
    away = prediction['away_team']
    spread = prediction['predicted_spread']
    uncertainty = prediction['uncertainty']
    win_prob = prediction['win_probability']
    confidence = prediction['confidence']
    
    if spread > 0:
        favorite = home
        underdog = away
        margin = spread
    else:
        favorite = away
        underdog = home
        margin = abs(spread)
    
    text = f"\n{'='*70}\n"
    text += f"🏀 {home} (HOME) vs {away} (AWAY)\n"
    text += f"{'='*70}\n\n"
    
    text += f"📊 PREDICTION:\n"
    text += f"   Spread: {spread:+.1f} points (±{uncertainty:.1f})\n"
    text += f"   Favorite: {favorite} by {margin:.1f} points\n"
    text += f"   Win Probability: {win_prob:.1%}\n"
    text += f"   Confidence: {confidence}\n\n"
    
    # EPAA info if available
    if 'epaa_diff' in prediction:
        text += f"🎯 EPAA ADJUSTMENT:\n"
        text += f"   {home} EPAA: {prediction['home_epaa']:+.2f}\n"
        text += f"   {away} EPAA: {prediction['away_epaa']:+.2f}\n"
        text += f"   Differential: {prediction['epaa_diff']:+.2f}\n"
        text += f"   Weight Used: {prediction['epaa_weight_used']:.0%}\n\n"
    
    text += f"📈 95% CONFIDENCE INTERVAL:\n"
    text += f"   {prediction['lower_bound']:.1f} to {prediction['upper_bound']:.1f} points\n"
    text += f"{'='*70}\n"
    
    return text


if __name__ == "__main__":
    print("🏀 Predictor module loaded successfully!")
    print("✅ Available functions:")
    print("   - predict_game_gp(): GP model predictions")
    print("   - predict_game_with_epaa(): GP + MCMC predictions")
    print("   - format_prediction_text(): Format output")
