"""
Monte Carlo Simulator for NBA Game Predictions
Runs thousands of simulations per game using model quantile outputs.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats as scipy_stats

# Import model and data loader
from learners.train_lgbm_model import LGBMWinPredictor
from loaders.data_loader import (
    fetch_nba_games,
    calculate_rolling_stats,
    get_all_nba_teams
)


def triangular_sample(q10, q50, q90, n_samples=1):
    """
    Sample from asymmetric triangular distribution using three quantiles.
    
    Parameters:
    - q10, q50, q90: The 10th, 50th, and 90th percentiles
    - n_samples: Number of samples to draw
    
    Returns:
    - Array of sampled values
    """
    # Use scipy triangular distribution
    # Mode parameter c = (mode - left) / (right - left)
    left = q10
    right = q90
    mode = q50
    
    if right <= left:
        # Degenerate case: return mode
        return np.full(n_samples, mode)
    
    c = (mode - left) / (right - left)
    c = np.clip(c, 0.01, 0.99)  # Ensure valid triangular distribution
    
    # Sample from triangular
    samples = scipy_stats.triang.rvs(c, loc=left, scale=right - left, size=n_samples)
    return samples


def get_team_latest_features(team_name, team_name_to_id, games_df_with_stats):
    """
    Get the most recent rolling stats for a team.
    
    Args:
        team_name: Full team name (e.g., 'Philadelphia 76ers')
        team_name_to_id: Dict mapping team names to IDs
        games_df_with_stats: DataFrame with rolling stats
    
    Returns:
        dict of feature_name: value
    """
    import pandas as pd
    
    team_id = team_name_to_id.get(team_name)
    if not team_id:
        return {}
    
    # Get team's games, sorted by date
    team_games = games_df_with_stats[games_df_with_stats['TEAM_ID'] == team_id].sort_values('GAME_DATE')
    
    if len(team_games) == 0:
        return {}
    
    # Get most recent game's stats
    latest = team_games.iloc[-1]
    
    # Extract rolling features
    features = {}
    for col in games_df_with_stats.columns:
        if '_ROLL' in col or col in ['WIN_STREAK', 'REST_DAYS', 'IS_BACK_TO_BACK', 'WIN_RATE_10']:
            features[col] = latest[col] if pd.notna(latest[col]) else 0.0
    
    return features


def run_monte_carlo_for_game(model, home_team, away_team, team_name_to_id, games_df_with_stats, n_simulations=10000):
    """
    Run Monte Carlo simulation for a single game.
    
    Parameters:
    - model: LGBMWinPredictor instance
    - home_team: Home team name
    - away_team: Away team name
    - team_name_to_id: Dict mapping team names to IDs
    - games_df_with_stats: DataFrame with rolling stats
    - n_simulations: Number of simulations to run
    
    Returns:
    - Dictionary with simulation results
    """
    import pandas as pd
    
    # Get latest features for both teams
    home_features = get_team_latest_features(home_team, team_name_to_id, games_df_with_stats)
    away_features = get_team_latest_features(away_team, team_name_to_id, games_df_with_stats)
    
    if not home_features or not away_features:
        raise ValueError(f"Missing features for {home_team} or {away_team}")
    
    # Create matchup feature vector in expected order
    feature_row = {}
    for col, value in home_features.items():
        feature_row[f'HOME_{col}'] = value
    for col, value in away_features.items():
        feature_row[f'AWAY_{col}'] = value
    
    # Convert to DataFrame with correct feature order
    X = pd.DataFrame([feature_row])
    
    # Align with model's feature names (fill missing with 0)
    for feat in model.feature_names:
        if feat not in X.columns:
            X[feat] = 0.0
    X = X[model.feature_names]
    
    # Get quantile predictions from model
    X_scaled = model.scaler.transform(X)
    quantiles = model.quantile_model.predict(X_scaled)
    
    q10 = quantiles['q10'][0]
    q50 = quantiles['q50'][0]
    q90 = quantiles['q90'][0]
    
    # Run simulations
    point_diffs = triangular_sample(q10, q50, q90, n_simulations)
    
    # Convert to win outcomes (home team wins if point_diff > 0)
    home_wins = (point_diffs > 0).astype(int)
    
    # Calculate statistics
    mc_win_prob = home_wins.mean()
    mc_median_spread = np.median(point_diffs)
    mc_mean_spread = np.mean(point_diffs)
    mc_std_spread = np.std(point_diffs)
    
    # Confidence intervals
    ci_95_low, ci_95_high = np.percentile(point_diffs, [2.5, 97.5])
    ci_90_low, ci_90_high = np.percentile(point_diffs, [5, 95])
    ci_80_low, ci_80_high = np.percentile(point_diffs, [10, 90])
    
    # Model's calibrated prediction for comparison
    model_pred = model.predict_win_probability(X)
    
    return {
        'mc_win_prob': mc_win_prob,
        'mc_median_spread': mc_median_spread,
        'mc_mean_spread': mc_mean_spread,
        'mc_std_spread': mc_std_spread,
        'mc_ci_95': (ci_95_low, ci_95_high),
        'mc_ci_90': (ci_90_low, ci_90_high),
        'mc_ci_80': (ci_80_low, ci_80_high),
        'model_win_prob': model_pred['win_prob'][0],
        'model_confidence': model_pred['confidence_label'][0],
        'model_confidence_score': model_pred['confidence_score'][0],
        'quantiles': {'q10': q10, 'q50': q50, 'q90': q90},
        'point_diff_samples': point_diffs,
    }


def main():
    parser = argparse.ArgumentParser(description='Run Monte Carlo simulations for NBA games')
    parser.add_argument('--model', type=str, default='machine_learning/models/lgbm_win_predictor_latest.pkl',
                        help='Path to trained LGBMWinPredictor model')
    parser.add_argument('--date', type=str, default='2026-02-19',
                        help='Date for predictions (YYYY-MM-DD)')
    parser.add_argument('--n', type=int, default=10000,
                        help='Number of Monte Carlo simulations per game')
    parser.add_argument('--out', type=str, default='diagnostics/mc_simulations.csv',
                        help='Output CSV file path')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"🎲 MONTE CARLO SIMULATOR")
    print(f"{'='*80}\n")
    print(f"📅 Target Date: {args.date}")
    print(f"🔢 Simulations per game: {args.n:,}")
    print(f"📦 Model: {args.model}\n")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = LGBMWinPredictor.load(args.model)
    print("✅ Model loaded\n")
    
    # Load latest team stats
    print("Loading latest team statistics...")
    games_df = fetch_nba_games(seasons=['2024-25'], season_type='Regular Season', verbose=False)
    games_with_stats = calculate_rolling_stats(games_df, window=5)
    print(f"✅ Loaded {len(games_with_stats)} games with rolling stats")
    print(f"   Date range: {games_with_stats['GAME_DATE'].min()} → {games_with_stats['GAME_DATE'].max()}")
    
    # Get team mappings
    team_data = get_all_nba_teams()
    team_name_to_id = {team['full_name']: team['id'] for team in team_data['teams']}
    print(f"✅ Team mappings ready for {len(team_name_to_id)} teams\n")
    
    # Define games for Feb 19, 2026 (from the notebook output)
    games = [
        {'away': 'Atlanta Hawks', 'home': 'Philadelphia 76ers', 'time': '7:00p'},
        {'away': 'Indiana Pacers', 'home': 'Washington Wizards', 'time': '7:00p'},
        {'away': 'Detroit Pistons', 'home': 'New York Knicks', 'time': '7:30p'},
        {'away': 'Toronto Raptors', 'home': 'Chicago Bulls', 'time': '8:00p'},
        {'away': 'Phoenix Suns', 'home': 'San Antonio Spurs', 'time': '8:30p'},
        {'away': 'Boston Celtics', 'home': 'Golden State Warriors', 'time': '10:00p'},
        {'away': 'Orlando Magic', 'home': 'Sacramento Kings', 'time': '10:00p'},
        {'away': 'Denver Nuggets', 'home': 'Los Angeles Clippers', 'time': '10:30p'},
    ]
    
    print(f"Running Monte Carlo simulations for {len(games)} games...\n")
    
    # Run simulations for each game
    results = []
    
    for i, game in enumerate(games, 1):
        away_team = game['away']
        home_team = game['home']
        
        print(f"{'─'*80}")
        print(f"GAME {i}/{len(games)}: {away_team} @ {home_team}")
        print(f"{'─'*80}")
        
        # Check if teams exist
        if away_team not in team_name_to_id or home_team not in team_name_to_id:
            print(f"⚠️  Missing team mapping for {away_team} or {home_team}, skipping...")
            continue
        
        # Run Monte Carlo
        print(f"Running {args.n:,} simulations...")
        try:
            mc_result = run_monte_carlo_for_game(
                model, home_team, away_team, team_name_to_id, games_with_stats, args.n
            )
        except Exception as e:
            print(f"⚠️  Error running simulation: {e}, skipping...")
            continue
        
        # Determine predicted winner
        if mc_result['mc_win_prob'] > 0.5:
            winner = home_team
            pred_winner_prob = mc_result['mc_win_prob']
        else:
            winner = away_team
            pred_winner_prob = 1 - mc_result['mc_win_prob']
        
        print(f"\n📊 SIMULATION RESULTS:")
        print(f"  🏆 Predicted Winner:    {winner}")
        print(f"  🎯 MC Win Probability:  {mc_result['mc_win_prob']:.1%} (home)")
        print(f"  📈 Winner Probability:  {pred_winner_prob:.1%}")
        print(f"  📊 Expected Spread:     {mc_result['mc_median_spread']:+.1f} points (median)")
        print(f"  📊 Mean Spread:         {mc_result['mc_mean_spread']:+.1f} ± {mc_result['mc_std_spread']:.1f}")
        print(f"  📉 95% CI:              [{mc_result['mc_ci_95'][0]:+.1f}, {mc_result['mc_ci_95'][1]:+.1f}]")
        print(f"  📉 90% CI:              [{mc_result['mc_ci_90'][0]:+.1f}, {mc_result['mc_ci_90'][1]:+.1f}]")
        print(f"  📉 80% CI:              [{mc_result['mc_ci_80'][0]:+.1f}, {mc_result['mc_ci_80'][1]:+.1f}]")
        print(f"  🔮 Model Quantiles:     Q10={mc_result['quantiles']['q10']:+.1f}, "
              f"Q50={mc_result['quantiles']['q50']:+.1f}, Q90={mc_result['quantiles']['q90']:+.1f}")
        print(f"  💪 Model Confidence:    {mc_result['model_confidence']} "
              f"(score={mc_result['model_confidence_score']:.3f})")
        print(f"  🎲 Simulations:         {args.n:,}")
        print()
        
        # Store result
        results.append({
            'game_num': i,
            'away_team': away_team,
            'home_team': home_team,
            'time': game['time'],
            'predicted_winner': winner,
            'mc_win_prob_home': mc_result['mc_win_prob'],
            'mc_win_prob_winner': pred_winner_prob,
            'mc_median_spread': mc_result['mc_median_spread'],
            'mc_mean_spread': mc_result['mc_mean_spread'],
            'mc_std_spread': mc_result['mc_std_spread'],
            'ci_95_low': mc_result['mc_ci_95'][0],
            'ci_95_high': mc_result['mc_ci_95'][1],
            'ci_90_low': mc_result['mc_ci_90'][0],
            'ci_90_high': mc_result['mc_ci_90'][1],
            'ci_80_low': mc_result['mc_ci_80'][0],
            'ci_80_high': mc_result['mc_ci_80'][1],
            'q10': mc_result['quantiles']['q10'],
            'q50': mc_result['quantiles']['q50'],
            'q90': mc_result['quantiles']['q90'],
            'model_win_prob': mc_result['model_win_prob'],
            'model_confidence': mc_result['model_confidence'],
            'model_confidence_score': mc_result['model_confidence_score'],
            'n_simulations': args.n,
        })
    
    # Save results
    df_results = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df_results.to_csv(args.out, index=False)
    
    print(f"\n{'='*80}")
    print(f"✅ SIMULATIONS COMPLETE")
    print(f"{'='*80}\n")
    print(f"📁 Results saved to: {args.out}")
    print(f"📊 Total games simulated: {len(results)}")
    print(f"🎲 Total simulations run: {len(results) * args.n:,}")
    
    # Summary statistics
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"  Average winner probability: {df_results['mc_win_prob_winner'].mean():.1%}")
    print(f"  Highest confidence game:    {df_results.loc[df_results['mc_win_prob_winner'].idxmax(), 'predicted_winner']} "
          f"({df_results['mc_win_prob_winner'].max():.1%})")
    print(f"  Lowest confidence game:     {df_results.loc[df_results['mc_win_prob_winner'].idxmin(), 'predicted_winner']} "
          f"({df_results['mc_win_prob_winner'].min():.1%})")
    print(f"  Average spread magnitude:   {df_results['mc_median_spread'].abs().mean():.1f} points")
    print(f"  Model confidence counts:")
    print(f"    HIGH:   {(df_results['model_confidence'] == 'HIGH').sum()} games")
    print(f"    MEDIUM: {(df_results['model_confidence'] == 'MEDIUM').sum()} games")
    print(f"    LOW:    {(df_results['model_confidence'] == 'LOW').sum()} games")
    print()


if __name__ == '__main__':
    main()
