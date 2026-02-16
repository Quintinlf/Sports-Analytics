"""
Simple predictions using pre-trained LightGBM model
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# Add paths
parent_dir = r'c:\Users\Windows User\My_folder\gamble_code\sports_analytics'
sys.path.insert(0, parent_dir)

from machine_learning.data_loader import get_all_nba_teams, get_team_latest_stats, fetch_nba_games, calculate_rolling_stats, create_matchup_features
from machine_learning.game_parser import parse_game_data_from_text

#Game data
GAME_DATA_CSV = """Date,Start (ET),Visitor/Neutral,PTS,Home/Neutral,PTS,,,Attend.,LOG,Arena,Notes
Thu Feb 19 2026,7:00p,Atlanta Hawks,,Philadelphia 76ers,,,,,,Xfinity Mobile Arena,
Thu Feb 19 2026,7:00p,Indiana Pacers,,Washington Wizards,,,,,,Capital One Arena,
Thu Feb 19 2026,7:30p,Detroit Pistons,,New York Knicks,,,,,,Madison Square Garden (IV),
Thu Feb 19 2026,8:00p,Toronto Raptors,,Chicago Bulls,,,,,,United Center,
Thu Feb 19 2026,8:30p,Phoenix Suns,,San Antonio Spurs,,,,,,Moody Center,
Thu Feb 19 2026,10:00p,Boston Celtics,,Golden State Warriors,,,,,,Chase Center,
Thu Feb 19 2026,10:00p,Orlando Magic,,Sacramento Kings,,,,,,Golden 1 Center,
Thu Feb 19 2026,10:30p,Denver Nuggets,,Los Angeles Clippers,,,,,,Intuit Dome,"""


def simple_prediction(away_team, home_team):
    """Make a simple prediction based on basic logic"""
    
    # Team strength ratings (very simplified)
    strength = {
        'Boston Celtics': 95,
        'Cleveland Cavaliers': 92,
        'Oklahoma City Thunder': 91,
        'Denver Nuggets': 90,
        'Milwaukee Bucks': 88,
        'Phoenix Suns': 87,
        'Los Angeles Clippers': 86,
        'Golden State Warriors': 85,
        'Los Angeles Lakers': 84,
        'Miami Heat': 82,
        'Philadelphia 76ers': 81,
        'New York Knicks': 80,
        'Sacramento Kings': 78,
        'Dallas Mavericks': 77,
        'Indiana Pacers': 76,
        'Chicago Bulls': 75,
        'Orlando Magic': 74,
        'Atlanta Hawks': 73,
        'Toronto Raptors': 72,
        'San Antonio Spurs': 70,
        'New Orleans Pelicans': 69,
        'Utah Jazz': 68,
        'Brooklyn Nets': 67,
        'Memphis Grizzlies': 66,
        'Houston Rockets': 65,
        'Washington Wizards': 64,
        'Charlotte Hornets': 63,
        'Portland Trail Blazers': 62,
        'Detroit Pistons': 60,
    }
    
    away_str = strength.get(away_team, 70)
    home_str = strength.get(home_team, 70)
    
    # Home court advantage
home_str += 3
    
    # Calculate predicted spread
    spread = home_str - away_str
    
    # Determine winner
    if spread > 0:
        winner = home_team
        win_prob = min(0.5 + (spread / 40), 0.95)
    else:
        winner = away_team
        win_prob = min(0.5 + (abs(spread) / 40), 0.95)
    
    return {
        'predicted_winner': winner,
        'predicted_spread': spread,
        'win_probability': win_prob,
        'confidence_level': 'HIGH' if abs(spread) > 8 else 'MEDIUM' if abs(spread) > 4 else 'LOW'
    }


def main():
    print("\n" + "="*70)
    print("🏀 NBA PREDICTIONS FOR FEBRUARY 19, 2026")
    print("="*70)
    print(f"📅 Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Parse game data
    print("📋 Parsing game data...")
    parsed_data = parse_game_data_from_text(GAME_DATA_CSV, verbose=False)
    upcoming_games = parsed_data['upcoming_list']
    
    print(f"✅ Found {len(upcoming_games)} games to predict\n")
    
    # Make predictions
    print("="*70)
    print("🎯 PREDICTIONS")
    print("="*70)
    
    for i, game in enumerate(upcoming_games, 1):
        away = game['away_team']
        home = game['home_team']
        time = game.get('start_time', 'TBD')
        
        prediction = simple_prediction(away, home)
        
        print(f"\n{i}. {away} @ {home} ({time})")
        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"   🏆 Winner: {prediction['predicted_winner']}")
        print(f"   📊 Spread: {prediction['predicted_spread']:.1f}")
        print(f"   🎯 Win Probability: {prediction['win_probability']:.1%}")
        print(f"   💪 Confidence: {prediction['confidence_level']}")
    
    print("\n" + "="*70)
    print("✅ PREDICTIONS COMPLETE!")
    print("="*70)
    print("\n⚠️  Note: These are simplified predictions based on team strength ratings.")
    print("For more accurate predictions with machine learning, use the full iterative system.\n")


if __name__ == "__main__":
    main()
