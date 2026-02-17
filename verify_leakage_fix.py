"""
Verification Script: Check that Rolling Stats Leakage is Fixed

Run this to confirm that rolling features only use PRIOR games, not current game.
"""

import sys
sys.path.insert(0, r'c:\Users\Windows User\My_folder\gamble_code\sports_analytics')

import pandas as pd
from machine_learning.data_loader import fetch_nba_games, calculate_rolling_stats

print("="*80)
print("🔍 VERIFICATION: Rolling Stats Leakage Check")
print("="*80)

# Fetch a small sample to inspect manually
print("\n📥 Fetching 2024-25 season data...")
games_df = fetch_nba_games(seasons=['2024-25'], season_type='Regular Season', verbose=False)
games_with_stats = calculate_rolling_stats(games_df, window=5)

# Focus on ONE team's first 15 games
sample_team = games_with_stats['TEAM_ID'].iloc[0]
team_games = games_with_stats[games_with_stats['TEAM_ID'] == sample_team].sort_values('GAME_DATE').head(15).copy()

print(f"\n📊 Sample Team: {team_games['TEAM_ABBREVIATION'].iloc[0]}")
print(f"   First 15 games of season\n")

# Show key columns
display_cols = ['GAME_DATE', 'TEAM_ABBREVIATION', 'WL', 'PTS', 'PTS_ROLL', 'WIN_RATE_10', 'WIN_STREAK']
print(team_games[display_cols].to_string())

# Manual verification for Game 10
print("\n" + "="*80)
print("🧮 MANUAL VERIFICATION: Game 10")
print("="*80)

if len(team_games) >= 10:
    game_10 = team_games.iloc[9]  # 0-indexed
    games_1_to_9 = team_games.iloc[0:9]
    
    # Calculate what PTS_ROLL SHOULD be (average of Games 1-9, excluding NaN)
    expected_pts_roll = games_1_to_9['PTS'].mean()
    actual_pts_roll = game_10['PTS_ROLL']
    
    # Calculate what WIN_RATE_10 SHOULD be (win % of Games 1-9)
    expected_win_rate = (games_1_to_9['WL'] == 'W').mean()
    actual_win_rate = game_10['WIN_RATE_10']
    
    print(f"\nGame 10 PTS: {game_10['PTS']:.1f}")
    print(f"\nExpected PTS_ROLL (avg of Games 1-9): {expected_pts_roll:.2f}")
    print(f"Actual PTS_ROLL:                       {actual_pts_roll:.2f}")
    
    if abs(expected_pts_roll - actual_pts_roll) < 0.01:
        print("✅ PASS: PTS_ROLL does NOT include Game 10")
    else:
        print("❌ FAIL: PTS_ROLL includes Game 10's data (LEAKAGE DETECTED)")
    
    print(f"\nExpected WIN_RATE_10 (Games 1-9): {expected_win_rate:.3f}")
    print(f"Actual WIN_RATE_10:                {actual_win_rate:.3f}")
    
    if abs(expected_win_rate - actual_win_rate) < 0.01:
        print("✅ PASS: WIN_RATE_10 does NOT include Game 10")
    else:
        print("❌ FAIL: WIN_RATE_10 includes Game 10's data (LEAKAGE DETECTED)")

print("\n" + "="*80)
print("✅ WHAT TO CHECK:")
print("="*80)
print("""
Game 1:
  ✓ PTS_ROLL should be NaN (no prior games)
  ✓ WIN_RATE_10 should be NaN (no prior games)
  ✓ WIN_STREAK should be 0.0 (no prior games)

Game 2:
  ✓ PTS_ROLL should equal Game 1's PTS (only 1 prior game)
  ✓ WIN_RATE_10 should be 1.0 or 0.0 (based on Game 1's result)

Game 10:
  ✓ PTS_ROLL should average Games 1-9 ONLY (not include Game 10's PTS)
  ✓ WIN_RATE_10 should be win % of Games 1-9 (not include Game 10's result)
  ✓ WIN_STREAK should reflect Game 9's result (not Game 10)

If ALL checks pass → CLEAN ✅
If ANY fail → LEAKAGE STILL EXISTS ❌
""")

print("\n" + "="*80)
print("🎯 NEXT STEPS")
print("="*80)
print("""
1. If verification PASSES:
   → Rebuild full dataset with fixed features
   → Retrain models from scratch
   → Re-run backtest (expect 55-65% accuracy, NOT 94%)

2. If verification FAILS:
   → Check data_loader.py for correct .shift(1) placement
   → Verify no caching issues
   → Re-run this script

3. After retraining:
   → Compare new backtest accuracy to live 60-62%
   → If they match → leakage is fixed ✅
   → If backtest still inflated → investigate further
""")
