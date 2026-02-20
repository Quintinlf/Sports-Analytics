"""
REBUILD SCRIPT: Fresh dataset + retrained models with FIXED features

This script:
1. Clears all caches
2. Rebuilds dataset from scratch (2023-24 + 2024-25)
3. Retrains ML models with clean features (no leakage)

Run this once, then report the results.
"""

import os
import sys
import shutil
sys.path.insert(0, r'c:\Users\Windows User\My_folder\gamble_code\sports_analytics')

import pandas as pd
from machine_learning.data_loader import fetch_nba_games, calculate_rolling_stats
from machine_learning.model_trainer import GPPredictor

print("\n" + "="*80)
print("🧹 STEP 1: CLEARING CACHES")
print("="*80)

# Clear Python cache
cache_dirs = ['.cache', '__pycache__', 'machine_learning/__pycache__']
for cache_dir in cache_dirs:
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"   ✅ Removed {cache_dir}")

# Clear database
if os.path.exists('sports_analytics.db'):
    os.remove('sports_analytics.db')
    print(f"   ✅ Removed sports_analytics.db")

print("✅ Caches cleared\n")

print("="*80)
print("🔄 STEP 2: REBUILDING DATASET (2023-24 + 2024-25)")
print("="*80)

try:
    games_df = fetch_nba_games(
        seasons=['2023-24', '2024-25'], 
        season_type='Regular Season', 
        verbose=True
    )
    print(f"\n✅ Fetched {len(games_df)} total game records")
    
    print("\n📊 Calculating rolling stats with FIXED features (shift applied)...")
    games_with_stats = calculate_rolling_stats(games_df, window=5)
    print(f"✅ Dataset built: {len(games_with_stats)} games")
    print(f"   Date range: {games_with_stats['GAME_DATE'].min()} → {games_with_stats['GAME_DATE'].max()}\n")
    
except Exception as e:
    print(f"❌ Error rebuilding dataset: {e}")
    sys.exit(1)

print("="*80)
print("📊 STEP 3: CHRONOLOGICAL TRAIN/TEST SPLIT")
print("="*80)

# Sort by date and split
games_sorted = games_with_stats.sort_values('GAME_DATE').reset_index(drop=True)
split_idx = int(len(games_sorted) * 0.80)

train_df = games_sorted.iloc[:split_idx].copy()
test_df = games_sorted.iloc[split_idx:].copy()

print(f"\n📚 Training set:")
print(f"   Games: {len(train_df)}")
print(f"   Dates: {train_df['GAME_DATE'].min()} → {train_df['GAME_DATE'].max()}")

print(f"\n🧪 Test set:")
print(f"   Games: {len(test_df)}")
print(f"   Dates: {test_df['GAME_DATE'].min()} → {test_df['GAME_DATE'].max()}")

print("\n" + "="*80)
print("🧠 STEP 4: RETRAINING MODELS FROM SCRATCH")
print("="*80)

try:
    print("\n⏳ Training Gaussian Process model...")
    print("   (This may take a few minutes...)")
    
    gp_model = GPPredictor(kernel_type='matern', length_scale=1.0, noise_level=0.1)
    gp_model.fit(train_df, test_df)
    
    print("✅ Gaussian Process model trained")
    
except Exception as e:
    print(f"❌ Error training model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✨ REBUILD COMPLETE")
print("="*80)

print(f"""
Summary:
  ✅ Caches cleared
  ✅ Dataset rebuilt
  ✅ Features recalculated with shift(1) (NO LEAKAGE)
  ✅ Models retrained from scratch

Next: Run backtest and compare new accuracy to 94.3%
Expected: ~55-65% accuracy (matching your live 60%)
""")