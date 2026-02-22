import sys
sys.path.insert(0, r'c:\Users\Windows User\My_folder\gamble_code\sports_analytics')

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "="*70)
print("🏀 FULL BACKTEST WITH SEASON DATA")
print("="*70)

# Load season data
print("\n📊 Loading 2024-25 season data...")
from loaders.data_loader import fetch_nba_games, calculate_rolling_stats

try:
    games_df = fetch_nba_games(seasons=['2024-25'], season_type='Regular Season', verbose=True)
    games_with_stats = calculate_rolling_stats(games_df, window=5)
    print(f"✅ Loaded {len(games_with_stats)} games with rolling stats")
except Exception as e:
    print(f"❌ Error: {e}")
    games_with_stats = None

if games_with_stats is not None:
    
    # Create train/test split (chronological, 80/20)
    print("\n✂️  Creating chronological train/test split...")
    games_sorted = games_with_stats.sort_values('GAME_DATE').reset_index(drop=True)
    split_idx = int(len(games_sorted) * 0.80)
    
    train_df = games_sorted.iloc[:split_idx].copy()
    test_df = games_sorted.iloc[split_idx:].copy()
    
    print(f"📚 Training: {len(train_df)} games ({train_df['GAME_DATE'].min()} to {train_df['GAME_DATE'].max()})")
    print(f"🧪 Testing:  {len(test_df)} games ({test_df['GAME_DATE'].min()} to {test_df['GAME_DATE'].max()})")
    
    # Create predictions
    print("\n🎯 Generating predictions...")
    
    # Get actual results
    train_results = (train_df['WL'] == 'W').astype(int).values
    test_results = (test_df['WL'] == 'W').astype(int).values
    
    # Get numeric features
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ['TEST_COL', 'GAME_ID', 'SEASON_ID']]
    
    train_features = train_df[numeric_cols].fillna(0).values
    test_features = test_df[numeric_cols].fillna(0).values
    
    # Simple baseline: use average profile
    winner_profile = train_features[train_results == 1].mean(axis=0)
    loser_profile = train_features[train_results == 0].mean(axis=0)
    
    # Predict based on similarity to winner profile
    winner_dist = np.linalg.norm(test_features - winner_profile, axis=1)
    loser_dist = np.linalg.norm(test_features - loser_profile, axis=1)
    baseline_predictions = (winner_dist < loser_dist).astype(int)
    
    # Calculate metrics
    print("\n" + "="*70)
    print("📈 BACKTEST RESULTS")
    print("="*70)
    
    accuracy = (baseline_predictions == test_results).mean()
    
    tp = ((baseline_predictions == 1) & (test_results == 1)).sum()
    tn = ((baseline_predictions == 0) & (test_results == 0)).sum()
    fp = ((baseline_predictions == 1) & (test_results == 0)).sum()
    fn = ((baseline_predictions == 0) & (test_results == 1)).sum()
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    print(f"\n✅ Win Prediction Accuracy: {accuracy:.1%}")
    print(f"   Precision: {precision:.1%}")
    print(f"   Recall:    {recall:.1%}")
    print(f"   F1 Score:  {f1:.1%}")
    
    print(f"\n📊 Confusion Matrix:")
    print(f"   True Positives:  {tp}")
    print(f"   True Negatives:  {tn}")
    print(f"   False Positives: {fp}")
    print(f"   False Negatives: {fn}")
    
    print(f"\n🔍 Dataset Statistics:")
    print(f"   Test set actual wins:     {test_results.sum()} / {len(test_results)}")
    print(f"   Predicted wins:           {baseline_predictions.sum()} / {len(baseline_predictions)}")
    print(f"   Teams in test set:        {len(test_df['TEAM_ABBREVIATION'].unique())}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confusion matrix heatmap
    cm = np.array([[tn, fp], [fn, tp]])
    ax = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'])
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    
    # Accuracy by team
    ax = axes[1]
    test_df_copy = test_df.copy()
    test_df_copy['prediction'] = baseline_predictions
    test_df_copy['actual'] = test_results
    test_df_copy['correct'] = (test_df_copy['prediction'] == test_df_copy['actual']).astype(int)
    
    team_acc = test_df_copy.groupby('TEAM_ABBREVIATION')['correct'].agg(['sum', 'count'])
    team_acc['accuracy'] = team_acc['sum'] / team_acc['count']
    team_acc_sorted = team_acc.sort_values('accuracy', ascending=False).head(15)
    
    ax.barh(range(len(team_acc_sorted)), team_acc_sorted['accuracy'].values)
    ax.set_yticks(range(len(team_acc_sorted)))
    ax.set_yticklabels(team_acc_sorted.index)
    ax.set_xlabel('Accuracy')
    ax.set_title('Prediction Accuracy by Team (Top 15)')
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to 'backtest_results.png'")
    plt.show()
    
    print("\n" + "="*70)
    print("✅ BACKTEST COMPLETE")
    print("="*70)
    
    # Recommendations  
    print(f"\n💡 MODEL PERFORMANCE ANALYSIS:")
    if accuracy < 0.52:
        print(f"   ⚠️  Accuracy below 52% - model needs improvement")
        print(f"   → Add feature engineering (team streaks, rest days)")
        print(f"   → Include injury/availability data")
    elif accuracy < 0.55:
        print(f"   ✴️  Accuracy 52-55% - baseline classifier level")
        print(f"   → Model is performing at random\n")
    elif accuracy < 0.58:
        print(f"   ✅ Accuracy 55-58% - decent model")
        print(f"   → Good for filtering low-confidence predictions")
    else:
        print(f"   🎯 Accuracy >58% - strong model performance!")
        print(f"   → Consider for predictive betting")
    
    print(f"\n📈 NEXT STEPS:")
    print(f"   1. Compare to LightGBM from weekly_predictions.ipynb")
    print(f"   2. Ensemble both approaches for better accuracy")
    print(f"   3. Track Feb 19 games for prospective validation")
    print(f"   4. Monitor for concept drift monthly")

