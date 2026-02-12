# ============================================================
# DIAGNOSTIC CELL 1: Check Current Variables
# ============================================================
print("=" * 100)
print("🔍 DIAGNOSTIC: Current Variable State")
print("=" * 100)

print("\n📊 Data Shapes:")
print(f"   X_train_corrected: {X_train_corrected.shape if 'X_train_corrected' in locals() else 'NOT DEFINED'}")
print(f"   X_test_corrected: {X_test_corrected.shape if 'X_test_corrected' in locals() else 'NOT DEFINED'}")
print(f"   y_train: {y_train.shape if 'y_train' in locals() else 'NOT DEFINED'}")
print(f"   y_test: {y_test.shape if 'y_test' in locals() else 'NOT DEFINED'}")
print(f"   games_with_stats: {games_with_stats.shape if 'games_with_stats' in locals() else 'NOT DEFINED'}")
print(f"   matchup_df_sorted: {matchup_df_sorted.shape if 'matchup_df_sorted' in locals() else 'NOT DEFINED'}")

print("\n📊 Split Indices:")
print(f"   train_end: {train_end if 'train_end' in locals() else 'NOT DEFINED'}")
print(f"   calib_end: {calib_end if 'calib_end' in locals() else 'NOT DEFINED'}")
print(f"   Total matchups: {len(matchup_df_sorted) if 'matchup_df_sorted' in locals() else 'NOT DEFINED'}")

if 'matchup_df_sorted' in locals() and 'train_end' in locals() and 'calib_end' in locals():
    print("\n📅 Date Ranges:")
    train_games = matchup_df_sorted.iloc[:train_end]
    test_games = matchup_df_sorted.iloc[train_end:calib_end]
    val_games = matchup_df_sorted.iloc[calib_end:]
    
    print(f"   Training: {len(train_games)} games, {train_games['GAME_DATE'].min().date()} to {train_games['GAME_DATE'].max().date()}")
    print(f"   Test: {len(test_games)} games, {test_games['GAME_DATE'].min().date()} to {test_games['GAME_DATE'].max().date()}")
    print(f"   Validation: {len(val_games)} games, {val_games['GAME_DATE'].min().date()} to {val_games['GAME_DATE'].max().date()}")

print("\n📊 Feature Information:")
print(f"   feature_cols_fixed: {len(feature_cols_fixed) if 'feature_cols_fixed' in locals() else 'NOT DEFINED'} features")
if 'feature_cols_fixed' in locals():
    print(f"   First 10: {feature_cols_fixed[:10]}")

print("\n📊 Models Available:")
print(f"   predictor: {'✅' if 'predictor' in locals() else '❌'}")
print(f"   model_corrected: {'✅' if 'model_corrected' in locals() else '❌'}")
print(f"   production_model: {'✅' if 'production_model' in locals() else '❌'}")

print("\n📊 Validation Data:")
print(f"   df_val: {len(df_val) if 'df_val' in locals() else 'NOT DEFINED'} games")
print(f "   val_corrected_df: {len(val_corrected_df) if 'val_corrected_df' in locals() else 'NOT DEFINED'}")

print("\n📊 Calibration:")
print(f"   CALIBRATION_ALPHA: {CALIBRATION_ALPHA if 'CALIBRATION_ALPHA' in locals() else 'NOT DEFINED'}")
print(f"   CALIBRATION_BETA: {CALIBRATION_BETA if 'CALIBRATION_BETA' in locals() else 'NOT DEFINED'}")

print("\n=" * 100)
