# NBA Game Outcome Prediction Model - PATH A Implementation

## Executive Summary
Production-ready system for NBA game spread prediction using 130 engineered features combining team-level and player-level statistics. Current baseline: 54.9% test accuracy (random: 50%). Target post-PATH A: 57-59%.

## Model Architecture

### 1. Core Prediction Engine
- **Algorithm**: LightGBM Quantile Regression (predicts spread distribution)
- **Task**: Binary classification disguised as regression (predict spread > 0 = home win)
- **Test Accuracy**: 54.9% → Target: 57-59% with player features added
- **Validation Accuracy**: 59.3% (calibration dataset)

### 2. Feature Engineering (130 total features)

#### A. Team-Level Features (94 features)
**Baseline Statistics (38 features):**
- Rolling averages, pct metrics: PTS, FG%, 3P%, REB, AST, STL, BLK, TOV (per team: 19)
- Advanced metrics: TS%, EFG%, AST/TO ratio, Pace, OFF_RTG approximation, FT rate, Plus/Minus
- Each metric computed for both HOME and AWAY teams

**Adjusted Statistics (30 features):**
- All baseline metrics adjusted by opponents (opponent-relative strength)
- Rationale: Controls for strength of schedule

**Advanced Metrics (26 features):**
- OFF RTG, DEF RTG, NET RTG, PACE, PIE
- TS%, AST%, AST/TO, OREB%, DREB%, REB%, TOV%, EFG%
- Computed per team using 5-game rolling windows

**Contextual Features (6 features):**
- Win streak, REST days, Back-to-back flag, Win rate (last 10 games) [per team]

#### B. Player-Level Features (36 features: 18 per team)
**Minutes-Weighted Aggregates (6 features per team):**
- Points, Rebounds, Assists, FGA, FG%, Points/FGA weighted by avg minutes

**Top Performer Metrics (4 features per team):**
- PPG of highest scorer, RPG of rebounder, APG of playmaker
- Star player scoring share (top scorer PPG / team PPG)

**Rotation Metrics (2 features per team):**
- Number of active players (MIN > 5 min avg)
- Rotation stability (std dev of minutes across 5 games)

**Injury Proxies (2 features per team):**
- Count of key players with sudden minutes drop >40%
- Flag if starter avg minutes dropped >5 minutes vs rolling avg

**Scoring Efficiency (4 features per team):**
- Scoring concentration (variance in scorer contributions)
- Count of defensive contributors (STL>1 or BLK>0.5)
- Bench scoring % of total team points
- Count of bench scorers (5-15 min players)

### 3. Data Pipeline

#### Input Data
- **Source**: nba_api.stats.endpoints (playergamelog, commonteamroster)
- **Coverage**: 2024-25 NBA season (Oct 2024 - Mar 2025)
- **Games**: 812 total (487 train, 162 calib, 163 test)

#### Chronological Safety (CRITICAL)
- **No leakage**: Player logs filtered to games STRICTLY before prediction date
- **Rolling windows**: 5-game rolling averages (min_periods=1)
- **Before-date enforcement**: All API calls use `before_date` parameter
- **Verification**: Diagnostic tests confirm no forward-fill contamination

#### Performance Options
**Option 1: Direct API (Slower)**
- Runtime: 2-4 hours per full dataset rebuild
- Rate limit: 0.6 seconds per API call
- Use case: One-time build or debugging

**Option 2: Database Cache (RECOMMENDED)**
- Setup: One-time 2-4 hour investment (SQLite database)
- Runtime: 10-15 minutes per full dataset rebuild
- Cache schema: player_game_logs(player_id, team_id, game_date, season, stats)
- Speedup: 100x faster than API calls

### 4. Feature Assembly Process (Per Game)

```
Input: game_date, home_team_id, away_team_id
       ↓
[Extract 94 team features from matchup_df]
       ↓
[Fetch HOME player logs before game_date → Calculate rolling stats → Aggregate 18 features]
       ↓
[Fetch AWAY player logs before game_date → Calculate rolling stats → Aggregate 18 features]
       ↓
[Combine: 94 + 18 + 18 = 130 feature vector]
       ↓
Output: (130,) numpy array, chronologically safe, no NaN/Inf
```

### 5. Production Deployment Checklist

**Phase 1: Model Training (1 day)**
- [ ] Build X_train (487 × 130), X_calib (162 × 130), X_test (163 × 130)
- [ ] Retrain LightGBM on 130 features
- [ ] Validate test accuracy ≥ 57% (benchmark: 54.9%)
- [ ] Check feature importance: player features in top 20?

**Phase 2: Database Setup (2-4 hours)**
- [ ] Run `PlayerDataCache('player_logs.db').create_tables()`
- [ ] Populate cache with all 30 teams for 2024-25 season
- [ ] Verify cache stats: 2,000-3,000 game records per team expected
- [ ] Query benchmark: <1 second for 1 team's games

**Phase 3: Daily Refresh Pipeline (Automated)**
```python
# Daily at 00:30 UTC (after all previous day games concluded)
cache = PlayerDataCache('player_logs.db')
yesterday = datetime.now() - timedelta(days=1)

for team_id in range(30):
    logs = fetch_player_logs_for_team(team_id, before_date=yesterday + timedelta(days=1))
    cache.cache_team_logs(logs, season='2024-25')

# Retrain on accumulated data (optional: weekly if validation drift >1pp)
```

**Phase 4: Prediction API (Real-time)**
```python
from machine_learning.integration_pipeline import build_enhanced_game_features

# For upcoming game:
features, feat_dict, feat_names = build_enhanced_game_features(
    game_date=datetime(2026, 2, 14),  # Game date
    home_team_id=6,  # Mavericks (normalized 0-29 format)
    away_team_id=13,  # Lakers
    matchup_df=matchup_df_sorted,  # Precomputed team features
    team_feature_cols=feature_cols_enhanced,  # 94 team features (IDs removed)
    player_data_source=cache,  # Use cached DB queries
    verbose=False
)

# Feed to model:
pred_spread = model.predict(features.reshape(1, -1))
pred_proba = model.predict_proba(features.reshape(1, -1))
```

### 6. Expected Performance & Monitoring

**Success Criteria (Post-PATH A)**
- ✅ Test accuracy: 57-59% (vs 54.9% baseline) = +2-5pp improvement
- ✅ Validation accuracy: Stable ~59% (no catastrophic overfitting)
- ✅ Brier score: < 0.24 (vs current 0.25)
- ✅ Calibration: Platt + temperature scaling validated on calib set
- ✅ Feature coverage: >95% of games have complete player data

**Monitoring (Daily)**
- [ ] Accuracy on previous day games (should stay ≥57%)
- [ ] % games with complete player data (flag if <90%)
- [ ] API error rate (should be <1%)
- [ ] Cache query latency (should be <1 second/team)

**Failure Modes & Recovery**
- If test_acc < 54.9%: Regression detected → Revert to 94-team-feature model, investigate leakage
- If accuracy decays >2pp: Model drift → Retrain on latest 200 games
- If >10% games missing player data: Team roster API failed → Manual roster update needed
- If API rate limit errors > 5%: Implement exponential backoff (already in code)

### 7. Files & Modules

**Production Modules** (in machine_learning/):
- `player_fetcher.py` (250 lines): nba_api integration + Team ID mapping
- `player_features.py` (350 lines): Rolling stats + 18-feature aggregation
- `player_cache.py` (200 lines): SQLite caching layer
- `integration_pipeline.py` (300 lines): 130-feature vector assembly

**Data Files** (versioned in git):
- Precomputed `matchup_df_sorted.pkl`: 812 games × 94 team features (updated weekly)
- Feature column list `feature_cols_enhanced.pkl`: 94 column names (remove HOME_TEAM_ID, AWAY_TEAM_ID)

**Database** (local, regenerated as needed):
- `player_logs.db`: SQLite cache (2-4 GB), not version controlled

### 8. Next Steps (For Implementation Team)

1. **Test on sample games** (15 minutes)
   ```python
   sample_game = matchup_df_sorted.iloc[700]  # Random test game
   features, _, _ = build_enhanced_game_features(...)
   assert features.shape == (130,), f"Expected (130,), got {features.shape}"
   assert not np.any(np.isnan(features))
   ```

2. **Build full 130-feature training set** (30-40 min with API, 15 min with cache)
   ```python
   X_train_enhanced = np.array([
       build_enhanced_game_features(game_date, home_id, away_id)[0]
       for _, game in train_games.iterrows()
   ])
   ```

3. **Retrain LightGBM** (5 minutes)
   ```python
   enhanced_model = LGBMQuantilePredictor()
   enhanced_model.fit(X_train_enhanced, y_train)
   test_acc = ((enhanced_model.predict(X_test_enhanced) > 0) == (y_test > 0)).mean()
   print(f"Test accuracy: {test_acc:.1%}")  # Target: ≥57%
   ```

4. **Deploy if test_acc ≥ 57%**
   - Commit feature list and trained model to git
   - Set up daily refresh pipeline (cron job or Airflow)
   - Monitor accuracy daily

---

## Risk Assessment

**Technical Risks**
- [ ] Player data incomplete for certain games (mitigated: graceful zeros)
- [ ] API rate limiting during initial cache build (mitigated: exponential backoff + 0.6s sleep)
- [ ] Feature leakage from forward-fill (mitigated: before_date enforcement + unit tests)
- [ ] Model decay over season (mitigated: monitor daily, retrain weekly if drift)

**Business Risks**
- [ ] Improvement may be marginal (2-3pp) if player variance doesn't predict spreads well
- [ ] Requires 2-3 week development + testing timeline
- [ ] Player injury data not integrated (manual roster updates needed)

---

## Glossary

- **Quantile Regression**: Predicts spread range, not just point estimate
- **Platt Scaling**: Logistic calibration method for probability correction
- **Temperature Scaling**: Entropy-based calibration (tunes overconfidence)
- **Chronological Integrity**: No future data leaks into past predictions
- **Rolling Window**: 5-game average, recomputed daily (min_periods=1)
- **Before-Date Filter**: GAME_DATE < prediction_date (prevents cheating)

---

## Contact & Questions

For implementation assistance, refer to:
- `INTELLIGENT_PREDICTIONS_README.md` — Overview
- `PREDICTION_GUIDE.md` — Usage guide
- `ADAPTIVE_LEARNING_README.md` — Advanced features
- Github: `/sports_analytics/machine_learning/`
