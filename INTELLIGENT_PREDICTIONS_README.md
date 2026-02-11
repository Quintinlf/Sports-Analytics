# Intelligent Iterative NBA Predictions System

## 🚀 Overview

This system implements an autonomous, confidence-driven prediction pipeline that:

- **Loads all models**: GP, Ensemble, Bayesian, MCMC
- **Trains on 3 seasons** of historical data (2022-23, 2023-24, 2024-25)
- **Iteratively retrains** when confidence < 0.6 (up to 10 iterations)
- **Stores everything in SQLite** for future learning and analysis
- **Predicts all games** from your provided CSV data

## 📁 New Files Created

### Core Engine Files (in `machine_learning/`)

1. **`database_handler.py`** - SQLite database management
   - Tables: predictions, prediction_results, model_history, games_cache
   - Full CRUD operations for predictions and results
   - Performance statistics tracking

2. **`extended_data_loader.py`** - 3-season data fetching
   - Fetches 2022-23, 2023-24, 2024-25 seasons
   - Database caching to avoid repeated API calls
   - Rolling stats calculation across extended dataset

3. **`iterative_predictor.py`** - Core prediction engine
   - Confidence-driven retraining (threshold: 0.6)
   - Maximum 10 iterations per prediction
   - Integrates all models (GP, Ensemble, MCMC)
   - Logs every prediction attempt to database

4. **`game_parser.py`** - CSV game data parser
   - Parses your game schedule format
   - Separates completed vs. upcoming games
   - Extracts actual results for validation

### Execution Scripts (in project root)

5. **`run_predictions.py`** - Main execution script
   - Parses all 162 games from your CSV data
   - Validates on Feb 1-7 completed games
   - Predicts Feb 8-28 upcoming games
   - Console output + SQLite storage

6. **`test_system.py`** - System validation script
   - Tests all components before running
   - Verifies imports and database setup

## 🎯 How It Works

### Prediction Flow

```
1. Parse Game Data
   └─> Separate completed (with scores) vs. upcoming games

2. Initialize Predictor
   └─> Load/train GP, Ensemble, MCMC models on 3-season data

3. For Each Game:
   ├─> Make initial prediction with confidence score
   ├─> IF confidence < 0.6:
   │   ├─> Retrain models with fresh data
   │   ├─> Re-predict with updated models
   │   ├─> Repeat up to 10 iterations
   │   └─> Log each iteration to database
   └─> Save final prediction to SQLite

4. Validation (for completed games):
   └─> Compare predictions vs actual results
       ├─> Calculate error metrics
       ├─> Track winner accuracy
       └─> Store results for future training
```

### Retraining Strategy

When confidence < 0.6, the system:

1. **Refreshes data** - Fetches last 14 days of games
2. **Retrains GP model** - Updates with bias corrections
3. **Retrains Ensemble** - Updates Bayesian weights
4. **Updates MCMC** - Refines EPAA parameters (if available)
5. **Re-predicts** - Generates new prediction with updated models
6. **Logs iteration** - Stores confidence improvement metrics

## 📊 Database Schema

### `predictions` table
- Game metadata (teams, date, venue)
- Predicted spread, winner, scores
- Confidence score and level
- Win probability, CI bounds
- Model versions used
- Iteration count
- Timestamp

### `prediction_results` table  
- Actual game outcomes
- Error metrics
- Winner accuracy
- CI coverage

### `model_history` table
- Each retrain step logged
- Parameters changed
- Confidence before/after
- Metrics tracked

### `games_cache` table
- Historical game data cached
- Avoids repeated API calls
- Includes all stats

## 🚀 How to Run

### Option 1: Run the Main Script

```powershell
cd "c:\Users\Windows User\My_folder\gamble_code\sports_analytics"
& "C:/Users/Windows User/My_folder/gamble_code/sports_analytics/.venv/Scripts/python.exe" run_predictions.py
```

This will:
1. Parse all 162 games from your CSV data
2. Load/train all models (may take 15-20 minutes first time)
3. Validate on completed games (Feb 1-7)
4. Predict upcoming games (Feb 8-28)
5. Display results in console
6. Save everything to `sports_analytics.db`

### Option 2: Test First

```powershell
& "C:/Users/Windows User/My_folder/gamble_code/sports_analytics/.venv/Scripts/python.exe" test_system.py
```

Validates all components work before full run.

### Option 3: Use in Notebook

Add this to your `weekly_predictions.ipynb`:

```python
import sys
sys.path.append('machine_learning')

from iterative_predictor import IterativePredictor
from database_handler import SportsAnalyticsDB

# Initialize predictor
predictor = IterativePredictor(
    confidence_threshold=0.6,
    max_iterations=10,
    verbose=True
)

# Load models (first time may take 15-20 min)
predictor.load_models()

# Make prediction
prediction = predictor.predict_with_retraining(
    home_team="Boston Celtics",
    away_team="New York Knicks",
    game_date="2026-02-08"
)

print(f"Winner: {prediction['predicted_winner']}")
print(f"Spread: {prediction['predicted_spread']:.1f}")
print(f"Confidence: {prediction['confidence_score']:.3f}")
print(f"Iterations: {prediction['iteration_count']}")

# Save to database
db = SportsAnalyticsDB()
pred_id = db.insert_prediction(prediction)
db.close()
```

## 📈 Expected Output

### Console Output Example

```
======================================================================
🚀 INTELLIGENT ITERATIVE NBA PREDICTIONS
======================================================================
📅 Execution Date: 2026-02-08 15:30:00
🤖 Confidence Threshold: 0.6
🔄 Max Iterations: 10
📊 Using 3-season training data (2022-23, 2023-24, 2024-25)
======================================================================

======================================================================
📊 COMPREHENSIVE DATA FETCH: 3 Seasons
======================================================================
Seasons: 2022-23, 2023-24, 2024-25
...
✅ COMPREHENSIVE DATA READY: 2,460 game records

======================================================================
🔧 LOADING/TRAINING MODELS
======================================================================
🔮 Gaussian Process Model:
   ✅ Trained and saved: gp_model_20260208_153000.pkl
🎯 Bayesian Ensemble Model:
   ✅ Ensemble trained
⚡ Bayesian MCMC Model:
   ✅ MCMC trained

======================================================================
🏀 PREDICTING: New York Knicks @ Boston Celtics
📅 Date: 2026-02-08
======================================================================
🔄 Iteration 1/10
   📊 Confidence: 0.450 (MEDIUM)
   🎯 Prediction: Boston Celtics by 5.2
   📈 Win Probability: 68.5%
   🔄 Confidence below threshold, triggering retraining...
   🔧 Retraining models...
   ✅ Models retrained

🔄 Iteration 2/10
   📊 Confidence: 0.625 (HIGH)
   🎯 Prediction: Boston Celtics by 5.8
   📈 Win Probability: 72.1%
   ✅ Confidence threshold met (0.60)

======================================================================
✅ FINAL PREDICTION
======================================================================
Winner: Boston Celtics
Spread: 5.8
Confidence: 0.625 (HIGH)
Iterations: 2
======================================================================
```

### Final Statistics

```
======================================================================
📈 FINAL STATISTICS
======================================================================

🤖 Predictor Performance:
   Total Predictions: 162
   Retraining Triggered: 45
   Retraining Rate: 27.8%

📊 Confidence Distribution:
   Average: 0.687
   Min: 0.453
   Max: 0.891
   HIGH (≥0.6): 142 (87.7%)
   MEDIUM (0.3-0.6): 18 (11.1%)
   LOW (<0.3): 2 (1.2%)

🔄 Iteration Statistics:
   Average Iterations: 1.8
   Max Iterations: 7
   Single Pass (1 iter): 117
   Multiple Iterations: 45

======================================================================
✅ EXECUTION COMPLETE
======================================================================
💾 Database: sports_analytics.db
📊 All predictions stored and ready for analysis
======================================================================
```

## 🔍 Querying Results

### Using Python

```python
from database_handler import SportsAnalyticsDB

db = SportsAnalyticsDB()

# Get all predictions from Feb 8-28
predictions = db.get_predictions_by_date('2026-02-08', '2026-02-28')

# Get performance stats
stats = db.get_performance_stats(days=30)
print(f"Win Accuracy: {stats['win_accuracy']:.1f}%")
print(f"Avg Error: {stats['avg_error']:.2f} points")

db.close()
```

### Using SQL

```sql
-- Top 10 highest confidence predictions
SELECT home_team, away_team, predicted_winner, 
       confidence_score, iteration_count
FROM predictions
WHERE game_date >= '2026-02-08'
ORDER BY confidence_score DESC
LIMIT 10;

-- Games that needed multiple retraining iterations
SELECT game_date, home_team, away_team,
       iteration_count, confidence_score
FROM predictions
WHERE iteration_count > 1
ORDER BY iteration_count DESC;

-- Validation accuracy for completed games
SELECT 
  COUNT(*) as total,
  SUM(CASE WHEN correct_winner = 1 THEN 1 ELSE 0 END) as correct,
  AVG(prediction_error) as avg_error
FROM predictions p
JOIN prediction_results r ON p.prediction_id = r.prediction_id;
```

## 📝 Configuration

You can adjust parameters in `run_predictions.py` or when initializing `IterativePredictor`:

```python
predictor = IterativePredictor(
    confidence_threshold=0.6,  # Min confidence to accept
    max_iterations=10,         # Max retraining attempts
    db_path="sports_analytics.db",
    verbose=True              # Print detailed progress
)
```

## 🎓 Learning from Results

The system automatically learns from completed games:

1. **Actual results stored** - Game outcomes logged to database
2. **Error analysis** - Prediction vs. actual spread calculated
3. **Model bias detection** - Team-specific adjustments identified
4. **Future retraining** - Next run uses improved models

To manually trigger learning from stored results:

```python
from validation_tracker import PredictionValidator
from model_updater import apply_learning_pipeline

validator = PredictionValidator('basketball/predictions_log.json')
results = validator.get_recent_performance(days=7)

# Apply learning
update_results = apply_learning_pipeline(
    validation_results=results,
    mcmc_model=mcmc_model,
    gp_model=gp_model,
    team_data=team_data,
    learning_rate=0.1,
    save_models=True
)
```

## 🔧 Troubleshooting

### Installation Issues

If packages fail to install:

```powershell
# Install core packages individually
& "./.venv/Scripts/python.exe" -m pip install numpy pandas scipy
& "./.venv/Scripts/python.exe" -m pip install scikit-learn xgboost
& "./.venv/Scripts/python.exe" -m pip install nba_api
```

### Memory Issues

If MCMC training fails due to memory:

```python
# In iterative_predictor.py, reduce MCMC parameters:
self.mcmc_model = BayesianBasketballHierarchical(
    L=5,   # Reduced from 10
    J=5,   # Reduced from 10
    K=7
)

self.mcmc_model.fit_gibbs(
    team_stats=team_stats,
    n_iterations=2000,  # Reduced from 5000
    burn_in=500,        # Reduced from 1500
    verbose=False
)
```

### Slow Execution

First run trains all models (~15-20 min). Subsequent runs are faster:
- Models cached and reloaded
- Database caching reduces API calls
- Only retrain when confidence low

## 📚 System Architecture

```
sports_analytics/
├── machine_learning/
│   ├── database_handler.py      ← SQLite operations
│   ├── extended_data_loader.py  ← 3-season data fetch
│   ├── iterative_predictor.py   ← Core engine
│   ├── game_parser.py           ← CSV parsing
│   ├── data_loader.py           ← Base data functions
│   ├── model_trainer.py         ← GP & Ensemble models
│   ├── mcmc_sampler.py          ← Bayesian MCMC
│   ├── predictor.py             ← Prediction functions
│   ├── model_updater.py         ← Learning pipeline
│   └── validation_tracker.py    ← Metrics tracking
├── run_predictions.py           ← Main execution script
├── test_system.py               ← System tests
└── sports_analytics.db          ← SQLite database (created on first run)
```

## 🎯 Next Steps

1. **Run the system**: Execute `run_predictions.py`
2. **Review predictions**: Check console output and database
3. **Track results**: As games complete, log actual outcomes
4. **Analyze performance**: Query database for insights
5. **Iterate**: System learns and improves with each run

---

**Built with**: Python 3.14, scikit-learn, XGBoost, PyMC, pandas, numpy, SQLite
**Data Source**: NBA API (via nba_api library)
**Confidence Threshold**: 0.6 (HIGH/MEDIUM boundary)
**Max Iterations**: 10 per prediction
**Training Data**: 3 NBA seasons (2022-23, 2023-24, 2024-25)
