# 🎯 Adaptive Learning System for NBA Predictions

## Overview

This system provides **continuous improvement** for NBA game predictions through:
- ✅ **Validation**: Compare predictions to actual game results
- 📊 **Error Analysis**: Identify systematic biases and patterns  
- 🔬 **MCMC Refinement**: Use Bayesian methods to update model parameters
- 🔄 **Backpropagation**: Apply learned corrections to improve future predictions

## Architecture

```
┌─────────────────┐
│  Predictions    │
│  (GP + MCMC)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Actual Results  │
│  (Sports API)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Validation & Error Analysis   │
│   • Match predictions to games  │
│   • Calculate accuracy metrics  │
│   • Identify biases             │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│     Adaptive Learning           │
│   • Team-specific adjustments   │
│   • EPAA weight optimization    │
│   • Confidence recalibration    │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│ Model Updates   │
│ (MCMC + GP)     │
└─────────────────┘
```

## Key Components

### 1. `adaptive_learner.py`

**Main validation and learning module** with these classes:

#### `GameResultParser`
- Parses game results from CSV (Sports Reference format)
- Handles various data formats
- Extracts home/away teams and scores

#### `PredictionMatcher`
- Matches predictions to actual game results
- Handles team name variations (e.g., "LA Clippers" → "Los Angeles Clippers")
- Calculates prediction errors and accuracy

#### `ErrorAnalyzer`
- Comprehensive error analysis:
  - Overall accuracy metrics (win rate, MAE, RMSE)
  - Systematic bias detection (home/away bias)
  - Confidence calibration analysis
  - Identification of worst/best predictions
- Statistical tests for significance

#### `AdaptiveLearner`
- **Team Error Adjustments**: Calculates per-team bias corrections using t-tests
- **MCMC Refinement**: Proposes EPAA weight adjustments based on performance
- **Learning Report**: Generates comprehensive markdown reports with actionable insights

### 2. `model_updater.py`

**Applies learned insights back to models:**

#### `ModelUpdater`
- `apply_team_adjustments()`: Updates EPAA values for teams with systematic biases
- `update_epaa_weight()`: Adjusts EPAA weighting in hybrid predictions
- `retrain_gp_with_corrections()`: Retrains GP model with bias-corrected data
- `save_updated_models()`: Saves versioned model updates

#### Helper Functions
- `apply_learning_pipeline()`: Complete end-to-end update pipeline
- `create_feedback_loop_config()`: Generates config for automated retraining

### 3. Notebook Integration

New cells in `weekly_predictions.ipynb`:
1. **Load Results**: Parse actual game data from Sports Reference
2. **Run Validation**: Match predictions to results and analyze errors
3. **View Report**: Display comprehensive learning insights
4. **Error Analysis**: Show detailed match-by-match comparison
5. **Visualizations**: Plot error distributions and accuracy by confidence
6. **Apply Updates**: Update models with learned improvements

## Usage

### Basic Workflow

```python
from adaptive_learner import validate_and_learn
from model_updater import apply_learning_pipeline

# Step 1: Validate predictions
results = validate_and_learn(
    predictions=my_predictions,
    results_csv=actual_game_results,
    team_data=team_data,
    current_epaa_weight=0.5
)

# Step 2: Review learning report
print(results['report'])

# Step 3: Apply learned improvements
update_results = apply_learning_pipeline(
    validation_results=results,
    mcmc_model=mcmc_model,
    gp_model=gp_model,
    team_data=team_data,
    learning_rate=0.1
)
```

### Getting Game Results

#### Option 1: Sports Reference CSV
1. Go to [Basketball Reference](https://www.basketball-reference.com/leagues/NBA_2026_games-february.html)
2. Click "Share & Export" → "Get table as CSV"
3. Copy CSV text into notebook

#### Option 2: NBA API (automated)
```python
from results_fetcher import fetch_live_scores

results = fetch_live_scores(target_date='2026-02-01')
```

## Learning Mechanisms

### 1. Team Bias Correction

**Problem**: Model consistently over/under-predicts certain teams

**Solution**: 
- Track prediction errors by team
- Perform t-test for statistical significance (p < 0.10)
- Apply conservative adjustment (10% learning rate)

**Example**:
```
Atlanta Hawks: Consistently over-predicted by 8 points
→ Reduce Hawks EPAA by 0.8 points
```

### 2. EPAA Weight Optimization

**Problem**: Balance between MCMC (EPAA) and GP models

**Solution**:
- If win accuracy < 55%: Reduce EPAA weight (rely more on rolling stats)
- If win accuracy > 70% but MAE > 10: Increase EPAA weight (direction is right)
- If MAE < 8: Maintain or slight increase (good performance)

**Example**:
```
Current weight: 0.50
Win accuracy: 52% → Low
Proposed: 0.40 (reduce EPAA influence)
```

### 3. Confidence Recalibration

**Problem**: Confidence levels don't match actual accuracy

**Solution**:
- Track accuracy by confidence level (HIGH/MEDIUM/LOW)
- Identify overconfident errors (HIGH confidence, wrong winner)
- Identify underconfident successes (LOW confidence, accurate prediction)
- Propose threshold adjustments

### 4. Uncertainty Calibration

**Problem**: Uncertainty estimates don't correlate with actual errors

**Solution**:
- Check if high-uncertainty games have larger errors
- If not correlated: Retune GP kernel hyperparameters
- Use isotonic regression for confidence calibration

## Key Metrics

### Overall Performance
- **Win Accuracy**: % of games where predicted winner was correct
- **MAE**: Mean absolute error in point spreads
- **RMSE**: Root mean squared error
- **R²**: Correlation between predicted and actual spreads

### Bias Metrics
- **Mean Bias**: Average (predicted - actual) spread
- **Error Skew**: Asymmetry in error distribution
- **Confidence Justified Rate**: % where confidence level matched accuracy

### Calibration Metrics
- **CI Coverage**: % of games within prediction confidence interval
- **Uncertainty Correlation**: Do uncertain predictions have larger errors?

## Output Files

### Generated Files

```
json/
  ├── validation_matches.json       # Matched predictions and results
  ├── learning_summary_*.json       # Learning insights and recommendations
  └── predictions_log.json          # Historical prediction log

models/updated/
  ├── mcmc_model_v*.pkl            # Updated MCMC model
  ├── gp_model_v*.pkl              # Updated GP model
  └── update_history_v*.json       # Change log

config/
  └── feedback_config.json          # Automated feedback loop config
```

### Learning Report Format

```markdown
# 🎯 Adaptive Learning Report

## 📊 Overall Performance
- Win Accuracy: 62.5% (15/24)
- MAE: 9.2 points
- RMSE: 11.5 points

## 🎲 Bias Analysis
- Mean Bias: +2.3 points (home-favoring)
- Systematic bias detected!

## 🎯 Confidence Calibration
- HIGH: 80% accurate (justified)
- MEDIUM: 55% accurate (needs adjustment)
- LOW: 45% accurate (appropriate uncertainty)

## 🔧 Proposed Adjustments
1. Update EPAA weight: 0.50 → 0.45
2. Correct +2.3 point home bias
3. Adjust confidence thresholds

## 🔍 Teams Needing Adjustment
- Atlanta Hawks: -8.2 points (overrated)
- Sacramento Kings: +5.1 points (underrated)
```

## Best Practices

### 1. Validation Frequency
- **Minimum**: Every 20 games
- **Recommended**: Every 10 games  
- **Optimal**: After each game day (for high-frequency betting)

### 2. Learning Rate
- **Conservative**: 0.05 (slow, stable learning)
- **Moderate**: 0.10 (balanced, recommended)
- **Aggressive**: 0.20 (fast adaptation, riskier)

### 3. Minimum Sample Size
- Need **at least 10 matched predictions** for meaningful analysis
- Statistical tests require n ≥ 5 per team for significance

### 4. Model Versioning
- Save models before and after updates
- Keep update history for rollback
- Compare performance across versions

## Advanced Features

### Statistical Rigor
- **t-tests** for team bias significance
- **Isotonic regression** for confidence calibration
- **Bayesian posterior sampling** for uncertainty quantification
- **Cross-validation** for hyperparameter tuning

### Continuous Learning Loop
1. Make predictions
2. Collect results
3. Validate and analyze
4. Update models
5. Repeat with improved models

### Integration with Production
```python
# Automated daily pipeline
predictions = generate_todays_predictions()
validator.log_predictions(predictions)

# Next day
results = fetch_yesterdays_results()
validation = validate_and_learn(predictions, results)

if validation['n_matches'] >= 10:
    apply_learning_pipeline(validation, models)
    retrain_models()
```

## Troubleshooting

### No Matches Found
**Cause**: Team name mismatch
**Solution**: Check `PredictionMatcher.TEAM_ALIASES` and add mappings

### High Error Rates  
**Cause**: Model drift or poor data
**Solution**: 
- Check for injuries, trades (external factors)
- Verify data quality
- Consider full model retraining

### Overconfident Predictions
**Cause**: Underestimating uncertainty
**Solution**:
- Increase GP kernel noise parameter
- Reduce confidence thresholds
- Add ensemble uncertainty

## Theory: Why This Works

### Bayesian Framework
The system uses **Bayesian updating**:
- **Prior**: Current model parameters (MCMC posterior)
- **Likelihood**: New game outcomes  
- **Posterior**: Updated parameters incorporating new evidence

### Online Learning
Instead of batch retraining:
- **Incremental updates** preserve learned patterns
- **Exponential moving average** balances old/new information
- **Conservative learning rate** prevents overfitting to recent noise

### Ensemble Calibration
Combines two complementary models:
- **GP**: Captures short-term dynamics (rolling stats)
- **MCMC**: Captures long-term patterns (shooting efficiency)
- **Adaptive weighting**: Adjusts based on which performs better

## Future Improvements

### Potential Enhancements
1. **Online MCMC**: True incremental Bayesian updating
2. **Neural Network Integration**: Deep learning for feature extraction
3. **External Data**: Injuries, rest days, travel schedules
4. **Multi-objective Optimization**: Balance multiple loss functions
5. **Automated A/B Testing**: Compare model variants systematically

### Research Directions
- **Meta-learning**: Learn how to learn faster
- **Transfer learning**: Cross-sport knowledge transfer
- **Causal inference**: Understand why predictions fail
- **Explainable AI**: Detailed feature importance

## References

### Academic Papers
- Gibbs Sampling for Bayesian Hierarchical Models
- Gaussian Processes for Regression
- Online Learning and Stochastic Optimization
- Isotonic Regression for Calibration

### Libraries Used
- `scikit-learn`: Machine learning models
- `scipy`: Statistical functions
- `numpy/pandas`: Data manipulation
- `nba_api`: Real-time NBA data

## Support

For issues or questions:
1. Check error messages in validation output
2. Review learning report for insights
3. Examine match-by-match results for patterns
4. Consult update history for recent changes

---

**Built with 🏀 for continuous improvement in sports analytics**
