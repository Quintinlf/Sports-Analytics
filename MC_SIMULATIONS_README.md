# Monte Carlo Simulation Results - February 19, 2026

## Executive Summary

**80,000 simulations run** across 8 games using quantile-based triangular distributions (Q10/Q50/Q90) from our trained LightGBM model.

### Key Findings

- **Average Winner Probability:** 63.1% — moderate confidence
- **Winner Probability Range:** 50.5% to 80.1%
- **Model Confidence:** ALL 8 games rated LOW confidence
- **Average Spread:** 2.9 points

---

## What This Means for Your Watching Experience

When you watch these games on February 19th, here's what to expect based on **10,000 simulations per game**:

### 🎯 **Strong Predictions (>70% confidence)**

**1. Indiana Pacers @ Washington Wizards**
- **Winner:** Indiana Pacers (80.1% win probability)
- **Spread:** Pacers by 8.2 points (median)
- **95% Confidence Interval:** Pacers by 21.2 to Wizards by 8.9
- **What to expect:** This is our strongest pick. In 8,010 out of 10,000 simulations, the Pacers won. However, the model still rates this LOW confidence due to high scoring variance (±8.0 points).

**2. Atlanta Hawks @ Philadelphia 76ers**
- **Winner:** Atlanta Hawks (72.7% win probability)
- **Spread:** Hawks by 4.6 points (median)
- **95% Confidence Interval:** Hawks by 18.6 to 76ers by 9.0
- **What to expect:** In 7,270 simulations, the Hawks won on the road. Wide interval reflects game unpredictability.

---

### 🤝 **Moderate Predictions (60-70% confidence)**

**3. Sacramento Kings @ Orlando Magic**
- **Winner:** Orlando Magic (66.6%)
- **Spread:** Magic by 3.3 points
- **Simulations:** Magic won in 6,660 out of 10,000

**4. Boston Celtics @ Golden State Warriors**
- **Winner:** Boston Celtics (62.8%)
- **Spread:** Celtics by 2.5 points
- **Simulations:** Celtics won in 6,280 out of 10,000

**5. Los Angeles Clippers vs Denver Nuggets**
- **Winner:** Clippers (61.7%)
- **Spread:** Clippers by 2.3 points
- **Simulations:** Clippers won in 6,170 out of 10,000

**6. New York Knicks vs Detroit Pistons**
- **Winner:** Knicks (59.5%)
- **Spread:** Knicks by 1.8 points
- **Simulations:** Knicks won in 5,950 out of 10,000

---

### 🎲 **Toss-Up Games (<55% confidence)**

**7. Phoenix Suns @ San Antonio Spurs**
- **Winner:** Phoenix Suns (50.8%)
- **Spread:** Basically even (-0.1 median)
- **95% CI:** [-16.3, +14.7] — massive variance
- **What to expect:** This is essentially a coin flip. Don't be surprised by any outcome.

**8. Toronto Raptors @ Chicago Bulls**
- **Winner:** Toronto Raptors (50.5%)
- **Spread:** Dead even (-0.1 median)
- **95% CI:** [-15.3, +14.6]
- **What to expect:** Another coin flip. 5,050 simulations picked Raptors, 4,950 picked Bulls.

---

## The "I Ran Thousands of Simulations" Statement

### What You CAN Say:

✅ **"I ran 10,000 Monte Carlo simulations per game using a trained machine learning model. Based on those simulations, the Indiana Pacers have an 80% chance of beating the Wizards, winning in 8 out of 10 scenarios."**

✅ **"The Suns-Spurs game is a true toss-up — in 10,000 simulations, the outcomes were split almost 50-50, and the 95% confidence interval spans 31 points."**

✅ **"Five of the eight games have winner probabilities between 60-73%, which means the predicted winner should happen roughly 6-7 times out of 10."**

✅ **"The average winner probability across all games is 63%, which aligns with our model's test accuracy of 64.5%."**

### What You CANNOT Say:

❌ **"These predictions are guaranteed" or "near-certain"**
- Reality: Even our best pick (80%) fails 1 in 5 times.

❌ **"The model is highly confident in these predictions"**
- Reality: All 8 games rated LOW confidence due to high uncertainty intervals.

❌ **"I expect all 8 winners to be correct"**
- Reality: At 63% average, you'd expect ~5 correct out of 8 (range: 3-7).

---

## Why All Games Show "LOW" Confidence

The model's confidence score combines:
1. **Probability margin** (how far from 50-50)
2. **Predictive interval width** (uncertainty in point differential)

Even though some games have 70-80% win probabilities, the **wide confidence intervals** (typical 95% CI spans ~25-35 points) pull down the confidence score. This reflects:

- High intrinsic variance in NBA scoring
- Model uncertainty about exact margins
- Lack of opponent-adjusted features
- Limited training data (2 seasons)

---

## Comparison: Ensemble vs Monte Carlo

| Prediction | Ensemble Winner | MC Winner | Ensemble Prob | MC Prob | Agreement |
|------------|----------------|-----------|---------------|---------|-----------|
| Game 1     | 76ers          | **Hawks** | 53.3%         | 72.7%   | ❌ Disagree |
| Game 2     | Pacers         | Pacers    | 58.6%         | 80.1%   | ✅ Agree |
| Game 3     | Knicks         | Knicks    | 66.2%         | 59.5%   | ✅ Agree |
| Game 4     | Bulls          | **Raptors** | 56.2%       | 50.5%   | ❌ Disagree |
| Game 5     | Suns           | Suns      | 58.7%         | 50.8%   | ✅ Agree |
| Game 6     | Celtics        | Celtics   | 50.4%         | 62.8%   | ✅ Agree |
| Game 7     | Kings          | **Magic** | 52.2%         | 66.6%   | ❌ Disagree |
| Game 8     | Nuggets        | **Clippers** | 56.4%     | 61.7%   | ❌ Disagree |

**Agreement Rate:** 50% (4 out of 8 games)

The ensemble blends LightGBM with simpler team strength ratings, which causes divergence. Monte Carlo uses pure quantile outputs and may be more aligned with model internals.

---

## Next Steps to Improve Confidence

1. **Add opponent-adjusted features** (e.g., opponent offensive rating, defensive matchup stats)
2. **Increase training data** (add 2022-23 season or earlier)
3. **Conditional interval estimation** (fit separate quantile models for different matchup types)
4. **Residual bootstrap** (sample from historical errors to get empirical distribution)
5. **Bayesian hierarchical model** (if you want principled posterior intervals and partial pooling)

---

## Files Generated

- **CSV Results:** [diagnostics/mc_simulations_20260219.csv](diagnostics/mc_simulations_20260219.csv)
- **Simulator Script:** [tools/monte_carlo_simulator.py](tools/monte_carlo_simulator.py)
- **This Summary:** MC_SIMULATIONS_README.md

---

## How to Re-Run

```bash
python tools/monte_carlo_simulator.py \
  --model machine_learning/models/lgbm_win_predictor_latest.pkl \
  --date 2026-02-19 \
  --n 10000 \
  --out diagnostics/mc_simulations_20260219.csv
```

Increase `--n` to 50,000 or 100,000 for more stable percentiles (diminishing returns after 10k).
