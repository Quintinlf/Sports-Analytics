# Sports Analytics — NBA Prediction System

A machine-learning NBA prediction system that blends **LightGBM quantile regression**, **Gaussian Process** uncertainty, **Bayesian Ridge**, **XGBoost**, and a **Random Forest** ensemble. The system was enhanced to incorporate a possession-based decision-making framework built on three foundational concepts from basketball analytics.

---

## What Was Built

### The Friend's Framework (Why It Matters)

Your friend's insight is that basketball is fundamentally a game of **decision-making per possession**. There are only so many possessions per game (~100), and teams that make better *decisions* with each one consistently win over time. This idea is captured by three complementary lenses:

| Lens | What It Measures | Why It Matters |
|------|-----------------|----------------|
| **Dean Oliver's Four Factors** | How efficiently a team uses each possession | The four levers every coach actually controls |
| **Net Rating** | Points scored minus allowed per 100 possessions | The "north star" — cuts through variance better than win-loss record |
| **Pythagorean Win %** | Expected win rate based on scoring efficiency | Detects teams that are over- or under-performing their true quality |

---

## Dean Oliver's Four Factors

Dean Oliver (author of *Basketball on Paper*) showed that four statistics explain ~96% of the variance in a team's winning percentage. Each possession goes through all four:

### 1. eFG% — Effective Field Goal Percentage
**What it is:** A shot quality metric that weights 3-pointers 1.5× a 2-pointer, because they're worth more.

```
eFG% = (FGM + 0.5 × 3PM) / FGA
```

**What it tells you:** A team with 55% eFG is generating better shot opportunities than one at 50%, regardless of shot volume. The best offenses in NBA history rank near the top here. **Weight: 40%** — it's the single biggest factor.

**In the output table:** Top value = away team / bottom = home team. **Green bold = winner of this factor.**

---

### 2. TOV% — Turnover Rate per Possession
**What it is:** How often a team gives the ball away before even attempting a shot.

```
TOV% = TOV / (FGA + 0.44 × FTA + TOV)
```

**What it tells you:** **Lower is better.** A turnover is the worst outcome of any possession — you gave the opponent a free chance to score without any defense needing to be played. Teams that protect the ball are making better decisions. **Weight: 25%.**

**In the output table:** Green bold = the team with the *lower* (better) turnover rate.

> **Example:** If MIA has TOV% 12.8% and CHA has 14.1%, Miami is making smarter decisions on offense — every ~7 possessions they save one extra shot attempt that Charlotte throws away.

---

### 3. OREB% — Offensive Rebound Rate
**What it is:** Of all missed shots, what percentage does the offensive team grab back?

```
OREB% = OREB / (OREB + Opponent DREB)
```

**What it tells you:** Offensive rebounds turn "failed possessions" into second chances. A team grabbing 30% of its own misses is effectively getting more possessions per game. **Weight: 20%.**

---

### 4. FT Rate — Free Throw Rate
**What it is:** How often a team gets to the free throw line relative to field goal attempts.

```
FT Rate = FTA / FGA
```

**What it tells you:** Free throws are the most efficient "shot" in basketball (no defender, uncontested). Teams that draw fouls score free points and also put opponents in foul trouble. **Weight: 15%.**

---

## Net Rating — The "North Star"

**What it is:** Points scored minus points allowed per 100 possessions.

```
Net Rating = (Points For − Points Against) / Possessions × 100
```

**What it tells you:** A team with +6.0 Net Rating outscores opponents by 6 points per 100 possessions on average. This is the cleanest single-number measure of how good a team actually is — it smooths out scheduling noise, late-game garbage time, and win-loss record variance (a team can go 3-7 in close games despite being a top-10 unit).

**In the output table:** The **Net Rtg Δ** column is *home minus away*. Green = home team has the net rating edge. The bigger the gap, the clearer the talent separation.

> **Rule of thumb:** 0 to +3 = average, +4 to +7 = playoff contender, +8 and above = title contender.

---

## Pythagorean Win Percentage

**What it is:** An expected win percentage calculated purely from points scored and allowed, using the NBA-tuned exponent of 1.67 (Bill James originally developed this for baseball).

```
Pythagorean Win% = PF^1.67 / (PF^1.67 + PA^1.67)
```

**What it tells you:** This is a *prediction* of what a team's record *should* be given their scoring efficiency. When compared to their actual record:

| Situation | What It Means |
|-----------|--------------|
| Actual W-L **much better** than Pythagorean | Team is over-performing — likely been lucky in close games. Regression toward .500 is coming. |
| Actual W-L **much worse** than Pythagorean | Team is under-performing — probably unlucky in close games. "Buy low" opportunity. |
| Actual ≈ Pythagorean | Team is playing exactly to their talent level. |

**In the output table:** The **Pythagorean** column shows:
- `✅ STRONG` — Pythagorean agrees with the model *and* the gap is large (>8 pts). High confidence signal.
- `✅ CONFIRM` — Pythagorean agrees with the model (smaller gap). Moderate validation.
- `⚠️ CONFLICT` — Pythagorean expected wins point to the *opposite* team winning. Confidence penalty applied.

---

## Output Column Guide

| Column | Definition | Values |
|--------|-----------|--------|
| **Time / TV** | Tip-off time (ET) and broadcast network | e.g. `7:30 PM ET / ESPN` |
| **Matchup** | Away team @ Home team | Top = away, bottom = home |
| **Model Pick** | Predicted winner and margin | `● Celtics by 14.8 pts` — ● agrees with Vegas, ◆ disagrees |
| **Conf** | Confidence percentage with tier label | See section below |
| **Vegas Line** | DraftKings point spread | `Celtics -15.5` = Boston favored by 15.5 pts. A negative number always follows the favored team. |
| **O/U** | Over/Under total | `226.5` = Vegas estimates 226–227 total points scored. Bet over if you think offenses dominate; under if defense/pace is slow. |
| **Net Rtg Δ** | Home minus away Net Rating | `+5.4` green = home team is clearly better per-possession; `-2.1` red = away team has NRtg edge despite home court |
| **eFG%** | Effective field goal % per team | Top = away, bottom = home. Green bold = winner of this factor |
| **TOV%** | Turnover rate per possession | **Lower is better.** Green bold = team with fewer turnovers (better decision-making) |
| **OREB%** | Offensive rebound rate | Green bold = team grabbing more second chances |
| **FT Rate** | Free throw rate (FTA/FGA) | Green bold = team getting to the line more |
| **4F Edge** | Summary of how many Four Factors the *predicted winner* controls | `HOME 3/4` = home team wins 3 of the 4 factors. `EVEN 2/4` = split — low confidence situation |
| **Pythagorean** | Expected-win confirmation + per-team pct | `✅ STRONG` / `✅ CONFIRM` / `⚠️ CONFLICT` with both teams' expected win rates |

---

## Confidence Levels — Why They Are What They Are

Confidence (58%–93%) is computed from three independent signals that must all point the same direction for a HIGH rating:

```
Confidence = 64
           + (predicted margin) × 2.2     ← how many points does the model see?
           + Pythagorean bonus/penalty     ← −5 conflict / 0 confirm / +5 strong
           + Winner Four Factor bonus      ← −6 (0/4 factors) to +6 (4/4 factors)

Capped at min 58%, max 93%
```

### Thresholds

| Label | Range | Meaning |
|-------|-------|---------|
| **HIGH** | ≥ 82% | Large margin + winner dominates Four Factors + Pythagorean strongly confirms |
| **MED** | 72–81% | One team clearly better in most metrics, but not a clean sweep |
| **LOW** | < 72% | Close game, conflicting signals, or Pythagorean contradicts the model |

### Why confidence may look "low"

**This is correct, not a bug.** NBA games are genuinely close. In any given season, 30–40% of games are decided by 5 points or fewer. When the model predicts a 2-point margin, confidence should be modest — because there really isn't a dominant favorite.

**LOW confidence means:** "The signals don't align strongly enough to bet confidently. Pass this game or watch the number."

**Things that will NOT raise confidence:**
- Running the prediction cell again (it uses fixed season-to-date profiles — same result every time)
- Hoping the model "converges" — it's not iterating, it's computing

**Things that WILL raise confidence in the future:**
- Running the full `IterativePredictor` pipeline (`machine_learning/run_basketball_today.py`) which pulls live game-by-game NBA API data, retrains LightGBM + Gaussian Process up to 10 times until GP uncertainty < 0.4, and calibrates confidence from the ensemble spread across 5+ models
- Inputing more recent team data closer to tip-off (injury reports change profiles materially)

---

## How Each File Contributes

```
feature_selection/
  advanced_features.py    — Calculates rolling advanced stats from box scores.
                            NEW: TOV_PCT, OREB_PCT, NET_RTG, PYT_WIN added here.
                            All stats computed as 10-game rolling averages (±SE).

post_processors/
  h2h_adjuster.py         — Post-prediction layer. Blends 85% model / 15% H2H history.
                            NEW: compute_four_factors_edge() method added — computes
                            per-factor winner, net rating delta, Pythagorean edge, and
                            merges them into every prediction's output dict.

machine_learning/
  basketball_model.ipynb  — Main prediction notebook.
                            Cell 87: Full March 6 engine with team profiles, Four Factors
                            scoring, Pythagorean alignment, confidence formula, HTML report.

  run_basketball_today.py — Full pipeline runner (LightGBM + GP + Bayesian ensemble).
                            Use this for the highest-quality, data-driven predictions.

learners/
  model_trainer.py        — Trains LightGBM quantile model (Q10/Q50/Q90 spreads)
  adaptive_learner.py     — Handles iterative retraining loop
  mcmc_sampler.py         — Bayesian posterior sampling for uncertainty

predictors/
  lgbm_predictor.py       — LightGBM inference layer
  predictor.py            — Main prediction orchestrator

evaluators/
  validation_tracker.py   — Logs predictions; compares to actual outcomes over time
  backtest_validation.py  — Historical backtesting framework
```

---

## Running Predictions for Future Dates

### Quick predictions (notebook cell, no API calls):
1. Open `machine_learning/basketball_model.ipynb`
2. Update the `GAME_DATE` string and `vegas_lines` dict with tomorrow's games and DraftKings lines
3. Update any `team_profiles` entries where a key player is injured (reduce `str` and `net_rtg` accordingly)
4. Run cell 87 — HTML report auto-saves to the `machine_learning/` folder

### Full ML pipeline (highest accuracy):
```bash
python machine_learning/run_basketball_today.py
```
This ingests fresh NBA API game logs, retrains LightGBM/GP/Bayesian models, and produces calibrated confidence via iterative uncertainty reduction.

---

## Vegas Line Primer (for reference)

| Term | Meaning | Example |
|------|---------|---------|
| **Favorite** | Team Vegas expects to win | Celtics -15.5 |
| **Spread** | Points the favorite must win by to "cover" | BOS must win by 16+ |
| **Dog** | Underdog (positive line) | Mavericks +15.5 |
| **Moneyline** | Bet on who wins outright (no spread) | Usually listed separately |
| **Over/Under** | Total combined points | O/U 226.5: bet over 227 or under 226 |
| **Model ● agrees** | Model's predicted winner matches Vegas favorite | Low-risk confirmation |
| **Model ◆ disagrees** | Model picks the underdog | High-risk / high-reward; check confidence + Four Factors carefully |

---

*Models are trained on 2022–23, 2023–24, and 2024–25 NBA seasons. Always cross-reference injury reports before wagering.*
