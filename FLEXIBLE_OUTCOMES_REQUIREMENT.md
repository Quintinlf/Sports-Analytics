---
name: Flexible Prediction Outcomes - Requirement Summary
---

# 🔄 Flexible Prediction Outcomes — Implementation Guide

## New Requirement

**Support an arbitrary number of prediction outcomes**, not just hardcoded binary/ternary.

### Examples

```
NBA (Binary):
  ✓ Knicks Win (62%)
  ✓ Spurs Win (38%)

MLB (Binary):
  ✓ Dodgers Win (55%)
  ✓ Giants Win (45%)

Soccer (Ternary):
  ✓ Sweden Win (52%)
  ✓ Draw (28%)
  ✓ Tunisia Win (20%)

Future: Over/Under Markets (Binary):
  ✓ Over 45.5 (58%)
  ✓ Under 45.5 (42%)

Future: Playoff Series (Quaternary+):
  ✓ Team A Wins Series (35%)
  ✓ Team B Wins Series (40%)
  ✓ Team C Wins Series (25%)
```

### Key Constraint

**NO hardcoding:**
- ❌ `home_win_probability`
- ❌ `away_win_probability`
- ❌ Sport-specific prediction fields

**YES:**
- ✅ `prediction_options` table (flexible)
- ✅ `option_name` + `probability` + `rank`
- ✅ Dynamic email & form rendering

---

## Database Design

### Two-Table Architecture

#### `predictions` (Core)
```sql
prediction_id          -- PK
sport                  -- "NBA", "MLB", "SOCCER"
game_date              -- Date of game
home_team, away_team   -- Matchup
predicted_winner       -- Top option (option_name) by probability
confidence_level       -- "HIGH" / "MEDIUM" / "LOW"
confidence_score       -- Max probability (0-1)
actual_winner          -- Result (option_name that occurred)
correct                -- 1 if predicted_winner == actual_winner
feature_snapshot       -- JSON (features)
model_version          -- Model identifier
```

#### `prediction_options` (Flexible Outcomes) ← NEW
```sql
option_id              -- PK
prediction_id          -- FK to predictions
option_name            -- "Knicks Win", "Draw", "Over 45.5", etc.
probability            -- P(outcome) in [0, 1]
rank                   -- 1 = most likely, 2 = second, etc.
implied_odds           -- Optional (for betting: -110, +150)
description            -- Optional ("Knicks win by any margin")
```

### Example Data

```
Prediction (NBA):
  prediction_id=1
  sport=NBA
  game_date=2026-06-13
  home_team=Knicks
  away_team=Spurs
  predicted_winner=Knicks Win
  confidence_score=0.62
  confidence_level=HIGH

Prediction Options:
  option_id=1, prediction_id=1, option_name=Knicks Win, probability=0.62, rank=1
  option_id=2, prediction_id=1, option_name=Spurs Win, probability=0.38, rank=2

Prediction (Soccer):
  prediction_id=2
  sport=SOCCER
  game_date=2026-06-13
  home_team=Sweden
  away_team=Tunisia
  predicted_winner=Sweden Win
  confidence_score=0.52
  confidence_level=MEDIUM

Prediction Options:
  option_id=3, prediction_id=2, option_name=Sweden Win, probability=0.52, rank=1
  option_id=4, prediction_id=2, option_name=Draw, probability=0.28, rank=2
  option_id=5, prediction_id=2, option_name=Tunisia Win, probability=0.20, rank=3
```

---

## Implementation Impact (2-Week Timeline)

### Days 1–2: Database Schema (+2 hours)

**Create `prediction_options` table:**

```python
# In data/database/database_handler.py

cursor.execute("""
    CREATE TABLE IF NOT EXISTS prediction_options (
        option_id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_id INTEGER NOT NULL,
        option_name TEXT NOT NULL,
        probability REAL NOT NULL,
        rank INTEGER NOT NULL,
        implied_odds REAL,
        description TEXT,
        FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id),
        UNIQUE(prediction_id, option_name)
    )
""")

cursor.execute("CREATE INDEX IF NOT EXISTS idx_options_prediction ON prediction_options(prediction_id)")
```

**Update `predictions` table:**
- Remove: `win_probability` (derived from options)
- Keep: `predicted_winner`, `confidence_score`, `confidence_level`

**Add method to `SportsAnalyticsDB`:**

```python
def insert_option(self, prediction_id: int, option_data: Dict) -> int:
    """Insert a prediction option."""
    cursor = self.conn.cursor()
    cursor.execute("""
        INSERT INTO prediction_options (
            prediction_id, option_name, probability, rank, implied_odds, description
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        prediction_id,
        option_data["option_name"],
        option_data["probability"],
        option_data["rank"],
        option_data.get("implied_odds"),
        option_data.get("description"),
    ))
    self.conn.commit()
    return cursor.lastrowid
```

### Days 3–8: Update Models (+4 hours)

**Change model output format:**

```python
# OLD (hardcoded)
def predict(self, features):
    return {
        "home_win_prob": 0.62,
        "away_win_prob": 0.38,
    }

# NEW (flexible)
def predict(self, features):
    return {
        "options": [
            {"option_name": "Home Team", "probability": 0.62, "rank": 1},
            {"option_name": "Away Team", "probability": 0.38, "rank": 2},
        ],
        "predicted_winner": "Home Team",
        "confidence_score": 0.62,
    }
```

**Apply to:**
- `machine_learning/mlb_models.py` (MLBEloModel, MLBXGBoostPredictor)
- `machine_learning/soccer_models.py` (SoccerEloModel, SoccerXGBoostPredictor)
- NBA ensemble: convert output at insertion time (see Days 9–10)

### Days 9–10: Store Options (+3 hours)

**Insert both prediction + options:**

```python
def store_prediction(prediction_row, options_list):
    """Store prediction and all options."""
    
    db = SportsAnalyticsDB()
    
    # Insert prediction
    prediction_id = db.insert_prediction({
        "sport": prediction_row["sport"],
        "game_date": prediction_row["game_date"],
        "home_team": prediction_row["home_team"],
        "away_team": prediction_row["away_team"],
        "predicted_winner": options_list[0]["option_name"],  # Top option
        "confidence_score": options_list[0]["probability"],
        "confidence_level": get_confidence_level(options_list[0]["probability"]),
        "feature_snapshot": json.dumps(prediction_row.get("features", {})),
        "model_version": prediction_row.get("model_version"),
    })
    
    # Insert options (one row per outcome)
    for rank, option in enumerate(options_list, start=1):
        db.insert_option(prediction_id, {
            "option_name": option["option_name"],
            "probability": option["probability"],
            "rank": rank,
            "implied_odds": option.get("implied_odds"),
            "description": option.get("description"),
        })
    
    return prediction_id

def get_confidence_level(max_prob):
    if max_prob >= 0.65:
        return "HIGH"
    elif max_prob >= 0.55:
        return "MEDIUM"
    else:
        return "LOW"
```

### Days 9–10: Email Report (+2 hours)

**Dynamic rendering (NO sport-specific hardcoding):**

```python
# In reports/unified_daily_report.py

def generate_unified_report() -> str:
    """Generate email with dynamic options."""
    
    db = SportsAnalyticsDB()
    today = datetime.utcnow().date().isoformat()
    
    predictions = pd.read_sql_query(
        f"SELECT * FROM predictions WHERE game_date = '{today}' ORDER BY sport, game_date",
        db.conn
    )
    
    html_sections = {}
    for sport in ["NBA", "MLB", "SOCCER"]:
        sport_preds = predictions[predictions["sport"] == sport]
        
        rows_html = ""
        for _, pred in sport_preds.iterrows():
            # Fetch options from database (NOT hardcoded)
            options = pd.read_sql_query(
                "SELECT * FROM prediction_options WHERE prediction_id = ? ORDER BY rank",
                db.conn,
                params=(pred["prediction_id"],)
            )
            
            # Dynamically render all options
            options_html = " | ".join([
                f"{row['option_name']}: {row['probability']*100:.0f}%"
                for _, row in options.iterrows()
            ])
            
            rows_html += f"""
            <tr>
                <td>{pred['home_team']} vs {pred['away_team']}</td>
                <td><strong>{pred['predicted_winner']}</strong></td>
                <td>{pred['confidence_level']}</td>
                <td>{options_html}</td>
            </tr>
            """
        
        html_sections[sport] = rows_html if rows_html else "No predictions"
    
    # Same template for all sports
    html_template = """
    <html><head><style>
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; }
    </style></head><body>
        <h1>Daily Sports Predictions - {{ date }}</h1>
        <h2>🏀 NBA</h2>
        <table><tr><th>Matchup</th><th>Prediction</th><th>Confidence</th><th>All Options</th></tr>
        {{ nba_rows }}
        </table>
        <h2>⚾ MLB</h2>
        <table><tr><th>Matchup</th><th>Prediction</th><th>Confidence</th><th>All Options</th></tr>
        {{ mlb_rows }}
        </table>
        <h2>⚽ Soccer</h2>
        <table><tr><th>Matchup</th><th>Prediction</th><th>Confidence</th><th>All Options</th></tr>
        {{ soccer_rows }}
        </table>
    </body></html>
    """
    
    from jinja2 import Template
    template = Template(html_template)
    return template.render(
        date=today,
        nba_rows=html_sections["NBA"],
        mlb_rows=html_sections["MLB"],
        soccer_rows=html_sections["SOCCER"],
    )
```

### Days 11–12: Feedback Forms (+3 hours)

**Forms dynamically fetch and render options:**

```html
<!-- backend/static/feedback_form_v2.html -->

<form id="feedbackForm">
    <input type="hidden" id="prediction_id" />
    
    <div class="form-group">
        <label>What do you think will happen?</label>
        <!-- DYNAMIC: Options populated from API -->
        <select id="user_option">
            <option value="">Loading options...</option>
        </select>
    </div>
    
    <div class="form-group">
        <label>Your Confidence (1-10):</label>
        <input type="range" id="user_confidence" min="1" max="10" value="5" />
    </div>
    
    <button type="submit">Submit Feedback</button>
</form>

<script>
    // Get prediction_id from URL
    const prediction_id = new URLSearchParams(window.location.search).get("prediction_id");
    document.getElementById("prediction_id").value = prediction_id;
    
    // Fetch options from API (NO hardcoding)
    fetch(`/api/v2/predictions/${prediction_id}/options`)
        .then(res => res.json())
        .then(options => {
            const select = document.getElementById("user_option");
            select.innerHTML = "";  // Clear "Loading"
            options.forEach(opt => {
                const option = document.createElement("option");
                option.value = opt.option_name;  // User picks option_name
                option.textContent = `${opt.option_name} (Model: ${(opt.probability*100).toFixed(0)}%)`;
                select.appendChild(option);
            });
        });
    
    // Submit
    document.getElementById("feedbackForm").addEventListener("submit", async (e) => {
        e.preventDefault();
        
        const feedback = {
            prediction_id: prediction_id,
            user_option: document.getElementById("user_option").value,  // option_name
            user_confidence: parseInt(document.getElementById("user_confidence").value),
        };
        
        await fetch("/api/v2/feedback", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(feedback),
        });
        
        alert("Feedback submitted!");
    });
</script>
```

**API endpoint (NEW):**

```python
# In backend/main_v2.py

@app.get("/api/v2/predictions/{prediction_id}/options")
def get_prediction_options(prediction_id: int):
    """Fetch options for a prediction."""
    from data.database.database_handler import SportsAnalyticsDB
    db = SportsAnalyticsDB()
    
    options = db.conn.execute(
        "SELECT option_name, probability, rank FROM prediction_options WHERE prediction_id = ? ORDER BY rank",
        (prediction_id,)
    ).fetchall()
    
    return [dict(opt) for opt in options]
```

**Update feedback ORM:**

```python
# In backend/models_v2.py

class UserFeedback(Base):
    __tablename__ = "user_feedback"
    
    feedback_id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.prediction_id"))
    user_option: Mapped[str] = mapped_column(String(100))  # "Knicks Win", "Draw", etc.
    user_confidence: Mapped[int] = mapped_column(Integer, nullable=True)
    disagreement_reason: Mapped[str] = mapped_column(Text, nullable=True)
    form_responses: Mapped[dict] = mapped_column(JSON().with_variant(JSONB, "postgresql"))
    submitted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
```

### Days 13–14: Testing (+2 hours)

**Verify:**
- Options inserted correctly (N outcomes per prediction)
- Email renders all options dynamically
- Feedback forms populate options from API
- User feedback correctly stores option_name

---

## Benefits

### ✅ Flexibility
- **Binary:** NBA, MLB → 2 options
- **Ternary:** Soccer → 3 options
- **Quaternary+:** Playoffs, tournaments → N options
- **Future:** Over/under, prop bets, etc.

### ✅ No Hardcoding
- Email template is **identical for all sports**
- Feedback form **auto-populates options from API**
- No sport-specific if/else statements

### ✅ Future-Proof
- Add new outcome type? Just add more options.
- Change outcome names? Update option_name in options table.
- Add odds/props? Add implied_odds field.

### ✅ Analytics
- Compare human picks vs. model picks (per option)
- Track calibration by option
- Identify disagreements at option level

---

## Summary

| Component | Change | Effort |
|-----------|--------|--------|
| **Database** | Add `prediction_options` table | 2 hrs |
| **Models** | Return options array | 4 hrs |
| **Storage** | Insert prediction + N options | 3 hrs |
| **Email** | Render options dynamically | 2 hrs |
| **Feedback Forms** | Fetch options from API | 3 hrs |
| **API** | Add options endpoint | 1 hr |
| **Testing** | Verify all pieces | 2 hrs |
| **Total** | **Flexible architecture** | **~17 hours** |

**Fits comfortably in 2-week MVP (60 hours available). ✅**

---

## Key Files (Updated)

### Create (New)
- None new beyond original MVP

### Modify
- `data/database/database_handler.py` (add `prediction_options` table + `insert_option()`)
- `machine_learning/mlb_models.py` (return options array)
- `machine_learning/soccer_models.py` (return options array)
- `scripts/run_daily_all_sports.py` (insert options)
- `reports/unified_daily_report.py` (dynamic rendering)
- `backend/main_v2.py` (add GET `/api/v2/predictions/{id}/options`)
- `backend/models_v2.py` (update `UserFeedback` schema)
- `backend/static/feedback_form_v2.html` (fetch options from API)

### Keep As-Is
- All other files unchanged

---

## Next Steps

1. ✅ Review this document
2. ✅ Read [FLEXIBLE_OUTCOMES_ARCHITECTURE.md](/.cursor/plans/flexible_outcomes_architecture.md) for detailed implementation
3. ✅ Update implementation plan with +15 hours estimated
4. 🚀 **Ship with flexible outcomes from day 1**

**No redesigns after MVP. This architecture scales. 🎯**
