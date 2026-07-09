# MLB Data Sources

## Phase 6a — MLB-StatsAPI (production cron)

| Data | Source | Status |
|------|--------|--------|
| Schedule (teams, date, game_id) | `statsapi.schedule()` | In use |
| Probable starting pitchers (name) | Schedule payload (`home_probable_pitcher`, `away_probable_pitcher`) | In use |
| Pitcher ERA / WHIP / K9 | `statsapi.player_stat_data()` | Best-effort per starter |
| Bullpen 3-day workload | — | Not available via schedule; flagged `bullpen_workload_unavailable` |
| Confirmed lineups | — | Not ingested; `lineups.confirmed=false` |
| Park factors / weather | — | Future; may require boxscore or third-party |

Stored in `predictions.feature_snapshot` JSON:

- `starting_pitchers.home` / `.away`
- `bullpen` (placeholder)
- `lineups` (placeholder)
- `missing_data_warnings[]`

## Phase 6b — pybaseball (optional, automation only)

| Data | Source | Status |
|------|--------|--------|
| Season pitching stats (ERA, WHIP, SO9) | `pybaseball.pitching_stats()` | Optional fallback in `data/mlb_context.py` |
| Statcast / game logs | pybaseball | Not wired; requires name matching |
| Bullpen usage | Statcast pitch-level | Requires scraping/parsing; document only |

Install: `requirements-automation.txt` includes `pybaseball` for GitHub Actions cron only. Render web service does not require it.

## Scraping / paid (not implemented)

- Baseball Savant bullpen fatigue (3-day IP)
- Official MLB lineups API (confirmed batting order)
- Injury feeds (IL list scraping)

When data is missing, the dashboard shows `missing_data_warnings` so analysts know what the model could not see.
