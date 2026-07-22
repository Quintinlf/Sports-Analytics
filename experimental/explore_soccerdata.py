"""Exploratory script (temporary) — validate soccerdata/FBref behavior for
international tournament data before building training/fifa_trainer.py.

Not part of the production pipeline. Safe to delete once Phase 2's data
source approach is confirmed.
"""
import soccerdata as sd

# Test grabbing World Cup data using FBref's actual international league id
try:
    fbref = sd.FBref(leagues="INT-World Cup", seasons="2022")
    print("Successfully initialized FBref for World Cup!")
    print("Available leagues:", fbref.leagues)

    for stat_type in ["standard", "passing", "shooting", "defense"]:
        try:
            stats = fbref.read_player_season_stats(stat_type=stat_type)
            print(f"\n--- {stat_type} ({len(stats)} rows) ---")
            print(stats.head())
            print("columns:", list(stats.columns)[:20])
            print("index names:", stats.index.names)
        except Exception as e:
            print(f"read_player_season_stats(stat_type={stat_type!r}) failed: {e}")
except Exception as e:
    print(f"Direct tournament pull failed: {e}")

# Also inspect what competitions FBref actually exposes, since the World Cup
# constructor call above may fail if "WC" isn't a valid league id.
try:
    print("\nAvailable FBref competitions:")
    print(sd.FBref.available_leagues())
except Exception as e:
    print(f"Could not list available leagues: {e}")

# Check available seasons for international competitions, and whether
# read_schedule() gives usable match results (labels) for training.
try:
    seasons = sd.FBref.available_seasons(leagues=["INT-World Cup", "INT-European Championship"])
    print("\nAvailable seasons:")
    print(seasons)
except Exception as e:
    print(f"Could not list available seasons: {e}")

try:
    fbref_multi = sd.FBref(leagues="INT-World Cup", seasons=["2018", "2022"])
    schedule = fbref_multi.read_schedule()
    print(f"\n--- schedule ({len(schedule)} rows) ---")
    print(schedule.head())
    print("columns:", list(schedule.columns))
except Exception as e:
    print(f"read_schedule failed: {e}")
