"""
FIFA International Squad-Profile Training Pipeline

Ingests player-season stats for national squads at major tournaments via
soccerdata's FBref scraper, aggregates them into squad-level profiles,
reduces dimensionality with PCA, and trains a classifier on historical
match outcomes (home win / draw / away win) — per the Al-Bustami & Ghazal
player-attribute-aggregation methodology.

v1 scope (confirmed after exploratory testing — see
experimental/explore_soccerdata.py): 'standard' + 'shooting' + 'misc' season
stat categories only. soccerdata 1.9.0's read_player_season_stats() does not
expose season-level passing or defensive-action tables (only
read_player_match_stats(stat_type='summary'), which would require scraping
and aggregating every individual match — deferred to a later iteration).

Known limitation: only a handful of major tournaments are available per
competition (World Cup: 2018, 2022; Euros: 2016, 2020, 2024), so total
training data is small (on the order of 100-300 matches). High overfitting
risk is expected — PCA dimensionality reduction plus a shallow, regularized
classifier is a deliberate response to that, matching the paper's approach.
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

import soccerdata as sd
from machine_learning.fifa_ensemble_model import FIFAEnsembleModel

COMPETITIONS: Dict[str, List[str]] = {
    "INT-World Cup": ["2018", "2022"],
    "INT-European Championship": ["2016", "2020", "2024"],
}

SEASON_STAT_TYPES = ["standard", "shooting", "misc"]

MODEL_DIR = os.path.join("machine_learning", "models")
MODEL_FILENAME = "fifa_ensemble.pkl"


# ---------------------------------------------------------------------------
# 1) Ingestion
# ---------------------------------------------------------------------------

def fetch_squad_player_stats(verbose: bool = True) -> pd.DataFrame:
    """
    Fetch player-season stats for every configured tournament, across the
    v1 stat_type set, and concatenate into one long DataFrame with columns
    league, season, team, player, plus prefixed stat columns
    (e.g. standard__Standard_Gls, shooting__Standard_Sh, ...).
    """
    frames = []
    for league, seasons in COMPETITIONS.items():
        if verbose:
            print(f"  Fetching {league} {seasons}...", flush=True)
        try:
            fbref = sd.FBref(leagues=league, seasons=seasons)
        except Exception as exc:
            if verbose:
                print(f"    ERROR initializing FBref for {league}: {exc}", flush=True)
            continue

        stat_frames = []
        for stat_type in SEASON_STAT_TYPES:
            try:
                df = fbref.read_player_season_stats(stat_type=stat_type)
            except Exception as exc:
                if verbose:
                    print(f"    ERROR fetching {stat_type} for {league}: {exc}", flush=True)
                continue
            # Flatten MultiIndex columns (e.g. ('Standard', 'Gls') -> 'Standard_Gls').
            df = df.copy()
            df.columns = [
                '_'.join(str(p) for p in col if p).strip('_') if isinstance(col, tuple) else str(col)
                for col in df.columns
            ]
            df = df.add_prefix(f'{stat_type}__')
            stat_frames.append(df)

        if not stat_frames:
            continue

        merged = stat_frames[0]
        for extra in stat_frames[1:]:
            merged = merged.join(extra, how='outer')
        frames.append(merged.reset_index())

    if not frames:
        raise ValueError("No FIFA player-season stats fetched for any competition.")

    return pd.concat(frames, ignore_index=True)


def _parse_fbref_score(raw: object) -> Optional[Tuple[int, int]]:
    """Parse an FBref score string like '4-2' or '1-1 (4-2)' (penalty
    shootouts) into (home_goals, away_goals) using the regulation/ET score
    before the parenthesis — shootouts don't change the match outcome label
    used here (still a draw for classification purposes)."""
    if not isinstance(raw, str):
        return None
    base = raw.split('(')[0].strip()
    match = re.match(r'(\d+)\s*[-–]\s*(\d+)', base)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def fetch_match_results(verbose: bool = True) -> pd.DataFrame:
    """
    Fetch match schedules/results for every configured tournament.

    Returns
    -------
    pd.DataFrame with columns: league, season, home_team, away_team,
    home_goals, away_goals, outcome ('HOME_WIN' | 'DRAW' | 'AWAY_WIN').
    """
    frames = []
    for league, seasons in COMPETITIONS.items():
        if verbose:
            print(f"  Fetching schedule for {league} {seasons}...", flush=True)
        try:
            fbref = sd.FBref(leagues=league, seasons=seasons)
            schedule = fbref.read_schedule().reset_index()
        except Exception as exc:
            if verbose:
                print(f"    ERROR fetching schedule for {league}: {exc}", flush=True)
            continue
        frames.append(schedule)

    if not frames:
        raise ValueError("No FIFA match schedules fetched for any competition.")

    schedule = pd.concat(frames, ignore_index=True)
    schedule = schedule.dropna(subset=['score']).copy()

    parsed = schedule['score'].apply(_parse_fbref_score)
    schedule = schedule[parsed.notna()].copy()
    parsed = parsed[parsed.notna()]
    schedule['home_goals'] = parsed.apply(lambda t: t[0])
    schedule['away_goals'] = parsed.apply(lambda t: t[1])

    def _outcome(row) -> str:
        if row['home_goals'] > row['away_goals']:
            return 'HOME_WIN'
        if row['home_goals'] < row['away_goals']:
            return 'AWAY_WIN'
        return 'DRAW'

    schedule['outcome'] = schedule.apply(_outcome, axis=1)
    if verbose:
        print(f"  Parsed {len(schedule)} matches with results", flush=True)
    return schedule[['league', 'season', 'home_team', 'away_team', 'home_goals', 'away_goals', 'outcome']]


# ---------------------------------------------------------------------------
# 2) Squad-level aggregation
# ---------------------------------------------------------------------------

def build_squad_profiles(player_stats: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Aggregate player-season stats to one row per (league, season, team) squad.

    Uses the mean across players for every numeric column. FBref's season
    tables mix rate and counting stats inconsistently, and the downstream
    StandardScaler + PCA step absorbs scale differences regardless, so a
    single aggregation rule keeps this step simple for v1.
    """
    numeric_cols = player_stats.select_dtypes(include=[np.number]).columns.tolist()
    if verbose:
        print(f"  Aggregating {len(numeric_cols)} numeric player columns to squad level...", flush=True)

    grouped = player_stats.groupby(['league', 'season', 'team'])[numeric_cols].mean()
    grouped = grouped.fillna(0.0)
    return grouped.reset_index()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def full_retrain(verbose: bool = True) -> Dict:
    """Run the full FIFA training pipeline and persist machine_learning/models/fifa_ensemble.pkl."""
    if verbose:
        print('\n' + '=' * 70, flush=True)
        print('FIFA FULL RETRAIN STARTED', flush=True)
        print('=' * 70, flush=True)

    version = datetime.now().strftime('v%Y%m%d_%H%M%S')

    if verbose:
        print('\n[1/4] Fetching player-season stats...', flush=True)
    player_stats = fetch_squad_player_stats(verbose=verbose)

    if verbose:
        print('\n[2/4] Fetching match results...', flush=True)
    match_results = fetch_match_results(verbose=verbose)

    if verbose:
        print('\n[3/4] Aggregating squad profiles...', flush=True)
    squad_profiles = build_squad_profiles(player_stats, verbose=verbose)

    if verbose:
        print('\n[4/4] Training PCA + ensemble classifier...', flush=True)
    model = FIFAEnsembleModel(n_components=8)
    metrics = model.fit(squad_profiles, match_results, verbose=verbose)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    joblib.dump({
        'model': model,
        'squad_profiles': squad_profiles,
        'version': version,
        'trained_at': datetime.now().isoformat(),
        'metrics': metrics,
    }, model_path)

    if verbose:
        print('\n' + '=' * 70)
        print('FIFA FULL RETRAIN COMPLETE')
        print(f'Model version: {version}')
        print(f"Accuracy: {metrics['accuracy']:.1%}")
        print(f'Saved -> {model_path}')
        print('=' * 70 + '\n')

    return {'model_version': version, 'model_path': model_path, 'metrics': metrics}


if __name__ == '__main__':
    full_retrain()
