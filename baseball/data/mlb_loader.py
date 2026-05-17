"""MLB data loader using the MLB Stats API (MLB-StatsAPI package)."""

from datetime import datetime, timedelta

import pandas as pd
import statsapi


_OPENING_DAYS = {
    2022: '2022-04-07',
    2023: '2023-03-30',
    2024: '2024-03-20',
    2025: '2025-03-27',
}


def current_season() -> int:
    today = datetime.now()
    return today.year if today.month >= 3 else today.year - 1


def _season_start(season: int) -> str:
    return _OPENING_DAYS.get(season, f'{season}-03-27')


def fetch_todays_games() -> list:
    """
    Return list of today's scheduled MLB games.

    Each entry dict: game_id, game_date, game_time, home_team, home_team_id,
                     away_team, away_team_id, home_pitcher, away_pitcher, venue
    """
    today = datetime.now().strftime('%Y-%m-%d')
    try:
        raw = statsapi.schedule(date=today, sportId=1)
    except Exception as exc:
        print(f'  Warning: could not fetch today\'s schedule: {exc}', flush=True)
        return []

    games = []
    for g in raw:
        if g.get('status') in (
            'Preview', 'Pre-Game', 'Scheduled', 'Warmup',
            'Delayed', 'Delayed Start',
        ):
            games.append({
                'game_id': g['game_id'],
                'game_date': g['game_date'],
                'game_time': g.get('game_datetime', ''),
                'home_team': g['home_name'],
                'home_team_id': g['home_id'],
                'away_team': g['away_name'],
                'away_team_id': g['away_id'],
                'home_pitcher': g.get('home_probable_pitcher', 'TBD'),
                'away_pitcher': g.get('away_probable_pitcher', 'TBD'),
                'venue': g.get('venue_name', ''),
            })

    return games


def fetch_season_game_log(season: int = None, verbose: bool = False) -> pd.DataFrame:
    """
    Return a DataFrame of all completed MLB games this season up through yesterday.

    Columns: game_id, game_date (datetime), home_team_id, away_team_id,
             home_team, away_team, home_score, away_score, home_win (int 0/1)
    """
    if season is None:
        season = current_season()

    start_date = _season_start(season)
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    if verbose:
        print(f'  Fetching {season} MLB season: {start_date} → {yesterday}', flush=True)

    try:
        raw = statsapi.schedule(start_date=start_date, end_date=yesterday, sportId=1)
    except Exception as exc:
        print(f'  Warning: statsapi.schedule failed: {exc}', flush=True)
        return pd.DataFrame()

    rows = []
    for g in raw:
        if g.get('status') != 'Final':
            continue
        try:
            home_score = int(g.get('home_score', 0))
            away_score = int(g.get('away_score', 0))
        except (ValueError, TypeError):
            continue
        rows.append({
            'game_id': g['game_id'],
            'game_date': g['game_date'],
            'home_team_id': g['home_id'],
            'away_team_id': g['away_id'],
            'home_team': g['home_name'],
            'away_team': g['away_name'],
            'home_score': home_score,
            'away_score': away_score,
            'home_win': 1 if home_score > away_score else 0,
        })

    if not rows:
        if verbose:
            print('  No completed games found.', flush=True)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values('game_date').reset_index(drop=True)

    if verbose:
        print(f'  Loaded {len(df)} completed games', flush=True)

    return df
