"""Run NBA predictions for today's games using nba_api and a Bayesian Ridge model.

This script fetches recent seasons' game logs, engineers rolling features,
trains a BayesianRidge model on historical games (excluding today), and
predicts point differential and win probability for games occurring today.

Usage: python run_basketball_today.py
"""
from datetime import datetime, timedelta
import time
import pandas as pd
import numpy as np
import warnings

from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams

from sklearn.linear_model import BayesianRidge

warnings.filterwarnings('ignore')


def fetch_with_retry(season, max_retries=3):
    for attempt in range(max_retries):
        try:
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable='Regular Season',
                league_id_nullable='00',
                timeout=60,
            )
            games = gamefinder.get_data_frames()[0]
            return games
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None


def calculate_rolling_stats(df, window=5):
    df = df.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    for col in ['PTS', 'FG_PCT', 'FG3_PCT', 'REB', 'AST', 'STL', 'BLK', 'TOV']:
        if col in df.columns:
            df[f'{col}_ROLL'] = df.groupby('TEAM_ID')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )

    def calculate_streak(wl_series):
        streak = []
        current_streak = 0
        for wl in wl_series:
            if wl == 'W':
                current_streak = current_streak + 1 if current_streak >= 0 else 1
            else:
                current_streak = current_streak - 1 if current_streak <= 0 else -1
            streak.append(current_streak)
        return pd.Series(streak, index=wl_series.index)

    if 'WL' in df.columns:
        df['WIN_STREAK'] = df.groupby('TEAM_ID')['WL'].transform(calculate_streak)

    df['REST_DAYS'] = df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days.fillna(2)
    df['IS_BACK_TO_BACK'] = (df['REST_DAYS'] == 1).astype(int)

    if 'WL' in df.columns:
        df['WIN_RATE_10'] = df.groupby('TEAM_ID')['WL'].transform(
            lambda x: (x == 'W').rolling(window=10, min_periods=1).mean()
        )

    return df


def create_matchup_features(games_df, team_names):
    matchups = []
    for game_id, game_group in games_df.groupby('GAME_ID'):
        if len(game_group) == 2:
            # ensure a consistent order: home first
            game_group = game_group.sort_values('MATCHUP', ascending=False)
            home = game_group.iloc[0]
            away = game_group.iloc[1]

            def get_feat(row, prefix):
                return {
                    f'{prefix}_PTS_ROLL': row.get('PTS_ROLL', 0),
                    f'{prefix}_FG_PCT_ROLL': row.get('FG_PCT_ROLL', 0),
                    f'{prefix}_FG3_PCT_ROLL': row.get('FG3_PCT_ROLL', 0),
                    f'{prefix}_REB_ROLL': row.get('REB_ROLL', 0),
                    f'{prefix}_AST_ROLL': row.get('AST_ROLL', 0),
                    f'{prefix}_STL_ROLL': row.get('STL_ROLL', 0),
                    f'{prefix}_BLK_ROLL': row.get('BLK_ROLL', 0),
                    f'{prefix}_TOV_ROLL': row.get('TOV_ROLL', 0),
                    f'{prefix}_WIN_STREAK': row.get('WIN_STREAK', 0),
                    f'{prefix}_REST_DAYS': row.get('REST_DAYS', 2),
                    f'{prefix}_IS_BACK_TO_BACK': row.get('IS_BACK_TO_BACK', 0),
                    f'{prefix}_WIN_RATE_10': row.get('WIN_RATE_10', 0),
                }

            matchup = {
                'GAME_ID': game_id,
                'GAME_DATE': home['GAME_DATE'],
                'HOME_TEAM': home['TEAM_ID'],
                'AWAY_TEAM': away['TEAM_ID'],
                'HOME_TEAM_NAME': team_names.get(home['TEAM_ID'], 'Unknown'),
                'AWAY_TEAM_NAME': team_names.get(away['TEAM_ID'], 'Unknown'),
            }
            matchup.update(get_feat(home, 'HOME'))
            matchup.update(get_feat(away, 'AWAY'))
            # targets if available
            if 'PTS' in home and 'PTS' in away:
                matchup['HOME_PTS'] = home['PTS']
                matchup['AWAY_PTS'] = away['PTS']
                matchup['POINT_DIFF'] = home['PTS'] - away['PTS']
                matchup['HOME_WIN'] = 1 if home.get('WL') == 'W' else 0

            matchups.append(matchup)

    return pd.DataFrame(matchups)


def main():
    print('🏀 Starting NBA today runner')

    nba_teams = teams.get_teams()
    team_ids = [t['id'] for t in nba_teams]
    team_names = {t['id']: t['full_name'] for t in nba_teams}

    seasons = ['2023-24', '2024-25']
    frames = []
    for s in seasons:
        print(f'📥 Fetching {s}...')
        df = fetch_with_retry(s)
        if df is not None:
            frames.append(df)

    if not frames:
        print('⚠️ NBA API unavailable; exiting')
        return

    games = pd.concat(frames, ignore_index=True)
    games = games.sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)

    games = calculate_rolling_stats(games, window=5)

    matchup_df = create_matchup_features(games, team_names)

    # Ensure GAME_DATE is datetime.date for filtering
    matchup_df['GAME_DATE'] = pd.to_datetime(matchup_df['GAME_DATE']).dt.date
    today = datetime.utcnow().date()

    todays = matchup_df[matchup_df['GAME_DATE'] == today]
    if todays.empty:
        # If no games matched UTC date, also try local date
        local_today = datetime.now().date()
        todays = matchup_df[matchup_df['GAME_DATE'] == local_today]

    if todays.empty:
        print('ℹ️ No games found for today in fetched data.')
        print('Available date range: ', matchup_df['GAME_DATE'].min(), 'to', matchup_df['GAME_DATE'].max())
        return

    print(f'🔎 Found {len(todays)} games for today')

    # Prepare training set (exclude today's games)
    training = matchup_df[~matchup_df['GAME_DATE'].isin(todays['GAME_DATE'])].dropna(subset=['POINT_DIFF'])
    feature_cols = [c for c in matchup_df.columns if c.endswith('_ROLL') or 'WIN_STREAK' in c or 'REST_DAYS' in c or 'IS_BACK_TO_BACK' in c or 'WIN_RATE_10' in c]

    X_train = training[feature_cols].fillna(0)
    y_train = training['POINT_DIFF']

    print('📚 Training model on historical games:', X_train.shape)
    model = BayesianRidge()
    model.fit(X_train, y_train)

    X_today = todays[feature_cols].fillna(0)
    mu, std = model.predict(X_today, return_std=True)

    from scipy.stats import norm

    probs = 1 - norm.cdf(0, loc=mu, scale=std)

    results = todays[['GAME_ID', 'HOME_TEAM_NAME', 'AWAY_TEAM_NAME']].copy()
    results['PRED_POINT_DIFF'] = mu
    results['PRED_STD'] = std
    results['HOME_WIN_PROB'] = probs

    pd.set_option('display.float_format', '{:.3f}'.format)
    print('\n🏁 Predictions for today:')
    print(results.sort_values('HOME_WIN_PROB', ascending=False).to_string(index=False))


if __name__ == '__main__':
    main()
