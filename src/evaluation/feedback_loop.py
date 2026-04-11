"""SQLite feedback loop utilities for prediction logging, result updates, and evaluation."""

from __future__ import annotations

import json
from datetime import datetime
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from data.database.database_handler import SportsAnalyticsDB
from data.nba_loader import fetch_nba_games
from evaluators.prediction_logger import PredictionLogger
from src.evaluation.vectorized_features import compute_game_theory_matchup_features
from src.utils.timezone_utils import utc_to_pst_fields

# Replace this constant when you settle on a production feature name.
NEW_FEATURE_COLUMN = 'new_feature'
RETRAIN_EVERY_N_DEFAULT = 7
HIGH_SIGNAL_FEATURE_COLUMNS = [
    'elo_diff',
    'last5_win_pct_home',
    'last5_win_pct_away',
    'last5_point_diff_home',
    'last5_point_diff_away',
    'rest_days_home',
    'rest_days_away',
    'rest_diff',
    'is_back_to_back_home',
    'is_back_to_back_away',
    'home_away_strength_diff',
    'schedule_density_diff',
    'pace_diff',
    'injury_proxy',
    'expected_payoff_matrix',
    'optimal_path_delta',
    'signal_consistency_score',
]
BASELINE = {
    'accuracy': 0.5278,
    'brier': 0.2906,
    'mae': 13.40,
}

_TEAM_ALIASES: Dict[str, str] = {
    'LA Clippers': 'Los Angeles Clippers',
    'LA Lakers': 'Los Angeles Lakers',
    'L.A. Clippers': 'Los Angeles Clippers',
    'L.A. Lakers': 'Los Angeles Lakers',
}


def _normalize_team_name(name: str) -> str:
    cleaned = str(name or '').strip()
    return _TEAM_ALIASES.get(cleaned, cleaned)


def _to_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _coerce_feature_value(value: Any) -> Optional[str]:
    """Normalize a feature value into a stable string for SQLite grouping."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.6f}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, bool):
        return '1' if value else '0'
    text = str(value).strip()
    return text if text else None


def derive_new_feature_placeholder(
    prediction: Dict[str, Any],
    features: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Scaffold for feature derivation.

    Replace this logic with your real feature calculation.
    Current fallback order:
    1) prediction[new_feature]
    2) features[new_feature]
    3) a simple spread bucket derived from prediction spread
    """
    if NEW_FEATURE_COLUMN in prediction:
        return _coerce_feature_value(prediction.get(NEW_FEATURE_COLUMN))

    if features and NEW_FEATURE_COLUMN in features:
        return _coerce_feature_value(features.get(NEW_FEATURE_COLUMN))

    spread = prediction.get('spread')
    if spread is None:
        return None

    try:
        spread_val = float(spread)
    except Exception:
        return None

    if spread_val >= 8:
        return 'spread_bucket:heavy_favorite'
    if spread_val >= 3:
        return 'spread_bucket:moderate_favorite'
    if spread_val > -3:
        return 'spread_bucket:coin_flip'
    return 'spread_bucket:underdog'


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _group_by_game_id(history: pd.DataFrame) -> pd.DataFrame:
    required = {'GAME_ID', 'TEAM_ID', 'GAME_DATE', 'PTS'}
    if history is None or history.empty or not required.issubset(set(history.columns)):
        return pd.DataFrame(columns=['GAME_ID', 'home_team_id', 'away_team_id', 'home_pts', 'away_pts', 'GAME_DATE'])

    rows: List[Dict[str, Any]] = []
    hist = history.copy()
    hist['GAME_DATE'] = pd.to_datetime(hist['GAME_DATE'], errors='coerce')
    hist['PTS'] = pd.to_numeric(hist['PTS'], errors='coerce')
    hist = hist.dropna(subset=['GAME_ID', 'TEAM_ID', 'GAME_DATE', 'PTS'])
    if hist.empty:
        return pd.DataFrame(columns=['GAME_ID', 'home_team_id', 'away_team_id', 'home_pts', 'away_pts', 'GAME_DATE'])

    for _, grp in hist.groupby('GAME_ID', sort=False):
        if len(grp) < 2:
            continue
        home_mask = grp['MATCHUP'].astype(str).str.contains(r'vs\.', na=False) if 'MATCHUP' in grp.columns else pd.Series([False] * len(grp), index=grp.index)
        if int(home_mask.sum()) == 1:
            home = grp[home_mask].iloc[0]
            away = grp[~home_mask].iloc[0]
        else:
            sorted_grp = grp.sort_values('TEAM_ID')
            home = sorted_grp.iloc[0]
            away = sorted_grp.iloc[1]

        rows.append(
            {
                'GAME_ID': str(home['GAME_ID']),
                'home_team_id': _safe_int(home['TEAM_ID']),
                'away_team_id': _safe_int(away['TEAM_ID']),
                'home_pts': _safe_float(home['PTS']),
                'away_pts': _safe_float(away['PTS']),
                'GAME_DATE': home['GAME_DATE'],
            }
        )

    if not rows:
        return pd.DataFrame(columns=['GAME_ID', 'home_team_id', 'away_team_id', 'home_pts', 'away_pts', 'GAME_DATE'])

    out = pd.DataFrame(rows).drop_duplicates(subset=['GAME_ID']).sort_values('GAME_DATE').reset_index(drop=True)
    return out


def _games_before_date(team_history_df: pd.DataFrame, prediction_date: pd.Timestamp) -> pd.DataFrame:
    if team_history_df is None or team_history_df.empty:
        return pd.DataFrame()
    hist = team_history_df.copy()
    hist['GAME_DATE'] = pd.to_datetime(hist['GAME_DATE'], errors='coerce')
    if 'PTS' not in hist.columns:
        return pd.DataFrame()
    hist['PTS'] = pd.to_numeric(hist['PTS'], errors='coerce')
    hist = hist.dropna(subset=['GAME_DATE', 'PTS'])
    hist = hist[hist['GAME_DATE'] < prediction_date]
    return hist.sort_values('GAME_DATE').reset_index(drop=True)


def _team_recent_rows(hist: pd.DataFrame, team_id: Optional[int], team_name: Optional[str]) -> pd.DataFrame:
    if hist.empty:
        return hist

    team_rows = pd.DataFrame()
    if team_id is not None and 'TEAM_ID' in hist.columns:
        team_rows = hist[pd.to_numeric(hist['TEAM_ID'], errors='coerce') == int(team_id)]

    if team_rows.empty and team_name and 'TEAM_NAME' in hist.columns:
        norm_name = _normalize_team_name(team_name)
        team_rows = hist[hist['TEAM_NAME'].astype(str).map(_normalize_team_name) == norm_name]

    return team_rows.sort_values('GAME_DATE').reset_index(drop=True)


def _compute_elo_diff(game_rows: pd.DataFrame, home_team_id: int, away_team_id: int, k_factor: float = 20.0) -> float:
    if game_rows.empty:
        return 0.0

    elo: Dict[int, float] = {}
    for _, row in game_rows.iterrows():
        home_id = int(row['home_team_id'])
        away_id = int(row['away_team_id'])
        home_elo = elo.get(home_id, 1500.0)
        away_elo = elo.get(away_id, 1500.0)

        expected_home = 1.0 / (1.0 + 10.0 ** ((away_elo - home_elo) / 400.0))
        expected_away = 1.0 - expected_home

        if float(row['home_pts']) > float(row['away_pts']):
            result_home, result_away = 1.0, 0.0
        elif float(row['home_pts']) < float(row['away_pts']):
            result_home, result_away = 0.0, 1.0
        else:
            result_home, result_away = 0.5, 0.5

        elo[home_id] = home_elo + k_factor * (result_home - expected_home)
        elo[away_id] = away_elo + k_factor * (result_away - expected_away)

    return float(elo.get(home_team_id, 1500.0) - elo.get(away_team_id, 1500.0))


def derive_game_features(game: Dict[str, Any], team_history_df: pd.DataFrame) -> Dict[str, float]:
    """Derive deterministic high-signal features from settled historical games only.

    Rules enforced:
    - only settled games (scores known)
    - only historical rows strictly before target game date
    - numeric null-safe outputs with deterministic defaults
    """
    game_date_raw = game.get('game_date') or game.get('game_date_utc')
    game_date = pd.to_datetime(str(game_date_raw) if game_date_raw is not None else '', errors='coerce')
    if pd.isna(game_date):
        game_date = pd.Timestamp.utcnow().tz_localize(None)
    else:
        game_date = game_date.tz_localize(None) if getattr(game_date, 'tzinfo', None) else game_date

    home_team_id = game.get('home_team_id')
    away_team_id = game.get('away_team_id')
    home_team_name = game.get('home_team')
    away_team_name = game.get('away_team')

    settled_hist = _games_before_date(team_history_df, game_date)
    if settled_hist.empty:
        return {
            'elo_diff': 0.0,
            'last5_win_pct_home': 0.5,
            'last5_win_pct_away': 0.5,
            'last5_point_diff_home': 0.0,
            'last5_point_diff_away': 0.0,
            'rest_days_home': 2.0,
            'rest_days_away': 2.0,
            'rest_diff': 0.0,
            'is_back_to_back_home': 0.0,
            'is_back_to_back_away': 0.0,
            'home_away_strength_diff': 0.0,
            'schedule_density_diff': 0.0,
            'pace_diff': 0.0,
            'injury_proxy': 0.0,
            'expected_payoff_matrix': 0.0,
            'optimal_path_delta': 0.0,
            'signal_consistency_score': 0.5,
        }

    home_team_id_int = _safe_int(home_team_id, default=-1)
    away_team_id_int = _safe_int(away_team_id, default=-2)
    game_rows = _group_by_game_id(settled_hist)
    elo_diff = _compute_elo_diff(game_rows, home_team_id_int, away_team_id_int, k_factor=20.0)

    home_rows = _team_recent_rows(settled_hist, home_team_id if home_team_id is not None else None, home_team_name)
    away_rows = _team_recent_rows(settled_hist, away_team_id if away_team_id is not None else None, away_team_name)

    def _last5_metrics(team_rows: pd.DataFrame) -> tuple[float, float]:
        if team_rows.empty:
            return 0.5, 0.0
        recent = team_rows.tail(5).copy()
        if 'WL' in recent.columns:
            win_pct = float((recent['WL'].astype(str) == 'W').mean()) if len(recent) else 0.5
        else:
            win_pct = 0.5

        point_diff = 0.0
        if 'MATCHUP' in recent.columns and 'PTS' in recent.columns:
            # Team-centric point differential proxy from available game logs.
            point_diff = float(pd.to_numeric(recent['PTS'], errors='coerce').fillna(0.0).mean() - pd.to_numeric(recent['PTS'], errors='coerce').fillna(0.0).median())
        return win_pct, point_diff

    last5_win_pct_home, last5_point_diff_home = _last5_metrics(home_rows)
    last5_win_pct_away, last5_point_diff_away = _last5_metrics(away_rows)

    def _rest_days(team_rows: pd.DataFrame) -> float:
        if team_rows.empty:
            return 2.0
        last_date = pd.to_datetime(team_rows.iloc[-1]['GAME_DATE'], errors='coerce')
        if pd.isna(last_date):
            return 2.0
        delta_days = float((game_date.normalize() - last_date.normalize()).days)
        return max(delta_days - 1.0, 0.0)

    rest_days_home = _rest_days(home_rows)
    rest_days_away = _rest_days(away_rows)
    rest_diff = rest_days_home - rest_days_away
    is_back_to_back_home = 1.0 if rest_days_home == 0.0 else 0.0
    is_back_to_back_away = 1.0 if rest_days_away == 0.0 else 0.0

    def _home_away_strength(team_rows: pd.DataFrame, is_home_split: bool) -> float:
        if team_rows.empty:
            return 0.0
        if 'MATCHUP' not in team_rows.columns or 'WL' not in team_rows.columns:
            return 0.0
        if is_home_split:
            split = team_rows[team_rows['MATCHUP'].astype(str).str.contains(r'vs\.', na=False)]
        else:
            split = team_rows[team_rows['MATCHUP'].astype(str).str.contains('@', na=False)]
        if split.empty:
            return 0.0
        return float((split['WL'].astype(str) == 'W').mean())

    home_strength = _home_away_strength(home_rows, is_home_split=True)
    away_strength = _home_away_strength(away_rows, is_home_split=False)
    home_away_strength_diff = home_strength - away_strength

    lookback_days = 7
    home_density = 0.0
    away_density = 0.0
    if not home_rows.empty:
        home_window_start = game_date - pd.Timedelta(days=lookback_days)
        home_density = float(home_rows[home_rows['GAME_DATE'] >= home_window_start]['GAME_ID'].nunique())
    if not away_rows.empty:
        away_window_start = game_date - pd.Timedelta(days=lookback_days)
        away_density = float(away_rows[away_rows['GAME_DATE'] >= away_window_start]['GAME_ID'].nunique())
    schedule_density_diff = home_density - away_density

    def _pace_proxy(team_rows: pd.DataFrame) -> float:
        if team_rows.empty:
            return 0.0
        if 'PTS' not in team_rows.columns:
            return 0.0
        pts = pd.to_numeric(team_rows.tail(5)['PTS'], errors='coerce').dropna()
        if pts.empty:
            return 0.0
        return float(pts.mean())

    pace_diff = _pace_proxy(home_rows) - _pace_proxy(away_rows)
    pace_home = _pace_proxy(home_rows)
    pace_away = _pace_proxy(away_rows)

    def _off_def_rating(team_rows: pd.DataFrame, team_id_int: int) -> tuple[float, float]:
        if team_rows.empty:
            return 100.0, 100.0

        off = float(pd.to_numeric(team_rows.tail(10)['PTS'], errors='coerce').dropna().mean())
        if not np.isfinite(off):
            off = 100.0

        if game_rows.empty:
            return off, off

        team_games = game_rows[
            (pd.to_numeric(game_rows['home_team_id'], errors='coerce') == float(team_id_int))
            | (pd.to_numeric(game_rows['away_team_id'], errors='coerce') == float(team_id_int))
        ].copy()
        if team_games.empty:
            return off, off

        allowed = np.where(
            pd.to_numeric(team_games['home_team_id'], errors='coerce') == float(team_id_int),
            pd.to_numeric(team_games['away_pts'], errors='coerce'),
            pd.to_numeric(team_games['home_pts'], errors='coerce'),
        )
        allowed = pd.Series(allowed).dropna()
        if allowed.empty:
            return off, off

        deff = float(allowed.tail(10).mean())
        if not np.isfinite(deff):
            deff = off
        return off, deff

    home_off, home_def = _off_def_rating(home_rows, home_team_id_int)
    away_off, away_def = _off_def_rating(away_rows, away_team_id_int)

    gt = compute_game_theory_matchup_features(
        home_off_rating=home_off,
        home_def_rating=home_def,
        away_off_rating=away_off,
        away_def_rating=away_def,
        rest_days_home=rest_days_home,
        rest_days_away=rest_days_away,
        schedule_density_home=home_density,
        schedule_density_away=away_density,
        is_back_to_back_home=is_back_to_back_home,
        is_back_to_back_away=is_back_to_back_away,
        pace_home=pace_home,
        pace_away=pace_away,
    )

    # TODO: replace with real injury/availability integration once roster signal is available.
    injury_proxy = 0.0

    return {
        'elo_diff': float(elo_diff),
        'last5_win_pct_home': float(last5_win_pct_home),
        'last5_win_pct_away': float(last5_win_pct_away),
        'last5_point_diff_home': float(last5_point_diff_home),
        'last5_point_diff_away': float(last5_point_diff_away),
        'rest_days_home': float(rest_days_home),
        'rest_days_away': float(rest_days_away),
        'rest_diff': float(rest_diff),
        'is_back_to_back_home': float(is_back_to_back_home),
        'is_back_to_back_away': float(is_back_to_back_away),
        'home_away_strength_diff': float(home_away_strength_diff),
        'schedule_density_diff': float(schedule_density_diff),
        'pace_diff': float(pace_diff),
        'injury_proxy': float(injury_proxy),
        'expected_payoff_matrix': float(gt.get('expected_payoff_matrix', 0.0)),
        'optimal_path_delta': float(gt.get('optimal_path_delta', 0.0)),
        'signal_consistency_score': float(gt.get('signal_consistency_score', 0.5)),
    }


def _season_from_date(dt: datetime) -> str:
    year = dt.year
    month = dt.month
    start_year = year if month >= 7 else (year - 1)
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _local_date_key(row: Dict[str, Any]) -> str:
    """Best-effort YYYY-MM-DD key for a prediction row."""
    raw = row.get('game_date_local_date') or row.get('game_date') or ''
    return str(raw)[:10]


def _build_result_data(
    prediction_row: Dict[str, Any],
    actual_home_score: int,
    actual_away_score: int,
    actual_winner: str,
) -> Dict[str, Any]:
    """Build normalized result payload for prediction_results rows."""
    pred_spread = float(prediction_row.get('predicted_spread') or 0.0)
    actual_spread = float(actual_home_score - actual_away_score)

    ci_lower_raw = prediction_row.get('ci_lower')
    ci_upper_raw = prediction_row.get('ci_upper')
    ci_lower = float(ci_lower_raw) if ci_lower_raw is not None else (pred_spread - 6.0)
    ci_upper = float(ci_upper_raw) if ci_upper_raw is not None else (pred_spread + 6.0)

    return {
        'actual_home_score': int(actual_home_score),
        'actual_away_score': int(actual_away_score),
        'actual_spread': actual_spread,
        'actual_winner': str(actual_winner),
        'prediction_error': abs(pred_spread - actual_spread),
        'correct_winner': (pred_spread >= 0) == (actual_spread >= 0),
        'within_ci': ci_lower <= actual_spread <= ci_upper,
    }


def _upsert_prediction_result_row(
    db: SportsAnalyticsDB,
    prediction_id: int,
    result_data: Dict[str, Any],
) -> None:
    """Insert or update a normalized prediction_results row."""
    existing = db.get_prediction_with_result(prediction_id)
    if not existing or existing.get('result_id') is None:
        db.insert_result(prediction_id, result_data)
        return

    if db.conn is None:
        raise RuntimeError('Database connection is not available')

    cur = db.conn.cursor()
    cur.execute(
        """
        UPDATE prediction_results
        SET actual_home_score = ?,
            actual_away_score = ?,
            actual_spread = ?,
            actual_winner = ?,
            prediction_error = ?,
            correct_winner = ?,
            within_ci = ?,
            result_timestamp = ?
        WHERE result_id = ?
        """,
        (
            int(result_data['actual_home_score']),
            int(result_data['actual_away_score']),
            float(result_data['actual_spread']),
            str(result_data['actual_winner']),
            float(result_data['prediction_error']),
            int(bool(result_data['correct_winner'])),
            int(bool(result_data['within_ci'])),
            datetime.now().isoformat(),
            int(existing['result_id']),
        ),
    )
    db.conn.commit()


def _format_display_time_military(value: Any) -> str:
    """Format datetime-ish values as 24-hour HH:MM for notebook display."""
    text = str(value or '').strip()
    if not text:
        return ''

    # Preserve date-only values if there is no time component in the source.
    if len(text) == 10 and text.count('-') == 2:
        return text

    try:
        parsed = pd.to_datetime(text, errors='coerce')
        if pd.isna(parsed):
            return text
        return parsed.strftime('%H:%M')
    except Exception:
        return text


def _build_result_map_from_logs(games_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Build {game_id: result_dict} from LeagueGameFinder two-row game logs."""
    if games_df is None or games_df.empty:
        return {}

    df = games_df.copy()
    if 'GAME_DATE' in df.columns:
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')

    out: Dict[str, Dict[str, Any]] = {}

    for game_id, g in df.groupby('GAME_ID'):
        home_rows = g[g['MATCHUP'].astype(str).str.contains('vs\\.', na=False)]
        away_rows = g[g['MATCHUP'].astype(str).str.contains('@', na=False)]

        if home_rows.empty or away_rows.empty:
            continue

        home_row = home_rows.iloc[0]
        away_row = away_rows.iloc[0]

        home_score = int(home_row.get('PTS', 0) or 0)
        away_score = int(away_row.get('PTS', 0) or 0)

        home_team = _normalize_team_name(str(home_row.get('TEAM_NAME', '')))
        away_team = _normalize_team_name(str(away_row.get('TEAM_NAME', '')))
        if not home_team or not away_team:
            continue

        if home_score > away_score:
            actual_winner = home_team
        elif away_score > home_score:
            actual_winner = away_team
        else:
            actual_winner = ''

        game_date = home_row.get('GAME_DATE')
        game_date_iso = pd.to_datetime(game_date).date().isoformat() if pd.notna(game_date) else ''

        out[str(game_id)] = {
            'game_id': str(game_id),
            'game_date': game_date_iso,
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'actual_winner': actual_winner,
            'actual_spread': float(home_score - away_score),
        }

    return out


class PredictionFeedbackManager:
    """Wrap prediction logging with timezone-aware fields and feedback loop helpers."""

    def __init__(
        self,
        db: SportsAnalyticsDB,
        model_version: Optional[str] = None,
    ):
        self._db = db
        self._logger = PredictionLogger(db, model_version=model_version)

    def log_prediction(
        self,
        prediction: Any,
        features: Optional[Any] = None,
        game_date_utc: Optional[str] = None,
        new_feature: Optional[Any] = None,
    ) -> int:
        """Log prediction and stamp UTC/PST date fields on the predictions row."""
        pred = dict(prediction or {})
        feature_dict = dict(features or {}) if isinstance(features, dict) else None

        feature_value = _coerce_feature_value(new_feature)
        if feature_value is None:
            feature_value = derive_new_feature_placeholder(prediction=pred, features=feature_dict)
        pred[NEW_FEATURE_COLUMN] = feature_value

        utc_value = game_date_utc or pred.get('game_date_utc') or pred.get('game_date')
        if utc_value:
            try:
                game_date_pst, game_date_local_date = utc_to_pst_fields(str(utc_value))
            except Exception:
                game_date_pst = None
                game_date_local_date = pred.get('game_date')
        else:
            game_date_pst = None
            game_date_local_date = pred.get('game_date')

        if game_date_local_date and not pred.get('game_date'):
            pred['game_date'] = game_date_local_date

        pred_id = self._logger.log_prediction(
            prediction=_to_native(pred),
            features=_to_native(features) if features is not None else None,
        )

        if self._db.conn is None:
            raise RuntimeError('Database connection is not available')

        cur = self._db.conn.cursor()
        cur.execute(
            """
            UPDATE predictions
            SET game_date_utc = ?,
                game_date_pst = ?,
                game_date_local_date = ?,
                new_feature = ?
            WHERE prediction_id = ?
            """,
            (
                str(utc_value) if utc_value else None,
                game_date_pst,
                game_date_local_date,
                feature_value,
                pred_id,
            ),
        )
        self._db.conn.commit()

        return int(pred_id)


def fetch_and_update_results(
    db_path: str = 'sports_analytics.db',
    days: int = 14,
    seasons: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Backfill actual results for recent unsettled predictions.

    Resolution order:
    1) games_cache by game_id / matchup+date
    2) nba_api LeagueGameFinder game logs (via fetch_nba_games)

    Returns both legacy counters and a richer diagnostics block with:
    - per-source resolved counts
    - per-reason unresolved counts
    """
    with SportsAnalyticsDB(db_path) as db:
        unsettled = db.get_unsettled_predictions(days=days)
        if not unsettled:
            return {
                'updated': 0,
                'skipped': 0,
                'pending_or_future': 0,
                'unresolved_past': 0,
                'resolved_from_cache': 0,
                'resolved_from_api_game_id': 0,
                'resolved_from_api_matchup': 0,
                'unresolved_reason_counts': {},
                'unresolved_examples': [],
                'errors': [],
                'source': 'none',
                'diagnostics': {
                    'inputs': {
                        'total_unsettled': 0,
                        'cache_game_ids': 0,
                        'api_game_ids': 0,
                        'api_matchup_keys': 0,
                    },
                    'resolved': {
                        'cache': 0,
                        'api_game_id': 0,
                        'api_matchup': 0,
                    },
                    'unresolved_reasons': {},
                },
            }

        # Build season list dynamically from pending prediction dates if not provided.
        if seasons is None:
            season_set = set()
            for row in unsettled:
                date_val = row.get('game_date_utc') or row.get('game_date_local_date') or row.get('game_date')
                if not date_val:
                    continue
                try:
                    dt = pd.to_datetime(date_val, utc=True, errors='coerce')
                    if pd.isna(dt):
                        continue
                    season_set.add(_season_from_date(dt.to_pydatetime()))
                except Exception:
                    continue
            seasons = sorted(season_set) or ['2024-25', '2025-26']

        if verbose:
            print(f"Checking unsettled predictions: {len(unsettled)}")
            print(f"Result lookup seasons: {seasons}")

        results_map: Dict[str, Dict[str, Any]] = {}
        try:
            games_df = fetch_nba_games(seasons=seasons, season_type='Regular Season', verbose=False)
            results_map = _build_result_map_from_logs(games_df)
        except Exception as exc:
            if verbose:
                print(f"Warning: API fallback fetch failed: {exc}")

        cache_rows = db.get_cached_games()
        cache_by_game_id: Dict[str, Dict[str, Any]] = {}
        for c in cache_rows:
            cache_game_id = str(c.get('game_id') or '')
            if cache_game_id and cache_game_id not in cache_by_game_id:
                cache_by_game_id[cache_game_id] = c

        api_matchup_index: Dict[tuple[str, str, str], Dict[str, Any]] = {}
        for m in results_map.values():
            date_key = str(m.get('game_date') or '')[:10]
            if not date_key:
                continue
            away_key = _normalize_team_name(m.get('away_team', ''))
            home_key = _normalize_team_name(m.get('home_team', ''))
            api_matchup_index[(date_key, away_key, home_key)] = m

        if verbose:
            print(
                "Lookup inventory:",
                {
                    'cache_game_ids': len(cache_by_game_id),
                    'api_game_ids': len(results_map),
                    'api_matchup_keys': len(api_matchup_index),
                },
            )

        updated = 0
        skipped = 0
        pending_or_future = 0
        unresolved_past = 0
        resolved_from_cache = 0
        resolved_from_api_game_id = 0
        resolved_from_api_matchup = 0
        unresolved_reason_counts: Dict[str, int] = {}
        unresolved_examples: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        today_pst = pd.Timestamp.now(tz='America/Los_Angeles').date().isoformat()

        for row in unsettled:
            try:
                prediction_id = int(row['prediction_id'])
                game_id = str(row.get('game_id') or '')
                home_team = _normalize_team_name(row.get('home_team', ''))
                away_team = _normalize_team_name(row.get('away_team', ''))
                local_date = _local_date_key(row)
                resolution_source = None
                unresolved_reason = None

                resolved = None

                # 1) Cache lookup by game_id.
                if game_id:
                    c = cache_by_game_id.get(game_id)
                    if c is not None:
                        if c.get('home_score') is not None and c.get('away_score') is not None:
                            c_home = _normalize_team_name(str(c.get('home_team', '')))
                            c_away = _normalize_team_name(str(c.get('away_team', '')))
                            h_score = int(c.get('home_score') or 0)
                            a_score = int(c.get('away_score') or 0)
                            winner = c_home if h_score > a_score else c_away
                            resolved = {
                                'actual_home_score': h_score,
                                'actual_away_score': a_score,
                                'actual_spread': float(h_score - a_score),
                                'actual_winner': winner,
                            }
                            resolution_source = 'cache'
                        else:
                            unresolved_reason = 'cache_missing_final_score'
                    else:
                        unresolved_reason = 'cache_game_id_not_found'

                # 2) API map lookup by game_id.
                if resolved is None and game_id and game_id in results_map:
                    m = results_map[game_id]
                    resolved = {
                        'actual_home_score': int(m['home_score']),
                        'actual_away_score': int(m['away_score']),
                        'actual_spread': float(m['actual_spread']),
                        'actual_winner': m['actual_winner'],
                    }
                    resolution_source = 'api_game_id'
                elif resolved is None and game_id and unresolved_reason is None:
                    unresolved_reason = 'api_game_id_not_found'

                # 3) API map lookup by date + matchup if game_id missing.
                if resolved is None:
                    m = api_matchup_index.get((local_date[:10], away_team, home_team))
                    if m is not None:
                        resolved = {
                            'actual_home_score': int(m['home_score']),
                            'actual_away_score': int(m['away_score']),
                            'actual_spread': float(m['actual_spread']),
                            'actual_winner': m['actual_winner'],
                        }
                        resolution_source = 'api_matchup'
                    else:
                        if not game_id:
                            unresolved_reason = 'missing_game_id_and_matchup_not_found'
                        elif unresolved_reason is None:
                            unresolved_reason = 'api_matchup_not_found'

                if resolved is None:
                    skipped += 1
                    is_pending_or_future = bool(local_date and local_date >= today_pst)
                    if is_pending_or_future:
                        pending_or_future += 1
                    else:
                        unresolved_past += 1

                    reason_key = unresolved_reason or 'unknown_unresolved_reason'
                    unresolved_reason_counts[reason_key] = unresolved_reason_counts.get(reason_key, 0) + 1

                    if len(unresolved_examples) < 8:
                        unresolved_examples.append(
                            {
                                'prediction_id': prediction_id,
                                'game_id': game_id,
                                'local_date': local_date,
                                'matchup': f"{away_team} @ {home_team}",
                                'reason': reason_key,
                                'pending_or_future': is_pending_or_future,
                            }
                        )
                    continue

                if resolution_source == 'cache':
                    resolved_from_cache += 1
                elif resolution_source == 'api_game_id':
                    resolved_from_api_game_id += 1
                elif resolution_source == 'api_matchup':
                    resolved_from_api_matchup += 1

                db.update_prediction_actual(
                    prediction_id=prediction_id,
                    actual_winner=resolved['actual_winner'],
                    home_score=resolved['actual_home_score'],
                    away_score=resolved['actual_away_score'],
                )

                result_data = _build_result_data(
                    prediction_row=row,
                    actual_home_score=int(resolved['actual_home_score']),
                    actual_away_score=int(resolved['actual_away_score']),
                    actual_winner=str(resolved['actual_winner']),
                )
                _upsert_prediction_result_row(db, prediction_id=prediction_id, result_data=result_data)

                updated += 1
            except Exception as exc:
                errors.append({'prediction_id': row.get('prediction_id'), 'error': str(exc)})

        return {
            'updated': updated,
            'skipped': skipped,
            'pending_or_future': pending_or_future,
            'unresolved_past': unresolved_past,
            'resolved_from_cache': resolved_from_cache,
            'resolved_from_api_game_id': resolved_from_api_game_id,
            'resolved_from_api_matchup': resolved_from_api_matchup,
            'unresolved_reason_counts': unresolved_reason_counts,
            'unresolved_examples': unresolved_examples,
            'errors': errors,
            'source': 'games_cache + nba_api',
            'diagnostics': {
                'inputs': {
                    'total_unsettled': len(unsettled),
                    'cache_game_ids': len(cache_by_game_id),
                    'api_game_ids': len(results_map),
                    'api_matchup_keys': len(api_matchup_index),
                },
                'resolved': {
                    'cache': resolved_from_cache,
                    'api_game_id': resolved_from_api_game_id,
                    'api_matchup': resolved_from_api_matchup,
                },
                'unresolved_reasons': unresolved_reason_counts,
            },
        }


def apply_manual_result(
    db_path: str = 'sports_analytics.db',
    *,
    prediction_id: Optional[int] = None,
    game_id: Optional[str] = None,
    home_score: int,
    away_score: int,
    actual_winner: Optional[str] = None,
) -> Dict[str, Any]:
    """Manually backfill one prediction outcome by prediction_id or game_id."""
    if prediction_id is None and not game_id:
        return {'status': 'error', 'reason': 'Provide prediction_id or game_id'}

    with SportsAnalyticsDB(db_path) as db:
        if db.conn is None:
            raise RuntimeError('Database connection is not available')

        cur = db.conn.cursor()
        if prediction_id is not None:
            cur.execute("SELECT * FROM predictions WHERE prediction_id = ?", (int(prediction_id),))
        else:
            cur.execute(
                """
                SELECT *
                FROM predictions
                WHERE game_id = ?
                ORDER BY prediction_id DESC
                LIMIT 1
                """,
                (str(game_id),),
            )

        row = cur.fetchone()
        if row is None:
            return {'status': 'error', 'reason': 'Prediction not found', 'prediction_id': prediction_id, 'game_id': game_id}

        prediction = dict(row)
        resolved_prediction_id = int(prediction['prediction_id'])

        h_score = int(home_score)
        a_score = int(away_score)
        if actual_winner is None:
            if h_score > a_score:
                winner = str(prediction.get('home_team') or '')
            elif a_score > h_score:
                winner = str(prediction.get('away_team') or '')
            else:
                winner = ''
        else:
            winner = str(actual_winner)

        db.update_prediction_actual(
            prediction_id=resolved_prediction_id,
            actual_winner=winner,
            home_score=h_score,
            away_score=a_score,
        )

        result_data = _build_result_data(
            prediction_row=prediction,
            actual_home_score=h_score,
            actual_away_score=a_score,
            actual_winner=winner,
        )
        _upsert_prediction_result_row(db, prediction_id=resolved_prediction_id, result_data=result_data)

        return {
            'status': 'updated',
            'prediction_id': resolved_prediction_id,
            'game_id': prediction.get('game_id'),
            'matchup': f"{prediction.get('away_team')} @ {prediction.get('home_team')}",
            'actual_winner': winner,
            'home_score': h_score,
            'away_score': a_score,
            'correct': int(prediction.get('predicted_winner') == winner),
        }


def get_training_data_for_calibration(db: SportsAnalyticsDB, limit: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Return recent probability/outcome pairs for calibration fitting."""
    if db.conn is None:
        return np.array([], dtype=float), np.array([], dtype=float)

    query = """
        SELECT
            win_probability,
            COALESCE(correct, CASE WHEN predicted_winner = actual_winner THEN 1 ELSE 0 END) AS correct_flag
        FROM predictions
        WHERE win_probability IS NOT NULL
          AND actual_winner IS NOT NULL
        ORDER BY prediction_timestamp DESC
        LIMIT ?
    """
    df = pd.read_sql_query(query, db.conn, params=(int(limit),))
    if df.empty:
        return np.array([], dtype=float), np.array([], dtype=float)

    df = df.dropna(subset=['win_probability', 'correct_flag'])
    if df.empty:
        return np.array([], dtype=float), np.array([], dtype=float)

    return (
        pd.to_numeric(df['win_probability'], errors='coerce').dropna().to_numpy(dtype=float),
        pd.to_numeric(df['correct_flag'], errors='coerce').dropna().to_numpy(dtype=float),
    )


def evaluate_recent_predictions(db_path: str = 'sports_analytics.db', n: int = 100) -> Dict[str, Any]:
    """Evaluate recent settled predictions with calibration details."""
    with SportsAnalyticsDB(db_path) as db:
        if db.conn is None:
            raise RuntimeError('Database connection is not available')

        query = """
            SELECT
                prediction_id,
                home_team,
                away_team,
                predicted_winner,
                win_probability,
                win_probability_calibrated,
                confidence_level,
                new_feature,
                elo_diff,
                rest_diff,
                last5_win_pct_home,
                last5_win_pct_away,
                last5_point_diff_home,
                last5_point_diff_away,
                predicted_spread,
                actual_winner,
                home_score,
                away_score,
                correct,
                COALESCE(game_date_local_date, game_date) AS game_date_local_date
            FROM predictions
            WHERE actual_winner IS NOT NULL
              AND home_score IS NOT NULL
              AND away_score IS NOT NULL
            ORDER BY COALESCE(game_date_local_date, game_date) DESC, prediction_id DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, db.conn, params=(int(n),))

    if df.empty:
        return {
            'n_total': 0,
            'accuracy': 0.0,
            'brier_score': None,
            'mae_spread': None,
            'by_confidence': {},
            'calibration': [],
        }

    df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
    df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')
    df['predicted_spread'] = pd.to_numeric(df['predicted_spread'], errors='coerce').fillna(0.0)
    df['win_probability'] = pd.to_numeric(df['win_probability'], errors='coerce').fillna(0.5).clip(1e-6, 1 - 1e-6)
    df['win_probability_calibrated'] = pd.to_numeric(df['win_probability_calibrated'], errors='coerce')
    eval_prob = df['win_probability_calibrated'].fillna(df['win_probability']).clip(1e-6, 1 - 1e-6)

    # Accuracy
    if 'correct' in df.columns and df['correct'].notna().any():
        accuracy = float(df['correct'].fillna(0).astype(int).mean())
    else:
        accuracy = float((df['predicted_winner'] == df['actual_winner']).mean())

    # Brier score on home-win probability.
    actual_home_win = (df['home_score'] > df['away_score']).astype(float)
    probs_raw = df['win_probability']
    probs_calibrated = eval_prob
    brier_raw = float(np.mean((probs_raw - actual_home_win) ** 2))
    brier_calibrated = float(np.mean((probs_calibrated - actual_home_win) ** 2))

    # MAE on spread if score data exists.
    actual_spread = df['home_score'] - df['away_score']
    mae_spread = float(np.mean(np.abs(df['predicted_spread'] - actual_spread)))

    by_confidence: Dict[str, Dict[str, Any]] = {}
    for level in ('HIGH', 'MEDIUM', 'LOW'):
        sub = df[df['confidence_level'] == level]
        if sub.empty:
            continue
        sub_acc = float((sub['predicted_winner'] == sub['actual_winner']).mean())
        sub_eval_prob = sub['win_probability_calibrated'].fillna(sub['win_probability']).clip(1e-6, 1 - 1e-6)
        pred_prob = np.where(
            sub['predicted_winner'] == sub['home_team'],
            sub_eval_prob,
            1.0 - sub_eval_prob,
        )
        by_confidence[level] = {
            'count': int(len(sub)),
            'accuracy': sub_acc,
            'avg_predicted_probability': float(np.mean(pred_prob)),
        }

    # Calibration buckets on home-win probabilities.
    bins = [i / 10.0 for i in range(11)]
    bucket = pd.cut(eval_prob, bins=bins, include_lowest=True)
    cal_df = (
        df.assign(actual_home_win=actual_home_win, bucket=bucket, eval_prob=eval_prob)
        .groupby('bucket', observed=False)
        .agg(
            count=('prediction_id', 'size'),
            avg_predicted_probability=('eval_prob', 'mean'),
            actual_home_win_rate=('actual_home_win', 'mean'),
        )
        .reset_index()
    )
    cal_df = cal_df[cal_df['count'] > 0]
    calibration = [
        {
            'bucket': str(r['bucket']),
            'count': int(r['count']),
            'avg_predicted_probability': float(r['avg_predicted_probability']),
            'actual_home_win_rate': float(r['actual_home_win_rate']),
        }
        for _, r in cal_df.iterrows()
    ]

    # Feature-aware breakdown for MLflow/notebook analytics.
    by_new_feature: Dict[str, Dict[str, Any]] = {}
    feature_df = df[df['new_feature'].notna() & (df['new_feature'].astype(str).str.len() > 0)].copy()
    if not feature_df.empty:
        feature_df['actual_home_win'] = (feature_df['home_score'] > feature_df['away_score']).astype(float)
        feature_df['actual_spread'] = feature_df['home_score'] - feature_df['away_score']
        feature_df['eval_prob'] = feature_df['win_probability_calibrated'].fillna(feature_df['win_probability']).clip(1e-6, 1 - 1e-6)

        for feature_value, sub in feature_df.groupby('new_feature'):
            sub_acc = float((sub['predicted_winner'] == sub['actual_winner']).mean())
            sub_brier_raw = float(np.mean((sub['win_probability'].clip(1e-6, 1 - 1e-6) - sub['actual_home_win']) ** 2))
            sub_brier_calibrated = float(np.mean((sub['eval_prob'] - sub['actual_home_win']) ** 2))
            sub_mae = float(np.mean(np.abs(sub['predicted_spread'] - sub['actual_spread'])))

            sub_bins = [i / 5.0 for i in range(6)]
            sub_bucket = pd.cut(sub['win_probability'], bins=sub_bins, include_lowest=True)
            sub_cal_df = (
                sub.assign(bucket=sub_bucket)
                .groupby('bucket', observed=False)
                .agg(
                    count=('prediction_id', 'size'),
                    avg_predicted_probability=('eval_prob', 'mean'),
                    actual_home_win_rate=('actual_home_win', 'mean'),
                )
                .reset_index()
            )
            sub_cal_df = sub_cal_df[sub_cal_df['count'] > 0]
            sub_calibration = [
                {
                    'bucket': str(r['bucket']),
                    'count': int(r['count']),
                    'avg_predicted_probability': float(r['avg_predicted_probability']),
                    'actual_home_win_rate': float(r['actual_home_win_rate']),
                }
                for _, r in sub_cal_df.iterrows()
            ]

            # Simple rolling accuracy scaffold over the latest 20 rows for this feature.
            rolling_series = (
                (sub['predicted_winner'] == sub['actual_winner'])
                .astype(float)
                .rolling(window=min(20, len(sub)), min_periods=1)
                .mean()
            )
            by_new_feature[str(feature_value)] = {
                'count': int(len(sub)),
                'accuracy': sub_acc,
                'brier_raw': sub_brier_raw,
                'brier_calibrated': sub_brier_calibrated,
                'brier_score': sub_brier_calibrated,
                'mae_spread': sub_mae,
                'rolling_accuracy_latest': float(rolling_series.iloc[-1]),
                'calibration': sub_calibration,
            }

    def _bucket_stats(metric_df: pd.DataFrame, bucket_col: str) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for bucket_value, sub in metric_df.groupby(bucket_col):
            if pd.isna(bucket_value) or len(sub) == 0:
                continue
            bucket_acc = float((sub['predicted_winner'] == sub['actual_winner']).mean())
            bucket_brier = float(np.mean((sub['eval_prob'] - sub['actual_home_win']) ** 2))
            out[str(bucket_value)] = {
                'count': int(len(sub)),
                'accuracy': bucket_acc,
                'brier_calibrated': bucket_brier,
            }
        return out

    bucket_df = df.copy()
    bucket_df['actual_home_win'] = (bucket_df['home_score'] > bucket_df['away_score']).astype(float)
    bucket_df['eval_prob'] = bucket_df['win_probability_calibrated'].fillna(bucket_df['win_probability']).clip(1e-6, 1 - 1e-6)

    bucket_df['elo_bucket'] = pd.cut(
        pd.to_numeric(bucket_df['elo_diff'], errors='coerce'),
        bins=[-np.inf, -80, -30, 30, 80, np.inf],
        labels=['strong_away', 'away_edge', 'even', 'home_edge', 'strong_home'],
    )
    bucket_df['rest_bucket'] = pd.cut(
        pd.to_numeric(bucket_df['rest_diff'], errors='coerce'),
        bins=[-np.inf, -1, 0, 1, np.inf],
        labels=['away_rest_adv', 'away_slight_adv', 'even_rest', 'home_rest_adv'],
    )
    last5_diff = pd.to_numeric(bucket_df['last5_win_pct_home'], errors='coerce').fillna(0.5) - pd.to_numeric(bucket_df['last5_win_pct_away'], errors='coerce').fillna(0.5)
    bucket_df['last5_form_bucket'] = pd.cut(
        last5_diff,
        bins=[-np.inf, -0.2, -0.05, 0.05, 0.2, np.inf],
        labels=['away_hot', 'away_edge', 'neutral_form', 'home_edge', 'home_hot'],
    )

    by_elo_bucket = _bucket_stats(bucket_df, 'elo_bucket')
    by_rest_diff_bucket = _bucket_stats(bucket_df, 'rest_bucket')
    by_last5_form_bucket = _bucket_stats(bucket_df, 'last5_form_bucket')

    return {
        'n_total': int(len(df)),
        'accuracy': accuracy,
        'brier_raw': brier_raw,
        'brier_calibrated': brier_calibrated,
        'brier_score': brier_calibrated,
        'mae_spread': mae_spread,
        'by_confidence': by_confidence,
        'by_new_feature': by_new_feature,
        'by_elo_bucket': by_elo_bucket,
        'by_rest_diff_bucket': by_rest_diff_bucket,
        'by_last5_form_bucket': by_last5_form_bucket,
        'calibration': calibration,
    }


def get_recent_predictions(db_path: str = 'sports_analytics.db', n: int = 20) -> pd.DataFrame:
    """Return recent predictions DataFrame for notebook display."""
    with SportsAnalyticsDB(db_path) as db:
        rows = db.get_recent_predictions_for_display(n=n)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                'game_date_pst',
                'matchup',
                'new_feature',
                'elo_diff',
                'rest_diff',
                'last5_win_pct_home',
                'last5_win_pct_away',
                'last5_point_diff_home',
                'last5_point_diff_away',
                'predicted_winner',
                'actual_winner',
                'probability_raw',
                'probability_calibrated',
                'probability',
                'correct',
            ]
        )

    df = df.rename(
        columns={
            'predicted_probability': 'probability_raw',
            'predicted_probability_calibrated': 'probability_calibrated',
        }
    )
    df['probability_calibrated'] = pd.to_numeric(df['probability_calibrated'], errors='coerce')
    df['probability'] = df['probability_calibrated'].fillna(df['probability_raw'])
    df['game_date_pst'] = df['game_date_pst'].map(_format_display_time_military)
    df['actual_winner'] = df['actual_winner'].fillna('Pending')
    ordered_cols = [
        'game_date_pst',
        'matchup',
        'new_feature',
        'elo_diff',
        'rest_diff',
        'last5_win_pct_home',
        'last5_win_pct_away',
        'last5_point_diff_home',
        'last5_point_diff_away',
        'predicted_winner',
        'actual_winner',
        'probability_raw',
        'probability_calibrated',
        'probability',
        'correct',
    ]
    return df[ordered_cols]


def compare_to_baseline(metrics: Dict[str, Any], baseline: Dict[str, float] = BASELINE) -> Dict[str, Optional[float]]:
    """Compare metrics against locked baseline values."""
    accuracy = metrics.get('accuracy')
    brier = metrics.get('brier_calibrated', metrics.get('brier_score'))
    mae = metrics.get('mae_spread')

    return {
        'accuracy_delta': (float(accuracy) - float(baseline['accuracy'])) if accuracy is not None else None,
        'brier_delta': (float(baseline['brier']) - float(brier)) if brier is not None else None,
        'mae_delta': (float(baseline['mae']) - float(mae)) if mae is not None else None,
    }


def _parse_baseline_snapshot(raw_value: Any) -> Optional[Dict[str, Any]]:
    if raw_value is None:
        return None
    if isinstance(raw_value, dict):
        return raw_value
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _ensure_baseline_snapshot(db_path: str) -> Dict[str, Any]:
    """Persist a baseline snapshot once and return the effective baseline object."""
    with SportsAnalyticsDB(db_path) as db:
        state = db.get_retraining_state()
        baseline_snapshot = _parse_baseline_snapshot(state.get('baseline_snapshot'))
        baseline_locked = int(state.get('baseline_locked') or 0)

        if baseline_locked and baseline_snapshot:
            return baseline_snapshot

        locked_snapshot = {
            **BASELINE,
            'locked_at': datetime.now().isoformat(),
            'source': 'phase0_default',
        }
        db.update_retraining_state(
            incremental_count=int(state.get('incremental_count') or 0),
            baseline_snapshot=locked_snapshot,
            baseline_locked=True,
        )
        return locked_snapshot


def _get_retrain_progress(db_path: str, retrain_every_n: int) -> Dict[str, Any]:
    """Return retrain progress metadata from retraining_metadata."""
    threshold = max(int(retrain_every_n or RETRAIN_EVERY_N_DEFAULT), 1)

    try:
        with SportsAnalyticsDB(db_path) as db:
            state = db.get_retraining_state()
    except Exception:
        state = {}

    current_count = int(state.get('incremental_count') or 0)
    progress_mod = current_count % threshold
    ready_to_retrain = bool(current_count > 0 and progress_mod == 0)
    progress_count = threshold if ready_to_retrain else progress_mod
    remaining_batches = 0 if ready_to_retrain else max(threshold - progress_count, 0)

    return {
        'current_count': current_count,
        'progress_count': progress_count,
        'retrain_every_n': threshold,
        'remaining_batches': remaining_batches,
        'progress_pct': (progress_count / threshold) * 100.0,
        'ready_to_retrain': ready_to_retrain,
        'model_version': state.get('model_version'),
        'last_full_retrain': state.get('last_full_retrain'),
        'last_incremental': state.get('last_incremental'),
    }


def get_accuracy_summary(
    db_path: str = 'sports_analytics.db',
    n: int = 100,
    retrain_every_n: int = RETRAIN_EVERY_N_DEFAULT,
) -> Dict[str, Any]:
    """Return compact accuracy summary that is easy to render in notebooks."""
    metrics = evaluate_recent_predictions(db_path=db_path, n=n)
    effective_baseline = _ensure_baseline_snapshot(db_path=db_path)
    baseline_compare = compare_to_baseline(metrics=metrics, baseline=effective_baseline)
    retrain_progress = _get_retrain_progress(db_path=db_path, retrain_every_n=retrain_every_n)
    if metrics.get('n_total', 0) == 0:
        return {
            'n_total': 0,
            'headline': 'No settled predictions yet',
            'accuracy_pct': 0.0,
            'brier_raw': None,
            'brier_calibrated': None,
            'brier_score': None,
            'mae_spread': None,
            'baseline': effective_baseline,
            'baseline_comparison': baseline_compare,
            'by_confidence': {},
            'by_new_feature': {},
            'by_elo_bucket': {},
            'by_rest_diff_bucket': {},
            'by_last5_form_bucket': {},
            'calibration': [],
            'retrain_progress': retrain_progress,
        }

    # Optional color hints for notebook HTML rendering.
    feature_impact: Dict[str, Dict[str, Any]] = {}
    for value, m in metrics.get('by_new_feature', {}).items():
        acc = float(m.get('accuracy', 0.0))
        if acc >= 0.55:
            color = '#16a34a'
        elif acc >= 0.50:
            color = '#f59e0b'
        else:
            color = '#dc2626'
        feature_impact[value] = {
            **m,
            'color': color,
        }

    return {
        'n_total': metrics['n_total'],
        'headline': f"Accuracy: {metrics['accuracy'] * 100:.1f}% over last {metrics['n_total']} settled predictions",
        'accuracy_pct': metrics['accuracy'] * 100.0,
        'brier_raw': metrics.get('brier_raw'),
        'brier_calibrated': metrics.get('brier_calibrated'),
        'brier_score': metrics['brier_score'],
        'mae_spread': metrics['mae_spread'],
        'baseline': effective_baseline,
        'baseline_comparison': baseline_compare,
        'by_confidence': metrics['by_confidence'],
        'by_new_feature': metrics.get('by_new_feature', {}),
        'by_elo_bucket': metrics.get('by_elo_bucket', {}),
        'by_rest_diff_bucket': metrics.get('by_rest_diff_bucket', {}),
        'by_last5_form_bucket': metrics.get('by_last5_form_bucket', {}),
        'feature_impact': feature_impact,
        'calibration': metrics['calibration'],
        'retrain_progress': retrain_progress,
    }


def render_accuracy_summary_html(summary: Dict[str, Any]) -> str:
    """Return a compact HTML block suitable for Jupyter display."""
    if not summary or int(summary.get('n_total', 0)) == 0:
        return '<div><strong>No settled predictions yet.</strong></div>'

    def _color_for_acc(acc: float) -> str:
        if acc >= 0.55:
            return '#16a34a'
        if acc >= 0.50:
            return '#f59e0b'
        return '#dc2626'

    headline = str(summary.get('headline', '')).strip()
    accuracy_pct = float(summary.get('accuracy_pct', 0.0))
    overall_color = _color_for_acc(accuracy_pct / 100.0)
    brier_raw = summary.get('brier_raw')
    brier_calibrated = summary.get('brier_calibrated')
    brier = summary.get('brier_score')
    mae = summary.get('mae_spread')
    baseline_compare = summary.get('baseline_comparison') or {}

    retrain_progress = summary.get('retrain_progress') or {}
    retrain_html = ''
    if retrain_progress:
        progress_count = int(retrain_progress.get('progress_count', 0))
        threshold = int(retrain_progress.get('retrain_every_n', RETRAIN_EVERY_N_DEFAULT))
        remaining = int(retrain_progress.get('remaining_batches', 0))
        progress_pct = float(retrain_progress.get('progress_pct', 0.0))
        ready_to_retrain = bool(retrain_progress.get('ready_to_retrain', False))

        badge_text = 'Retrain due now' if ready_to_retrain else f'Retrain Progress: {progress_count}/{threshold}'
        badge_bg = '#991b1b' if ready_to_retrain else '#1d4ed8'
        status_text = (
            'Auto full retrain will trigger on the next prediction batch.'
            if ready_to_retrain
            else f'{remaining} prediction batch(es) remaining until next auto full retrain.'
        )

        retrain_html = f"""
        <div style="margin:0 0 12px 0;padding:10px 12px;border:1px solid #d1d5db;border-radius:10px;background:#f8fafc;">
            <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                <span style="display:inline-block;padding:4px 10px;border-radius:999px;background:{badge_bg};color:#f9fafb;font-weight:700;font-size:12px;letter-spacing:0.2px;">{badge_text}</span>
                <span style="color:#1f2937;font-size:13px;">{status_text}</span>
            </div>
            <div style="margin-top:8px;height:8px;border-radius:999px;background:#e5e7eb;overflow:hidden;">
                <div style="height:100%;width:{progress_pct:.1f}%;background:{badge_bg};"></div>
            </div>
        </div>
        """

    feature_rows = ''
    for value, row in (summary.get('feature_impact') or {}).items():
        acc = float(row.get('accuracy', 0.0))
        color = row.get('color', _color_for_acc(acc))
        feature_rows += (
            "<tr style='background:linear-gradient(90deg,#ffffff,#f8fafc);'>"
            f"<td style='padding:10px 8px;border-bottom:1px solid #e5e7eb;color:#111827;'>{value}</td>"
            f"<td style='padding:10px 8px;border-bottom:1px solid #e5e7eb;color:#111827;'>{row.get('count', 0)}</td>"
            f"<td style='padding:10px 8px;border-bottom:1px solid #e5e7eb;color:{color};font-weight:700'>{acc * 100:.1f}%</td>"
            f"<td style='padding:10px 8px;border-bottom:1px solid #e5e7eb;color:#111827;'>{row.get('brier_score', 0.0):.4f}</td>"
            f"<td style='padding:10px 8px;border-bottom:1px solid #e5e7eb;color:#111827;'>{row.get('mae_spread', 0.0):.2f}</td>"
            "</tr>"
        )

    if not feature_rows:
        feature_rows = (
            "<tr><td colspan='5' style='padding:12px;color:#4b5563;'>"
            "<em>No new_feature values populated yet.</em></td></tr>"
        )

    return f"""
    <div style="border:1px solid #111827;border-radius:12px;overflow:hidden;background:#ffffff;box-shadow:0 10px 26px rgba(2,6,23,0.15);">
      <div style="padding:12px 14px;background:linear-gradient(120deg,#111827,#1f2937);">
        <h3 style="margin:0;color:#f9fafb;letter-spacing:0.2px;">Prediction Feedback Summary</h3>
      </div>
      <div style="padding:12px 14px;background:#ffffff;">
        {retrain_html}
        <p style="margin:0 0 6px 0;color:#111827;font-weight:600;">{headline}</p>
        <p style="margin:0 0 10px 0;color:#111827;">
          <span style="font-weight:700;color:#111827;">Accuracy:</span>
          <span style="font-weight:800;color:{overall_color};"> {accuracy_pct:.1f}%</span>
                    <span style="color:#111827;"> | Brier Raw: {float(brier_raw) if brier_raw is not None else 0.0:.4f} | Brier Calibrated: {float(brier_calibrated) if brier_calibrated is not None else float(brier) if brier is not None else 0.0:.4f} | Spread MAE: {float(mae) if mae is not None else 0.0:.2f}</span>
                </p>
                <p style="margin:0 0 10px 0;color:#111827;">
                    <span style="font-weight:700;color:#111827;">Baseline Δ:</span>
                    <span style="color:#111827;"> Accuracy {float(baseline_compare.get('accuracy_delta') or 0.0) * 100.0:+.2f} pp | Brier {float(baseline_compare.get('brier_delta') or 0.0):+.4f} | MAE {float(baseline_compare.get('mae_delta') or 0.0):+.2f}</span>
        </p>
      </div>
      <table style="border-collapse:collapse;width:100%;background:#ffffff;">
        <thead>
          <tr>
            <th style="text-align:left;padding:10px 8px;border-bottom:1px solid #d1d5db;background:#f3f4f6;color:#111827;">new_feature</th>
            <th style="text-align:left;padding:10px 8px;border-bottom:1px solid #d1d5db;background:#f3f4f6;color:#111827;">Count</th>
            <th style="text-align:left;padding:10px 8px;border-bottom:1px solid #d1d5db;background:#f3f4f6;color:#111827;">Accuracy</th>
            <th style="text-align:left;padding:10px 8px;border-bottom:1px solid #d1d5db;background:#f3f4f6;color:#111827;">Brier</th>
            <th style="text-align:left;padding:10px 8px;border-bottom:1px solid #d1d5db;background:#f3f4f6;color:#111827;">MAE</th>
          </tr>
        </thead>
        <tbody>{feature_rows}</tbody>
      </table>
    </div>
    """


def log_feature_metrics_to_mlflow(
    summary: Dict[str, Any],
    run_name: str = 'prediction-feedback-loop',
    enable: bool = False,
    tags: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Optional MLflow hook for feature-aware evaluation metrics.

    MLflow logs experiment metrics outside SQLite so you can compare runs over time.
    Keep this disabled during normal notebook debugging, and enable only after
    your MLflow tracking backend is configured.
    """
    if not enable:
        return {'status': 'skipped', 'reason': 'enable=False'}

    try:
        import mlflow  # type: ignore
    except Exception as exc:
        return {'status': 'skipped', 'reason': f'mlflow unavailable: {exc}'}

    with mlflow.start_run(run_name=run_name):
        if tags:
            mlflow.set_tags(tags)

        mlflow.log_metric('feedback.n_total', float(summary.get('n_total', 0)))
        mlflow.log_metric('feedback.accuracy_pct', float(summary.get('accuracy_pct', 0.0)))
        if summary.get('brier_raw') is not None:
            mlflow.log_metric('feedback.brier_raw', float(summary['brier_raw']))
        if summary.get('brier_calibrated') is not None:
            mlflow.log_metric('feedback.brier_calibrated', float(summary['brier_calibrated']))
        if summary.get('brier_score') is not None:
            mlflow.log_metric('feedback.brier_score', float(summary['brier_score']))
        if summary.get('mae_spread') is not None:
            mlflow.log_metric('feedback.mae_spread', float(summary['mae_spread']))

        baseline_compare = summary.get('baseline_comparison') or {}
        if baseline_compare.get('accuracy_delta') is not None:
            mlflow.log_metric('feedback.baseline.accuracy_delta', float(baseline_compare['accuracy_delta']))
        if baseline_compare.get('brier_delta') is not None:
            mlflow.log_metric('feedback.baseline.brier_delta', float(baseline_compare['brier_delta']))
        if baseline_compare.get('mae_delta') is not None:
            mlflow.log_metric('feedback.baseline.mae_delta', float(baseline_compare['mae_delta']))

        retrain_progress = summary.get('retrain_progress') or {}
        if retrain_progress:
            mlflow.log_metric('training.retrain_every_n', float(retrain_progress.get('retrain_every_n', 0)))
            mlflow.log_metric('training.retrain_progress_count', float(retrain_progress.get('progress_count', 0)))
            mlflow.log_metric('training.retrain_remaining_batches', float(retrain_progress.get('remaining_batches', 0)))
            mlflow.log_metric('training.retrain_ready', 1.0 if retrain_progress.get('ready_to_retrain') else 0.0)
            mlflow.log_metric('training.retrain_counter_raw', float(retrain_progress.get('current_count', 0)))

        def _sanitize_metric_key_segment(text: str) -> str:
            # MLflow metric names allow alphanumerics, _, -, ., spaces and /.
            cleaned = re.sub(r'[^A-Za-z0-9_./\- ]+', '_', str(text))
            cleaned = re.sub(r'_+', '_', cleaned).strip(' ._')
            return cleaned or 'unknown_feature'

        for feature_value, metrics in (summary.get('by_new_feature') or {}).items():
            safe_feature = _sanitize_metric_key_segment(feature_value).replace(' ', '_')
            prefix = f"feedback.new_feature.{safe_feature}"
            mlflow.log_metric(f"{prefix}.count", float(metrics.get('count', 0)))
            mlflow.log_metric(f"{prefix}.accuracy", float(metrics.get('accuracy', 0.0)))
            mlflow.log_metric(f"{prefix}.brier_raw", float(metrics.get('brier_raw', 0.0)))
            mlflow.log_metric(f"{prefix}.brier_calibrated", float(metrics.get('brier_calibrated', metrics.get('brier_score', 0.0))))
            mlflow.log_metric(f"{prefix}.brier_score", float(metrics.get('brier_score', 0.0)))
            mlflow.log_metric(f"{prefix}.mae_spread", float(metrics.get('mae_spread', 0.0)))

        for bucket_name, bucket_data in (
            ('elo', summary.get('by_elo_bucket') or {}),
            ('rest', summary.get('by_rest_diff_bucket') or {}),
            ('last5', summary.get('by_last5_form_bucket') or {}),
        ):
            for bucket_value, metrics in bucket_data.items():
                safe_bucket = _sanitize_metric_key_segment(bucket_value).replace(' ', '_')
                prefix = f"feedback.bucket.{bucket_name}.{safe_bucket}"
                mlflow.log_metric(f"{prefix}.count", float(metrics.get('count', 0)))
                mlflow.log_metric(f"{prefix}.accuracy", float(metrics.get('accuracy', 0.0)))
                mlflow.log_metric(f"{prefix}.brier_calibrated", float(metrics.get('brier_calibrated', 0.0)))

        mlflow.log_dict(summary, 'feedback_summary.json')

    return {'status': 'logged'}
