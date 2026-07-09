"""
SQLite Database Handler for Sports Analytics Predictions
Manages storage of predictions, results, model history, and cached game data
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import numpy as np
except ImportError:
    np = None


def _to_json_serializable(obj: Any) -> Any:
    """Recursively convert pandas/numpy objects to JSON-safe values."""
    if obj is None:
        return None

    if isinstance(obj, datetime):
        return obj.isoformat()

    if pd is not None:
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.Series):
            return {k: _to_json_serializable(v) for k, v in obj.to_dict().items()}
        if isinstance(obj, pd.DataFrame):
            return [{k: _to_json_serializable(v) for k, v in rec.items()} for rec in obj.to_dict(orient='records')]

    if np is not None:
        if isinstance(obj, np.ndarray):
            return _to_json_serializable(obj.tolist())
        if isinstance(obj, np.generic):
            return obj.item()

    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_to_json_serializable(v) for v in obj]

    if pd is not None:
        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass

    return obj


class SportsAnalyticsDB:
    """Handler for SQLite database operations"""
    
    def __init__(self, db_path: str = "sports_analytics.db"):
        """Initialize database connection and create tables if needed"""
        self.db_path = db_path
        self.conn = None
        self.connect()
        self.create_tables()
    
    def connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        
    def create_tables(self):
        """Create all necessary tables with indexes"""
        cursor = self.conn.cursor()
        
        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT,
                game_date TEXT NOT NULL,
                game_date_utc TEXT,
                game_date_pst TEXT,
                game_date_local_date TEXT,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                predicted_spread REAL NOT NULL,
                predicted_home_score REAL,
                predicted_away_score REAL,
                predicted_winner TEXT NOT NULL,
                win_probability REAL NOT NULL,
                win_probability_calibrated REAL,
                confidence_score REAL NOT NULL,
                confidence_level TEXT NOT NULL,
                new_feature TEXT,
                elo_diff REAL,
                last5_win_pct_home REAL,
                last5_win_pct_away REAL,
                last5_point_diff_home REAL,
                last5_point_diff_away REAL,
                rest_days_home INTEGER,
                rest_days_away INTEGER,
                rest_diff REAL,
                is_back_to_back_home INTEGER,
                is_back_to_back_away INTEGER,
                home_away_strength_diff REAL,
                schedule_density_diff REAL,
                pace_diff REAL,
                injury_proxy REAL,
                pred_std REAL,
                ci_lower REAL,
                ci_upper REAL,
                actual_winner TEXT,
                home_score INTEGER,
                away_score INTEGER,
                correct INTEGER,
                epaa_weight REAL,
                model_versions TEXT,
                iteration_count INTEGER DEFAULT 1,
                retraining_triggered BOOLEAN DEFAULT FALSE,
                prediction_timestamp TEXT NOT NULL,
                notes TEXT,
                model_version TEXT
            )
        """)
        
        # Prediction results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                actual_home_score INTEGER NOT NULL,
                actual_away_score INTEGER NOT NULL,
                actual_spread REAL NOT NULL,
                actual_winner TEXT NOT NULL,
                prediction_error REAL NOT NULL,
                correct_winner BOOLEAN NOT NULL,
                within_ci BOOLEAN NOT NULL,
                result_timestamp TEXT NOT NULL,
                FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
            )
        """)
        
        # Model history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_history (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER,
                iteration INTEGER NOT NULL,
                model_type TEXT NOT NULL,
                action TEXT NOT NULL,
                confidence_before REAL,
                confidence_after REAL,
                parameters_changed TEXT,
                metrics TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
            )
        """)
        
        # Cached games data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS games_cache (
                cache_id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT UNIQUE,
                game_date TEXT NOT NULL,
                season TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_team_id INTEGER,
                away_team_id INTEGER,
                home_score INTEGER,
                away_score INTEGER,
                game_status TEXT,
                stats_json TEXT,
                cached_timestamp TEXT NOT NULL
            )
        """)
        
        # Feature snapshots for failure analysis
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_features (
                feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                feature_snapshot TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
            )
        """)

        # Retraining metadata / incremental counter
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retraining_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incremental_count INTEGER DEFAULT 0,
                last_incremental TEXT,
                last_full_retrain TEXT,
                model_version TEXT,
                ensemble_weights TEXT,
                baseline_snapshot TEXT,
                baseline_locked INTEGER DEFAULT 0,
                updated_at TEXT NOT NULL
            )
        """)

        # Add model_version column to predictions if it doesn't exist yet
        try:
            cursor.execute("ALTER TABLE predictions ADD COLUMN model_version TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Add timezone and lightweight outcome columns if missing.
        # These are used by the notebook feedback loop and kept on predictions
        # for fast querying without mandatory joins.
        prediction_columns = [
            ('game_date_utc', 'TEXT'),
            ('game_date_pst', 'TEXT'),
            ('game_date_local_date', 'TEXT'),
            ('new_feature', 'TEXT'),
            ('actual_winner', 'TEXT'),
            ('home_score', 'INTEGER'),
            ('away_score', 'INTEGER'),
            ('correct', 'INTEGER'),
            ('win_probability_calibrated', 'REAL'),
            ('elo_diff', 'REAL'),
            ('last5_win_pct_home', 'REAL'),
            ('last5_win_pct_away', 'REAL'),
            ('last5_point_diff_home', 'REAL'),
            ('last5_point_diff_away', 'REAL'),
            ('rest_days_home', 'INTEGER'),
            ('rest_days_away', 'INTEGER'),
            ('rest_diff', 'REAL'),
            ('is_back_to_back_home', 'INTEGER'),
            ('is_back_to_back_away', 'INTEGER'),
            ('home_away_strength_diff', 'REAL'),
            ('schedule_density_diff', 'REAL'),
            ('pace_diff', 'REAL'),
            ('injury_proxy', 'REAL'),
            ('model_version', 'TEXT'),
        ]
        for col_name, col_type in prediction_columns:
            try:
                cursor.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")
            except sqlite3.OperationalError:
                pass

        # Keep baseline lock fields available for legacy DBs.
        retrain_columns = [
            ('baseline_snapshot', 'TEXT'),
            ('baseline_locked', 'INTEGER DEFAULT 0'),
        ]
        for col_name, col_type in retrain_columns:
            try:
                cursor.execute(f"ALTER TABLE retraining_metadata ADD COLUMN {col_name} {col_type}")
            except sqlite3.OperationalError:
                pass

        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_game_id ON predictions(game_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(game_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_date_local ON predictions(game_date_local_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_new_feature ON predictions(new_feature)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_elo_diff ON predictions(elo_diff)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_rest_diff ON predictions(rest_diff)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_teams ON predictions(home_team, away_team)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_prediction ON prediction_results(prediction_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_prediction ON model_history(prediction_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_cache_id ON games_cache(game_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_cache_date ON games_cache(game_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_features_prediction ON prediction_features(prediction_id)")

        self._ensure_unified_schema(cursor)

        self.conn.commit()

    def _ensure_unified_schema(self, cursor) -> None:
        """Add unified multi-sport columns and prediction_options table."""
        unified_columns = [
            ('sport', 'TEXT'),
            ('league', 'TEXT'),
            ('feature_snapshot', 'TEXT'),
            ('actual_home_score', 'INTEGER'),
            ('actual_away_score', 'INTEGER'),
            ('created_at', 'TEXT'),
        ]
        for col_name, col_type in unified_columns:
            try:
                cursor.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")
            except sqlite3.OperationalError:
                pass

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_options (
                option_id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                option_name TEXT NOT NULL,
                probability REAL NOT NULL,
                rank INTEGER NOT NULL,
                FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id),
                UNIQUE(prediction_id, option_name)
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_sport_date "
            "ON predictions (sport, game_date)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_options_prediction "
            "ON prediction_options (prediction_id)"
        )
        
    def insert_prediction(self, prediction_data: Dict[str, Any]) -> int:
        """Insert a new prediction record and return its ID"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO predictions (
                game_id, game_date, game_date_utc, game_date_pst, game_date_local_date,
                home_team, away_team,
                predicted_spread, predicted_home_score, predicted_away_score,
                predicted_winner, win_probability, confidence_score, confidence_level,
                win_probability_calibrated,
                new_feature,
                elo_diff,
                last5_win_pct_home, last5_win_pct_away,
                last5_point_diff_home, last5_point_diff_away,
                rest_days_home, rest_days_away, rest_diff,
                is_back_to_back_home, is_back_to_back_away,
                home_away_strength_diff, schedule_density_diff, pace_diff, injury_proxy,
                pred_std, ci_lower, ci_upper,
                actual_winner, home_score, away_score, correct,
                epaa_weight, model_versions,
                iteration_count, retraining_triggered, prediction_timestamp, notes,
                model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction_data.get('game_id'),
            prediction_data['game_date'],
            prediction_data.get('game_date_utc'),
            prediction_data.get('game_date_pst'),
            prediction_data.get('game_date_local_date'),
            prediction_data['home_team'],
            prediction_data['away_team'],
            prediction_data['predicted_spread'],
            prediction_data.get('predicted_home_score'),
            prediction_data.get('predicted_away_score'),
            prediction_data['predicted_winner'],
            prediction_data['win_probability'],
            prediction_data['confidence_score'],
            prediction_data['confidence_level'],
            prediction_data.get('win_probability_calibrated'),
            prediction_data.get('new_feature'),
            prediction_data.get('elo_diff'),
            prediction_data.get('last5_win_pct_home'),
            prediction_data.get('last5_win_pct_away'),
            prediction_data.get('last5_point_diff_home'),
            prediction_data.get('last5_point_diff_away'),
            prediction_data.get('rest_days_home'),
            prediction_data.get('rest_days_away'),
            prediction_data.get('rest_diff'),
            prediction_data.get('is_back_to_back_home'),
            prediction_data.get('is_back_to_back_away'),
            prediction_data.get('home_away_strength_diff'),
            prediction_data.get('schedule_density_diff'),
            prediction_data.get('pace_diff'),
            prediction_data.get('injury_proxy'),
            prediction_data.get('pred_std'),
            prediction_data.get('ci_lower'),
            prediction_data.get('ci_upper'),
            prediction_data.get('actual_winner'),
            prediction_data.get('home_score'),
            prediction_data.get('away_score'),
            prediction_data.get('correct'),
            prediction_data.get('epaa_weight'),
            json.dumps(_to_json_serializable(prediction_data.get('model_versions', {}))),
            prediction_data.get('iteration_count', 1),
            prediction_data.get('retraining_triggered', False),
            datetime.now().isoformat(),
            prediction_data.get('notes'),
            prediction_data.get('model_version')
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def insert_result(self, prediction_id: int, result_data: Dict[str, Any]) -> int:
        """Insert actual game result for a prediction"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO prediction_results (
                prediction_id, actual_home_score, actual_away_score,
                actual_spread, actual_winner, prediction_error,
                correct_winner, within_ci, result_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction_id,
            result_data['actual_home_score'],
            result_data['actual_away_score'],
            result_data['actual_spread'],
            result_data['actual_winner'],
            result_data['prediction_error'],
            result_data['correct_winner'],
            result_data['within_ci'],
            datetime.now().isoformat()
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def log_model_action(self, log_data: Dict[str, Any]) -> int:
        """Log a model training/update action"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO model_history (
                prediction_id, iteration, model_type, action,
                confidence_before, confidence_after, parameters_changed,
                metrics, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            log_data.get('prediction_id'),
            log_data['iteration'],
            log_data['model_type'],
            log_data['action'],
            log_data.get('confidence_before'),
            log_data.get('confidence_after'),
            json.dumps(_to_json_serializable(log_data.get('parameters_changed', {}))),
            json.dumps(_to_json_serializable(log_data.get('metrics', {}))),
            datetime.now().isoformat()
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def cache_game(self, game_data: Dict[str, Any]) -> int:
        """Cache game data to avoid repeated API calls"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO games_cache (
                game_id, game_date, season, home_team, away_team,
                home_team_id, away_team_id, home_score, away_score,
                game_status, stats_json, cached_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            game_data['game_id'],
            game_data['game_date'],
            game_data.get('season'),
            game_data['home_team'],
            game_data['away_team'],
            game_data.get('home_team_id'),
            game_data.get('away_team_id'),
            game_data.get('home_score'),
            game_data.get('away_score'),
            game_data.get('game_status'),
            json.dumps(_to_json_serializable(game_data.get('stats', {}))),
            datetime.now().isoformat()
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_cached_games(self, season: Optional[str] = None, 
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> List[Dict]:
        """Retrieve cached games with optional filters"""
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM games_cache WHERE 1=1"
        params = []
        
        if season:
            query += " AND season = ?"
            params.append(season)
        if start_date:
            query += " AND game_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND game_date <= ?"
            params.append(end_date)
            
        query += " ORDER BY game_date DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    def get_predictions_by_date(self, start_date: str, end_date: str) -> List[Dict]:
        """Get all predictions within a date range"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM predictions
            WHERE game_date BETWEEN ? AND ?
            ORDER BY game_date, prediction_timestamp
        """, (start_date, end_date))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_prediction_with_result(self, prediction_id: int) -> Optional[Dict]:
        """Get prediction with its actual result if available"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT p.*, r.* FROM predictions p
            LEFT JOIN prediction_results r ON p.prediction_id = r.prediction_id
            WHERE p.prediction_id = ?
        """, (prediction_id,))
        
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_unsettled_predictions(self, days: int = 14) -> List[Dict]:
        """Return recent predictions that still do not have actual outcomes."""
        cursor = self.conn.cursor()
        cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        cursor.execute(
            """
            SELECT p.*
            FROM predictions p
            WHERE p.actual_winner IS NULL
              AND COALESCE(p.game_date_local_date, p.game_date) >= ?
            ORDER BY COALESCE(p.game_date_local_date, p.game_date), p.prediction_id
            """,
            (cutoff,),
        )

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def update_prediction_actual(
        self,
        prediction_id: int,
        actual_winner: str,
        home_score: int,
        away_score: int,
    ) -> None:
        """Update lightweight actual fields on predictions table."""
        cursor = self.conn.cursor()

        cursor.execute(
            "SELECT predicted_winner FROM predictions WHERE prediction_id = ?",
            (prediction_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return

        predicted_winner = row['predicted_winner']
        correct = 1 if predicted_winner == actual_winner else 0

        cursor.execute(
            """
            UPDATE predictions
            SET actual_winner = ?,
                home_score = ?,
                away_score = ?,
                correct = ?
            WHERE prediction_id = ?
            """,
            (actual_winner, int(home_score), int(away_score), correct, prediction_id),
        )
        self.conn.commit()

    def get_recent_predictions_for_display(self, n: int = 20) -> List[Dict]:
        """Return recent predictions in a notebook-friendly shape."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT
                COALESCE(game_date_pst, game_date) AS game_date_pst,
                away_team || ' @ ' || home_team AS matchup,
                predicted_winner,
                actual_winner,
                win_probability AS predicted_probability,
                win_probability_calibrated AS predicted_probability_calibrated,
                new_feature,
                elo_diff,
                rest_diff,
                last5_win_pct_home,
                last5_win_pct_away,
                last5_point_diff_home,
                last5_point_diff_away,
                correct,
                confidence_level,
                predicted_spread
            FROM predictions
            ORDER BY COALESCE(game_date_local_date, game_date) DESC, prediction_id DESC
            LIMIT ?
            """,
            (int(n),),
        )
        return [dict(r) for r in cursor.fetchall()]

    def get_predictions_by_new_feature(
        self,
        feature_value: Optional[str] = None,
        n: int = 100,
    ) -> List[Dict]:
        """Return recent predictions filtered by the scaffold new_feature value."""
        cursor = self.conn.cursor()
        if feature_value is None:
            cursor.execute(
                """
                SELECT *
                FROM predictions
                ORDER BY COALESCE(game_date_local_date, game_date) DESC, prediction_id DESC
                LIMIT ?
                """,
                (int(n),),
            )
        else:
            cursor.execute(
                """
                SELECT *
                FROM predictions
                WHERE new_feature = ?
                ORDER BY COALESCE(game_date_local_date, game_date) DESC, prediction_id DESC
                LIMIT ?
                """,
                (str(feature_value), int(n)),
            )

        return [dict(r) for r in cursor.fetchall()]
    
    def get_model_history(self, prediction_id: int) -> List[Dict]:
        """Get all model actions for a prediction"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM model_history
            WHERE prediction_id = ?
            ORDER BY iteration, timestamp
        """, (prediction_id,))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_batch_accuracy_comparison(
        self,
        window_batches: int = 7,
        point_tolerance: float = 5.0,
    ) -> Dict[str, Any]:
        """Compare current vs previous batch-window prediction accuracy.

        A "batch" is inferred from prediction date (YYYY-MM-DD part of
        prediction_timestamp), which matches the notebook's one-run/day usage.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT
                p.prediction_timestamp,
                p.predicted_spread,
                gc.home_score,
                gc.away_score
            FROM predictions p
            JOIN games_cache gc ON gc.game_id = p.game_id
            WHERE gc.home_score IS NOT NULL
              AND gc.away_score IS NOT NULL
            """
        )
        rows = [dict(r) for r in cursor.fetchall()]
        if not rows:
            return {
                'current': None,
                'previous': None,
                'delta': None,
                'window_batches': window_batches,
                'point_tolerance': point_tolerance,
            }

        for r in rows:
            ts = str(r.get('prediction_timestamp') or '')
            r['batch_day'] = ts[:10] if len(ts) >= 10 else ''

        batch_days = sorted({r['batch_day'] for r in rows if r.get('batch_day')}, reverse=True)
        current_days = set(batch_days[:window_batches])
        previous_days = set(batch_days[window_batches: window_batches * 2])

        def _calc(day_set: set) -> Optional[Dict[str, Any]]:
            if not day_set:
                return None
            items = [r for r in rows if r.get('batch_day') in day_set]
            if not items:
                return None

            spread_errors = []
            winner_correct = 0
            point_hits = 0

            for r in items:
                pred_spread = float(r.get('predicted_spread') or 0.0)
                actual_spread = float((r.get('home_score') or 0) - (r.get('away_score') or 0))

                err = abs(pred_spread - actual_spread)
                spread_errors.append(err)
                if (pred_spread >= 0) == (actual_spread >= 0):
                    winner_correct += 1
                if err <= point_tolerance:
                    point_hits += 1

            total = len(items)
            return {
                'batches': len(day_set),
                'samples': total,
                'winner_accuracy': (winner_correct * 100.0 / total) if total else 0.0,
                'point_accuracy': (point_hits * 100.0 / total) if total else 0.0,
                'spread_mae': (sum(spread_errors) / total) if total else None,
            }

        current = _calc(current_days)
        previous = _calc(previous_days)

        delta = None
        if current and previous:
            delta = {
                'winner_accuracy': current['winner_accuracy'] - previous['winner_accuracy'],
                'point_accuracy': current['point_accuracy'] - previous['point_accuracy'],
                'spread_mae': previous['spread_mae'] - current['spread_mae'],
            }

        return {
            'current': current,
            'previous': previous,
            'delta': delta,
            'window_batches': window_batches,
            'point_tolerance': point_tolerance,
        }
    
    def get_performance_stats(self, days: int = 30) -> Dict[str, Any]:
        """Calculate aggregate performance statistics"""
        cursor = self.conn.cursor()
        
        # Get predictions with results
        cursor.execute("""
            SELECT 
                COUNT(*) as total_predictions,
                AVG(p.confidence_score) as avg_confidence,
                AVG(p.iteration_count) as avg_iterations,
                SUM(CASE WHEN p.retraining_triggered THEN 1 ELSE 0 END) as retraining_count,
                AVG(r.prediction_error) as avg_error,
                SUM(CASE WHEN r.correct_winner THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_accuracy,
                SUM(CASE WHEN r.within_ci THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as ci_coverage
            FROM predictions p
            LEFT JOIN prediction_results r ON p.prediction_id = r.prediction_id
            WHERE date(p.game_date) >= date('now', '-' || ? || ' days')
        """, (days,))
        
        row = cursor.fetchone()
        return dict(row) if row else {}
    
    def log_prediction_features(self, prediction_id: int, features: dict) -> int:
        """Store raw feature snapshot for failure analysis"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO prediction_features (prediction_id, feature_snapshot, created_at)
            VALUES (?, ?, ?)
        """, (prediction_id, json.dumps(_to_json_serializable(features)), datetime.now().isoformat()))
        self.conn.commit()
        return cursor.lastrowid

    def get_retraining_state(self) -> Dict[str, Any]:
        """Return the most recent retraining metadata row"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM retraining_metadata ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        return dict(row) if row else {'incremental_count': 0, 'model_version': None}

    def update_retraining_state(
        self,
        incremental_count: int,
        model_version: Optional[str] = None,
        full_retrain: bool = False,
        ensemble_weights: Optional[dict] = None,
        baseline_snapshot: Optional[dict] = None,
        baseline_locked: Optional[bool] = None,
    ) -> None:
        """Insert a new retraining metadata record.

        Notes
        -----
        This table is append-only and callers frequently update just one field
        (e.g. bumping incremental_count). Because other modules (like
        WeightManager) read the MOST RECENT row, we must preserve prior state
        for fields not explicitly provided.
        """
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()

        prev = self.get_retraining_state() if self.conn is not None else {}

        effective_model_version = (
            model_version if model_version is not None else prev.get('model_version')
        )

        # Preserve ensemble weights unless explicitly overridden.
        prev_weights_raw = prev.get('ensemble_weights')
        if ensemble_weights is None:
            effective_weights_raw = prev_weights_raw
        else:
            effective_weights_raw = json.dumps(ensemble_weights)

        # Preserve timestamps unless explicitly updated.
        prev_count = int(prev.get('incremental_count') or 0)
        effective_last_incremental = (
            now
            if (not full_retrain and int(incremental_count) != prev_count)
            else prev.get('last_incremental')
        )
        effective_last_full_retrain = (
            now if full_retrain else prev.get('last_full_retrain')
        )

        prev_baseline_snapshot_raw = prev.get('baseline_snapshot')
        if baseline_snapshot is None:
            effective_baseline_snapshot_raw = prev_baseline_snapshot_raw
        else:
            effective_baseline_snapshot_raw = json.dumps(_to_json_serializable(baseline_snapshot))

        if baseline_locked is None:
            effective_baseline_locked = int(prev.get('baseline_locked') or 0)
        else:
            effective_baseline_locked = int(bool(baseline_locked))

        cursor.execute(
            """
            INSERT INTO retraining_metadata
                (incremental_count, last_incremental, last_full_retrain,
                 model_version, ensemble_weights, baseline_snapshot,
                 baseline_locked, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(incremental_count),
                effective_last_incremental,
                effective_last_full_retrain,
                effective_model_version,
                effective_weights_raw,
                effective_baseline_snapshot_raw,
                effective_baseline_locked,
                now,
            ),
        )
        self.conn.commit()

    def insert_unified_prediction(self, prediction_data: Dict[str, Any]) -> int:
        """Insert a multi-sport prediction row (unified schema). Returns prediction_id."""
        cursor = self.conn.cursor()
        created_at = prediction_data.get('created_at') or datetime.now().isoformat()
        feature_snapshot = prediction_data.get('feature_snapshot')
        if feature_snapshot is not None and not isinstance(feature_snapshot, str):
            feature_snapshot = json.dumps(_to_json_serializable(feature_snapshot))

        # Detect legacy table (no sport column populated path uses legacy NOT NULL cols)
        cursor.execute("PRAGMA table_info(predictions)")
        columns = {row[1] for row in cursor.fetchall()}
        is_legacy = 'predicted_spread' in columns

        if is_legacy:
            win_prob = float(prediction_data.get('win_probability') or 0.5)
            cursor.execute(
                """
                INSERT INTO predictions (
                    sport, league, game_date, home_team, away_team,
                    predicted_winner, confidence_level, feature_snapshot,
                    actual_home_score, actual_away_score, actual_winner, correct,
                    created_at,
                    predicted_spread, win_probability, confidence_score,
                    prediction_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction_data['sport'],
                    prediction_data.get('league'),
                    prediction_data['game_date'],
                    prediction_data['home_team'],
                    prediction_data['away_team'],
                    prediction_data['predicted_winner'],
                    prediction_data['confidence_level'],
                    feature_snapshot,
                    prediction_data.get('actual_home_score'),
                    prediction_data.get('actual_away_score'),
                    prediction_data.get('actual_winner'),
                    prediction_data.get('correct'),
                    created_at,
                    float(prediction_data.get('predicted_spread') or 0.0),
                    win_prob,
                    float(prediction_data.get('confidence_score') or win_prob),
                    created_at,
                ),
            )
        else:
            cursor.execute(
                """
                INSERT INTO predictions (
                    sport, league, game_date, home_team, away_team,
                    predicted_winner, confidence_level, feature_snapshot,
                    actual_home_score, actual_away_score, actual_winner, correct,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction_data['sport'],
                    prediction_data.get('league'),
                    prediction_data['game_date'],
                    prediction_data['home_team'],
                    prediction_data['away_team'],
                    prediction_data['predicted_winner'],
                    prediction_data['confidence_level'],
                    feature_snapshot,
                    prediction_data.get('actual_home_score'),
                    prediction_data.get('actual_away_score'),
                    prediction_data.get('actual_winner'),
                    prediction_data.get('correct'),
                    created_at,
                ),
            )

        self.conn.commit()
        return cursor.lastrowid

    def insert_prediction_options(
        self,
        prediction_id: int,
        options: List[Dict[str, Any]],
    ) -> int:
        """Insert outcome options for a prediction. Returns number of rows inserted."""
        if not options:
            return 0

        cursor = self.conn.cursor()
        for idx, opt in enumerate(options, start=1):
            cursor.execute(
                """
                INSERT INTO prediction_options (
                    prediction_id, option_name, probability, rank
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    prediction_id,
                    opt['option_name'],
                    float(opt['probability']),
                    int(opt.get('rank') or idx),
                ),
            )
        self.conn.commit()
        return len(options)

    def get_unified_predictions_by_date(
        self,
        start_date: str,
        end_date: str,
        sport: Optional[str] = None,
    ) -> List[Dict]:
        """Return unified prediction rows within a date range."""
        cursor = self.conn.cursor()
        sql = """
            SELECT
                prediction_id, sport, league, game_date,
                home_team, away_team, predicted_winner, confidence_level,
                feature_snapshot, actual_home_score, actual_away_score,
                actual_winner, correct, created_at
            FROM predictions
            WHERE game_date BETWEEN ? AND ?
        """
        params: List[Any] = [start_date, end_date]
        if sport:
            sql += " AND sport = ?"
            params.append(sport)
        sql += " ORDER BY game_date, prediction_id"
        cursor.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_prediction_options(self, prediction_id: int) -> List[Dict]:
        """Return all outcome options for a prediction ordered by rank."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT option_id, prediction_id, option_name, probability, rank
            FROM prediction_options
            WHERE prediction_id = ?
            ORDER BY rank, option_id
            """,
            (prediction_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_database(db_path: str = "sports_analytics.db") -> SportsAnalyticsDB:
    """Factory function to get database instance"""
    return SportsAnalyticsDB(db_path)
