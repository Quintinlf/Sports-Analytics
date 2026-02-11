"""
SQLite Database Handler for Sports Analytics Predictions
Manages storage of predictions, results, model history, and cached game data
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import os


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
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                predicted_spread REAL NOT NULL,
                predicted_home_score REAL,
                predicted_away_score REAL,
                predicted_winner TEXT NOT NULL,
                win_probability REAL NOT NULL,
                confidence_score REAL NOT NULL,
                confidence_level TEXT NOT NULL,
                pred_std REAL,
                ci_lower REAL,
                ci_upper REAL,
                epaa_weight REAL,
                model_versions TEXT,
                iteration_count INTEGER DEFAULT 1,
                retraining_triggered BOOLEAN DEFAULT 0,
                prediction_timestamp TEXT NOT NULL,
                notes TEXT
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
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_game_id ON predictions(game_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(game_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_teams ON predictions(home_team, away_team)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_prediction ON prediction_results(prediction_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_prediction ON model_history(prediction_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_cache_id ON games_cache(game_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_cache_date ON games_cache(game_date)")
        
        self.conn.commit()
        
    def insert_prediction(self, prediction_data: Dict[str, Any]) -> int:
        """Insert a new prediction record and return its ID"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO predictions (
                game_id, game_date, home_team, away_team,
                predicted_spread, predicted_home_score, predicted_away_score,
                predicted_winner, win_probability, confidence_score, confidence_level,
                pred_std, ci_lower, ci_upper, epaa_weight, model_versions,
                iteration_count, retraining_triggered, prediction_timestamp, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction_data.get('game_id'),
            prediction_data['game_date'],
            prediction_data['home_team'],
            prediction_data['away_team'],
            prediction_data['predicted_spread'],
            prediction_data.get('predicted_home_score'),
            prediction_data.get('predicted_away_score'),
            prediction_data['predicted_winner'],
            prediction_data['win_probability'],
            prediction_data['confidence_score'],
            prediction_data['confidence_level'],
            prediction_data.get('pred_std'),
            prediction_data.get('ci_lower'),
            prediction_data.get('ci_upper'),
            prediction_data.get('epaa_weight'),
            json.dumps(prediction_data.get('model_versions', {})),
            prediction_data.get('iteration_count', 1),
            prediction_data.get('retraining_triggered', False),
            datetime.now().isoformat(),
            prediction_data.get('notes')
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
            json.dumps(log_data.get('parameters_changed', {})),
            json.dumps(log_data.get('metrics', {})),
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
            json.dumps(game_data.get('stats', {})),
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
