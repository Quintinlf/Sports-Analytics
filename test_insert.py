#!/usr/bin/env python
"""Test if the database insert works correctly."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from data.database.database_handler import SportsAnalyticsDB

db_path = str(ROOT / 'sports_analytics.db')

try:
    with SportsAnalyticsDB(db_path) as db:
        print("Database initialized successfully")
        
        # Test insert with sample data
        test_data = {
            'game_id': 'test_game_001',
            'game_date': '2025-03-18',
            'game_date_utc': '2025-03-18T23:30:00',
            'game_date_pst': '2025-03-18T16:30:00',
            'game_date_local_date': '2025-03-18',
            'home_team': 'Lakers',
            'away_team': 'Celtics',
            'predicted_spread': -5.5,
            'predicted_home_score': 105.0,
            'predicted_away_score': 110.5,
            'predicted_winner': 'Celtics',
            'win_probability': 0.65,
            'confidence_score': 0.75,
            'confidence_level': 'HIGH',
            'new_feature': 'test_feature_value',
            'pred_std': 2.0,
            'ci_lower': -2.0,
            'ci_upper': 9.0,
            'actual_winner': None,
            'home_score': None,
            'away_score': None,
            'correct': None,
            'epaa_weight': 1.0,
            'model_versions': {},
            'iteration_count': 1,
            'retraining_triggered': False,
            'notes': 'Test insert',
            'model_version': 'v1.2.3'
        }
        
        pred_id = db.insert_prediction(test_data)
        print(f"✓ Prediction inserted successfully with ID: {pred_id}")
        
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
