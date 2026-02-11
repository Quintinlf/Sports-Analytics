"""
Quick Test Script - Verify System Components
"""

import sys
import os

# Add machine_learning to path
sys.path.append('machine_learning')

print("=" * 70)
print("🧪 SYSTEM COMPONENT TEST")
print("=" * 70 + "\n")

# Test 1: Database Handler
print("1️⃣ Testing Database Handler...")
try:
    from database_handler import SportsAnalyticsDB
    db = SportsAnalyticsDB(":memory:")  # In-memory test
    db.create_tables()
    print("   ✅ Database handler working\n")
    db.close()
except Exception as e:
    print(f"   ❌ Error: {e}\n")
    sys.exit(1)

# Test 2: Game Parser
print("2️⃣ Testing Game Parser...")
try:
    from game_parser import parse_game_data_from_text
    sample_csv = """Date,Start (ET),Visitor/Neutral,PTS,Home/Neutral,PTS,,,Attend.,LOG,Arena,Notes
Sun Feb 1 2026,3:30p,Milwaukee Bucks,79,Boston Celtics,107,Box Score,,19156,2:09,TD Garden,
Sun Feb 8 2026,12:30p,New York Knicks,,Boston Celtics,,,,,,TD Garden,"""
    
    result = parse_game_data_from_text(sample_csv, verbose=False)
    assert result['total_games'] == 2
    assert result['completed_count'] == 1
    assert result['upcoming_count'] == 1
    print(f"   ✅ Game parser working (parsed {result['total_games']} games)\n")
except Exception as e:
    print(f"   ❌ Error: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Extended Data Loader
print("3️⃣ Testing Extended Data Loader...")
try:
    from extended_data_loader import fetch_comprehensive_nba_data
    print("   ✅ Extended data loader imports successfully\n")
except Exception as e:
    print(f"   ❌ Error: {e}\n")
    sys.exit(1)

# Test 4: Iterative Predictor
print("4️⃣ Testing Iterative Predictor...")
try:
    from iterative_predictor import IterativePredictor
    print("   ✅ Iterative predictor imports successfully\n")
except Exception as e:
    print(f"   ❌ Error: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check existing modules
print("5️⃣ Testing Existing Modules...")
try:
    from data_loader import get_all_nba_teams
    from model_trainer import GaussianProcessPredictor
    from validation_tracker import PredictionValidator
    print("   ✅ All existing modules accessible\n")
except Exception as e:
    print(f"   ❌ Error: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 70)
print("✅ ALL TESTS PASSED")
print("=" * 70)
print("System is ready to run predictions!")
print("=" * 70 + "\n")
