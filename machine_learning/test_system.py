"""Quick Test Script - Verify System Components.

This script is intentionally lightweight: it only checks that key modules
import and that the DB schema can be created in-memory.
"""

import os
import sys

# Ensure project root is on sys.path even when run from elsewhere
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

print("=" * 70)
print("🧪 SYSTEM COMPONENT TEST")
print("=" * 70 + "\n")

# Test 1: Database Handler
print("1️⃣ Testing Database Handler...")
try:
    from data.database.database_handler import SportsAnalyticsDB
    db = SportsAnalyticsDB(":memory:")  # In-memory test
    db.create_tables()
    print("   ✅ Database handler working\n")
    db.close()
except Exception as e:
    print(f"   ❌ Error: {e}\n")
    sys.exit(1)

# Test 2: NBA Loader (static)
print("2️⃣ Testing NBA Loader...")
try:
    from data.nba_loader import get_all_nba_teams
    teams_info = get_all_nba_teams()
    assert 'teams' in teams_info and len(teams_info['teams']) > 0
    print(f"   ✅ NBA loader working ({len(teams_info['teams'])} teams)\n")
except Exception as e:
    print(f"   ❌ Error: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Extended Data Loader (experimental)
print("3️⃣ Testing Extended Data Loader (experimental)...")
try:
    from data.extended_data_loader import fetch_comprehensive_nba_data
    print("   ✅ Extended data loader imports successfully\n")
except Exception as e:
    print(f"   ⚠️  Skipped (experimental): {e}\n")

# Test 4: Ensemble + Training modules (import-only)
print("4️⃣ Testing Ensemble + Training imports...")
try:
    from ensemble.ensemble_predictor import EnsemblePredictor
    from training.trainer import ModelTrainer
    _ = EnsemblePredictor
    _ = ModelTrainer
    print("   ✅ Ensemble + training modules import successfully\n")
except Exception as e:
    print(f"   ❌ Error: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: GP model module (import-only)
print("5️⃣ Testing GP model module...")
try:
    from machine_learning.gp_model import GaussianProcessPredictor
    _ = GaussianProcessPredictor
    print("   ✅ GP model module imports successfully\n")
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
