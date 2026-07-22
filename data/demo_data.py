"""Shared gate for non-production demo/placeholder prediction data.

Both the FastAPI startup seed (backend/routes/feedback.py) and FIFA's
off-season fabricated-fixture fallback (data/fifa_predictions_service.py)
insert synthetic rows into the same `predictions` table real live
predictions land in. Production must never do this silently — demo data is
only ever written when ENABLE_DEMO_PREDICTIONS is explicitly set.
"""
from __future__ import annotations

import os

_TRUTHY = {"1", "true", "yes", "on"}


def demo_predictions_enabled() -> bool:
    """True only when ENABLE_DEMO_PREDICTIONS is explicitly set truthy.

    Defaults to False so demo/seed/fabricated rows never reach a production
    database by accident.
    """
    return os.getenv("ENABLE_DEMO_PREDICTIONS", "").strip().lower() in _TRUTHY
