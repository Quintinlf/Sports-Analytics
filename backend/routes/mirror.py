"""You vs the AI — API router.

The owner's picks default to the model's (see ``backend.model_mirror``), so the
scoreboard needs no background job and writes nothing: it is computed from the
predictions table and whatever reviews the owner actually submitted.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Query
from sqlalchemy import text

from backend.db import engine, get_db_session
from backend.model_mirror import load_games, owner_ref, scoreboard
from scripts.db_utils import _column_names

router = APIRouter(prefix="/api/mirror", tags=["model-mirror"])


def _resolve_owner(session, ref: str) -> Optional[Dict[str, Any]]:
    """Exact id first, then the canonical account for that name.

    Same rule as the feedback platform's sign-in: among rows sharing a name,
    the oldest one with an email is the real account. Other rows with the same
    name are reported, not merged, so a split identity stays visible.
    """
    rows = session.execute(
        text("""
            SELECT reviewer_id, name FROM reviewers
            WHERE reviewer_id = :ref OR lower(trim(name)) = lower(trim(:ref))
            ORDER BY CASE WHEN reviewer_id = :ref THEN 0 ELSE 1 END,
                     CASE WHEN email IS NOT NULL AND trim(email) != '' THEN 0 ELSE 1 END,
                     created_at ASC
        """),
        {"ref": ref},
    ).mappings().all()
    if not rows:
        return None
    owner = dict(rows[0])
    owner["same_name_accounts"] = len(rows) - 1
    return owner


@router.get("/scoreboard")
def get_scoreboard(
    reviewer: Optional[str] = Query(default=None, description="Reviewer id or name"),
    recent: int = Query(default=10, ge=1, le=50),
) -> Dict[str, Any]:
    ref = (reviewer or owner_ref()).strip()
    with get_db_session() as session:
        owner = _resolve_owner(session, ref)
        games = load_games(session, owner["reviewer_id"] if owner else None,
                           _column_names(engine, "predictions"))
    board = scoreboard(games, recent=recent)
    board["owner"] = {
        "ref": ref,
        "found": owner is not None,
        "reviewer_id": owner["reviewer_id"] if owner else None,
        "name": owner["name"] if owner else None,
        "same_name_accounts": owner["same_name_accounts"] if owner else 0,
    }
    return board
