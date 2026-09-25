"""Powerball: model ticket vs your ticket, graded after every draw — API router."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text

from backend.db import engine
from backend.grading import resolve_owner_id
from backend.lottery_picks import (
    EASTERN, MODEL_OWNER, POWERBALL, draw_datetime, effective_owner_rows, ensure_model_pick,
    grade, ledger, next_draw, submit_pick, ungraded_since,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/lottery", tags=["lottery"])


class TicketRequest(BaseModel):
    whites: List[int] = Field(min_length=5, max_length=5)
    special: int
    reviewer_id: Optional[str] = None


def _backfill_model_picks(conn, through: date) -> None:
    """Fill any draws the model missed since its first stored ticket.

    Safe after the fact only because the model's ticket is a seeded function
    of the draw date: it is the same whenever it is computed.
    """
    first = conn.execute(text("SELECT MIN(draw_date) FROM lottery_picks WHERE game_key = :g AND owner = :o"),
                         {"g": POWERBALL.key, "o": MODEL_OWNER}).scalar()
    if not first:
        return
    day = date.fromisoformat(first)
    weekdays = (0, 2, 5)
    while day <= through:
        if day.weekday() in weekdays:
            ensure_model_pick(conn, POWERBALL, day)
        day += timedelta(days=1)


def refresh(conn, fetch=None) -> Dict[str, Any]:
    """Commit the model's next ticket and grade whatever has been drawn."""
    upcoming = next_draw(POWERBALL)
    ensure_model_pick(conn, POWERBALL, upcoming)
    _backfill_model_picks(conn, upcoming)
    status: Dict[str, Any] = {"graded_now": 0, "fetch_error": None}
    since = ungraded_since(conn, POWERBALL)
    if since:
        try:
            if fetch is None:
                from lottery.sources import fetch_draws
                fetch = fetch_draws
            start = (date.fromisoformat(since) - timedelta(days=1)).isoformat()
            status["graded_now"] = grade(conn, POWERBALL, fetch(POWERBALL, since=start, timeout=10))
        except Exception as exc:  # the page still works when the source is down
            logger.warning("Powerball grading skipped: %s", exc)
            status["fetch_error"] = str(exc)
    return status


def _rows(conn, owner: str) -> List[Dict[str, Any]]:
    return [dict(r) for r in conn.execute(
        text("SELECT * FROM lottery_picks WHERE game_key = :g AND owner = :o ORDER BY draw_date DESC"),
        {"g": POWERBALL.key, "o": owner},
    ).mappings()]


def _public(row: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("draw_date", "source", "whites", "special", "crowd_score", "drawn_whites", "drawn_special",
            "matched_white", "matched_special", "prize", "jackpot")
    out = {k: row.get(k) for k in keys}
    out["graded"] = row.get("graded_at") is not None
    return out


@router.get("/powerball")
def powerball(reviewer: Optional[str] = None) -> Dict[str, Any]:
    with engine.begin() as conn:
        status = refresh(conn)
        owner_id = resolve_owner_id(conn, reviewer)
        model_rows = _rows(conn, MODEL_OWNER)
        owner_rows = _rows(conn, owner_id) if owner_id else []
    upcoming = next_draw(POWERBALL)
    mine = effective_owner_rows(model_rows, owner_rows)
    price = POWERBALL.era_for(upcoming).ticket_price
    return {
        "game": POWERBALL.name,
        "next_draw": upcoming.isoformat(),
        "draw_time": draw_datetime(POWERBALL, upcoming).isoformat(),
        "odds_jackpot": POWERBALL.era_for(upcoming).matrix.total_combinations,
        "ticket_price": price,
        "model_next": next((_public(r) for r in model_rows if r["draw_date"] == upcoming.isoformat()), None),
        "your_next": next((_public(r) for r in mine if r["draw_date"] == upcoming.isoformat()), None),
        "owner": {"reviewer_id": owner_id},
        "history": [_public(r) for r in mine if r["draw_date"] < upcoming.isoformat()][:30],
        "ledger": {"model": ledger(model_rows, price), "you": ledger(mine, price)},
        "status": status,
        "note": ("Every ticket has the same 1 in 292,201,338 jackpot odds. The model picks the "
                 "least-crowded ticket, which only changes how many people you would split with."),
    }


@router.post("/powerball/picks")
def commit_ticket(payload: TicketRequest) -> Dict[str, Any]:
    with engine.begin() as conn:
        owner_id = resolve_owner_id(conn, payload.reviewer_id)
        if not owner_id:
            raise HTTPException(404, "Reviewer not found")
        try:
            draw_day = submit_pick(conn, POWERBALL, owner_id, payload.whites, payload.special)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    return {"status": "saved", "reviewer_id": owner_id, "draw_date": draw_day.isoformat(),
            "locks_at": draw_datetime(POWERBALL, draw_day).isoformat()}
