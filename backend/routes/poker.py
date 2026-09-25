"""Interactive poker laboratory — API router.

Endpoints under /api/poker/ (all additive; no existing route is touched).

Session storage is in-memory for Milestone 1. Sessions are keyed by an
unguessable UUID, matching this application's existing lightweight identity
model — there is no auth layer to hook into. Hand persistence arrives with the
replay/analysis milestone, at which point these payloads are written to the
database rather than discarded.
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from poker.engine import GameConfig
from poker.engine.rules import IllegalAction
from poker.opponents import PROFILES
from poker.session import LearningMode, PokerSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/poker", tags=["poker-laboratory"])

#: Oldest sessions are evicted past this many, so a long-lived web process
#: does not grow without bound.
MAX_SESSIONS = 200

_sessions: "OrderedDict[str, PokerSession]" = OrderedDict()
# Sessions are mutable and FastAPI runs sync handlers on a threadpool, so
# concurrent requests to one session must not interleave.
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    opponent_profile: str = Field(default="balanced")
    small_blind: int = Field(default=1, ge=1, le=10_000)
    big_blind: int = Field(default=2, ge=1, le=20_000)
    starting_stack: int = Field(default=200, ge=2, le=1_000_000)
    hero_name: str = Field(default="You", max_length=40)
    learning_mode: str = Field(default=LearningMode.BEGINNER)
    seed: Optional[int] = Field(default=None)


class ActionRequest(BaseModel):
    action: str = Field(description="fold | check | call | bet | raise")
    amount: int = Field(default=0, ge=0, le=100_000_000)


class LearningModeRequest(BaseModel):
    learning_mode: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_session(session_id: str) -> PokerSession:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired. Start a new one.",
        )
    _sessions.move_to_end(session_id)
    return session


def _evict_if_needed() -> None:
    while len(_sessions) > MAX_SESSIONS:
        evicted, _ = _sessions.popitem(last=False)
        logger.info("Evicted poker session %s (capacity)", evicted)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/opponents")
def list_opponents() -> Dict[str, List[Dict[str, Any]]]:
    """Available opponent archetypes and their behavioural parameters."""
    return {"opponents": [profile.to_dict() for profile in PROFILES.values()]}


@router.get("/learning-modes")
def list_learning_modes() -> Dict[str, List[Dict[str, str]]]:
    return {
        "modes": [
            {"key": key, "description": LearningMode.DESCRIPTIONS[key]}
            for key in LearningMode.ALL
        ]
    }


@router.post("/sessions")
def create_session(payload: CreateSessionRequest) -> Dict[str, Any]:
    """Sit down at a new heads-up table and deal the first hand."""
    if payload.small_blind > payload.big_blind:
        raise HTTPException(400, "Small blind cannot exceed the big blind")
    if payload.starting_stack < payload.big_blind:
        raise HTTPException(400, "Starting stack must cover the big blind")

    try:
        config = GameConfig(
            small_blind=payload.small_blind,
            big_blind=payload.big_blind,
            starting_stack=payload.starting_stack,
        )
        with _lock:
            session = PokerSession(
                config=config,
                opponent_profile=payload.opponent_profile,
                hero_name=payload.hero_name or "You",
                learning_mode=payload.learning_mode,
                seed=payload.seed,
            )
            session.start_hand()
            _sessions[session.session_id] = session
            _evict_if_needed()
            return session.to_dict()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/sessions/{session_id}")
def get_session(session_id: str) -> Dict[str, Any]:
    with _lock:
        return _get_session(session_id).to_dict()


@router.post("/sessions/{session_id}/hands")
def deal_hand(session_id: str) -> Dict[str, Any]:
    """Deal the next hand. The button alternates automatically."""
    with _lock:
        session = _get_session(session_id)
        try:
            session.start_hand()
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return session.to_dict()


@router.post("/sessions/{session_id}/actions")
def submit_action(session_id: str, payload: ActionRequest) -> Dict[str, Any]:
    """Submit the hero's action; the opponent responds before this returns."""
    with _lock:
        session = _get_session(session_id)
        try:
            session.hero_action(payload.action, payload.amount)
        except IllegalAction as exc:
            # A rules violation is a client error, not a server fault.
            raise HTTPException(422, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return session.to_dict()


@router.post("/sessions/{session_id}/learning-mode")
def set_learning_mode(session_id: str, payload: LearningModeRequest) -> Dict[str, Any]:
    with _lock:
        session = _get_session(session_id)
        try:
            session.set_learning_mode(payload.learning_mode)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return session.to_dict()


@router.delete("/sessions/{session_id}")
def end_session(session_id: str) -> Dict[str, str]:
    with _lock:
        _get_session(session_id)
        del _sessions[session_id]
    return {"status": "ended", "session_id": session_id}


@router.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "active_sessions": len(_sessions),
        "opponent_profiles": sorted(PROFILES),
    }
