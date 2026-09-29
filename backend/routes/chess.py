"""Chess: save each game with its review, list your games, and build your style profile.

Each game keeps the browser's engine data (evaluations and best moves) next to its review, so
when the coach improves (COACH_VERSION) older games are re-reviewed the next time they're read.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text

from backend.db import engine
from backend.grading import resolve_owner_id
from backend.model_mirror import owner_ref
from chess_coach import COACH_VERSION, PRINCIPLES, build_profile, review_game

router = APIRouter(prefix="/api/chess", tags=["chess"])


class GameIn(BaseModel):
    moves: List[str] = Field(min_length=1, max_length=600)
    evals: List[Optional[float]]
    best_moves: List[Optional[str]]
    player_color: str = Field(pattern="^(white|black)$")
    result: str = Field(pattern=r"^(1-0|0-1|1/2-1/2|\*)$")
    opponent: str = Field(default="stockfish", max_length=40)
    reviewer_id: Optional[str] = None


def _owner(conn, ref: Optional[str]) -> str:
    """Same identity as picks: a reviewer if one matches, otherwise the configured owner name."""
    return resolve_owner_id(conn, ref) or (ref or owner_ref()).strip()


def _summary_columns(review: Dict[str, Any]) -> Dict[str, Any]:
    opening = review["opening"] or {}
    return {"eco": opening.get("eco"), "opening": opening.get("name"), "acpl": review["acpl"]["all"],
            "outcome": review["outcome"]}


def _current(conn, row) -> Dict[str, Any]:
    """The stored review, re-run and saved first if the coach has changed since the game was played."""
    review = json.loads(row["review"])
    inputs = review.get("engine")
    if review.get("coach_version") == COACH_VERSION or not inputs:
        return review
    try:
        review = {**review_game(row["moves_uci"].split(), inputs["evals"], inputs["best_moves"],
                                row["player_color"], row["result"]), "engine": inputs}
    except ValueError:
        return review
    conn.execute(text("""
        UPDATE chess_games SET review = :review, eco = :eco, opening = :opening, acpl = :acpl, outcome = :outcome
        WHERE game_id = :id
    """), {"id": row["game_id"], "review": json.dumps(review), **_summary_columns(review)})
    return review


@router.post("/games")
def save_game(game: GameIn) -> Dict[str, Any]:
    try:
        review = review_game(game.moves, game.evals, game.best_moves, game.player_color, game.result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    stored = {**review, "engine": {"evals": game.evals, "best_moves": game.best_moves}}
    game_id = str(uuid.uuid4())
    with engine.begin() as conn:
        owner = _owner(conn, game.reviewer_id)
        conn.execute(text("""
            INSERT INTO chess_games (game_id, owner, played_at, player_color, opponent, result, outcome,
                                     moves_uci, eco, opening, acpl, review)
            VALUES (:id, :owner, :at, :color, :opp, :result, :outcome, :moves, :eco, :opening, :acpl, :review)
        """), {
            "id": game_id, "owner": owner, "at": datetime.utcnow(), "color": game.player_color,
            "opp": game.opponent, "result": game.result, "moves": " ".join(game.moves),
            "review": json.dumps(stored), **_summary_columns(review),
        })
        played = conn.execute(text("SELECT COUNT(*) FROM chess_games WHERE owner = :o"), {"o": owner}).scalar()
    return {"game_id": game_id, "games_played": played, "review": review}


@router.get("/games")
def list_games(reviewer_id: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
    with engine.begin() as conn:
        owner = _owner(conn, reviewer_id)
        rows = conn.execute(text("""
            SELECT game_id, played_at, player_color, opponent, result, moves_uci, review
            FROM chess_games WHERE owner = :o ORDER BY played_at DESC LIMIT :n
        """), {"o": owner, "n": max(1, min(limit, 500))}).mappings().all()
        total = conn.execute(text("SELECT COUNT(*) FROM chess_games WHERE owner = :o"), {"o": owner}).scalar()
        games = []
        for r in rows:
            review = _current(conn, r)
            games.append({"game_id": r["game_id"], "played_at": str(r["played_at"]), "color": r["player_color"],
                          "opponent": r["opponent"], "result": r["result"], "outcome": review["outcome"],
                          "opening": (review["opening"] or {}).get("name"), "acpl": review["acpl"]["all"],
                          "lessons": [l["title"] for l in review.get("lessons", [])]})
    return {"owner": owner, "total": total, "games": games}


@router.get("/games/{game_id}")
def get_game(game_id: str) -> Dict[str, Any]:
    with engine.begin() as conn:
        row = conn.execute(text("SELECT * FROM chess_games WHERE game_id = :g"), {"g": game_id}).mappings().first()
        if not row:
            raise HTTPException(status_code=404, detail="Game not found")
        review = _current(conn, row)
    inputs = review.pop("engine", None) or {}
    return {"game_id": row["game_id"], "played_at": str(row["played_at"]), "opponent": row["opponent"],
            "moves": row["moves_uci"].split(), "evals": inputs.get("evals"), "review": review}


@router.get("/profile")
def profile(reviewer_id: Optional[str] = None) -> Dict[str, Any]:
    with engine.begin() as conn:
        owner = _owner(conn, reviewer_id)
        rows = conn.execute(text("""
            SELECT game_id, player_color, result, opponent, moves_uci, review
            FROM chess_games WHERE owner = :o ORDER BY played_at ASC
        """), {"o": owner}).mappings().all()
        reviews = [_current(conn, r) for r in rows]
    return {"owner": owner, "games_played": len(reviews), **build_profile(reviews),
            "by_opponent": _by_opponent([r["opponent"] for r in rows], reviews)}


def _by_opponent(opponents: List[str], reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Your record against each opponent style ("rusher-3" counts under "rusher"), most played first."""
    table: Dict[str, Dict[str, int]] = {}
    for opp, review in zip(opponents, reviews):
        key = (opp or "unknown").rsplit("-", 1)[0]
        row = table.setdefault(key, {"games": 0, "win": 0, "draw": 0, "loss": 0})
        row["games"] += 1
        if review.get("outcome") in ("win", "draw", "loss"):
            row[review["outcome"]] += 1
    return sorted(({"opponent": k, **v} for k, v in table.items()), key=lambda r: -r["games"])


@router.get("/principles")
def principles() -> Dict[str, Any]:
    out: Dict[str, List[Dict[str, str]]] = {"opening": [], "middlegame": [], "endgame": []}
    for p in PRINCIPLES:
        out[p.phase].append({"key": p.key, "title": p.title, "why": p.why, "practice": p.practice})
    return out
