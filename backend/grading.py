"""Grade every pick after the fact and keep the record.

Once a game settles, each pick on it is checked against the real winner and
stored in ``pick_grades``: every reviewer's submitted pick, and the owner's
default picks (which are the model's picks, see ``backend.model_mirror``). The
table is derived and idempotent -- re-running only writes what changed -- so it
can be refreshed from the daily cron and from the leaderboard request alike.

The leaderboard compares each person with the model *on the games that person
picked*, so someone who reviewed ten games is not measured against the model's
record on a thousand.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import text

from backend.model_mirror import AUTO, LATE, OVERRODE, Game, _norm, owner_ref, sign_test_p


def _start_select(column_names: set[str]) -> str:
    return "p.start_time_utc" if "start_time_utc" in column_names else "NULL"


def _settled_where() -> str:
    return ("p.actual_winner IS NOT NULL AND TRIM(p.actual_winner) <> '' "
            "AND UPPER(COALESCE(p.prediction_status, '')) <> 'VOID'")


def resolve_owner_id(conn, ref: Optional[str] = None) -> Optional[str]:
    ref = (ref or owner_ref()).strip()
    row = conn.execute(
        text("""
            SELECT reviewer_id FROM reviewers
            WHERE reviewer_id = :ref OR lower(trim(name)) = lower(trim(:ref))
            ORDER BY CASE WHEN reviewer_id = :ref THEN 0 ELSE 1 END,
                     CASE WHEN email IS NOT NULL AND trim(email) != '' THEN 0 ELSE 1 END,
                     created_at ASC
            LIMIT 1
        """),
        {"ref": ref},
    ).first()
    return row[0] if row else None


def compute_grades(conn, column_names: set[str], owner_id: Optional[str]) -> List[Dict[str, Any]]:
    """Every gradeable pick on a settled game, as rows for ``pick_grades``."""
    game_cols = (f"p.prediction_id, p.sport, p.home_team, p.away_team, p.game_date, "
                 f"{_start_select(column_names)} AS start_time_utc, p.predicted_winner, p.actual_winner")
    reviews = conn.execute(text(f"""
        SELECT {game_cols}, pr.reviewer_id, pr.reviewer_pick, pr.created_at AS picked_at
        FROM prediction_reviews pr JOIN predictions p ON p.prediction_id = pr.prediction_id
        WHERE {_settled_where()}
    """)).mappings().all()

    rows: List[Dict[str, Any]] = []
    reviewed_by_owner = set()
    for r in reviews:
        g = Game.from_row(r)
        if r["reviewer_id"] == owner_id:
            reviewed_by_owner.add(g.prediction_id)
        rows.append(_grade_row(r["reviewer_id"], g, g.manual_pick))

    if owner_id:
        settled = conn.execute(text(
            f"SELECT {game_cols} FROM predictions p WHERE {_settled_where()}")).mappings().all()
        for r in settled:
            if r["prediction_id"] not in reviewed_by_owner:
                g = Game.from_row(r)
                rows.append(_grade_row(owner_id, g, g.model_pick))
    return rows


def _grade_row(reviewer_id: str, g: Game, pick: Optional[str]) -> Dict[str, Any]:
    pick = pick or g.model_pick
    return {
        "reviewer_id": reviewer_id,
        "prediction_id": g.prediction_id,
        "sport": g.sport,
        "game_date": g.game_date[:10],
        "pick": pick,
        "source": g.source,
        "model_pick": g.model_pick,
        "actual_winner": str(g.actual),
        "correct": _norm(pick) == _norm(g.actual),
        "model_correct": g.ai_correct,
    }


_FIELDS = ("sport", "game_date", "pick", "source", "model_pick", "actual_winner", "correct", "model_correct")


def grade_sports(engine, column_names: set[str], owner: Optional[str] = None) -> Dict[str, int]:
    """Bring ``pick_grades`` up to date. Returns counts of rows inserted and updated."""
    with engine.begin() as conn:
        owner_id = resolve_owner_id(conn, owner)
        rows = compute_grades(conn, column_names, owner_id)
        existing = {
            (r["reviewer_id"], r["prediction_id"]): r
            for r in conn.execute(text("SELECT * FROM pick_grades")).mappings()
        }
        inserted = updated = 0
        for row in rows:
            key = (row["reviewer_id"], row["prediction_id"])
            old = existing.get(key)
            if old is None:
                conn.execute(text("""
                    INSERT INTO pick_grades (reviewer_id, prediction_id, sport, game_date, pick, source,
                                             model_pick, actual_winner, correct, model_correct, graded_at)
                    VALUES (:reviewer_id, :prediction_id, :sport, :game_date, :pick, :source,
                            :model_pick, :actual_winner, :correct, :model_correct, CURRENT_TIMESTAMP)
                """), row)
                inserted += 1
            elif any(_differs(old[f], row[f]) for f in _FIELDS):
                conn.execute(text("""
                    UPDATE pick_grades SET sport = :sport, game_date = :game_date, pick = :pick,
                        source = :source, model_pick = :model_pick, actual_winner = :actual_winner,
                        correct = :correct, model_correct = :model_correct, graded_at = CURRENT_TIMESTAMP
                    WHERE reviewer_id = :reviewer_id AND prediction_id = :prediction_id
                """), row)
                updated += 1
    return {"inserted": inserted, "updated": updated, "graded": len(rows)}


def _differs(old: Any, new: Any) -> bool:
    if isinstance(new, bool):
        return bool(old) != new
    return str(old or "") != str(new or "")


def leaderboard(conn, owner_id: Optional[str]) -> List[Dict[str, Any]]:
    """Each person against the model, on the games that person picked."""
    names = {r[0]: r[1] for r in conn.execute(text("SELECT reviewer_id, name FROM reviewers"))}
    rows = conn.execute(text(
        f"SELECT reviewer_id, source, correct, model_correct FROM pick_grades WHERE source <> '{LATE}'"
    )).all()
    board: Dict[str, Dict[str, Any]] = {}
    for reviewer_id, source, correct, model_correct in rows:
        b = board.setdefault(reviewer_id, {"picks": 0, "correct": 0, "model_correct": 0,
                                           "auto": 0, "overrides": 0, "won": 0, "lost": 0})
        correct, model_correct = bool(correct), bool(model_correct)
        b["picks"] += 1
        b["correct"] += correct
        b["model_correct"] += model_correct
        b["auto"] += source == AUTO
        if source == OVERRODE:
            b["overrides"] += 1
            b["won"] += correct and not model_correct
            b["lost"] += model_correct and not correct
    out = []
    for reviewer_id, b in board.items():
        out.append({
            "reviewer_id": reviewer_id,
            "name": names.get(reviewer_id, reviewer_id),
            "is_owner": reviewer_id == owner_id,
            **b,
            "accuracy": b["correct"] / b["picks"] if b["picks"] else None,
            "model_accuracy": b["model_correct"] / b["picks"] if b["picks"] else None,
            "lead": b["correct"] - b["model_correct"],
            "p_value": sign_test_p(b["won"], b["lost"]),
        })
    out.sort(key=lambda r: (-r["lead"], -r["picks"], r["name"] or ""))
    return out
