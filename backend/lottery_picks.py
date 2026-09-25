"""Powerball: the model's ticket, your ticket, and what each actually won.

No ticket is more likely to win than any other -- ``lottery/README.md`` shows
the history is consistent with a fair machine. So the model does not try to
predict numbers. It picks the least-crowded ticket among a few thousand random
ones, which leaves the odds exactly where they were and only improves what a
jackpot would pay if it hit (fewer people to split with).

Its ticket is a seeded function of the game and draw date, so it is fixed in
advance: storing it late cannot change it. The owner's ticket defaults to the
model's, like the sports picks; anyone can commit their own before the draw.
After the draw, every stored ticket is graded against the published numbers.
"""
from __future__ import annotations

import random
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence
from zoneinfo import ZoneInfo

from sqlalchemy import text

from lottery.games import POWERBALL, Game
from lottery.history import Draw
from lottery.popularity import PopularityModel

EASTERN = ZoneInfo("America/New_York")

#: Powerball draws Monday, Wednesday and Saturday at 10:59 pm Eastern.
DRAW_SCHEDULE = {"powerball": ((0, 2, 5), time(22, 59))}
CANDIDATES = 2000
MODEL_OWNER = "model"


def draw_datetime(game: Game, draw_day: date) -> datetime:
    return datetime.combine(draw_day, DRAW_SCHEDULE[game.key][1], tzinfo=EASTERN)


def next_draw(game: Game, now: Optional[datetime] = None) -> date:
    """The next draw whose drawing time has not passed."""
    weekdays, _ = DRAW_SCHEDULE[game.key]
    now = (now or datetime.now(EASTERN)).astimezone(EASTERN)
    day = now.date()
    for _ in range(8):
        if day.weekday() in weekdays and draw_datetime(game, day) > now:
            return day
        day += timedelta(days=1)
    raise RuntimeError("no draw within a week")


def format_whites(whites: Sequence[int]) -> str:
    return " ".join(f"{n:02d}" for n in sorted(whites))


def parse_whites(raw: str) -> tuple[int, ...]:
    return tuple(sorted(int(x) for x in raw.split()))


@lru_cache(maxsize=8)
def _crowd(matrix) -> PopularityModel:
    return PopularityModel(matrix)   # its normaliser is a 40,000-ticket simulation; build once


def model_ticket(game: Game, draw_day: date) -> Dict[str, Any]:
    """The least-crowded of CANDIDATES random tickets, seeded by the draw date."""
    matrix = game.era_for(draw_day).matrix
    crowd = _crowd(matrix)
    rng = random.Random(f"{game.key}:{draw_day.isoformat()}")
    best = None
    for _ in range(CANDIDATES):
        whites = tuple(sorted(rng.sample(range(1, matrix.white_max + 1), matrix.white_count)))
        score = crowd.crowd_score(whites)
        if best is None or score < best[1]:
            best = (whites, score)
    special = rng.randint(1, matrix.special_max)
    return {"whites": best[0], "special": special, "crowd_score": best[1]}


def validate_ticket(game: Game, draw_day: date, whites: Sequence[int], special: int) -> tuple[int, ...]:
    matrix = game.era_for(draw_day).matrix
    picks = tuple(sorted(int(n) for n in whites))
    if len(picks) != matrix.white_count or len(set(picks)) != len(picks):
        raise ValueError(f"pick {matrix.white_count} different white balls")
    if not all(1 <= n <= matrix.white_max for n in picks):
        raise ValueError(f"white balls run 1-{matrix.white_max}")
    if not 1 <= int(special) <= matrix.special_max:
        raise ValueError(f"the {matrix.special_name} runs 1-{matrix.special_max}")
    return picks


def ensure_model_pick(conn, game: Game, draw_day: date) -> None:
    exists = conn.execute(
        text("SELECT 1 FROM lottery_picks WHERE game_key = :g AND draw_date = :d AND owner = :o"),
        {"g": game.key, "d": draw_day.isoformat(), "o": MODEL_OWNER},
    ).first()
    if exists:
        return
    t = model_ticket(game, draw_day)
    _insert(conn, game, draw_day, MODEL_OWNER, "model", t["whites"], t["special"], t["crowd_score"])


def _insert(conn, game, draw_day, owner, source, whites, special, crowd_score) -> None:
    import uuid

    conn.execute(text("""
        INSERT INTO lottery_picks (pick_id, game_key, draw_date, owner, source, whites, special,
                                   crowd_score, created_at)
        VALUES (:id, :g, :d, :o, :s, :w, :sp, :c, CURRENT_TIMESTAMP)
    """), {"id": str(uuid.uuid4()), "g": game.key, "d": draw_day.isoformat(), "o": owner,
           "s": source, "w": format_whites(whites), "sp": int(special), "c": crowd_score})


def submit_pick(conn, game: Game, owner: str, whites: Sequence[int], special: int,
                now: Optional[datetime] = None) -> date:
    """Commit (or replace) ``owner``'s ticket for the next draw, before it is drawn."""
    draw_day = next_draw(game, now)
    picks = validate_ticket(game, draw_day, whites, special)
    crowd = _crowd(game.era_for(draw_day).matrix).crowd_score(picks)
    conn.execute(text("DELETE FROM lottery_picks WHERE game_key = :g AND draw_date = :d AND owner = :o"),
                 {"g": game.key, "d": draw_day.isoformat(), "o": owner})
    _insert(conn, game, draw_day, owner, "manual", picks, special, crowd)
    return draw_day


def prize_for(game: Game, draw_day: date, matched_white: int, matched_special: bool) -> tuple[Optional[int], bool]:
    """(fixed prize in dollars or None, is_jackpot). Power Play is not modelled."""
    for tier in game.prize_tiers:
        if tier.matched_white == matched_white and tier.matched_special == matched_special:
            return (None, True) if tier.is_jackpot else (tier.prize, False)
    return 0, False


def grade(conn, game: Game, draws: Iterable[Draw]) -> int:
    """Grade every stored ticket whose draw is now published. Returns how many."""
    by_date = {d.draw_date.isoformat(): d for d in draws if d.game_key == game.key}
    pending = conn.execute(
        text("SELECT pick_id, draw_date, whites, special FROM lottery_picks "
             "WHERE game_key = :g AND graded_at IS NULL"),
        {"g": game.key},
    ).mappings().all()
    graded = 0
    for row in pending:
        draw = by_date.get(row["draw_date"])
        if draw is None:
            continue
        whites = set(parse_whites(row["whites"]))
        matched_white = len(whites & set(draw.whites))
        matched_special = int(row["special"]) == draw.special
        prize, jackpot = prize_for(game, draw.draw_date, matched_white, matched_special)
        conn.execute(text("""
            UPDATE lottery_picks SET drawn_whites = :dw, drawn_special = :ds, matched_white = :mw,
                matched_special = :ms, prize = :p, jackpot = :j, graded_at = CURRENT_TIMESTAMP
            WHERE pick_id = :id
        """), {"dw": format_whites(draw.whites), "ds": draw.special, "mw": matched_white,
               "ms": matched_special, "p": prize, "j": jackpot, "id": row["pick_id"]})
        graded += 1
    return graded


def ungraded_since(conn, game: Game) -> Optional[str]:
    today = datetime.now(EASTERN).date().isoformat()
    row = conn.execute(
        text("SELECT MIN(draw_date) FROM lottery_picks WHERE game_key = :g AND graded_at IS NULL "
             "AND draw_date <= :today"),
        {"g": game.key, "today": today},
    ).first()
    return row[0] if row and row[0] else None


def ledger(rows: List[Dict[str, Any]], ticket_price: float) -> Dict[str, Any]:
    graded = [r for r in rows if r.get("graded_at")]
    won = sum(r["prize"] or 0 for r in graded)
    return {
        "tickets": len(graded),
        "spent": len(graded) * ticket_price,
        "won": won,
        "net": won - len(graded) * ticket_price,
        "jackpots": sum(1 for r in graded if r.get("jackpot")),
        "best_match": max(((r["matched_white"], bool(r["matched_special"])) for r in graded), default=None),
    }


def effective_owner_rows(model_rows: List[Dict[str, Any]], owner_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The owner's tickets: their own where they picked, the model's everywhere else."""
    mine = {r["draw_date"]: {**r, "source": "manual"} for r in owner_rows}
    out = []
    for r in model_rows:
        out.append(mine.pop(r["draw_date"], {**r, "source": "auto"}))
    out.extend(mine.values())
    return sorted(out, key=lambda r: r["draw_date"], reverse=True)


__all__ = ["POWERBALL", "next_draw", "model_ticket", "submit_pick", "grade", "ensure_model_pick",
           "ledger", "effective_owner_rows", "ungraded_since", "draw_datetime", "validate_ticket"]
