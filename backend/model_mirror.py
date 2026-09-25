"""You against the model: the owner's picks default to the model's.

The owner account (``MODEL_MIRROR_REVIEWER``, default ``quintin``) has a pick
on every game without submitting anything: where there is no manual review, the
owner's pick *is* the model's pick. Nothing is written for those games. The
default is resolved when the scoreboard is computed, so analyst stats, weekly
emails and the postgame queue only ever see picks a person actually made.

Only overrides can separate the two records. A game the owner left alone is the
AI against itself — same pick, same result — so ``you_correct - ai_correct``
always equals ``overrides won - overrides lost``.

An override placed after the start time is not scored as the owner's; that game
falls back to the model's pick, because a pick made once the game is under way
is not a prediction. A game with no known start time is scored but counted as
``unverified_timing`` so the reader knows the check could not be made.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from math import comb
from zoneinfo import ZoneInfo
from typing import Any, Dict, Iterable, List, Mapping, Optional

from data.prediction_time import pacific_today, parse_start_time_utc

_PACIFIC = ZoneInfo("America/Los_Angeles")

#: Who the mirror belongs to. Set MODEL_MIRROR_REVIEWER to a reviewer id or name.
DEFAULT_OWNER = "quintin"

AUTO = "auto"          # no manual pick: the owner's pick is the model's
AGREED = "agreed"      # manual pick, same team as the model
OVERRODE = "overrode"  # manual pick, different team, placed before the start
LATE = "late"          # manual pick placed after the start: not scored


def owner_ref() -> str:
    return (os.getenv("MODEL_MIRROR_REVIEWER") or DEFAULT_OWNER).strip()


def _norm(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _has_clock_time(raw: Any) -> bool:
    """A date-only start ('2026-09-24') says nothing about tip-off.

    Nor does exactly 12:00:00 Pacific: ``data.prediction_time`` stores that as a
    stand-in when a provider gives only a date. Treating it as real would mark a
    3 pm pick on a 7 pm game "late", so it counts as unknown (scored, flagged
    unverified) -- which also covers the occasional real noon kickoff safely.
    """
    if raw is None:
        return False
    if isinstance(raw, datetime):
        start = raw
    else:
        text = str(raw).strip()
        if len(text) <= 10:
            return False
        start = parse_start_time_utc(text)
        if start is None:
            return False
    local = start.astimezone(_PACIFIC) if start.tzinfo else start.replace(tzinfo=timezone.utc).astimezone(_PACIFIC)
    return (local.hour, local.minute, local.second) != (12, 0, 0)


@dataclass(frozen=True)
class Game:
    """One prediction and the owner's pick on it."""

    prediction_id: int
    sport: str
    matchup: str
    game_date: str
    model_pick: str
    actual: Optional[str]
    manual_pick: Optional[str] = None
    picked_at: Optional[datetime] = None
    start: Optional[datetime] = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Game":
        start_raw = row.get("start_time_utc")
        return cls(
            prediction_id=int(row["prediction_id"]),
            sport=str(row.get("sport") or ""),
            matchup=f"{row.get('away_team')} @ {row.get('home_team')}",
            game_date=str(row.get("game_date") or ""),
            model_pick=str(row.get("predicted_winner") or ""),
            actual=row.get("actual_winner"),
            manual_pick=row.get("reviewer_pick"),
            picked_at=parse_start_time_utc(row.get("picked_at")),
            start=parse_start_time_utc(start_raw) if _has_clock_time(start_raw) else None,
        )

    @property
    def source(self) -> str:
        if not _norm(self.manual_pick):
            return AUTO
        if self.start is not None and self.picked_at is not None and self.picked_at >= self.start:
            return LATE
        if _norm(self.manual_pick) == _norm(self.model_pick):
            return AGREED
        return OVERRODE

    @property
    def pick(self) -> str:
        """The owner's scored pick."""
        return self.manual_pick if self.source in (AGREED, OVERRODE) else self.model_pick

    @property
    def timing_verified(self) -> bool:
        return self.start is not None

    @property
    def settled(self) -> bool:
        return bool(_norm(self.actual))

    @property
    def you_correct(self) -> bool:
        return self.settled and _norm(self.pick) == _norm(self.actual)

    @property
    def ai_correct(self) -> bool:
        return self.settled and _norm(self.model_pick) == _norm(self.actual)

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "prediction_id": self.prediction_id,
            "sport": self.sport,
            "matchup": self.matchup,
            "game_date": self.game_date,
            "model_pick": self.model_pick,
            "your_pick": self.pick,
            "source": self.source,
            "actual_winner": self.actual,
        }
        if self.settled:
            out["you_correct"] = self.you_correct
            out["ai_correct"] = self.ai_correct
        return out


def sign_test_p(wins: int, losses: int) -> Optional[float]:
    """Two-sided exact binomial test that overrides are a coin flip.

    Only decisive overrides count (one side right, the other wrong). Under the
    null that you and the model are equally good on the games where you
    disagree, wins ~ Binomial(wins + losses, 1/2).
    """
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def _verdict(lead: int, decisive: int) -> str:
    if decisive == 0:
        return "Level: every scored game is the AI against itself."
    if lead > 0:
        return f"You're ahead of the AI by {lead}."
    if lead < 0:
        return f"The AI is ahead of you by {-lead}."
    return "Level: your overrides have won as often as they've lost."


def _soonest_first(game: Game) -> tuple:
    timestamp = game.start.timestamp() if game.start else float("inf")
    return (game.game_date, timestamp, game.prediction_id)


def scoreboard(games: Iterable[Game], *, recent: int = 10,
               today: Optional[str] = None) -> Dict[str, Any]:
    """Head-to-head record of the owner against the model on settled games.

    ``upcoming`` lists unsettled games dated ``today`` (Pacific) or later; an
    older game that never settled is stale, not upcoming.
    """
    games = list(games)
    today = today or pacific_today()
    settled = [g for g in games if g.settled]
    counts = {AUTO: 0, AGREED: 0, OVERRODE: 0, LATE: 0}
    for g in settled:
        counts[g.source] += 1

    overrides = [g for g in settled if g.source == OVERRODE]
    won = sum(1 for g in overrides if g.you_correct and not g.ai_correct)
    lost = sum(1 for g in overrides if g.ai_correct and not g.you_correct)
    neither = len(overrides) - won - lost

    you = sum(1 for g in settled if g.you_correct)
    ai = sum(1 for g in settled if g.ai_correct)

    by_sport: Dict[str, Dict[str, int]] = {}
    for g in settled:
        row = by_sport.setdefault(g.sport, {"games": 0, "you_correct": 0, "ai_correct": 0,
                                             "overrides": 0})
        row["games"] += 1
        row["you_correct"] += g.you_correct
        row["ai_correct"] += g.ai_correct
        row["overrides"] += g.source == OVERRODE

    head_to_head = sorted(overrides, key=lambda g: (g.game_date, g.prediction_id), reverse=True)
    upcoming = sorted((g for g in games if not g.settled and g.game_date[:10] >= today),
                      key=_soonest_first)

    return {
        "settled_games": len(settled),
        "you_correct": you,
        "ai_correct": ai,
        "lead": you - ai,
        "verdict": _verdict(you - ai, won + lost),
        "picks": {
            "auto": counts[AUTO],
            "agreed": counts[AGREED],
            "overrode": counts[OVERRODE],
            "late_not_scored": counts[LATE],
        },
        "overrides": {"won": won, "lost": lost, "neither_right": neither,
                      "p_value": sign_test_p(won, lost)},
        "unverified_timing": sum(1 for g in overrides if not g.timing_verified),
        "by_sport": by_sport,
        "recent_overrides": [g.to_dict() for g in head_to_head[:recent]],
        "upcoming": [g.to_dict() for g in upcoming[:recent]],
    }


def load_games(session, reviewer_id: Optional[str], column_names: set[str]) -> List[Game]:
    """Every non-void prediction joined to the owner's review, if any."""
    from sqlalchemy import text

    start_col = "p.start_time_utc" if "start_time_utc" in column_names else "NULL"
    sql = text(f"""
        SELECT p.prediction_id, p.sport, p.home_team, p.away_team, p.game_date,
               {start_col} AS start_time_utc, p.predicted_winner, p.actual_winner,
               pr.reviewer_pick, pr.created_at AS picked_at
        FROM predictions p
        LEFT JOIN prediction_reviews pr
               ON pr.prediction_id = p.prediction_id AND pr.reviewer_id = :rid
        WHERE UPPER(COALESCE(p.prediction_status, '')) <> 'VOID'
    """)
    rows = session.execute(sql, {"rid": reviewer_id or ""}).mappings().all()
    return [Game.from_row(row) for row in rows]
