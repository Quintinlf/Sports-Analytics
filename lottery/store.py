"""Local cache for fetched prize breakdowns.

Draw histories come from an open-data API that is happy to serve them
repeatedly. Prize breakdowns come from one page fetch per draw, so they are
cached permanently: a breakdown is immutable once a drawing is settled, and
re-fetching hundreds of pages to recompute a regression would be rude.
"""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from typing import Iterable, Optional, Sequence

from lottery.prizes import PrizeBreakdown, TierResult

__all__ = ["DEFAULT_PATH", "connect", "save_breakdowns", "load_breakdowns", "cached_dates"]

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "lottery_history.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS prize_breakdowns (
    game_key        TEXT NOT NULL,
    draw_date       TEXT NOT NULL,
    matched_white   INTEGER NOT NULL,
    matched_special INTEGER NOT NULL,
    winners         INTEGER NOT NULL,
    prize           INTEGER,
    PRIMARY KEY (game_key, draw_date, matched_white, matched_special)
);
"""


def connect(path: Optional[Path] = None) -> sqlite3.Connection:
    target = Path(path or DEFAULT_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.executescript(SCHEMA)
    return conn


def save_breakdowns(
    conn: sqlite3.Connection, breakdowns: Iterable[PrizeBreakdown]
) -> int:
    """Persist breakdowns. Returns how many draws were written."""
    count = 0
    for breakdown in breakdowns:
        conn.executemany(
            "INSERT OR REPLACE INTO prize_breakdowns VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    breakdown.game_key,
                    breakdown.draw_date.isoformat(),
                    tier.matched_white,
                    int(tier.matched_special),
                    tier.winners,
                    tier.prize,
                )
                for tier in breakdown.tiers
            ],
        )
        count += 1
        conn.commit()
    return count


def cached_dates(conn: sqlite3.Connection, game_key: str) -> set[date]:
    rows = conn.execute(
        "SELECT DISTINCT draw_date FROM prize_breakdowns WHERE game_key = ?",
        (game_key,),
    ).fetchall()
    return {date.fromisoformat(row[0]) for row in rows}


def load_breakdowns(
    conn: sqlite3.Connection, game_key: str, *, only: Optional[Sequence[date]] = None
) -> list[PrizeBreakdown]:
    """Load cached breakdowns, oldest first. Incomplete draws are skipped."""
    rows = conn.execute(
        "SELECT draw_date, matched_white, matched_special, winners, prize "
        "FROM prize_breakdowns WHERE game_key = ? ORDER BY draw_date",
        (game_key,),
    ).fetchall()

    wanted = set(only) if only is not None else None
    grouped: dict[date, list[TierResult]] = {}
    for raw_date, matched_white, matched_special, winners, prize in rows:
        when = date.fromisoformat(raw_date)
        if wanted is not None and when not in wanted:
            continue
        grouped.setdefault(when, []).append(
            TierResult(matched_white, bool(matched_special), winners, prize)
        )

    return [
        PrizeBreakdown(game_key, when, tuple(tiers))
        for when, tiers in sorted(grouped.items())
        if len(tiers) == 9
    ]
