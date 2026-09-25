"""Fetch lottery history and report on it.

    python scripts/lottery_report.py                          # Powerball, current era
    python scripts/lottery_report.py --game mega_millions
    python scripts/lottery_report.py --era all                # every era separately
    python scripts/lottery_report.py --ticket "38 43 52 61 67" --jackpot 1.5e9

Draw history is cached in a local SQLite file so repeated runs do not re-hit the
open-data API. Pass --refresh to pull new drawings.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lottery import randomness as R  # noqa: E402
from lottery.games import GAMES, Game, get_game  # noqa: E402
from lottery.history import Draw, filter_era, validate_draws  # noqa: E402
from lottery.popularity import PopularityModel  # noqa: E402
from lottery.sources import fetch_draws  # noqa: E402
from lottery.value import ticket_ev  # noqa: E402

CACHE_PATH = REPO_ROOT / "data" / "lottery_history.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS draws (
    game_key   TEXT NOT NULL,
    draw_date  TEXT NOT NULL,
    whites     TEXT NOT NULL,
    special    INTEGER NOT NULL,
    era_index  INTEGER NOT NULL,
    multiplier INTEGER,
    PRIMARY KEY (game_key, draw_date)
);
"""


def _connect() -> sqlite3.Connection:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_PATH)
    conn.executescript(SCHEMA)
    return conn


def _save(conn: sqlite3.Connection, draws: Sequence[Draw]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO draws VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                d.game_key,
                d.draw_date.isoformat(),
                " ".join(str(n) for n in d.whites),
                d.special,
                d.era_index,
                d.multiplier,
            )
            for d in draws
        ],
    )
    conn.commit()


def _load(conn: sqlite3.Connection, game: Game) -> list[Draw]:
    from datetime import date as _date

    rows = conn.execute(
        "SELECT draw_date, whites, special, era_index, multiplier "
        "FROM draws WHERE game_key = ? ORDER BY draw_date",
        (game.key,),
    ).fetchall()
    return [
        Draw(
            game_key=game.key,
            draw_date=_date.fromisoformat(row[0]),
            whites=tuple(int(n) for n in row[1].split()),
            special=row[2],
            era_index=row[3],
            multiplier=row[4],
        )
        for row in rows
    ]


def load_history(game: Game, *, refresh: bool) -> list[Draw]:
    """Cached history, topped up from the API when asked."""
    conn = _connect()
    try:
        cached = _load(conn, game)
        if refresh or not cached:
            # Only a refresh can be incremental; an empty cache needs everything.
            since = cached[-1].draw_date.isoformat() if cached else None
            print(
                f"fetching {game.name} "
                + (f"since {since}" if since else "(full history)")
                + " ...",
                file=sys.stderr,
            )
            fetched = fetch_draws(game, since=since)
            if fetched:
                _save(conn, fetched)
            cached = _load(conn, game)
        return cached
    finally:
        conn.close()


def report_era(game: Game, draws: Sequence[Draw], era_index: int) -> None:
    era = game.eras[era_index]
    subset = filter_era(draws, era_index)
    if not subset:
        print(f"\n=== era {era_index}: {era} -- no draws in cache")
        return

    matrix = era.matrix
    print(f"\n=== {game.name} era {era_index}: {era}")
    print(
        f"    {len(subset)} draws, "
        f"{subset[0].draw_date} .. {subset[-1].draw_date}, "
        f"${era.ticket_price:.2f} ticket, 1 in {matrix.total_combinations:,}"
    )
    if era.note:
        print(f"    {era.note}")
    print()

    whites = R.white_counts(subset, matrix)
    print("  " + R.uniformity_test(whites, label="white balls").summary())
    print(
        "  "
        + R.uniformity_test(
            R.special_counts(subset, matrix), label=matrix.special_name
        ).summary()
    )
    if len(subset) > 1:
        print("  " + R.serial_independence_test(subset, matrix).summary())

    ranked = sorted(whites.items(), key=lambda kv: -kv[1])
    print()
    print("  hottest: " + ", ".join(f"{n}({c})" for n, c in ranked[:6]))
    print("  coldest: " + ", ".join(f"{n}({c})" for n, c in ranked[-6:]))

    extremes = R.expected_extremes(
        whites, len(subset), matrix.white_count, trials=3000
    )
    print("  " + extremes.summary())
    print("  " + R.gap_summary(subset, ranked[0][0], matrix).summary())


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--game", default="powerball", choices=sorted(GAMES))
    parser.add_argument(
        "--era",
        default="current",
        help="'current', 'all', or an era index",
    )
    parser.add_argument("--refresh", action="store_true", help="pull new drawings")
    parser.add_argument("--ticket", help='e.g. "38 43 52 61 67"')
    parser.add_argument("--jackpot", type=float, help="advertised jackpot, e.g. 1.5e9")
    parser.add_argument("--tickets-sold", type=int, default=300_000_000)
    args = parser.parse_args(argv)

    game = get_game(args.game)
    draws = load_history(game, refresh=args.refresh)
    if not draws:
        print("No draws available.", file=sys.stderr)
        return 1

    problems = validate_draws(game, draws)
    if problems:
        print(f"!! {len(problems)} draws contradict their era's matrix:", file=sys.stderr)
        for problem in problems[:10]:
            print(f"   {problem}", file=sys.stderr)
        print("   Frequency analysis is unsafe until this is fixed.", file=sys.stderr)
        return 2

    if args.era == "all":
        indices = range(len(game.eras))
    elif args.era == "current":
        indices = [len(game.eras) - 1]
    else:
        indices = [int(args.era)]

    for index in indices:
        report_era(game, draws, index)

    if args.ticket:
        numbers = [int(token) for token in args.ticket.replace(",", " ").split()]
        model = PopularityModel(game.matrix)
        assessment = model.assess(numbers, tickets_sold=args.tickets_sold)
        print("\n=== ticket assessment")
        print(assessment.summary())

        if args.jackpot:
            print()
            print(
                ticket_ev(
                    game,
                    args.jackpot,
                    tickets_sold=args.tickets_sold,
                    crowd_score=assessment.crowd_score,
                ).summary()
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
