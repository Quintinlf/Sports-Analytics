"""Draw records: parsing, era assignment, and validation.

A :class:`Draw` is one drawing of one game. It carries the era it belongs to so
that no analysis can accidentally pool across a matrix change.

``validate_draws`` is the load-bearing function here. It checks every drawn
number against the matrix its era claims to have used, so a wrong era boundary
in ``games.py`` surfaces immediately as a loud failure rather than silently
skewing a frequency table. The era dates in this package were confirmed against
the real data using exactly this check.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Optional, Sequence

from lottery.games import Game

__all__ = ["Draw", "parse_record", "parse_records", "validate_draws", "DrawProblem"]


@dataclass(frozen=True)
class Draw:
    """One drawing.

    ``whites`` is sorted ascending. The draw order is not published, and for
    every analysis in this package the drawn *set* is what matters.
    """

    game_key: str
    draw_date: date
    whites: tuple[int, ...]
    special: int
    era_index: int
    multiplier: Optional[int] = None

    def __post_init__(self) -> None:
        if tuple(sorted(self.whites)) != self.whites:
            raise ValueError("whites must be sorted ascending")
        if len(set(self.whites)) != len(self.whites):
            raise ValueError(f"duplicate white balls in {self.whites}")

    @property
    def numbers(self) -> tuple[int, ...]:
        """Whites plus the special, for display."""
        return self.whites + (self.special,)

    def __str__(self) -> str:
        whites = " ".join(f"{n:02d}" for n in self.whites)
        return f"{self.draw_date.isoformat()}  {whites}  [{self.special:02d}]"


@dataclass(frozen=True)
class DrawProblem:
    """A draw that contradicts the matrix its era claims."""

    draw: Draw
    reason: str

    def __str__(self) -> str:
        return f"{self.draw.draw_date.isoformat()}: {self.reason}"


def _parse_date(raw: str) -> date:
    """Socrata floating timestamps look like ``2026-08-19T00:00:00.000``."""
    return datetime.fromisoformat(raw.replace("Z", "")).date()


def _parse_int(raw: object, field: str) -> int:
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        raise ValueError(f"Cannot parse {field} from {raw!r}") from None


def parse_record(game: Game, record: dict) -> Draw:
    """Turn one Socrata JSON row into a :class:`Draw`.

    Two shapes exist in the wild:

    * Powerball packs all six balls into ``winning_numbers``, special last.
    * Mega Millions gives five in ``winning_numbers`` plus ``mega_ball``.

    Both are handled by token count rather than by game, so a schema change on
    either dataset degrades to a clear error instead of a silent mis-parse.
    """
    when = _parse_date(record["draw_date"])
    tokens = str(record.get("winning_numbers", "")).split()

    special_field = record.get("mega_ball")
    if special_field is not None:
        if len(tokens) != 5:
            raise ValueError(
                f"{when}: expected 5 white balls alongside mega_ball, got {tokens}"
            )
        whites = [_parse_int(t, "white ball") for t in tokens]
        special = _parse_int(special_field, "mega_ball")
    else:
        if len(tokens) != 6:
            raise ValueError(
                f"{when}: expected 6 packed numbers, got {len(tokens)}: {tokens}"
            )
        whites = [_parse_int(t, "white ball") for t in tokens[:5]]
        special = _parse_int(tokens[5], "special ball")

    multiplier_raw = record.get("multiplier")
    multiplier = (
        _parse_int(multiplier_raw, "multiplier") if multiplier_raw not in (None, "") else None
    )

    return Draw(
        game_key=game.key,
        draw_date=when,
        whites=tuple(sorted(whites)),
        special=special,
        era_index=game.era_index(when),
        multiplier=multiplier,
    )


def parse_records(game: Game, records: Iterable[dict]) -> list[Draw]:
    """Parse many rows, returning them sorted oldest first."""
    draws = [parse_record(game, record) for record in records]
    draws.sort(key=lambda d: d.draw_date)
    return draws


def validate_draws(game: Game, draws: Sequence[Draw]) -> list[DrawProblem]:
    """Check every draw against the matrix of the era it was assigned to.

    Returns the problems found; an empty list means the era table in
    ``games.py`` is consistent with every observed draw. Anything non-empty
    means the era boundaries are wrong and no frequency analysis should be
    trusted until they are fixed.
    """
    problems: list[DrawProblem] = []

    for draw in draws:
        matrix = game.eras[draw.era_index].matrix

        if len(draw.whites) != matrix.white_count:
            problems.append(
                DrawProblem(draw, f"expected {matrix.white_count} whites, got {len(draw.whites)}")
            )

        for number in draw.whites:
            if not 1 <= number <= matrix.white_max:
                problems.append(
                    DrawProblem(
                        draw,
                        f"white ball {number} outside 1..{matrix.white_max} for era {matrix}",
                    )
                )

        if not 1 <= draw.special <= matrix.special_max:
            problems.append(
                DrawProblem(
                    draw,
                    f"{matrix.special_name} {draw.special} outside "
                    f"1..{matrix.special_max} for era {matrix}",
                )
            )

    return problems


def filter_era(draws: Sequence[Draw], era_index: int) -> list[Draw]:
    """All draws belonging to one era. The gateway to every valid analysis."""
    return [d for d in draws if d.era_index == era_index]
