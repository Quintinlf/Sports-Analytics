"""Draw matrices, historical eras, and exact combinatorial odds.

A lottery *matrix* is the shape of a draw: how many balls are picked from how
large a pool, plus a separate special ball (Powerball / Mega Ball) drawn from
its own machine.

WHY ERAS EXIST IN THIS MODULE
-----------------------------
Lotteries change their matrix. Powerball moved from 5/59+1/35 to 5/69+1/26 on
2015-10-07. Before that date the number 63 could not be drawn: it was not on
the machine.

This makes frequency analysis pooled across a matrix change meaningless. Every
number above the old maximum looks "cold" purely because it did not exist for
part of the window, and every number at or below it looks "hot" because it had
more chances to appear. Essentially every "hot numbers" table published on the
internet makes exactly this mistake, and what it is really measuring is the
rule change rather than the balls.

Every analysis in this package therefore runs against a single era. That is a
hard constraint, not a convention: see ``history.validate_draws``, which
refuses a draw containing a number its era cannot produce.

Odds here are computed combinatorially rather than copied from a website; the
published figures are asserted against in ``tests/test_lottery_games.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import comb
from typing import Iterator, Optional, Sequence

__all__ = [
    "Matrix",
    "Era",
    "Game",
    "PrizeTier",
    "POWERBALL",
    "MEGA_MILLIONS",
    "GAMES",
    "get_game",
]


@dataclass(frozen=True)
class Matrix:
    """The shape of a single draw.

    ``white_count`` balls are drawn without replacement from ``1..white_max``,
    then one special ball is drawn independently from ``1..special_max``.
    """

    white_count: int
    white_max: int
    special_max: int
    special_name: str = "Powerball"

    def __post_init__(self) -> None:
        if not 1 <= self.white_count <= self.white_max:
            raise ValueError("white_count must be between 1 and white_max")
        if self.special_max < 1:
            raise ValueError("special_max must be positive")

    def __str__(self) -> str:
        return f"{self.white_count}/{self.white_max} + 1/{self.special_max}"

    @property
    def total_combinations(self) -> int:
        """Number of distinct tickets. Powerball today: 292,201,338."""
        return comb(self.white_max, self.white_count) * self.special_max

    def white_range(self) -> range:
        return range(1, self.white_max + 1)

    def special_range(self) -> range:
        return range(1, self.special_max + 1)

    def tier_probability(self, matched_white: int, matched_special: bool) -> float:
        """Exact probability of matching ``matched_white`` whites and the special.

        Hypergeometric on the white balls, independent Bernoulli on the special:
        of the ``white_max`` numbers, ``white_count`` are winners; a ticket picks
        ``white_count`` of them and we want exactly ``matched_white`` hits.
        """
        if not 0 <= matched_white <= self.white_count:
            raise ValueError(f"Cannot match {matched_white} of {self.white_count}")

        hits = comb(self.white_count, matched_white)
        misses = comb(
            self.white_max - self.white_count, self.white_count - matched_white
        )
        white_p = hits * misses / comb(self.white_max, self.white_count)

        special_p = (
            1 / self.special_max
            if matched_special
            else (self.special_max - 1) / self.special_max
        )
        return white_p * special_p

    def tier_odds(self, matched_white: int, matched_special: bool) -> float:
        """One-in-N form of :meth:`tier_probability`."""
        p = self.tier_probability(matched_white, matched_special)
        return float("inf") if p == 0 else 1 / p


@dataclass(frozen=True)
class PrizeTier:
    """A fixed-prize tier. ``prize is None`` marks the pari-mutuel jackpot."""

    matched_white: int
    matched_special: bool
    prize: Optional[int]

    @property
    def is_jackpot(self) -> bool:
        return self.prize is None

    def label(self) -> str:
        base = f"{self.matched_white} white"
        return f"{base} + special" if self.matched_special else base


@dataclass(frozen=True)
class Era:
    """A period during which the matrix and ticket price were constant.

    ``end`` is inclusive. ``None`` means "still current".
    """

    start: date
    end: Optional[date]
    matrix: Matrix
    ticket_price: float
    note: str = ""

    def contains(self, when: date) -> bool:
        if when < self.start:
            return False
        return self.end is None or when <= self.end

    def __str__(self) -> str:
        finish = self.end.isoformat() if self.end else "present"
        return f"{self.start.isoformat()}..{finish} ({self.matrix})"


@dataclass(frozen=True)
class Game:
    """A lottery game across its whole history."""

    key: str
    name: str
    resource_id: str
    eras: Sequence[Era]
    prize_tiers: Sequence[PrizeTier] = ()

    @property
    def current_era(self) -> Era:
        return self.eras[-1]

    @property
    def matrix(self) -> Matrix:
        """Shorthand for the current matrix."""
        return self.current_era.matrix

    def era_for(self, when: date) -> Era:
        for era in self.eras:
            if era.contains(when):
                return era
        raise ValueError(f"{self.name} has no defined era covering {when.isoformat()}")

    def era_index(self, when: date) -> int:
        for index, era in enumerate(self.eras):
            if era.contains(when):
                return index
        raise ValueError(f"{self.name} has no defined era covering {when.isoformat()}")

    def __iter__(self) -> Iterator[Era]:
        return iter(self.eras)


# --------------------------------------------------------------------------
# Powerball
#
# The NY open-data set begins 2010-02-03, which lands inside the 5/59+1/39
# era, so all three modern matrices appear in the data we ingest.
# --------------------------------------------------------------------------
POWERBALL = Game(
    key="powerball",
    name="Powerball",
    resource_id="d6yy-54nr",
    eras=(
        Era(
            start=date(2009, 1, 7),
            end=date(2012, 1, 14),
            matrix=Matrix(5, 59, 39),
            ticket_price=1.0,
            note="$1 ticket; Power Play a separate $1 add-on.",
        ),
        Era(
            start=date(2012, 1, 15),
            end=date(2015, 10, 3),
            matrix=Matrix(5, 59, 35),
            ticket_price=2.0,
            note="Price doubled to $2; special pool shrank, improving overall odds.",
        ),
        Era(
            start=date(2015, 10, 7),
            end=None,
            matrix=Matrix(5, 69, 26),
            ticket_price=2.0,
            note=(
                "White pool grew 59->69 and special pool shrank 35->26. This made "
                "the jackpot far harder (1 in 292.2M) while making the lowest tier "
                "easier - which is what drives the huge rollovers of this era."
            ),
        ),
    ),
    prize_tiers=(
        PrizeTier(5, True, None),
        PrizeTier(5, False, 1_000_000),
        PrizeTier(4, True, 50_000),
        PrizeTier(4, False, 100),
        PrizeTier(3, True, 100),
        PrizeTier(3, False, 7),
        PrizeTier(2, True, 7),
        PrizeTier(1, True, 4),
        PrizeTier(0, True, 4),
    ),
)


# --------------------------------------------------------------------------
# Mega Millions
#
# History reaches back to 2002, spanning five matrices - the widest era span in
# the data, and the clearest demonstration of why pooling is invalid.
# --------------------------------------------------------------------------
MEGA_MILLIONS = Game(
    key="mega_millions",
    name="Mega Millions",
    resource_id="5xaw-6ayf",
    eras=(
        Era(
            start=date(2002, 5, 17),
            end=date(2005, 6, 21),
            matrix=Matrix(5, 52, 52, special_name="Mega Ball"),
            ticket_price=1.0,
        ),
        Era(
            start=date(2005, 6, 22),
            end=date(2013, 10, 18),
            matrix=Matrix(5, 56, 46, special_name="Mega Ball"),
            ticket_price=1.0,
        ),
        Era(
            start=date(2013, 10, 19),
            end=date(2017, 10, 27),
            matrix=Matrix(5, 75, 15, special_name="Mega Ball"),
            ticket_price=1.0,
        ),
        Era(
            start=date(2017, 10, 28),
            end=date(2025, 4, 4),
            matrix=Matrix(5, 70, 25, special_name="Mega Ball"),
            ticket_price=2.0,
        ),
        Era(
            start=date(2025, 4, 5),
            end=None,
            matrix=Matrix(5, 70, 24, special_name="Mega Ball"),
            ticket_price=5.0,
            note="Price rose to $5; built-in multiplier and a smaller Mega Ball pool.",
        ),
    ),
    prize_tiers=(
        PrizeTier(5, True, None),
        PrizeTier(5, False, 1_000_000),
        PrizeTier(4, True, 10_000),
        PrizeTier(4, False, 500),
        PrizeTier(3, True, 200),
        PrizeTier(3, False, 10),
        PrizeTier(2, True, 10),
        PrizeTier(1, True, 4),
        PrizeTier(0, True, 2),
    ),
)


GAMES: dict[str, Game] = {g.key: g for g in (POWERBALL, MEGA_MILLIONS)}


def get_game(key: str) -> Game:
    try:
        return GAMES[key]
    except KeyError:
        known = ", ".join(sorted(GAMES))
        raise ValueError(f"Unknown game {key!r}. Known games: {known}") from None
