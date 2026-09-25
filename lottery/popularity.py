"""Modelling how *other players* pick numbers, and what that costs you.

THE ONE REAL EDGE IN LOTTERY
---------------------------
You cannot change your probability of winning. It is fixed by the matrix at
1 in 292,201,338 and nothing in this repository will move it.

You can change *how much a win is worth*. The jackpot is pari-mutuel: it is
split equally among all tickets matching the same combination. Since players do
not pick uniformly — they pick birthdays, lucky numbers, and lines drawn on the
playslip — some combinations are held by thousands of people and some by almost
nobody. Choosing an unpopular combination does not make you win more often. It
makes the win you are unlikely to get worth more when it happens.

That is the entire edge, and it applies only to the jackpot and the pari-mutuel
tiers. Fixed-prize tiers pay the same regardless of how many share them.

HOW STRONG IS THE EFFECT?
-------------------------
Large. The 2005 Powerball drawing where 110 players matched five numbers — an
event the lottery expected to produce four or five winners — happened because
those numbers appeared in a fortune cookie. Conversely, combinations of all
high, non-consecutive, patternless numbers are held by very few tickets.

A SCOPE NOTE, IN THE SAME SPIRIT AS poker/strength.py
-----------------------------------------------------
The weights below are a **documented heuristic prior, not measured data**.
Lotteries do not publish per-combination ticket sales, so nobody outside the
operator can measure pick frequency directly. Each factor here is justified by
published research on lottery number selection, but the magnitudes are
estimates.

This module therefore reports a *relative crowd score* — "this ticket is about
2.3x more popular than an average ticket" — and never a probability. Do not
present its output as one. The calibration path, if per-draw winner counts are
ever ingested, is described in ``calibration_notes()``.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from math import exp
from typing import Optional, Sequence

from lottery.games import Matrix

__all__ = [
    "PopularityModel",
    "TicketAssessment",
    "expected_cowinners",
    "expected_share",
    "calibration_notes",
]


@dataclass
class PopularityModel:
    """Relative propensity for players to choose each number and pattern.

    All factors are multiplicative and centred so that an average random ticket
    scores 1.0 after normalisation. Values above 1 mean over-picked by the
    crowd (bad for you); below 1 means under-picked (good for you).
    """

    matrix: Matrix

    #: Numbers 1-31 can be a day of the month. Birthday and anniversary picking
    #: is the single largest distortion in every lottery studied. Deliberately
    #: conservative: the per-number boost and ``all_dates_boost`` below overlap,
    #: and the qualitative conclusion holds for anything above ~1.3.
    date_boost: float = 1.65

    #: 1-12 double as months, compounding the effect.
    month_boost: float = 1.15

    #: Culturally lucky numbers, picked well above their date weight.
    lucky_numbers: tuple[int, ...] = (7, 3, 11, 21)
    lucky_boost: float = 1.20

    #: Widely avoided. Being unpopular, it is a genuinely good pick.
    unlucky_numbers: tuple[int, ...] = (13,)
    unlucky_penalty: float = 0.75

    #: Players under-pick consecutive numbers, believing they look "less random".
    consecutive_penalty: float = 0.88

    #: Evenly spaced picks are lines drawn down a playslip - heavily over-picked.
    arithmetic_boost: float = 2.6

    #: A ticket entirely inside 1-31 is almost certainly a set of dates.
    all_dates_boost: float = 1.5

    _normaliser: Optional[float] = field(default=None, repr=False, compare=False)

    # -- per-number weights -------------------------------------------------

    def number_weight(self, number: int) -> float:
        """Relative propensity of a single white ball being chosen."""
        if number not in self.matrix.white_range():
            raise ValueError(f"{number} outside 1..{self.matrix.white_max}")

        weight = 1.0
        if number <= 31:
            weight *= self.date_boost
        if number <= 12:
            weight *= self.month_boost
        if number in self.lucky_numbers:
            weight *= self.lucky_boost
        if number in self.unlucky_numbers:
            weight *= self.unlucky_penalty
        return weight

    # -- combination-level effects -----------------------------------------

    def _pattern_multiplier(self, whites: Sequence[int]) -> float:
        picks = sorted(whites)
        multiplier = 1.0

        if picks[-1] <= 31:
            multiplier *= self.all_dates_boost

        gaps = [b - a for a, b in zip(picks, picks[1:])]

        # A constant gap is a straight line on the playslip.
        if len(set(gaps)) == 1:
            multiplier *= self.arithmetic_boost

        consecutive = sum(1 for gap in gaps if gap == 1)
        multiplier *= self.consecutive_penalty**consecutive

        return multiplier

    def raw_score(self, whites: Sequence[int]) -> float:
        """Unnormalised crowd weight for a set of white balls."""
        if len(set(whites)) != len(whites):
            raise ValueError("duplicate numbers in ticket")
        if len(whites) != self.matrix.white_count:
            raise ValueError(
                f"expected {self.matrix.white_count} numbers, got {len(whites)}"
            )

        weight = 1.0
        for number in whites:
            weight *= self.number_weight(number)
        return weight * self._pattern_multiplier(whites)

    def normaliser(self, *, trials: int = 40_000, seed: int = 20260820) -> float:
        """Mean raw score of a uniformly random ticket, by simulation.

        Dividing by this turns raw weights into "times more popular than
        average", which is the only form worth reporting. Cached per instance.
        """
        if self._normaliser is None:
            rng = random.Random(seed)
            pool = list(self.matrix.white_range())
            total = 0.0
            for _ in range(trials):
                total += self.raw_score(rng.sample(pool, self.matrix.white_count))
            self._normaliser = total / trials
        return self._normaliser

    def crowd_score(self, whites: Sequence[int]) -> float:
        """How many times more popular than an average ticket this pick is."""
        return self.raw_score(whites) / self.normaliser()

    def assess(
        self, whites: Sequence[int], *, tickets_sold: Optional[int] = None
    ) -> "TicketAssessment":
        score = self.crowd_score(whites)
        picks = sorted(whites)

        reasons: list[str] = []
        if picks[-1] <= 31:
            reasons.append(
                "every number is 31 or below - reads as birthdays, the most "
                "crowded pattern there is"
            )
        gaps = [b - a for a, b in zip(picks, picks[1:])]
        if len(set(gaps)) == 1:
            reasons.append(
                f"evenly spaced by {gaps[0]} - a straight line on the playslip"
            )
        high = sum(1 for n in picks if n > 31)
        if high >= 3:
            reasons.append(f"{high} numbers above 31, which the crowd under-picks")
        if 13 in picks:
            reasons.append("includes 13, widely avoided and therefore valuable")
        if not reasons:
            reasons.append("no strong crowd pattern detected")

        cowinners = (
            expected_cowinners(score, tickets_sold, self.matrix)
            if tickets_sold
            else None
        )

        return TicketAssessment(
            whites=tuple(picks),
            crowd_score=score,
            reasons=reasons,
            tickets_sold=tickets_sold,
            expected_cowinners=cowinners,
            expected_share=expected_share(cowinners) if cowinners is not None else None,
        )


@dataclass(frozen=True)
class TicketAssessment:
    """What a ticket's number choice implies about splitting a jackpot."""

    whites: tuple[int, ...]
    crowd_score: float
    reasons: list[str]
    tickets_sold: Optional[int] = None
    expected_cowinners: Optional[float] = None
    expected_share: Optional[float] = None

    def summary(self) -> str:
        picks = " ".join(f"{n:02d}" for n in self.whites)
        verdict = (
            "more crowded than average"
            if self.crowd_score > 1.15
            else "less crowded than average"
            if self.crowd_score < 0.85
            else "about average"
        )
        lines = [
            f"{picks}  crowd score {self.crowd_score:.2f}x -> {verdict}",
            "  odds of winning: unchanged. Only the size of a win moves.",
        ]
        for reason in self.reasons:
            lines.append(f"  - {reason}")
        if self.expected_cowinners is not None:
            lines.append(
                f"  at {self.tickets_sold:,} tickets sold: expect "
                f"{self.expected_cowinners:.2f} other jackpot winners, "
                f"keeping ~{self.expected_share:.1%} of the pot"
            )
        return "\n".join(lines)


def expected_cowinners(
    crowd_score: float, tickets_sold: int, matrix: Matrix
) -> float:
    """Expected number of *other* tickets holding the same combination.

    Uniform selling would put ``tickets_sold / total_combinations`` tickets on
    each combination. The crowd score scales that for this specific pick.
    """
    if tickets_sold < 0:
        raise ValueError("tickets_sold cannot be negative")
    baseline = max(tickets_sold - 1, 0) / matrix.total_combinations
    return baseline * crowd_score


def expected_share(cowinners: float) -> float:
    """Expected fraction of the jackpot kept, given Poisson co-winners.

    With K other winners ~ Poisson(lambda), your share is 1/(1+K), and

        E[1/(1+K)] = (1 - e^-lambda) / lambda

    which is exact, not an approximation. Note it is *not* 1/(1+lambda) —
    using that instead understates your share, because the split is convex.
    """
    if cowinners < 0:
        raise ValueError("cowinners cannot be negative")
    if cowinners < 1e-9:
        return 1.0
    return (1 - exp(-cowinners)) / cowinners


def calibration_notes() -> str:
    """How to replace these heuristics with measured values."""
    return (
        "The weights in PopularityModel are estimates, not measurements.\n"
        "\n"
        "They become measurable if per-draw winner counts by tier are ingested\n"
        "alongside the drawn numbers. The logic: for a fixed number of tickets\n"
        "sold, draws whose winning numbers are all <= 31 should produce\n"
        "systematically more lower-tier winners than draws containing high\n"
        "numbers. Regressing observed winner counts on the drawn numbers'\n"
        "features recovers the crowd's real preferences.\n"
        "\n"
        "Most state lotteries publish winner counts per drawing, but the NY\n"
        "open-data sets used here do not carry them. Until that lands, treat\n"
        "every number this module produces as directional only."
    )
