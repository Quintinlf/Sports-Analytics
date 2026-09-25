"""Statistical tests for whether drawn numbers behave like a fair machine.

This module exists to answer one question honestly: *is any number actually
hot?* It is built to let you check the claim yourself rather than take anyone's
word for it, including mine.

The tests are all standard, and all implemented on the standard library only —
``requirements-web.txt`` deliberately carries no numpy or scipy, so the
chi-square p-value is computed here from the regularised incomplete gamma
function rather than imported.

WHAT "HOT NUMBERS" ACTUALLY ARE
-------------------------------
In roughly 1,400 Powerball draws of the current era, each of the 69 white balls
is expected to appear about 101 times. Some will appear 125 times and some 80.
That spread is not evidence of bias — it is what fair randomness looks like,
and ``expected_extremes`` quantifies exactly how large a spread to expect.

The trap is that people look at 69 numbers, find the largest count, and treat
it as if they had picked that number in advance. Testing the maximum of 69
values against the distribution of a *single* value is a multiple-comparisons
error, and it is the entire basis of every hot-number system ever sold.
``expected_extremes`` simulates the null distribution of the maximum, so the
observed leader can be compared against the right yardstick.

Even a genuinely biased machine would not help much: the edge would have to be
enormous to overcome a 1-in-292-million jackpot. The real reason to run these
tests is to understand your data, not to beat the game.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Sequence

from lottery.games import Matrix
from lottery.history import Draw
from stats_core.gof import ChiSquareResult, chi_square_pvalue, chi_square_uniform

__all__ = [
    "chi_square_pvalue",
    "ChiSquareResult",
    "white_counts",
    "special_counts",
    "uniformity_test",
    "expected_extremes",
    "ExtremesResult",
    "gap_summary",
    "GapResult",
    "serial_independence_test",
]


def white_counts(draws: Sequence[Draw], matrix: Matrix) -> dict[int, int]:
    """How many times each white ball was drawn. Zero-filled across the pool."""
    counts = dict.fromkeys(matrix.white_range(), 0)
    for draw in draws:
        for number in draw.whites:
            counts[number] += 1
    return counts


def special_counts(draws: Sequence[Draw], matrix: Matrix) -> dict[int, int]:
    """How many times each special ball was drawn. Zero-filled across the pool."""
    counts = dict.fromkeys(matrix.special_range(), 0)
    for draw in draws:
        counts[draw.special] += 1
    return counts


def uniformity_test(counts: dict[int, int], *, label: str = "") -> ChiSquareResult:
    """Chi-square goodness-of-fit for ball counts against a uniform machine.

    Thin wrapper over :func:`stats_core.gof.chi_square_uniform`. ``counts`` must
    be zero-filled over the whole pool, which is why :func:`white_counts` fills
    it: a missing key would quietly shrink the degrees of freedom and bias the
    test toward calling a machine fair.
    """
    return chi_square_uniform(counts, label=label)


@dataclass(frozen=True)
class ExtremesResult:
    """Observed hottest/coldest numbers against the null distribution of extremes.

    This is the antidote to hot-number reasoning. ``max_pvalue`` is the fraction
    of simulated fair histories whose *hottest* number was at least as hot as
    the one actually observed. A large value means the leader is unremarkable.
    """

    hottest: int
    hottest_count: int
    coldest: int
    coldest_count: int
    expected: float
    simulated_max_mean: float
    simulated_min_mean: float
    max_pvalue: float
    min_pvalue: float
    trials: int

    def summary(self) -> str:
        return (
            f"hottest {self.hottest} appeared {self.hottest_count}x "
            f"(expected {self.expected:.1f}); a fair machine's hottest number "
            f"averages {self.simulated_max_mean:.1f}. "
            f"p={self.max_pvalue:.3f} -> "
            + (
                "unremarkable"
                if self.max_pvalue > 0.05
                else "unusually hot even allowing for multiple comparisons"
            )
        )


def expected_extremes(
    counts: dict[int, int],
    draws_count: int,
    picks_per_draw: int,
    *,
    trials: int = 20_000,
    seed: Optional[int] = 20260820,
) -> ExtremesResult:
    """Simulate what the hottest and coldest numbers look like under fairness.

    Repeatedly deals ``draws_count`` fair draws over the same pool and records
    the largest and smallest per-number counts. Comparing the real leader to
    this distribution — rather than to the average — is the correction for
    having searched all numbers for the biggest one.
    """
    pool = sorted(counts)
    if picks_per_draw > len(pool):
        raise ValueError("cannot pick more numbers than the pool holds")

    observed_max = max(counts.values())
    observed_min = min(counts.values())
    expected = sum(counts.values()) / len(pool)

    rng = random.Random(seed)
    max_total = min_total = 0
    max_at_least = min_at_most = 0

    for _ in range(trials):
        tally = Counter()
        for _ in range(draws_count):
            tally.update(rng.sample(pool, picks_per_draw))

        # Numbers never drawn are absent from the Counter; pad to pool size.
        simulated = list(tally.values()) + [0] * (len(pool) - len(tally))
        sim_max = max(simulated)
        sim_min = min(simulated)

        max_total += sim_max
        min_total += sim_min
        if sim_max >= observed_max:
            max_at_least += 1
        if sim_min <= observed_min:
            min_at_most += 1

    hottest = max(counts, key=lambda n: (counts[n], -n))
    coldest = min(counts, key=lambda n: (counts[n], n))

    return ExtremesResult(
        hottest=hottest,
        hottest_count=observed_max,
        coldest=coldest,
        coldest_count=observed_min,
        expected=expected,
        simulated_max_mean=max_total / trials,
        simulated_min_mean=min_total / trials,
        max_pvalue=max_at_least / trials,
        min_pvalue=min_at_most / trials,
        trials=trials,
    )


@dataclass(frozen=True)
class GapResult:
    """Spacing between successive appearances of one number.

    Under fairness, gaps are geometric: the mean gap equals the pool size over
    the picks per draw, and — crucially — the distribution has no memory. A
    number being "due" is precisely the property a geometric distribution
    does not have.
    """

    number: int
    appearances: int
    mean_gap: float
    expected_gap: float
    longest_gap: int
    current_gap: int
    gaps: list[int] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"number {self.number}: {self.appearances} appearances, "
            f"mean gap {self.mean_gap:.1f} draws (expected {self.expected_gap:.1f}), "
            f"longest {self.longest_gap}, currently {self.current_gap} draws out. "
            "A gap carries no information about the next draw."
        )


def gap_summary(draws: Sequence[Draw], number: int, matrix: Matrix) -> GapResult:
    """Gaps between appearances of ``number`` among the white balls."""
    positions = [i for i, draw in enumerate(draws) if number in draw.whites]
    gaps = [b - a for a, b in zip(positions, positions[1:])]

    expected_gap = matrix.white_max / matrix.white_count
    current_gap = len(draws) - 1 - positions[-1] if positions else len(draws)

    return GapResult(
        number=number,
        appearances=len(positions),
        mean_gap=sum(gaps) / len(gaps) if gaps else float("nan"),
        expected_gap=expected_gap,
        longest_gap=max(gaps) if gaps else 0,
        current_gap=current_gap,
        gaps=gaps,
    )


def serial_independence_test(
    draws: Sequence[Draw], matrix: Matrix
) -> ChiSquareResult:
    """Does a number appearing in one draw change its chance in the next?

    Builds a 2x2 table over every (number, consecutive draw pair): was it in
    draw t, was it in draw t+1. Independence is the null. This is the direct
    test of "momentum" and "due" claims, and the one most worth running before
    trusting any streak-based system.
    """
    if len(draws) < 2:
        raise ValueError("need at least two draws to test serial independence")

    table = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    pool = list(matrix.white_range())

    for first, second in zip(draws, draws[1:]):
        in_first = set(first.whites)
        in_second = set(second.whites)
        for number in pool:
            table[(int(number in in_first), int(number in in_second))] += 1

    total = sum(table.values())
    row = {r: sum(v for (rr, _), v in table.items() if rr == r) for r in (0, 1)}
    col = {c: sum(v for (_, cc), v in table.items() if cc == c) for c in (0, 1)}

    statistic = 0.0
    for (r, c), observed in table.items():
        expected = row[r] * col[c] / total
        if expected > 0:
            statistic += (observed - expected) ** 2 / expected

    flat = {i: v for i, v in enumerate(table.values())}
    return ChiSquareResult(
        statistic=statistic,
        dof=1,
        p_value=chi_square_pvalue(statistic, 1),
        expected=total / 4,
        observed=flat,
        label="serial independence (draw t vs t+1)",
    )
