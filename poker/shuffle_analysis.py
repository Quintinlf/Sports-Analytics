"""How random is a shuffled deck? Exact answers where they exist.

Two independent approaches, which is the point of the module: an exact
closed-form result for riffle shuffles, and empirical tests that work on any
procedure including ones no theory covers.

THE EXACT RESULT
----------------
Bayer and Diaconis (1992) proved that after ``m`` riffle shuffles of ``n``
cards, the probability of any particular arrangement is

    P(arrangement) = C(2^m + n - r, n) / 2^(mn)

where ``r`` is the arrangement's number of *rising sequences*. Since the number
of arrangements with exactly ``r`` rising sequences is the Eulerian number
A(n, r-1), the total variation distance from a uniform deck is a finite sum of
52 terms — computable exactly in integer arithmetic, no simulation required.

That is what produces the famous result: **seven riffle shuffles**. Total
variation drops from 1.000 to 0.334 at seven and keeps halving after. Below
seven the deck is measurably not random; the fall is abrupt rather than gradual,
which is why "a few shuffles" is not a matter of taste.

THE EMPIRICAL TESTS
-------------------
The exact formula only covers pure riffling. Real procedures mix in strips,
cuts and overhands, so those are measured by simulation against the same
chi-square machinery the lottery analysis uses (``stats_core.gof``). Using one
implementation for a ball machine and a card deck is deliberate: they are the
same question.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import comb, factorial
from typing import List, Optional, Sequence

from poker.shuffle import Procedure
from stats_core.gof import ChiSquareResult, chi_square_uniform

__all__ = [
    "rising_sequences",
    "eulerian_row",
    "tv_distance_after_riffles",
    "riffles_needed",
    "position_uniformity",
    "significant_fraction",
    "order_preservation",
    "top_card_survival",
    "ShuffleReport",
    "assess",
    "reachable_deck_orders",
]

DECK_SIZE = 52


# ---------------------------------------------------------------------------
# Exact theory
# ---------------------------------------------------------------------------


def rising_sequences(arrangement: Sequence[int]) -> int:
    """Count the rising sequences of an arrangement.

    A rising sequence is a maximal run of consecutively-numbered cards that
    still appear in increasing order, though not necessarily adjacently. A
    fresh deck has one. A single riffle can produce at most two, because it
    only interleaves two ordered packets — which is exactly why one riffle
    leaves the original order almost fully recoverable.

    ``arrangement[i]`` is the original label of the card now at position ``i``.
    """
    count = len(arrangement)
    if count == 0:
        raise ValueError("cannot count rising sequences of an empty deck")

    position = {label: index for index, label in enumerate(arrangement)}
    if len(position) != count:
        raise ValueError("arrangement contains duplicate labels")

    labels = sorted(position)
    return 1 + sum(
        1
        for earlier, later in zip(labels, labels[1:])
        if position[later] < position[earlier]
    )


@lru_cache(maxsize=None)
def eulerian_row(n: int) -> tuple[int, ...]:
    """Eulerian numbers A(n, k) for k = 0..n-1.

    A(n, k) counts permutations of n items with exactly k descents, so the
    number with ``r`` rising sequences is A(n, r-1). Built by the standard
    recurrence A(n,k) = (k+1)A(n-1,k) + (n-k)A(n-1,k-1) in exact integers.
    """
    if n < 1:
        raise ValueError("n must be at least 1")

    row = [1]
    for size in range(2, n + 1):
        previous = row
        row = [0] * size
        for k in range(size):
            left = (k + 1) * previous[k] if k < len(previous) else 0
            right = (size - k) * previous[k - 1] if 0 < k <= len(previous) else 0
            row[k] = left + right
    return tuple(row)


def tv_distance_after_riffles(m: int, n: int = DECK_SIZE) -> float:
    """Exact total variation distance from uniform after ``m`` riffle shuffles.

    0 means indistinguishable from a perfectly shuffled deck; 1 means an
    observer can tell with certainty. This is computed in exact rational
    arithmetic and then converted, not simulated.

    The interpretation of total variation is concrete: it is the maximum edge
    any betting strategy could have over someone facing a truly random deck.
    """
    if m < 0:
        raise ValueError("cannot riffle a negative number of times")
    if n < 1:
        raise ValueError("deck must have at least one card")
    if m == 0:
        # An unshuffled deck is one specific arrangement out of n!.
        return float(1 - Fraction(1, factorial(n)))

    counts = eulerian_row(n)
    uniform = Fraction(1, factorial(n))
    denominator = Fraction(1, 2 ** (m * n))
    two_to_m = 2**m

    total = Fraction(0)
    for r in range(1, n + 1):
        arrangements = counts[r - 1]
        if arrangements == 0:
            continue
        probability = comb(two_to_m + n - r, n) * denominator
        total += arrangements * abs(probability - uniform)

    return float(total / 2)


def riffles_needed(threshold: float = 0.5, n: int = DECK_SIZE, *, limit: int = 40) -> int:
    """Fewest riffles bringing total variation below ``threshold``.

    At the conventional 0.5 the answer for a 52-card deck is 7.
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    for m in range(limit + 1):
        if tv_distance_after_riffles(m, n) < threshold:
            return m
    raise ValueError(f"threshold {threshold} not reached within {limit} riffles")


def reachable_deck_orders(seed_bits: int, n: int = DECK_SIZE) -> tuple[int, int]:
    """How many deck orders a seeded PRNG can actually produce.

    Returns ``(reachable, total)``. ``poker.cards.Deck`` seeds from
    ``randrange(2**63)``, so it can reach at most 2^63 of the 52! possible
    orders — about one in 10^48 of them.

    This is not a flaw for a learning tool, where a replayable seed is the
    entire point, and no player could exploit it. It is a real constraint on any
    system dealing for money, and it is why regulated shufflers use hardware
    entropy rather than a seeded generator.
    """
    if seed_bits < 0:
        raise ValueError("seed_bits cannot be negative")
    return min(2**seed_bits, factorial(n)), factorial(n)


# ---------------------------------------------------------------------------
# Empirical measurement of arbitrary procedures
# ---------------------------------------------------------------------------


def _run(procedure: Procedure, trials: int, rng: random.Random, n: int) -> List[List[int]]:
    original = list(range(n))
    return [procedure.apply(original, rng) for _ in range(trials)]


def position_uniformity(
    procedure: Procedure,
    *,
    trials: int = 5000,
    tracked_card: int = 0,
    n: int = DECK_SIZE,
    rng: Optional[random.Random] = None,
) -> ChiSquareResult:
    """Where does one specific card end up, over many shuffles?

    Under a fair shuffle its final position is uniform over all ``n`` slots.
    Tracking the original top card is the most sensitive choice, since weak
    procedures tend to leave it near the top.
    """
    rng = rng or random.Random(20260820)
    counts = dict.fromkeys(range(n), 0)
    for arrangement in _run(procedure, trials, rng, n):
        counts[arrangement.index(tracked_card)] += 1
    return chi_square_uniform(
        counts, label=f"{procedure.name}: position of card {tracked_card}"
    )


def _derotate(arrangement: Sequence[int]) -> List[int]:
    """Rotate so the original top card sits at index 0.

    A cut is a rotation, and a rotation adds no randomness whatsoever — it
    changes no card's position relative to any other in cyclic order. But it
    *does* reverse the linear order of every pair straddling the cut point,
    which is roughly half of them. Left uncorrected, a cut therefore drags
    :func:`order_preservation` to 50% and makes the worst procedures look
    perfect: a single riffle measures 0.75 alone but 0.51 after a cut.

    Undoing the rotation before measuring is what makes the metric report
    mixing rather than cutting.
    """
    pivot = arrangement.index(0)
    return list(arrangement[pivot:]) + list(arrangement[:pivot])


def order_preservation(
    procedure: Procedure,
    *,
    trials: int = 500,
    n: int = DECK_SIZE,
    rng: Optional[random.Random] = None,
    derotate: bool = True,
) -> float:
    """Fraction of card pairs still in their original relative order.

    A uniformly shuffled deck gives 0.5 — knowing that A came before B tells
    you nothing. Anything above that is information about the original order
    surviving the shuffle, which is what an advantage player or a colluding
    dealer would exploit.

    A single riffle scores about 0.75: three quarters of all pairs are still
    where they started.

    READ THIS TWO-SIDED. Scores *below* 0.5 indicate structure just as strongly
    as scores above it, because reversing a block flips the order of every pair
    spanning it. One overhand shuffle scores about 0.38 and two score about
    0.65, oscillating as blocks are reversed and re-reversed. The informative
    quantity is therefore ``abs(score - 0.5)``, and a procedure sitting at
    exactly 0.5 for the wrong reason is why :func:`significant_fraction` exists
    as an independent check.

    ``derotate`` removes the effect of any cut before measuring; see
    :func:`_derotate` for why leaving it in would be misleading.
    """
    rng = rng or random.Random(20260820)
    preserved = 0
    total = 0
    for arrangement in _run(procedure, trials, rng, n):
        if derotate:
            arrangement = _derotate(arrangement)
        position = {label: index for index, label in enumerate(arrangement)}
        for first in range(n):
            for second in range(first + 1, n):
                total += 1
                if position[first] < position[second]:
                    preserved += 1
    return preserved / total


def top_card_survival(
    procedure: Procedure,
    *,
    trials: int = 5000,
    window: int = 5,
    n: int = DECK_SIZE,
    rng: Optional[random.Random] = None,
) -> tuple[float, float]:
    """Chance the original top card is still within ``window`` of the top.

    Returns ``(observed, fair)``. The fair value is ``window / n``.

    CONFOUNDED BY CUTS, which is why :func:`assess` does not report it. A cut
    near the middle reliably moves the original top card to the middle, so a
    single riffle plus a cut scores 0% — appearing *better* than fair while
    being the least random procedure there is. Meaningful only for procedures
    with no cut; use :func:`order_preservation` or
    :func:`rising_sequences` otherwise.
    """
    rng = rng or random.Random(20260820)
    hits = sum(
        1
        for arrangement in _run(procedure, trials, rng, n)
        if arrangement.index(0) < window
    )
    return hits / trials, window / n


def significant_fraction(
    procedure: Procedure,
    *,
    replicates: int = 20,
    trials: int = 1500,
    n: int = DECK_SIZE,
    alpha: float = 0.05,
    seed: int = 20260820,
) -> float:
    """Fraction of independent replications whose position test is significant.

    A single p-value is one draw from Uniform[0,1] under the null, so a fair
    procedure will be flagged at 5% about one run in twenty purely by luck —
    and reporting that single number would brand a provably-uniform procedure
    as biased. (``CASINO_DECK_CHANGE`` starts with a wash, which is uniform by
    construction; composing anything independent with it stays uniform, so any
    single significant p-value there is guaranteed to be noise.)

    Replicating turns that coin flip into a rate. Near ``alpha`` means fair;
    near 1.0 means genuinely broken. This is the same lesson as
    ``lottery.randomness.expected_extremes``: one extreme number proves
    nothing until you know how often extremes happen anyway.
    """
    hits = sum(
        1
        for index in range(replicates)
        if position_uniformity(
            procedure, trials=trials, n=n, rng=random.Random(seed + index * 7919)
        ).p_value
        < alpha
    )
    return hits / replicates


@dataclass(frozen=True)
class ShuffleReport:
    """Everything measured about one procedure."""

    procedure: Procedure
    mean_rising_sequences: float
    order_preservation: float
    position_test: ChiSquareResult
    significant_fraction: float
    exact_tv: Optional[float]

    def summary(self) -> str:
        lines = [
            f"{self.procedure}",
            f"  rising sequences   {self.mean_rising_sequences:6.2f} "
            f"(uniform deck averages ~{DECK_SIZE / 2:.0f})",
            f"  order preserved    {self.order_preservation:6.1%} "
            "(fair = 50.0%, one riffle = 75%)",
            f"  position test flags {self.significant_fraction:.0%} of runs "
            "(fair = about 5%)",
        ]
        if self.exact_tv is not None:
            lines.append(
                f"  exact TV distance  {self.exact_tv:6.3f} "
                f"({self.procedure.riffle_count} riffles; 0 = perfectly mixed)"
            )
        return "\n".join(lines)


def assess(
    procedure: Procedure,
    *,
    trials: int = 3000,
    n: int = DECK_SIZE,
    seed: int = 20260820,
) -> ShuffleReport:
    """Measure a procedure every way this module knows how.

    ``exact_tv`` is filled in only when the procedure is pure riffling, since
    that is the only case the closed form covers. Strips and cuts are not
    modelled by the theory — but as ``order_preservation`` shows, they add
    almost nothing anyway.
    """
    rng = random.Random(seed)
    arrangements = _run(procedure, trials, rng, n)
    mean_rising = sum(rising_sequences(a) for a in arrangements) / len(arrangements)

    only_riffles = all(step.name == "riffle" for step in procedure.steps)
    exact = (
        tv_distance_after_riffles(procedure.riffle_count, n) if only_riffles else None
    )

    return ShuffleReport(
        procedure=procedure,
        mean_rising_sequences=mean_rising,
        order_preservation=order_preservation(
            procedure, trials=min(trials, 400), n=n, rng=random.Random(seed + 2)
        ),
        position_test=position_uniformity(
            procedure, trials=trials, n=n, rng=random.Random(seed + 3)
        ),
        significant_fraction=significant_fraction(
            procedure, trials=min(trials, 1500), n=n, seed=seed + 5
        ),
        exact_tv=exact,
    )
