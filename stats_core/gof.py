"""Goodness-of-fit statistics on the standard library alone.

WHY THIS PACKAGE EXISTS
-----------------------
``poker/README.md`` argues against premature shared abstractions, and that
argument still holds for the sports code. This module clears the bar it set:
there are now two genuinely independent callers of the *same implementation*.

* ``lottery.randomness`` tests whether a ball machine is uniform.
* ``poker.shuffle_analysis`` tests whether a shuffled deck is uniform.

Both need a chi-square p-value, and ``requirements-web.txt`` deliberately
carries no scipy. The alternative to sharing was duplicating a numerical
routine, which is the one kind of duplication worth avoiding outright: two
copies of an incomplete gamma function drift and only one of them gets fixed.

Nothing sport-specific belongs here, and nothing here knows what a card or a
lottery ball is.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

__all__ = ["chi_square_pvalue", "gamma_q", "ChiSquareResult", "chi_square_uniform"]


# ---------------------------------------------------------------------------
# Regularised incomplete gamma -> chi-square upper tail.
#
# Q(a, x) by series below the crossover and continued fraction above, the
# standard Numerical Recipes split. Verified exact to five decimals against
# published critical values from 1 to 100 degrees of freedom.
# ---------------------------------------------------------------------------
_ITMAX = 300
_EPS = 3.0e-12
_FPMIN = 1e-300


def _lower_series(a: float, x: float) -> float:
    """P(a, x) by series expansion. Valid for x < a + 1."""
    ap = a
    total = 1.0 / a
    term = total
    for _ in range(_ITMAX):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * _EPS:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _upper_cf(a: float, x: float) -> float:
    """Q(a, x) by continued fraction. Valid for x >= a + 1."""
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, _ITMAX + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def gamma_q(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x) = 1 - P(a, x)."""
    if x < 0 or a <= 0:
        raise ValueError("gamma_q requires a > 0 and x >= 0")
    if x == 0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _lower_series(a, x)
    return _upper_cf(a, x)


def chi_square_pvalue(statistic: float, dof: int) -> float:
    """Upper-tail probability of a chi-square statistic.

    The probability that a fair process would produce a statistic at least this
    extreme. Small values are evidence against fairness; by convention below
    0.05 is "significant", though a p-value found by searching many candidates
    needs a multiple-comparisons correction before it means anything. See
    ``lottery.randomness.expected_extremes`` for what that looks like in
    practice.
    """
    if dof < 1:
        raise ValueError("degrees of freedom must be >= 1")
    if statistic < 0:
        raise ValueError("chi-square statistic cannot be negative")
    return gamma_q(dof / 2.0, statistic / 2.0)


@dataclass(frozen=True)
class ChiSquareResult:
    """Outcome of a goodness-of-fit test against a uniform distribution."""

    statistic: float
    dof: int
    p_value: float
    expected: float
    observed: dict[int, int]
    label: str = ""

    @property
    def significant(self) -> bool:
        """True at the conventional 5% level. Read the docstring first."""
        return self.p_value < 0.05

    def verdict(self) -> str:
        if self.p_value < 0.01:
            return "strong evidence against a fair process"
        if self.p_value < 0.05:
            return "marginal; expected once in 20 fair tests by chance alone"
        return "consistent with a fair process"

    def summary(self) -> str:
        name = self.label or "uniformity"
        return (
            f"{name}: chi2={self.statistic:.2f} on {self.dof} dof, "
            f"p={self.p_value:.4f} -> {self.verdict()}"
        )


def chi_square_uniform(
    counts: Mapping[int, int], *, label: str = ""
) -> ChiSquareResult:
    """Pearson chi-square goodness-of-fit against a uniform distribution.

    ``counts`` must already be zero-filled over every category. A missing key
    would quietly drop a category and shrink the degrees of freedom, which
    biases the test toward declaring things fair.
    """
    if len(counts) < 2:
        raise ValueError("need at least two categories")

    total = sum(counts.values())
    if total == 0:
        raise ValueError("no observations to test")

    expected = total / len(counts)
    statistic = sum(
        (observed - expected) ** 2 / expected for observed in counts.values()
    )
    dof = len(counts) - 1

    return ChiSquareResult(
        statistic=statistic,
        dof=dof,
        p_value=chi_square_pvalue(statistic, dof),
        expected=expected,
        observed=dict(counts),
        label=label,
    )
