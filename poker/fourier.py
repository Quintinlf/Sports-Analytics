"""Fourier analysis on groups: where complex conjugation earns its place.

Complex numbers look like an odd thing to find in a card program, but they are
the standard tool for proving how fast a shuffle mixes — the seven-shuffle
result included. This module implements the case that can be done exactly, and
states precisely how it generalises.

THE IDEA
--------
A shuffle is a probability distribution over permutations. Performing two
shuffles in a row *convolves* their distributions, and convolution is awkward
to iterate directly. The Fourier transform turns convolution into
multiplication, so ``k`` shuffles becomes a ``k``-th power, and the question
"how random is the deck after k shuffles" becomes "how fast do these numbers
shrink".

THE CUT IS THE EXACTLY SOLVABLE CASE
------------------------------------
A cut is a rotation, so cutting repeatedly is a random walk on the cyclic group
Z_52 — an *abelian* group, whose characters are just roots of unity:

    chi_m(j) = omega^(j*m),   omega = exp(2*pi*i/52)

The transform of a cut distribution P is

    P_hat(m) = sum_j P(j) * omega^(j*m)

and the Diaconis-Shahshahani upper bound lemma says

    4 * ||P^(*k) - U||^2  <=  sum over m != 0 of |P_hat(m)|^(2k)

where **|z|^2 = z * conj(z)**. That is the complex conjugation, and it is not
decoration: the modulus is what makes the bound real and non-negative, and the
whole argument collapses without it.

WHAT IT PROVES ABOUT CUTTING
----------------------------
Two things, and the second is the interesting one.

1. Cutting does converge — to the uniform distribution *on rotations*.
2. Uniform on rotations is not uniform on decks. Cuts reach only 52 of the
   52! arrangements, so no number of cuts brings the deck closer than
   ``1 - 52/52!`` to random. :func:`cuts_never_mix` computes that exactly.

This is the rigorous version of the informal claim in ``shuffle_analysis``
that a cut adds no randomness.

THE NON-ABELIAN GENERALISATION
------------------------------
Riffles live in the symmetric group S_52, which is not abelian, so characters
become matrix-valued representations and the transform becomes a matrix:

    P_hat(rho) = sum over pi of P(pi) * rho(pi)

The same lemma holds with the same shape, except that ``|z|^2 = z * conj(z)``
becomes ``Tr(A A*)`` where ``A*`` is the **conjugate transpose**. Complex
conjugation survives the generalisation; it just grows a transpose.

Implementing the representations of S_52 is a research-scale undertaking, and
it is not needed here: ``shuffle_analysis.tv_distance_after_riffles`` already
gets the exact riffle answer by the Bayer-Diaconis combinatorial route. This
module deliberately does the abelian case properly rather than the non-abelian
case badly.
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

__all__ = [
    "character",
    "dft",
    "conjugate_square",
    "convolve",
    "walk_distribution",
    "total_variation",
    "upper_bound",
    "CutAnalysis",
    "analyse_cuts",
    "cuts_never_mix",
    "characters_orthogonal",
]


def _validate(distribution: Sequence[float]) -> None:
    if len(distribution) < 1:
        raise ValueError("distribution must be non-empty")
    if any(p < 0 for p in distribution):
        raise ValueError("probabilities cannot be negative")
    total = math.fsum(distribution)
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError(f"distribution must sum to 1, got {total}")


def character(n: int, m: int, j: int) -> complex:
    """The character chi_m(j) = omega^(j*m) of the cyclic group Z_n.

    These are the irreducible representations of Z_n. Every one is a single
    complex number of modulus 1 — a point on the unit circle — which is exactly
    why the abelian case stays this simple.
    """
    if n < 1:
        raise ValueError("group order must be positive")
    return cmath.exp(2j * cmath.pi * (j * m % n) / n)


def conjugate_square(value: complex) -> float:
    """|z|^2 computed as z * conj(z), which is real by construction.

    Written out rather than using ``abs(z)**2`` because this is the step the
    module exists to make visible: multiplying by the complex conjugate is what
    turns an oscillating complex amplitude into a magnitude that can decay.
    """
    product = value * value.conjugate()
    # The imaginary part is zero up to floating point; assert the maths.
    if abs(product.imag) > 1e-9:
        raise ArithmeticError(f"z * conj(z) should be real, got {product}")
    return product.real


def dft(distribution: Sequence[float]) -> list[complex]:
    """Fourier transform of a distribution on Z_n.

    ``result[m] = sum_j P(j) chi_m(j)``. Index 0 is the trivial character and
    is always 1 for a probability distribution, which is why the bound skips it.
    """
    _validate(distribution)
    n = len(distribution)
    return [
        sum(
            (probability * character(n, m, j) for j, probability in enumerate(distribution)),
            start=0j,
        )
        for m in range(n)
    ]


def convolve(first: Sequence[float], second: Sequence[float]) -> list[float]:
    """Distribution of the sum of two independent cuts, mod n.

    Doing one cut then another adds their amounts modulo the deck size, so the
    resulting distribution is the cyclic convolution.
    """
    if len(first) != len(second):
        raise ValueError("distributions must be over the same group")
    _validate(first)
    _validate(second)

    n = len(first)
    out = [0.0] * n
    for i, p in enumerate(first):
        if p == 0.0:
            continue
        for j, q in enumerate(second):
            out[(i + j) % n] += p * q
    return out


def walk_distribution(distribution: Sequence[float], steps: int) -> list[float]:
    """Exact distribution after ``steps`` repetitions, by direct convolution.

    Kept deliberately independent of the Fourier machinery so it can be used to
    check the bound rather than assume it.
    """
    if steps < 0:
        raise ValueError("steps cannot be negative")
    _validate(distribution)

    n = len(distribution)
    result = [0.0] * n
    result[0] = 1.0  # identity: no cut at all
    for _ in range(steps):
        result = convolve(result, distribution)
    return result


def total_variation(first: Sequence[float], second: Sequence[float]) -> float:
    """Total variation distance between two distributions on the same set."""
    if len(first) != len(second):
        raise ValueError("distributions must be over the same group")
    return sum(abs(p - q) for p, q in zip(first, second)) / 2


def upper_bound(distribution: Sequence[float], steps: int) -> float:
    """Diaconis-Shahshahani upper bound on distance from uniform on Z_n.

    ``||P^(*k) - U|| <= (1/2) * sqrt( sum over m != 0 of |P_hat(m)|^(2k) )``

    The bound is an upper bound only: it can exceed 1, in which case it says
    nothing. That happens exactly when the walk genuinely has not converged,
    which for a deterministic cut is always.
    """
    if steps < 0:
        raise ValueError("steps cannot be negative")

    transformed = dft(distribution)
    total = math.fsum(
        conjugate_square(coefficient) ** steps
        for coefficient in transformed[1:]
    )
    return math.sqrt(total) / 2


def characters_orthogonal(n: int, first: int, second: int) -> complex:
    """Inner product of two characters: ``sum_j chi_a(j) * conj(chi_b(j))``.

    Equals ``n`` when the characters match and 0 otherwise. This orthogonality
    is what makes the transform invertible, and it is stated in terms of the
    conjugate — another place the conjugation is load-bearing rather than
    cosmetic.
    """
    return sum(
        (
            character(n, first, j) * character(n, second, j).conjugate()
            for j in range(n)
        ),
        start=0j,
    )


@dataclass(frozen=True)
class CutAnalysis:
    """What repeated cutting does, measured two independent ways."""

    deck_size: int
    steps: int
    exact_on_cyclic: float
    bound_on_cyclic: float
    distance_from_shuffled: float

    def summary(self) -> str:
        return (
            f"  {self.steps:>2} cuts: distance from uniform *rotation* "
            f"{self.exact_on_cyclic:.6f} (bound {min(self.bound_on_cyclic, 1.0):.6f})"
            f"   |   distance from a shuffled deck {self.distance_from_shuffled:.6f}"
        )


def cuts_never_mix(deck_size: int = 52) -> float:
    """Distance from a genuinely shuffled deck that cutting can never beat.

    Cutting produces only ``deck_size`` of the ``deck_size!`` arrangements, so
    however the cut amounts are distributed, the result puts all its mass on a
    vanishing subset. Computed exactly:

        TV = 1 - deck_size / deck_size!

    For 52 cards that is 1 - 6.4e-66, which rounds to 1 in any float. Cutting
    is not a weak shuffle; it is not a shuffle.
    """
    if deck_size < 1:
        raise ValueError("deck must have at least one card")
    return float(1 - Fraction(deck_size, math.factorial(deck_size)))


def analyse_cuts(
    distribution: Sequence[float], steps: Sequence[int] = (1, 2, 3, 5, 10, 25)
) -> list[CutAnalysis]:
    """Run the whole analysis for a real cut distribution."""
    _validate(distribution)
    n = len(distribution)
    uniform = [1 / n] * n
    unreachable = cuts_never_mix(n)

    return [
        CutAnalysis(
            deck_size=n,
            steps=k,
            exact_on_cyclic=total_variation(walk_distribution(distribution, k), uniform),
            bound_on_cyclic=upper_bound(distribution, k),
            distance_from_shuffled=unreachable,
        )
        for k in steps
    ]
