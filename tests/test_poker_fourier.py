"""Tests for Fourier analysis of cutting.

The load-bearing test is ``test_bound_dominates_the_exact_distance``: the
Diaconis-Shahshahani bound is proved to be an upper bound, so if the
implementation ever produces a bound below the exactly-computed distance, one
of the two is wrong. They are computed by completely independent routes —
characters and conjugation on one side, direct convolution on the other — so
agreement is real evidence.
"""
from __future__ import annotations

import math
from fractions import Fraction

import pytest

from poker.fourier import (
    character,
    characters_orthogonal,
    conjugate_square,
    convolve,
    cuts_never_mix,
    dft,
    total_variation,
    upper_bound,
    walk_distribution,
)

N = 52


def uniform(n: int = N) -> list[float]:
    return [1 / n] * n


def point_mass(at: int, n: int = N) -> list[float]:
    out = [0.0] * n
    out[at] = 1.0
    return out


def lumpy(n: int = N) -> list[float]:
    """A plausible human cut: concentrated near the middle."""
    weights = [math.exp(-((j - n / 2) ** 2) / (2 * (n / 10) ** 2)) for j in range(n)]
    total = math.fsum(weights)
    return [w / total for w in weights]


# ---------------------------------------------------------------------------
# Characters and conjugation
# ---------------------------------------------------------------------------


def test_characters_have_modulus_one():
    for m in range(N):
        assert abs(character(N, m, 3)) == pytest.approx(1.0)


def test_trivial_character_is_one():
    assert character(N, 0, 7) == pytest.approx(1.0)


def test_characters_are_orthogonal():
    """sum_j chi_a(j) conj(chi_b(j)) = n when a == b, else 0."""
    assert characters_orthogonal(N, 3, 3).real == pytest.approx(N)
    assert abs(characters_orthogonal(N, 3, 7)) == pytest.approx(0.0, abs=1e-9)
    assert abs(characters_orthogonal(N, 0, 5)) == pytest.approx(0.0, abs=1e-9)


def test_conjugate_square_is_real_and_matches_abs():
    for value in (3 + 4j, -1j, 0.5 + 0.5j, 2 + 0j):
        assert conjugate_square(value) == pytest.approx(abs(value) ** 2)


def test_conjugate_square_of_a_character_is_one():
    assert conjugate_square(character(N, 5, 9)) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Transform properties
# ---------------------------------------------------------------------------


def test_transform_at_zero_is_one():
    """The trivial character sums the probabilities, which total 1."""
    assert dft(lumpy())[0] == pytest.approx(1.0)


def test_uniform_distribution_kills_every_other_coefficient():
    """This is why a *uniform* cut mixes Z_n in a single step."""
    transformed = dft(uniform())
    assert all(abs(c) < 1e-9 for c in transformed[1:])


def test_point_mass_has_coefficients_of_modulus_one():
    """A deterministic cut loses nothing, so nothing decays."""
    transformed = dft(point_mass(26))
    assert all(abs(c) == pytest.approx(1.0) for c in transformed)


def test_convolution_theorem():
    """Transform of a convolution is the product of the transforms."""
    first, second = lumpy(), uniform()
    combined = dft(convolve(first, second))
    product = [a * b for a, b in zip(dft(first), dft(second))]
    for left, right in zip(combined, product):
        assert left == pytest.approx(right, abs=1e-9)


def test_transform_rejects_non_distributions():
    with pytest.raises(ValueError):
        dft([0.5, 0.2])
    with pytest.raises(ValueError):
        dft([-1.0, 2.0])


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------


def test_walk_preserves_total_probability():
    for steps in (0, 1, 5):
        assert math.fsum(walk_distribution(lumpy(), steps)) == pytest.approx(1.0)


def test_zero_steps_is_the_identity():
    assert walk_distribution(lumpy(), 0) == point_mass(0)


def test_repeated_cuts_converge_on_the_cyclic_group():
    distances = [
        total_variation(walk_distribution(lumpy(), k), uniform())
        for k in (1, 3, 10, 25)
    ]
    assert distances == sorted(distances, reverse=True)
    assert distances[-1] < 0.05


def test_a_uniform_cut_converges_immediately():
    assert total_variation(walk_distribution(uniform(), 1), uniform()) == pytest.approx(
        0.0, abs=1e-12
    )


def test_a_deterministic_cut_never_converges():
    """Cutting by exactly 26 every time just alternates between two decks."""
    for steps in (1, 2, 5, 50):
        distance = total_variation(walk_distribution(point_mass(26), steps), uniform())
        assert distance == pytest.approx(1 - 1 / N)


# ---------------------------------------------------------------------------
# The bound
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("steps", list(range(1, 26)))
def test_bound_dominates_the_exact_distance(steps):
    """The whole point of the lemma, checked against direct convolution."""
    distribution = lumpy()
    exact = total_variation(walk_distribution(distribution, steps), uniform())
    assert upper_bound(distribution, steps) + 1e-12 >= exact


def test_bound_decreases_with_more_cuts():
    bounds = [upper_bound(lumpy(), k) for k in (1, 2, 5, 10, 20)]
    assert bounds == sorted(bounds, reverse=True)


def test_bound_is_vacuous_for_a_deterministic_cut():
    """It exceeds 1, which is the bound honestly saying it knows nothing."""
    assert upper_bound(point_mass(26), 10) > 1.0


def test_bound_rejects_negative_steps():
    with pytest.raises(ValueError):
        upper_bound(lumpy(), -1)


# ---------------------------------------------------------------------------
# The result that matters
# ---------------------------------------------------------------------------


def test_cuts_never_mix_small_cases_exactly():
    """1 - n/n!, computed in exact rationals before converting."""
    assert cuts_never_mix(3) == pytest.approx(float(1 - Fraction(3, 6)))
    assert cuts_never_mix(4) == pytest.approx(float(1 - Fraction(4, 24)))
    assert cuts_never_mix(5) == pytest.approx(float(1 - Fraction(5, 120)))


def test_cutting_a_52_card_deck_is_indistinguishable_from_not_shuffling():
    """1 - 52/52! is 1 to every digit a float carries."""
    assert cuts_never_mix(52) == 1.0


def test_converging_on_rotations_is_not_converging_on_decks():
    """The central point, stated as a test.

    After 25 cuts the deck is essentially a uniformly random *rotation* — and
    still exactly as far from shuffled as it was before the first cut.
    """
    distribution = lumpy()
    on_cyclic = total_variation(walk_distribution(distribution, 25), uniform())
    assert on_cyclic < 0.01
    assert cuts_never_mix(N) == 1.0


def test_cuts_never_mix_rejects_empty_decks():
    with pytest.raises(ValueError):
        cuts_never_mix(0)
