"""Tests for physical shuffle models and their analysis.

The Bayer-Diaconis table test is the important one: it checks an exact
closed-form computation against ten published values, so an error anywhere in
the Eulerian numbers or the probability formula cannot pass silently.
"""
from __future__ import annotations

import random
from math import factorial

import pytest

from poker.shuffle import (
    OVERHAND,
    CASINO_DECK_CHANGE,
    CASINO_STANDARD,
    HOME_GAME,
    LAZY_DEALER,
    PROCEDURES,
    SINGLE_RIFFLE,
    Procedure,
    RIFFLE,
    cut,
    overhand,
    riffle,
    strip,
    wash,
)
from poker.shuffle_analysis import (
    DECK_SIZE,
    assess,
    eulerian_row,
    order_preservation,
    position_uniformity,
    reachable_deck_orders,
    riffles_needed,
    rising_sequences,
    significant_fraction,
    tv_distance_after_riffles,
)

DECK = list(range(DECK_SIZE))


def _rng(seed: int = 1) -> random.Random:
    return random.Random(seed)


# ---------------------------------------------------------------------------
# The operations preserve the deck
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("operation", [riffle, overhand, strip, cut, wash])
def test_operations_preserve_the_deck(operation):
    """No operation may lose, duplicate, or invent a card."""
    result = operation(DECK, _rng())
    assert sorted(result) == DECK


@pytest.mark.parametrize("name", sorted(PROCEDURES))
def test_procedures_preserve_the_deck(name):
    assert sorted(PROCEDURES[name].apply(DECK, _rng())) == DECK


@pytest.mark.parametrize("operation", [riffle, overhand, strip, cut, wash])
def test_operations_handle_degenerate_decks(operation):
    assert operation([], _rng()) == []
    assert operation([7], _rng()) == [7]


# ---------------------------------------------------------------------------
# Riffle: the defining structural property
# ---------------------------------------------------------------------------


def test_one_riffle_leaves_at_most_two_rising_sequences():
    """The property the whole seven-shuffle result rests on.

    A riffle interleaves two ordered packets, so it cannot produce a third
    rising sequence. If this ever fails, the shuffle is not GSR.
    """
    rng = _rng(4)
    for _ in range(300):
        assert rising_sequences(riffle(DECK, rng)) <= 2


def test_m_riffles_leave_at_most_two_to_the_m_rising_sequences():
    rng = _rng(5)
    for riffles in (1, 2, 3):
        for _ in range(60):
            deck = DECK
            for _ in range(riffles):
                deck = riffle(deck, rng)
            assert rising_sequences(deck) <= 2**riffles


def test_riffle_actually_interleaves():
    """A riffle that always returned the deck unchanged would pass the tests
    above, so check it genuinely moves cards."""
    rng = _rng(6)
    changed = sum(1 for _ in range(50) if riffle(DECK, rng) != DECK)
    assert changed == 50


# ---------------------------------------------------------------------------
# Cut: a rotation, and therefore not a shuffle
# ---------------------------------------------------------------------------


def test_cut_is_a_rotation():
    """A cut must preserve cyclic order exactly - it adds no randomness."""
    rng = _rng(7)
    for _ in range(50):
        result = cut(DECK, rng)
        # DECK is 0..51, so the first card names the rotation amount.
        rotation = result[0]
        assert result == DECK[rotation:] + DECK[:rotation]


def test_cut_at_explicit_position():
    assert cut([0, 1, 2, 3], _rng(), position=1) == [1, 2, 3, 0]


def test_cut_does_not_change_rising_sequence_count_by_more_than_one():
    rng = _rng(8)
    for _ in range(50):
        shuffled = riffle(DECK, rng)
        assert abs(rising_sequences(cut(shuffled, rng)) - rising_sequences(shuffled)) <= 1


# ---------------------------------------------------------------------------
# Rising sequences
# ---------------------------------------------------------------------------


def test_rising_sequences_of_a_fresh_deck_is_one():
    assert rising_sequences(DECK) == 1


def test_rising_sequences_of_a_reversed_deck_is_the_deck_size():
    assert rising_sequences(list(reversed(DECK))) == DECK_SIZE


def test_rising_sequences_of_a_perfect_interleave_is_two():
    perfect = DECK[:26] + DECK[26:]
    interleaved = [c for pair in zip(DECK[:26], DECK[26:]) for c in pair]
    assert rising_sequences(perfect) == 1
    assert rising_sequences(interleaved) == 2


def test_rising_sequences_rejects_bad_input():
    with pytest.raises(ValueError):
        rising_sequences([])
    with pytest.raises(ValueError, match="duplicate"):
        rising_sequences([0, 1, 1])


# ---------------------------------------------------------------------------
# Eulerian numbers and the exact total variation distance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 5, 9, 52])
def test_eulerian_row_sums_to_factorial(n):
    """Every permutation has some number of descents, so the row must total n!."""
    assert sum(eulerian_row(n)) == factorial(n)


@pytest.mark.parametrize("n", [4, 7, 52])
def test_eulerian_row_is_symmetric(n):
    row = eulerian_row(n)
    assert row == tuple(reversed(row))


def test_eulerian_small_rows_are_known():
    assert eulerian_row(3) == (1, 4, 1)
    assert eulerian_row(4) == (1, 11, 11, 1)
    assert eulerian_row(5) == (1, 26, 66, 26, 1)


@pytest.mark.parametrize(
    "riffles,published",
    [
        (1, 1.000),
        (2, 1.000),
        (3, 1.000),
        (4, 1.000),
        (5, 0.924),
        (6, 0.614),
        (7, 0.334),
        (8, 0.167),
        (9, 0.085),
        (10, 0.043),
    ],
)
def test_total_variation_matches_bayer_diaconis(riffles, published):
    """The published table from Bayer & Diaconis (1992), to three decimals."""
    assert tv_distance_after_riffles(riffles) == pytest.approx(published, abs=0.0006)


def test_seven_riffles_is_the_answer():
    """The famous result, derived rather than asserted."""
    assert riffles_needed(0.5) == 7


def test_total_variation_decreases_monotonically():
    values = [tv_distance_after_riffles(m) for m in range(1, 15)]
    assert values == sorted(values, reverse=True)


def test_total_variation_bounds():
    assert tv_distance_after_riffles(0) == pytest.approx(1.0)
    assert 0.0 <= tv_distance_after_riffles(30) < 0.001


def test_total_variation_rejects_bad_input():
    with pytest.raises(ValueError):
        tv_distance_after_riffles(-1)
    with pytest.raises(ValueError):
        riffles_needed(0.0)


def test_smaller_decks_need_fewer_riffles():
    """Mixing time grows with deck size; a 20-card deck settles sooner."""
    assert riffles_needed(0.5, n=20) < riffles_needed(0.5, n=52)


# ---------------------------------------------------------------------------
# Empirical measures
# ---------------------------------------------------------------------------


def test_order_preservation_of_a_single_riffle_is_about_three_quarters():
    """Theory: both-left and both-right pairs keep order, cross pairs are even."""
    single = Procedure("bare riffle", (RIFFLE,))
    assert order_preservation(single, trials=200, rng=_rng(11)) == pytest.approx(
        0.75, abs=0.03
    )


def test_order_preservation_of_a_washed_deck_is_even():
    washed = PROCEDURES["casino deck change"]
    assert order_preservation(washed, trials=200, rng=_rng(12)) == pytest.approx(
        0.5, abs=0.03
    )


def test_order_preservation_ignores_cuts():
    """A cut adds no randomness, so it must not move the metric.

    Without de-rotation a cut drags this to ~0.5 and makes the least random
    procedure in the module look perfect.
    """
    bare = Procedure("bare riffle", (RIFFLE,))
    assert order_preservation(
        SINGLE_RIFFLE, trials=200, rng=_rng(13)
    ) == pytest.approx(order_preservation(bare, trials=200, rng=_rng(13)), abs=0.03)


def test_order_preservation_falls_as_riffles_accumulate():
    scores = [
        order_preservation(PROCEDURES[f"{m}x riffle"], trials=120, rng=_rng(14))
        for m in (1, 3, 7)
    ]
    assert scores[0] > scores[1] > scores[2]
    assert scores[2] == pytest.approx(0.5, abs=0.04)


def test_position_test_detects_a_single_riffle():
    assert position_uniformity(SINGLE_RIFFLE, trials=1000, rng=_rng(15)).p_value < 1e-6


def test_position_test_accepts_a_washed_deck():
    """A wash is uniform by construction, so composing with it must stay uniform."""
    assert significant_fraction(CASINO_DECK_CHANGE, replicates=20, trials=1000) < 0.30


def test_significant_fraction_flags_broken_procedures_every_time():
    for procedure in (SINGLE_RIFFLE, LAZY_DEALER, CASINO_STANDARD, HOME_GAME):
        assert (
            significant_fraction(procedure, replicates=10, trials=800) == 1.0
        ), f"{procedure.name} should be detectably non-random"


def test_ten_riffles_passes_where_the_casino_standard_fails():
    """The headline comparison: real dealing procedure vs enough riffles."""
    assert significant_fraction(PROCEDURES["10x riffle"], replicates=20, trials=1000) < 0.30
    assert significant_fraction(CASINO_STANDARD, replicates=20, trials=1000) > 0.90


def test_a_single_overhand_reverses_rather_than_mixes():
    """One overhand scores *below* 0.5: it reverses blocks wholesale.

    Read one-sided this looks better than a riffle's 0.75. It is far worse,
    which is why the metric is documented as two-sided.
    """
    single = Procedure("overhand", (OVERHAND,))
    score = order_preservation(single, trials=150, rng=_rng(16))
    assert score < 0.45
    assert abs(score - 0.5) > 0.05


def test_overhand_never_mixes_however_often_it_is_repeated():
    """Overhand shuffling is famously ineffective; 12 of them still fail."""
    many = Procedure("overhand x12", tuple(OVERHAND for _ in range(12)))
    assert significant_fraction(many, replicates=10, trials=800) == 1.0


def test_rising_sequences_grow_with_riffles():
    reports = [assess(PROCEDURES[f"{m}x riffle"], trials=300) for m in (1, 3, 7)]
    assert reports[0].mean_rising_sequences < reports[1].mean_rising_sequences
    assert reports[1].mean_rising_sequences < reports[2].mean_rising_sequences


def test_assess_only_reports_exact_tv_for_pure_riffling():
    """The closed form covers riffles alone; strips and cuts are not modelled."""
    assert assess(PROCEDURES["7x riffle"], trials=200).exact_tv == pytest.approx(0.334, abs=0.001)
    assert assess(CASINO_STANDARD, trials=200).exact_tv is None


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def test_reachable_deck_orders_is_a_tiny_fraction_of_all_of_them():
    reachable, total = reachable_deck_orders(63)
    assert reachable == 2**63
    assert total == factorial(52)
    assert reachable / total < 1e-45


def test_enough_seed_bits_covers_the_whole_space():
    reachable, total = reachable_deck_orders(1000)
    assert reachable == total


def test_shuffles_are_reproducible_from_a_seed():
    assert CASINO_STANDARD.apply(DECK, _rng(99)) == CASINO_STANDARD.apply(DECK, _rng(99))
