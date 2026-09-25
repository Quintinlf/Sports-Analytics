"""Tests for range set algebra and out counting.

The inclusion-exclusion tests are the ones that matter: they check the identity
holds on real card sets rather than on a remembered rule of thumb.
"""
from __future__ import annotations

import pytest

from poker.cards import Card, Rank, Suit, parse_cards
from poker.ranges import (
    ALL_COMBOS,
    Combo,
    Range,
    draw_analysis,
    flush_completing_cards,
    parse_range,
    straight_completing_cards,
)


# ---------------------------------------------------------------------------
# Combos
# ---------------------------------------------------------------------------


def test_there_are_1326_combos():
    """C(52,2). If this is wrong, every count downstream is wrong."""
    assert len(ALL_COMBOS) == 1326


def test_combo_is_an_unordered_set_of_two_cards():
    first, second = parse_cards("As Kh")
    assert Combo.of(first, second) == Combo.of(second, first)


def test_combo_rejects_a_duplicate_card():
    ace = Card(Rank.ACE, Suit.SPADES)
    with pytest.raises(ValueError):
        Combo.of(ace, ace)


def test_combo_ordering_works_for_pairs():
    """Pairs are the case that used to raise before Suit became orderable.

    Kept as a regression test: Combo.high/.low must work when both cards
    share a rank.
    """
    combo = Combo.parse("AsAh")
    assert combo.is_pair
    assert combo.notation() == "AA"


@pytest.mark.parametrize(
    "text,notation,suited,pair",
    [
        ("AsAh", "AA", False, True),
        ("AsKs", "AKs", True, False),
        ("AsKh", "AKo", False, False),
        ("7d2c", "72o", False, False),
    ],
)
def test_combo_notation(text, notation, suited, pair):
    combo = Combo.parse(text)
    assert combo.notation() == notation
    assert combo.is_suited is suited
    assert combo.is_pair is pair


# ---------------------------------------------------------------------------
# Notation parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,count",
    [
        ("AA", 6),
        ("AKs", 4),
        ("AKo", 12),
        ("AK", 16),
        ("JJ+", 24),
        ("ATs+", 16),
        ("22-55", 24),
        ("AA, KK", 12),
    ],
)
def test_range_sizes(text, count):
    """A pair is 6 combos, suited 4, offsuit 12. Everything else follows."""
    assert len(parse_range(text)) == count


def test_pairs_plus_walks_upward():
    assert set(parse_range("JJ+").notations()) == {"JJ", "QQ", "KK", "AA"}


def test_suited_plus_walks_the_kicker():
    assert set(parse_range("ATs+").notations()) == {"ATs", "AJs", "AQs", "AKs"}


def test_every_combo_notation_is_covered_by_its_own_range():
    for text in ("AA", "AKs", "AKo", "72o"):
        for combo in parse_range(text):
            assert combo.notation() in parse_range(text).notations()


def test_parse_rejects_nonsense():
    with pytest.raises(ValueError):
        parse_range("ZZ")


# ---------------------------------------------------------------------------
# Set algebra
# ---------------------------------------------------------------------------


def test_union_intersection_difference():
    broadway = parse_range("AKs")
    pairs = parse_range("AA")
    assert len(broadway | pairs) == 10
    assert len(broadway & pairs) == 0
    assert len(broadway - pairs) == 4


def test_complement_is_everything_else():
    aces = parse_range("AA")
    assert len(~aces) == 1326 - 6
    assert len(aces | ~aces) == 1326
    assert len(aces & ~aces) == 0


def test_de_morgan():
    """~(A | B) == ~A & ~B, on actual card sets."""
    a, b = parse_range("AA, KK"), parse_range("AKs, QQ")
    assert ~(a | b) == (~a) & (~b)
    assert ~(a & b) == (~a) | (~b)


def test_subset_relation():
    assert parse_range("AKs") <= parse_range("AK")
    assert not parse_range("AK") <= parse_range("AKs")


def test_symmetric_difference():
    a, b = parse_range("AK"), parse_range("AKs")
    assert a ^ b == parse_range("AKo")


def test_everything_is_all_combos():
    assert len(Range.everything()) == 1326
    assert Range.everything().percent == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Blockers
# ---------------------------------------------------------------------------


def test_holding_an_ace_halves_the_aces():
    """6 combos of AA become 3 once you hold one. Set difference, nothing more."""
    aces = parse_range("AA")
    assert len(aces) == 6
    assert len(aces.remove_dead(parse_cards("As"))) == 3


def test_blocking_an_unrelated_card_changes_nothing():
    kings = parse_range("KK")
    assert len(kings.remove_dead(parse_cards("As"))) == 6


def test_holding_two_aces_leaves_one_combo():
    assert len(parse_range("AA").remove_dead(parse_cards("As Ah"))) == 1


def test_blocker_on_suited_hand():
    """AKs has 4 combos; holding the ace of spades kills exactly one."""
    assert len(parse_range("AKs").remove_dead(parse_cards("As"))) == 3


def test_containing_selects_combos_with_a_card():
    ace = Card(Rank.ACE, Suit.SPADES)
    assert len(Range.everything().containing(ace)) == 51


# ---------------------------------------------------------------------------
# Outs and inclusion-exclusion
# ---------------------------------------------------------------------------

COMBO_DRAW_HOLE = "Jh Th"
COMBO_DRAW_BOARD = "9h 8s 2h"


def test_the_textbook_combo_draw_is_fifteen_outs():
    """Flush draw (9) + open-ended straight draw (8) - overlap (2) = 15."""
    analysis = draw_analysis(
        parse_cards(COMBO_DRAW_HOLE), parse_cards(COMBO_DRAW_BOARD)
    )
    assert len(analysis.flush_outs) == 9
    assert len(analysis.straight_outs) == 8
    assert len(analysis.both) == 2
    assert len(analysis.union) == 15


def test_naive_addition_overcounts():
    """The mistake this module exists to prevent."""
    analysis = draw_analysis(
        parse_cards(COMBO_DRAW_HOLE), parse_cards(COMBO_DRAW_BOARD)
    )
    assert analysis.naive_sum == 17
    assert analysis.naive_sum - analysis.double_counted == len(analysis.union)


def test_inclusion_exclusion_identity_always_holds():
    """|A union B| = |A| + |B| - |A intersection B| on many real boards."""
    boards = [
        ("Jh Th", "9h 8s 2h"),
        ("As Ks", "Qs Jd 2s"),
        ("7c 6c", "5c 4d Kh"),
        ("Ad Kd", "2c 7h 9s"),
        ("9s 8s", "7s 6h 2d"),
    ]
    for hole, board in boards:
        analysis = draw_analysis(parse_cards(hole), parse_cards(board))
        assert len(analysis.union) == (
            len(analysis.flush_outs)
            + len(analysis.straight_outs)
            - len(analysis.both)
        )


def test_a_made_flush_has_no_flush_outs():
    hole, board = parse_cards("Ah Kh"), parse_cards("Qh Jh 2h")
    assert flush_completing_cards(hole, board) == set()


def test_a_made_straight_has_no_straight_outs():
    hole, board = parse_cards("9c 8d"), parse_cards("7h 6s 5c")
    assert straight_completing_cards(hole, board) == set()


def test_wheel_draw_is_found():
    """A-2-3-4-5 counts as a straight; the ace plays low."""
    hole, board = parse_cards("Ad 2c"), parse_cards("3h 4s Kd")
    outs = straight_completing_cards(hole, board)
    assert {card.rank for card in outs} == {Rank.FIVE}


def test_probability_is_a_ratio_of_counts():
    analysis = draw_analysis(
        parse_cards(COMBO_DRAW_HOLE), parse_cards(COMBO_DRAW_BOARD)
    )
    assert analysis.unseen == 47
    assert analysis.probability_next_card() == pytest.approx(15 / 47)


def test_no_out_is_a_known_card():
    hole, board = parse_cards(COMBO_DRAW_HOLE), parse_cards(COMBO_DRAW_BOARD)
    known = set(hole) | set(board)
    analysis = draw_analysis(hole, board)
    assert not (analysis.union & known)
