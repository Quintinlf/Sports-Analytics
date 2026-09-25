"""Hand evaluator correctness.

The exhaustive frequency test is the backbone here: if every one of the
2,598,960 distinct five-card hands lands in the right category, essentially no
classification bug can survive. The targeted tests then pin down the
tiebreaker and kicker rules that frequency counts cannot see.
"""
from __future__ import annotations

import os
import unittest
from collections import Counter
from itertools import combinations

from poker.cards import FULL_DECK, parse_cards
from poker.evaluator import HandCategory, compare, evaluate


def value_of(text: str):
    return evaluate(parse_cards(text))


class TestHandCategories(unittest.TestCase):
    def test_straight_flush(self) -> None:
        self.assertEqual(
            value_of("9h 8h 7h 6h 5h").category, HandCategory.STRAIGHT_FLUSH
        )

    def test_royal_flush_described_separately(self) -> None:
        self.assertEqual(value_of("As Ks Qs Js Ts").describe(), "Royal flush")

    def test_steel_wheel_is_five_high_straight_flush(self) -> None:
        value = value_of("As 2s 3s 4s 5s")
        self.assertEqual(value.category, HandCategory.STRAIGHT_FLUSH)
        self.assertEqual(value.tiebreakers, (5,))

    def test_four_of_a_kind(self) -> None:
        self.assertEqual(value_of("7c 7d 7h 7s Kd").category, HandCategory.FOUR_OF_A_KIND)

    def test_full_house(self) -> None:
        self.assertEqual(value_of("7c 7d 7h Ks Kd").category, HandCategory.FULL_HOUSE)

    def test_flush(self) -> None:
        self.assertEqual(value_of("Ah Jh 9h 5h 3h").category, HandCategory.FLUSH)

    def test_straight(self) -> None:
        self.assertEqual(value_of("9h 8c 7d 6s 5h").category, HandCategory.STRAIGHT)

    def test_wheel_straight_ranks_as_five_high(self) -> None:
        value = value_of("Ah 2c 3d 4s 5h")
        self.assertEqual(value.category, HandCategory.STRAIGHT)
        self.assertEqual(value.tiebreakers, (5,))

    def test_ace_high_straight(self) -> None:
        self.assertEqual(value_of("Ah Kc Qd Js Th").tiebreakers, (14,))

    def test_three_of_a_kind(self) -> None:
        self.assertEqual(value_of("7c 7d 7h Ks 2d").category, HandCategory.THREE_OF_A_KIND)

    def test_two_pair(self) -> None:
        self.assertEqual(value_of("7c 7d Ks Kd 2h").category, HandCategory.TWO_PAIR)

    def test_pair(self) -> None:
        self.assertEqual(value_of("7c 7d Ks 9d 2h").category, HandCategory.PAIR)

    def test_high_card(self) -> None:
        self.assertEqual(value_of("Ac Kd 9c 7d 5h").category, HandCategory.HIGH_CARD)

    def test_almost_straight_is_not_a_straight(self) -> None:
        # 9-8-7-6 with no 5 or 10 is only high card.
        self.assertEqual(value_of("9h 8c 7d 6s 2h").category, HandCategory.HIGH_CARD)

    def test_ace_does_not_wrap_around(self) -> None:
        # Q-K-A-2-3 is not a straight; the ace cannot bridge both ends.
        self.assertEqual(value_of("Qh Kc As 2d 3h").category, HandCategory.HIGH_CARD)


class TestSevenCardSelection(unittest.TestCase):
    def test_picks_best_five_of_seven(self) -> None:
        value = value_of("As Ks Qs Js Ts 2c 3d")
        self.assertEqual(value.category, HandCategory.STRAIGHT_FLUSH)

    def test_two_trips_makes_a_full_house_from_the_higher_set(self) -> None:
        value = value_of("7c 7d 7h 8s 8c 8d 4h")
        self.assertEqual(value.category, HandCategory.FULL_HOUSE)
        self.assertEqual(value.tiebreakers, (8, 7))

    def test_trips_plus_two_pairs_uses_the_higher_pair(self) -> None:
        value = value_of("9c 9d 9h Ks Kd 4s 4d")
        self.assertEqual(value.tiebreakers, (9, 13))

    def test_three_pairs_plays_the_top_two_plus_kicker(self) -> None:
        value = value_of("Ac Ad Kc Kd Qc Qd 5h")
        self.assertEqual(value.category, HandCategory.TWO_PAIR)
        # The third pair cannot play, but a queen is still the best kicker.
        self.assertEqual(value.tiebreakers, (14, 13, 12))

    def test_six_card_flush_takes_the_top_five(self) -> None:
        value = value_of("Ah Kh Qh 9h 5h 3h 2c")
        self.assertEqual(value.category, HandCategory.FLUSH)
        self.assertEqual(value.tiebreakers, (14, 13, 12, 9, 5))

    def test_straight_flush_beats_a_higher_ordinary_flush(self) -> None:
        # Same suit contains both a flush and a straight flush.
        value = value_of("Ah Kh 5h 4h 3h 2h 7c")
        self.assertEqual(value.category, HandCategory.STRAIGHT_FLUSH)
        self.assertEqual(value.tiebreakers, (5,))

    def test_quads_on_board_uses_best_kicker(self) -> None:
        value = value_of("2c 2d 2h 2s Kd 9c 3h")
        self.assertEqual(value.category, HandCategory.FOUR_OF_A_KIND)
        self.assertEqual(value.tiebreakers, (2, 13))

    def test_five_card_and_seven_card_agree(self) -> None:
        self.assertEqual(
            value_of("Ah Kh Qh Jh Th").key, value_of("Ah Kh Qh Jh Th 2c 3d").key
        )


class TestComparisons(unittest.TestCase):
    def test_category_ordering(self) -> None:
        ordered = [
            "2c 3d 5h 7s 9c",       # high card
            "2c 2d 5h 7s 9c",       # pair
            "2c 2d 5h 5s 9c",       # two pair
            "2c 2d 2h 5s 9c",       # trips
            "5c 6d 7h 8s 9c",       # straight
            "2c 5c 7c 9c Jc",       # flush
            "2c 2d 2h 5s 5c",       # full house
            "2c 2d 2h 2s 5c",       # quads
            "5c 6c 7c 8c 9c",       # straight flush
        ]
        for weaker, stronger in zip(ordered, ordered[1:]):
            self.assertEqual(
                compare(parse_cards(stronger), parse_cards(weaker)),
                1,
                f"{stronger} should beat {weaker}",
            )

    def test_kicker_decides_equal_pairs(self) -> None:
        self.assertEqual(
            compare(parse_cards("Ac Ad Kh 5s 3c"), parse_cards("As Ah Qh 5s 3c")), 1
        )

    def test_identical_ranks_different_suits_tie(self) -> None:
        self.assertEqual(
            compare(parse_cards("Ac Kc Qd Jh 9s"), parse_cards("Ad Kd Qh Js 9h")), 0
        )

    def test_wheel_loses_to_six_high_straight(self) -> None:
        self.assertEqual(
            compare(parse_cards("Ah 2c 3d 4h 5s"), parse_cards("2s 3h 4c 5d 6h")), -1
        )

    def test_higher_full_house_wins_on_the_trips(self) -> None:
        self.assertEqual(
            compare(parse_cards("9c 9d 9h 2s 2d"), parse_cards("8c 8d 8h As Ad")), 1
        )

    def test_flush_compared_card_by_card(self) -> None:
        self.assertEqual(
            compare(parse_cards("Ah Kh 9h 5h 3h"), parse_cards("Ah Kh 9h 5h 2h".replace("h", "s"))),
            1,
        )


class TestValidation(unittest.TestCase):
    def test_rejects_too_few_cards(self) -> None:
        with self.assertRaises(ValueError):
            evaluate(parse_cards("Ah Kh Qh Jh"))

    def test_rejects_too_many_cards(self) -> None:
        with self.assertRaises(ValueError):
            evaluate(parse_cards("Ah Kh Qh Jh Th 9h 8h 7h"))

    def test_rejects_duplicate_cards(self) -> None:
        with self.assertRaises(ValueError):
            evaluate([*parse_cards("Ah Kh Qh Jh"), *parse_cards("Ah")])


@unittest.skipUnless(
    os.getenv("POKER_EXHAUSTIVE") == "1",
    "Exhaustive 2.6M-hand sweep takes ~2 minutes; run with POKER_EXHAUSTIVE=1",
)
class TestExhaustiveFrequencies(unittest.TestCase):
    """Every 5-card hand must classify to the known textbook frequencies.

    Skipped by default to keep the suite fast. This is the highest-value test
    in the file, so run it before trusting any change to the evaluator:

        POKER_EXHAUSTIVE=1 python -m pytest tests/test_poker_evaluator.py
    """

    #: Standard 5-card poker hand frequencies over C(52,5) = 2,598,960.
    EXPECTED = {
        HandCategory.STRAIGHT_FLUSH: 40,
        HandCategory.FOUR_OF_A_KIND: 624,
        HandCategory.FULL_HOUSE: 3_744,
        HandCategory.FLUSH: 5_108,
        HandCategory.STRAIGHT: 10_200,
        HandCategory.THREE_OF_A_KIND: 54_912,
        HandCategory.TWO_PAIR: 123_552,
        HandCategory.PAIR: 1_098_240,
        HandCategory.HIGH_CARD: 1_302_540,
    }

    def test_all_five_card_hands(self) -> None:
        counts: Counter = Counter()
        for hand in combinations(FULL_DECK, 5):
            counts[evaluate(hand).category] += 1

        self.assertEqual(sum(counts.values()), 2_598_960)
        for category, expected in self.EXPECTED.items():
            self.assertEqual(
                counts[category], expected, f"{category.name} frequency is wrong"
            )


if __name__ == "__main__":
    unittest.main()
