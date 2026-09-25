"""Card primitives and deck reproducibility."""
from __future__ import annotations

import unittest

from poker.cards import FULL_DECK, Card, Deck, Rank, Suit, parse_cards


class TestCardParsing(unittest.TestCase):
    def test_round_trip(self) -> None:
        for card in FULL_DECK:
            self.assertEqual(Card.from_str(str(card)), card)

    def test_parses_ten_both_ways(self) -> None:
        self.assertEqual(Card.from_str("Th"), Card.from_str("10h"))

    def test_parses_suit_glyphs(self) -> None:
        self.assertEqual(Card.from_str("A♠"), Card(Rank.ACE, Suit.SPADES))

    def test_case_insensitive_rank(self) -> None:
        self.assertEqual(Card.from_str("as"), Card(Rank.ACE, Suit.SPADES))

    def test_display_uses_glyphs(self) -> None:
        self.assertEqual(Card(Rank.ACE, Suit.SPADES).display, "A♠")

    def test_rejects_nonsense(self) -> None:
        for bad in ("", "X", "1h", "Ax"):
            with self.assertRaises(ValueError, msg=bad):
                Card.from_str(bad)

    def test_parse_cards_rejects_duplicates(self) -> None:
        with self.assertRaises(ValueError):
            parse_cards("As As")

    def test_parse_cards_accepts_commas_and_lists(self) -> None:
        self.assertEqual(parse_cards("As, Kd"), parse_cards(["As", "Kd"]))


class TestRanksAndSuits(unittest.TestCase):
    def test_ace_is_high(self) -> None:
        self.assertGreater(Rank.ACE, Rank.KING)

    def test_pluralisation(self) -> None:
        self.assertEqual(Rank.SIX.name_plural, "Sixes")
        self.assertEqual(Rank.ACE.name_plural, "Aces")
        self.assertEqual(Rank.TWO.name_plural, "Twos")

    def test_red_suits(self) -> None:
        self.assertTrue(Suit.HEARTS.is_red)
        self.assertFalse(Suit.SPADES.is_red)


class TestCardOrdering(unittest.TestCase):
    """Card declares order=True, so suits need a total order to back it.

    Without one, any comparison of two equal-rank cards raised TypeError and
    sorting a deck was impossible. The ordering is a deterministic tie-break
    for presentation only; TestSuitOrderIsNotAHandRanking pins down that it
    never affects who wins.
    """

    def test_sorting_a_full_deck_succeeds(self) -> None:
        ordered = sorted(FULL_DECK)
        self.assertEqual(len(ordered), 52)
        self.assertEqual(str(ordered[0]), "2c")
        self.assertEqual(str(ordered[-1]), "As")

    def test_equal_rank_cards_compare_without_raising(self) -> None:
        for left, right in (
            (Suit.SPADES, Suit.HEARTS),
            (Suit.CLUBS, Suit.CLUBS),
            (Suit.DIAMONDS, Suit.SPADES),
        ):
            first, second = Card(Rank.ACE, left), Card(Rank.ACE, right)
            self.assertIsInstance(first < second, bool)
            self.assertIsInstance(first >= second, bool)

    def test_sorting_a_hand_containing_a_pair(self) -> None:
        """The case that actually bit: any hand with two cards of one rank."""
        hand = parse_cards("As Ah Kd 2c")
        self.assertEqual(len(sorted(hand)), 4)

    def test_rank_dominates_suit(self) -> None:
        self.assertLess(Card(Rank.TWO, Suit.SPADES), Card(Rank.THREE, Suit.CLUBS))

    def test_suit_order_is_total_and_deterministic(self) -> None:
        self.assertEqual(
            sorted(Suit),
            [Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES],
        )
        self.assertLess(Suit.CLUBS, Suit.SPADES)
        self.assertGreaterEqual(Suit.SPADES, Suit.HEARTS)

    def test_suit_equality_is_unchanged(self) -> None:
        self.assertEqual(Suit.HEARTS, Suit.HEARTS)
        self.assertNotEqual(Suit.HEARTS, Suit.SPADES)

    def test_suits_refuse_comparison_with_other_types(self) -> None:
        with self.assertRaises(TypeError):
            Suit.CLUBS < 3


class TestSuitOrderIsNotAHandRanking(unittest.TestCase):
    """The ordering must not leak into who wins a pot."""

    def test_identical_hands_in_different_suits_tie(self) -> None:
        from poker.evaluator import compare

        spades = parse_cards("As Ks Qs Js 9s")
        hearts = parse_cards("Ah Kh Qh Jh 9h")
        self.assertEqual(compare(spades, hearts), 0)

    def test_the_higher_suit_does_not_win(self) -> None:
        from poker.evaluator import compare

        # Same ranks, only the suits differ; spades must not beat clubs.
        spades = parse_cards("As Kd Qh Jc 9s")
        clubs = parse_cards("Ac Kh Qs Jd 9c")
        self.assertEqual(compare(spades, clubs), 0)


class TestDeck(unittest.TestCase):
    def test_deck_has_52_unique_cards(self) -> None:
        deck = Deck(seed=1)
        self.assertEqual(len(deck), 52)
        self.assertEqual(len(set(deck.remaining)), 52)

    def test_same_seed_deals_the_same_cards(self) -> None:
        self.assertEqual(Deck(seed=99).deal(10), Deck(seed=99).deal(10))

    def test_different_seeds_differ(self) -> None:
        self.assertNotEqual(Deck(seed=1).deal(10), Deck(seed=2).deal(10))

    def test_dealing_removes_cards(self) -> None:
        deck = Deck(seed=1)
        drawn = deck.deal(5)
        self.assertEqual(len(deck), 47)
        for card in drawn:
            self.assertNotIn(card, deck.remaining)

    def test_cannot_overdraw(self) -> None:
        deck = Deck(seed=1)
        with self.assertRaises(ValueError):
            deck.deal(53)

    def test_exclude_removes_known_cards(self) -> None:
        excluded = parse_cards("As Kd")
        deck = Deck(seed=1, exclude=excluded)
        self.assertEqual(len(deck), 50)
        for card in excluded:
            self.assertNotIn(card, deck.remaining)

    def test_remove_specific_cards(self) -> None:
        deck = Deck(seed=1)
        deck.remove(parse_cards("As"))
        self.assertEqual(len(deck), 51)
        with self.assertRaises(ValueError):
            deck.remove(parse_cards("As"))

    def test_seed_is_recorded_for_replay(self) -> None:
        deck = Deck(seed=4242)
        self.assertEqual(deck.seed, 4242)
        # An unseeded deck still records whatever seed it chose.
        self.assertIsInstance(Deck().seed, int)


if __name__ == "__main__":
    unittest.main()
