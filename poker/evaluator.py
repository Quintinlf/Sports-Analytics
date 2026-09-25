"""Texas Hold'em hand evaluation.

Evaluates any 5, 6, or 7 card set to its single best five-card poker hand.

The approach is direct classification (count ranks, detect flush, detect
straight) rather than a precomputed lookup table. This is a deliberate choice:
the laboratory needs to *explain* why a hand beats another, and a table of
7462 magic integers cannot explain anything. It is also fast enough — no
combinatorial 21-subset search is performed.

Two hands compare via ``HandValue.key``: the category first, then a tuple of
tiebreaker ranks. Equal keys mean a genuine tie, which is what split pots
depend on, so the tiebreakers are exhaustive rather than approximate.
"""
from __future__ import annotations

from collections import Counter
from enum import IntEnum
from functools import total_ordering
from typing import Dict, Iterable, List, Sequence, Tuple

from poker.cards import Card, Rank, Suit

__all__ = [
    "HandCategory",
    "HandValue",
    "evaluate",
    "compare",
    "best_hand_indices",
]


class HandCategory(IntEnum):
    """Poker hand categories, ordered weakest to strongest."""

    HIGH_CARD = 1
    PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9

    @property
    def label(self) -> str:
        return _CATEGORY_LABELS[self]


_CATEGORY_LABELS = {
    HandCategory.HIGH_CARD: "High card",
    HandCategory.PAIR: "Pair",
    HandCategory.TWO_PAIR: "Two pair",
    HandCategory.THREE_OF_A_KIND: "Three of a kind",
    HandCategory.STRAIGHT: "Straight",
    HandCategory.FLUSH: "Flush",
    HandCategory.FULL_HOUSE: "Full house",
    HandCategory.FOUR_OF_A_KIND: "Four of a kind",
    HandCategory.STRAIGHT_FLUSH: "Straight flush",
}

#: The five-high straight, where the ace plays low.
WHEEL_RANKS: Tuple[int, ...] = (5, 4, 3, 2, 14)


@total_ordering
class HandValue:
    """The best five-card hand available from a card set.

    ``key`` is the full comparison ordering. ``cards`` is the specific five
    cards that make the hand, retained so the interface can highlight them.
    """

    __slots__ = ("category", "tiebreakers", "cards")

    def __init__(
        self,
        category: HandCategory,
        tiebreakers: Sequence[int],
        cards: Sequence[Card],
    ) -> None:
        self.category = category
        self.tiebreakers: Tuple[int, ...] = tuple(tiebreakers)
        self.cards: Tuple[Card, ...] = tuple(cards)

    @property
    def key(self) -> Tuple[int, ...]:
        """Total ordering key. Equal keys are exact ties (split pot)."""
        return (int(self.category),) + self.tiebreakers

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HandValue):
            return NotImplemented
        return self.key == other.key

    def __lt__(self, other: "HandValue") -> bool:
        if not isinstance(other, HandValue):
            return NotImplemented
        return self.key < other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        return f"HandValue({self.describe()})"

    def describe(self) -> str:
        """Plain-English name, e.g. ``Full house, Kings full of Threes``."""
        return _describe(self)

    def to_dict(self) -> Dict[str, object]:
        return {
            "category": self.category.name,
            "category_rank": int(self.category),
            "description": self.describe(),
            "tiebreakers": list(self.tiebreakers),
            "cards": [str(card) for card in self.cards],
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(cards: Iterable[Card]) -> HandValue:
    """Return the best five-card hand from 5, 6, or 7 cards."""
    card_list = list(cards)
    if not 5 <= len(card_list) <= 7:
        raise ValueError(
            f"Hand evaluation needs 5-7 cards, got {len(card_list)}"
        )
    if len(set(card_list)) != len(card_list):
        raise ValueError(f"Duplicate cards: {[str(c) for c in card_list]}")

    rank_counts = Counter(card.rank for card in card_list)
    suit_counts = Counter(card.suit for card in card_list)

    # With 7 cards at most one suit can reach five, so this is unambiguous.
    flush_suit: Suit | None = next(
        (suit for suit, count in suit_counts.items() if count >= 5), None
    )

    if flush_suit is not None:
        flush_cards = _sorted_desc(
            [card for card in card_list if card.suit == flush_suit]
        )
        straight_high = _straight_high({int(c.rank) for c in flush_cards})
        if straight_high:
            return HandValue(
                HandCategory.STRAIGHT_FLUSH,
                (straight_high,),
                _select_straight(flush_cards, straight_high),
            )

    # Ranks grouped by how many times they appear, each group high-to-low.
    by_count: Dict[int, List[int]] = {}
    for rank, count in rank_counts.items():
        by_count.setdefault(count, []).append(int(rank))
    for group in by_count.values():
        group.sort(reverse=True)

    quads = by_count.get(4, [])
    trips = by_count.get(3, [])
    pairs = by_count.get(2, [])

    if quads:
        quad_rank = quads[0]
        kicker = _best_kickers(card_list, exclude={quad_rank}, count=1)
        return HandValue(
            HandCategory.FOUR_OF_A_KIND,
            (quad_rank, *kicker),
            _cards_of_rank(card_list, quad_rank, 4)
            + _select_kickers(card_list, kicker),
        )

    # A full house can come from trips+pair, or from two sets of trips
    # (the lower set contributes only two cards).
    if trips and (len(trips) > 1 or pairs):
        trip_rank = trips[0]
        pair_candidates = trips[1:] + pairs
        pair_rank = max(pair_candidates)
        return HandValue(
            HandCategory.FULL_HOUSE,
            (trip_rank, pair_rank),
            _cards_of_rank(card_list, trip_rank, 3)
            + _cards_of_rank(card_list, pair_rank, 2),
        )

    if flush_suit is not None:
        flush_cards = _sorted_desc(
            [card for card in card_list if card.suit == flush_suit]
        )[:5]
        return HandValue(
            HandCategory.FLUSH,
            tuple(int(card.rank) for card in flush_cards),
            flush_cards,
        )

    straight_high = _straight_high({int(card.rank) for card in card_list})
    if straight_high:
        return HandValue(
            HandCategory.STRAIGHT,
            (straight_high,),
            _select_straight(_sorted_desc(card_list), straight_high),
        )

    if trips:
        trip_rank = trips[0]
        kickers = _best_kickers(card_list, exclude={trip_rank}, count=2)
        return HandValue(
            HandCategory.THREE_OF_A_KIND,
            (trip_rank, *kickers),
            _cards_of_rank(card_list, trip_rank, 3)
            + _select_kickers(card_list, kickers),
        )

    if len(pairs) >= 2:
        high_pair, low_pair = pairs[0], pairs[1]
        # A third pair cannot play as a pair, but its cards remain kicker
        # candidates — hence excluding only the two pairs that do play.
        kicker = _best_kickers(
            card_list, exclude={high_pair, low_pair}, count=1
        )
        return HandValue(
            HandCategory.TWO_PAIR,
            (high_pair, low_pair, *kicker),
            _cards_of_rank(card_list, high_pair, 2)
            + _cards_of_rank(card_list, low_pair, 2)
            + _select_kickers(card_list, kicker),
        )

    if len(pairs) == 1:
        pair_rank = pairs[0]
        kickers = _best_kickers(card_list, exclude={pair_rank}, count=3)
        return HandValue(
            HandCategory.PAIR,
            (pair_rank, *kickers),
            _cards_of_rank(card_list, pair_rank, 2)
            + _select_kickers(card_list, kickers),
        )

    best_five = _sorted_desc(card_list)[:5]
    return HandValue(
        HandCategory.HIGH_CARD,
        tuple(int(card.rank) for card in best_five),
        best_five,
    )


def compare(left: Iterable[Card], right: Iterable[Card]) -> int:
    """Compare two card sets: 1 if left wins, -1 if right wins, 0 on a tie."""
    left_value, right_value = evaluate(left), evaluate(right)
    if left_value > right_value:
        return 1
    if left_value < right_value:
        return -1
    return 0


def best_hand_indices(hands: Sequence[Iterable[Card]]) -> List[int]:
    """Indices of the winning hands. More than one index means a split pot."""
    if not hands:
        return []
    values = [evaluate(hand) for hand in hands]
    best = max(values)
    return [i for i, value in enumerate(values) if value == best]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _sorted_desc(cards: Sequence[Card]) -> List[Card]:
    return sorted(cards, key=lambda card: int(card.rank), reverse=True)


def _straight_high(ranks: set[int]) -> int:
    """Highest straight top-card in ``ranks``, or 0 if there is no straight.

    The wheel (A-2-3-4-5) is checked last and reports 5, since the ace plays
    low there and a five-high straight is the weakest straight.
    """
    for high in range(14, 5, -1):
        if all(high - offset in ranks for offset in range(5)):
            return high
    if all(rank in ranks for rank in WHEEL_RANKS):
        return 5
    return 0


def _select_straight(cards: Sequence[Card], high: int) -> List[Card]:
    """Pick one card per rank of the straight, highest rank first."""
    wanted = WHEEL_RANKS if high == 5 else tuple(range(high, high - 5, -1))
    chosen: List[Card] = []
    for rank in wanted:
        chosen.append(next(c for c in cards if int(c.rank) == rank))
    return chosen


def _cards_of_rank(cards: Sequence[Card], rank: int, count: int) -> List[Card]:
    return [card for card in cards if int(card.rank) == rank][:count]


def _best_kickers(
    cards: Sequence[Card], *, exclude: set[int], count: int
) -> Tuple[int, ...]:
    """Highest ``count`` kicker ranks, skipping ranks already used."""
    kickers = sorted(
        (int(c.rank) for c in cards if int(c.rank) not in exclude),
        reverse=True,
    )
    return tuple(kickers[:count])


def _select_kickers(
    cards: Sequence[Card], kicker_ranks: Sequence[int]
) -> List[Card]:
    """Resolve kicker ranks back to concrete cards without reusing one."""
    remaining = list(cards)
    chosen: List[Card] = []
    for rank in kicker_ranks:
        card = next(c for c in remaining if int(c.rank) == rank)
        remaining.remove(card)
        chosen.append(card)
    return chosen


def _describe(value: HandValue) -> str:
    tiebreakers = value.tiebreakers
    category = value.category

    def plural(rank_value: int) -> str:
        return Rank(rank_value).name_plural

    def singular(rank_value: int) -> str:
        return Rank(rank_value).name_singular

    if category is HandCategory.STRAIGHT_FLUSH:
        if tiebreakers[0] == 14:
            return "Royal flush"
        return f"Straight flush, {singular(tiebreakers[0])} high"
    if category is HandCategory.FOUR_OF_A_KIND:
        return f"Four of a kind, {plural(tiebreakers[0])}"
    if category is HandCategory.FULL_HOUSE:
        return (
            f"Full house, {plural(tiebreakers[0])} "
            f"full of {plural(tiebreakers[1])}"
        )
    if category is HandCategory.FLUSH:
        return f"Flush, {singular(tiebreakers[0])} high"
    if category is HandCategory.STRAIGHT:
        return f"Straight, {singular(tiebreakers[0])} high"
    if category is HandCategory.THREE_OF_A_KIND:
        return f"Three of a kind, {plural(tiebreakers[0])}"
    if category is HandCategory.TWO_PAIR:
        return f"Two pair, {plural(tiebreakers[0])} and {plural(tiebreakers[1])}"
    if category is HandCategory.PAIR:
        return f"Pair of {plural(tiebreakers[0])}"
    return f"{singular(tiebreakers[0])} high"
