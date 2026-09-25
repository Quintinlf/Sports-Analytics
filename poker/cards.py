"""Card primitives: ranks, suits, cards, and a reproducible deck.

Every deck carries the seed it was shuffled with so any hand can be replayed
card-for-card later (see the data-integrity requirement in the project brief).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum, IntEnum
from functools import total_ordering
from typing import Iterable, Iterator, List, Optional, Sequence

__all__ = [
    "Rank",
    "Suit",
    "Card",
    "Deck",
    "RANKS",
    "SUITS",
    "FULL_DECK",
    "parse_cards",
]


class Rank(IntEnum):
    """Card rank. Integer values order naturally, ace high."""

    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    @property
    def symbol(self) -> str:
        return _RANK_SYMBOLS[self]

    @property
    def name_singular(self) -> str:
        return _RANK_NAMES[self]

    @property
    def name_plural(self) -> str:
        name = _RANK_NAMES[self]
        return name + "es" if name == "Six" else name + "s"

    @classmethod
    def from_symbol(cls, symbol: str) -> "Rank":
        try:
            return _SYMBOL_TO_RANK[symbol.upper()]
        except KeyError:
            raise ValueError(f"Unknown rank symbol: {symbol!r}") from None


_RANK_SYMBOLS = {
    Rank.TWO: "2", Rank.THREE: "3", Rank.FOUR: "4", Rank.FIVE: "5",
    Rank.SIX: "6", Rank.SEVEN: "7", Rank.EIGHT: "8", Rank.NINE: "9",
    Rank.TEN: "T", Rank.JACK: "J", Rank.QUEEN: "Q", Rank.KING: "K",
    Rank.ACE: "A",
}

_RANK_NAMES = {
    Rank.TWO: "Two", Rank.THREE: "Three", Rank.FOUR: "Four", Rank.FIVE: "Five",
    Rank.SIX: "Six", Rank.SEVEN: "Seven", Rank.EIGHT: "Eight", Rank.NINE: "Nine",
    Rank.TEN: "Ten", Rank.JACK: "Jack", Rank.QUEEN: "Queen", Rank.KING: "King",
    Rank.ACE: "Ace",
}

_SYMBOL_TO_RANK = {symbol: rank for rank, symbol in _RANK_SYMBOLS.items()}
# Accept "10" alongside "T" when parsing user-supplied strings.
_SYMBOL_TO_RANK["10"] = Rank.TEN


@total_ordering
class Suit(Enum):
    """Card suit.

    SUITS ARE NEVER RANKED AGAINST EACH OTHER IN HOLD'EM. No hand beats another
    because of its suit, split pots are split, and ``sort_index`` must never be
    used to decide a pot. ``evaluator.py`` does not consult it, and nothing that
    awards chips should.

    They are nonetheless given a total order here, for one narrow reason:
    ``Card`` is declared ``order=True``, so comparing two cards falls back to
    comparing their suits whenever the ranks are equal. With suits unordered
    that raised ``TypeError``, which made ``sorted(FULL_DECK)`` — and any sort
    of a hand containing a pair — fail outright.

    The sequence below is the conventional bridge order (clubs, diamonds,
    hearts, spades). It exists so that sorting cards is deterministic and
    reproducible, and it carries no game meaning whatsoever.
    """

    CLUBS = ("c", "♣", "Clubs", 0)
    DIAMONDS = ("d", "♦", "Diamonds", 1)
    HEARTS = ("h", "♥", "Hearts", 2)
    SPADES = ("s", "♠", "Spades", 3)

    def __init__(self, letter: str, glyph: str, label: str, sort_index: int) -> None:
        self.letter = letter
        self.glyph = glyph
        self.label = label
        self.sort_index = sort_index

    def __lt__(self, other: object) -> bool:
        """Presentation-order tie-break only; never a ranking. See the docstring."""
        if not isinstance(other, Suit):
            return NotImplemented
        return self.sort_index < other.sort_index

    @property
    def is_red(self) -> bool:
        return self in (Suit.DIAMONDS, Suit.HEARTS)

    @classmethod
    def from_letter(cls, letter: str) -> "Suit":
        try:
            return _LETTER_TO_SUIT[letter.lower()]
        except KeyError:
            raise ValueError(f"Unknown suit letter: {letter!r}") from None


_LETTER_TO_SUIT = {suit.letter: suit for suit in Suit}
# Also accept the glyphs themselves so pasted hands parse.
_LETTER_TO_SUIT.update({suit.glyph: suit for suit in Suit})


@dataclass(frozen=True, order=True)
class Card:
    """A single playing card. Immutable and hashable so it works in sets."""

    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        """Compact machine form, e.g. ``As``. Used for storage and tests."""
        return f"{self.rank.symbol}{self.suit.letter}"

    def __repr__(self) -> str:
        return f"Card({self})"

    @property
    def display(self) -> str:
        """Human form with the suit glyph, e.g. ``A♠``."""
        return f"{self.rank.symbol}{self.suit.glyph}"

    @classmethod
    def from_str(cls, text: str) -> "Card":
        """Parse ``As``, ``10h``, or ``A♠`` into a card."""
        cleaned = text.strip()
        if len(cleaned) < 2:
            raise ValueError(f"Cannot parse card from {text!r}")
        return cls(Rank.from_symbol(cleaned[:-1]), Suit.from_letter(cleaned[-1]))


RANKS: tuple[Rank, ...] = tuple(Rank)
SUITS: tuple[Suit, ...] = tuple(Suit)

#: The canonical 52 cards in a fixed order. Never mutate this.
FULL_DECK: tuple[Card, ...] = tuple(
    Card(rank, suit) for suit in SUITS for rank in RANKS
)


def parse_cards(text: str | Iterable[str]) -> List[Card]:
    """Parse a whitespace- or comma-separated card list.

    ``parse_cards("As Kd")`` and ``parse_cards(["As", "Kd"])`` are equivalent.
    Raises on duplicates, which are always a bug rather than a valid input.
    """
    if isinstance(text, str):
        tokens = text.replace(",", " ").split()
    else:
        tokens = [str(token) for token in text]

    cards = [Card.from_str(token) for token in tokens]
    if len(set(cards)) != len(cards):
        raise ValueError(f"Duplicate cards in {text!r}")
    return cards


class Deck:
    """A shuffled 52-card deck that records its seed for replay.

    Cards are dealt from the end of the internal list, so dealing is O(1).
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        *,
        exclude: Sequence[Card] = (),
    ) -> None:
        if seed is None:
            seed = random.SystemRandom().randrange(2**63)
        self.seed = seed
        self._rng = random.Random(seed)

        excluded = set(exclude)
        self._cards: List[Card] = [c for c in FULL_DECK if c not in excluded]
        self._rng.shuffle(self._cards)
        self._dealt: List[Card] = []

    def __len__(self) -> int:
        return len(self._cards)

    def __iter__(self) -> Iterator[Card]:
        """Iterate undealt cards. Order is the shuffled order, not deal order."""
        return iter(self._cards)

    @property
    def remaining(self) -> tuple[Card, ...]:
        return tuple(self._cards)

    @property
    def dealt(self) -> tuple[Card, ...]:
        """Cards removed so far, in the order they were dealt."""
        return tuple(self._dealt)

    def deal(self, count: int = 1) -> List[Card]:
        """Remove and return ``count`` cards from the top of the deck."""
        if count < 0:
            raise ValueError("Cannot deal a negative number of cards")
        if count > len(self._cards):
            raise ValueError(
                f"Cannot deal {count} cards; only {len(self._cards)} remain"
            )
        drawn = [self._cards.pop() for _ in range(count)]
        self._dealt.extend(drawn)
        return drawn

    def deal_one(self) -> Card:
        return self.deal(1)[0]

    def remove(self, cards: Iterable[Card]) -> None:
        """Remove specific cards from the deck (for constructed scenarios).

        Used by tests and by the drill/scenario tooling to force a board.
        """
        for card in cards:
            try:
                self._cards.remove(card)
            except ValueError:
                raise ValueError(f"{card} is not in the deck") from None
            self._dealt.append(card)
