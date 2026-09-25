"""Hand ranges as sets, with the set algebra poker actually needs.

What poker literature calls "combinatorics" is applied set theory, and saying so
out loud makes the arithmetic much easier to get right. Concretely:

* A **combo** is a set of two cards. There are C(52,2) = 1326 of them.
* A **range** is a set of combos. Every range operation is a set operation.
* A **blocker** is set difference. Holding the A-spades removes every combo
  containing it, which is why A-A drops from 6 combos to 3 when you hold an ace.
* **Outs** are a union, and unions need inclusion-exclusion. A flush draw plus
  an open-ended straight draw is *not* 9 + 8 = 17 outs, because some cards
  belong to both sets.

The last point is where players lose money. |A union B| = |A| + |B| -
|A intersection B| is the whole correction, and :func:`draw_analysis` computes
each term from the actual card sets rather than from a remembered rule.

SCOPE
-----
This module is exact combinatorics on card sets. It counts combos and computes
outs; it does not estimate equity, which needs enumeration against a range and
belongs to Milestone 3. Nothing here produces a probability that was not
obtained by dividing one exact count by another.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import FrozenSet, Iterable, Iterator, Optional, Sequence

from poker.cards import FULL_DECK, Card, Rank, Suit, parse_cards

__all__ = [
    "Combo",
    "Range",
    "ALL_COMBOS",
    "parse_range",
    "draw_analysis",
    "DrawAnalysis",
    "flush_completing_cards",
    "straight_completing_cards",
]


@dataclass(frozen=True)
class Combo:
    """Two specific cards. A set of size two, with poker vocabulary attached."""

    cards: FrozenSet[Card]

    def __post_init__(self) -> None:
        if len(self.cards) != 2:
            raise ValueError(f"a combo holds exactly two cards, got {len(self.cards)}")

    @classmethod
    def of(cls, first: Card, second: Card) -> "Combo":
        if first == second:
            raise ValueError("a combo needs two distinct cards")
        return cls(frozenset((first, second)))

    @classmethod
    def parse(cls, text: str) -> "Combo":
        """Parse two concrete cards, e.g. ``AsKh``."""
        cards = parse_cards(text) if " " in text or "," in text else _split_pair(text)
        if len(cards) != 2:
            raise ValueError(f"expected two cards in {text!r}")
        return cls.of(*cards)

    @property
    def _ordered(self) -> list[Card]:
        """High card first.

        ``Card`` sorts by rank, then by suit as a deterministic tie-break that
        carries no game meaning (see ``cards.Suit``). This needed an explicit
        key until equal-rank comparison stopped raising.
        """
        return sorted(self.cards, reverse=True)

    @property
    def high(self) -> Card:
        return self._ordered[0]

    @property
    def low(self) -> Card:
        return self._ordered[1]

    @property
    def ranks(self) -> tuple[Rank, Rank]:
        return (self.high.rank, self.low.rank)

    @property
    def is_pair(self) -> bool:
        return self.high.rank == self.low.rank

    @property
    def is_suited(self) -> bool:
        return self.high.suit == self.low.suit and not self.is_pair

    def notation(self) -> str:
        """Canonical shorthand: ``AA``, ``AKs``, ``AKo``."""
        high, low = self.ranks
        if self.is_pair:
            return f"{high.symbol}{low.symbol}"
        return f"{high.symbol}{low.symbol}{'s' if self.is_suited else 'o'}"

    def __iter__(self) -> Iterator[Card]:
        return iter(sorted(self.cards, reverse=True))

    def __str__(self) -> str:
        return f"{self.high}{self.low}"


def _split_pair(text: str) -> list[Card]:
    cleaned = text.strip()
    for cut in (2, 3):
        if len(cleaned) > cut:
            try:
                return [
                    Card.from_str(cleaned[:cut]),
                    Card.from_str(cleaned[cut:]),
                ]
            except ValueError:
                continue
    raise ValueError(f"cannot parse two cards from {text!r}")


#: Every one of the 1326 two-card combinations.
ALL_COMBOS: FrozenSet[Combo] = frozenset(
    Combo.of(a, b) for a, b in combinations(FULL_DECK, 2)
)


class Range:
    """A set of combos, supporting the full set algebra.

    ``|`` union, ``&`` intersection, ``-`` difference, ``^`` symmetric
    difference, ``~`` complement (within all 1326), ``<=`` subset.
    """

    __slots__ = ("_combos", "_label")

    def __init__(
        self, combos: Iterable[Combo] = (), *, label: str = ""
    ) -> None:
        self._combos: FrozenSet[Combo] = frozenset(combos)
        self._label = label

    # -- construction -------------------------------------------------------

    @classmethod
    def parse(cls, text: str) -> "Range":
        return parse_range(text)

    @classmethod
    def everything(cls) -> "Range":
        return cls(ALL_COMBOS, label="any two")

    # -- set algebra --------------------------------------------------------

    @property
    def combos(self) -> FrozenSet[Combo]:
        return self._combos

    def __len__(self) -> int:
        return len(self._combos)

    def __iter__(self) -> Iterator[Combo]:
        return iter(self._combos)

    def __contains__(self, combo: object) -> bool:
        return combo in self._combos

    def __or__(self, other: "Range") -> "Range":
        return Range(self._combos | other._combos)

    def __and__(self, other: "Range") -> "Range":
        return Range(self._combos & other._combos)

    def __sub__(self, other: "Range") -> "Range":
        return Range(self._combos - other._combos)

    def __xor__(self, other: "Range") -> "Range":
        return Range(self._combos ^ other._combos)

    def __invert__(self) -> "Range":
        return Range(ALL_COMBOS - self._combos)

    def __le__(self, other: "Range") -> bool:
        return self._combos <= other._combos

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Range) and self._combos == other._combos

    def __hash__(self) -> int:
        return hash(self._combos)

    # -- poker-specific ------------------------------------------------------

    def remove_dead(self, cards: Iterable[Card]) -> "Range":
        """Drop every combo touching a known card. Blockers, as set difference.

        This is the single most under-used piece of arithmetic in poker. If you
        hold an ace, your opponent's A-A combos fall from 6 to 3 — their range
        is literally half as likely to hold aces as the naive count suggests.
        """
        dead = set(cards)
        return Range(
            combo for combo in self._combos if not (combo.cards & dead)
        )

    def containing(self, card: Card) -> "Range":
        """The sub-range whose combos include a specific card."""
        return Range(combo for combo in self._combos if card in combo.cards)

    @property
    def percent(self) -> float:
        """Share of all 1326 combos, the usual way ranges are quoted."""
        return len(self._combos) / len(ALL_COMBOS)

    def notations(self) -> dict[str, int]:
        """Combo count per shorthand, e.g. ``{'AA': 6, 'AKs': 4}``."""
        counter: Counter[str] = Counter(c.notation() for c in self._combos)
        return dict(sorted(counter.items()))

    def __repr__(self) -> str:
        name = self._label or f"{len(self._combos)} combos"
        return f"<Range {name} ({self.percent:.1%})>"


# ---------------------------------------------------------------------------
# Range notation
# ---------------------------------------------------------------------------

_RANK_ORDER = sorted(Rank, reverse=True)


def _combos_for(high: Rank, low: Rank, form: str) -> set[Combo]:
    """All concrete combos for a shorthand like AKs / AKo / AK / AA."""
    result: set[Combo] = set()
    if high == low:
        for a, b in combinations([Card(high, s) for s in Suit], 2):
            result.add(Combo.of(a, b))
        return result

    for first in Suit:
        for second in Suit:
            suited = first == second
            if form == "s" and not suited:
                continue
            if form == "o" and suited:
                continue
            result.add(Combo.of(Card(high, first), Card(low, second)))
    return result


def _parse_token(token: str) -> set[Combo]:
    text = token.strip()
    if not text:
        return set()

    plus = text.endswith("+")
    if plus:
        text = text[:-1]

    if "-" in text and not plus:
        start, _, end = text.partition("-")
        low_end = _parse_token(end)
        high_end = _parse_token(start)
        if not (low_end and high_end):
            raise ValueError(f"cannot parse range span {token!r}")
        spans = sorted(
            {c.ranks[0] for c in low_end} | {c.ranks[0] for c in high_end}
        )
        result: set[Combo] = set()
        for rank in _RANK_ORDER:
            if spans[0] <= rank <= spans[-1]:
                result |= _parse_token(_retag(text, rank))
        return result

    form = ""
    if text and text[-1] in "so":
        form = text[-1]
        text = text[:-1]

    symbols = _split_symbols(text)
    if len(symbols) != 2:
        raise ValueError(f"cannot parse hand {token!r}")
    high, low = (Rank.from_symbol(symbols[0]), Rank.from_symbol(symbols[1]))
    if low > high:
        high, low = low, high

    if not plus:
        return _combos_for(high, low, form)

    # "JJ+" walks the pairs upward; "ATs+" walks the kicker upward.
    result = set()
    if high == low:
        for rank in _RANK_ORDER:
            if rank >= high:
                result |= _combos_for(rank, rank, form)
    else:
        for rank in _RANK_ORDER:
            if low <= rank < high:
                result |= _combos_for(high, rank, form)
    return result


def _retag(template: str, rank: Rank) -> str:
    form = template[-1] if template and template[-1] in "so" else ""
    return f"{rank.symbol}{rank.symbol}{form}"


def _split_symbols(text: str) -> list[str]:
    if text.startswith("10"):
        return ["10", text[2:]]
    if text.endswith("10"):
        return [text[:-2], "10"]
    return list(text)


def parse_range(text: str) -> Range:
    """Parse standard range notation.

    Accepts ``AA``, ``AKs``, ``AKo``, ``AK`` (both forms), ``JJ+``, ``ATs+``,
    and ``22-88``, comma- or space-separated.
    """
    combos: set[Combo] = set()
    for token in text.replace(",", " ").split():
        combos |= _parse_token(token)
    return Range(combos, label=text.strip())


# ---------------------------------------------------------------------------
# Outs, and why they must be unioned rather than added
# ---------------------------------------------------------------------------


def _has_flush(cards: Sequence[Card]) -> bool:
    return max(Counter(card.suit for card in cards).values(), default=0) >= 5


def _has_straight(cards: Sequence[Card]) -> bool:
    ranks = {int(card.rank) for card in cards}
    if Rank.ACE in {card.rank for card in cards}:
        ranks.add(1)  # the wheel: A-2-3-4-5
    return any(
        all(low + offset in ranks for offset in range(5))
        for low in range(1, 11)
    )


def _live_cards(known: Iterable[Card]) -> list[Card]:
    dead = set(known)
    return [card for card in FULL_DECK if card not in dead]


def flush_completing_cards(hole: Sequence[Card], board: Sequence[Card]) -> set[Card]:
    """Cards that put five of one suit into your seven."""
    known = list(hole) + list(board)
    if _has_flush(known):
        return set()
    return {card for card in _live_cards(known) if _has_flush(known + [card])}


def straight_completing_cards(
    hole: Sequence[Card], board: Sequence[Card]
) -> set[Card]:
    """Cards that put a five-card run into your seven."""
    known = list(hole) + list(board)
    if _has_straight(known):
        return set()
    return {card for card in _live_cards(known) if _has_straight(known + [card])}


@dataclass(frozen=True)
class DrawAnalysis:
    """Outs decomposed as sets, so the overlap is visible rather than assumed."""

    flush_outs: frozenset[Card]
    straight_outs: frozenset[Card]
    both: frozenset[Card]
    union: frozenset[Card]
    unseen: int

    @property
    def naive_sum(self) -> int:
        """What adding the two draws together would wrongly give."""
        return len(self.flush_outs) + len(self.straight_outs)

    @property
    def double_counted(self) -> int:
        return len(self.both)

    def probability_next_card(self) -> float:
        """Exact chance the very next card is an out. A ratio of counts."""
        return len(self.union) / self.unseen if self.unseen else 0.0

    def summary(self) -> str:
        return "\n".join(
            [
                f"  flush outs      {len(self.flush_outs):>3}  "
                + " ".join(sorted(str(c) for c in self.flush_outs)),
                f"  straight outs   {len(self.straight_outs):>3}  "
                + " ".join(sorted(str(c) for c in self.straight_outs)),
                f"  in both sets    {len(self.both):>3}  "
                + " ".join(sorted(str(c) for c in self.both)),
                f"  naive sum       {self.naive_sum:>3}  "
                "(wrong: double-counts the intersection)",
                f"  |A union B|     {len(self.union):>3}  "
                f"= {len(self.flush_outs)} + {len(self.straight_outs)} "
                f"- {self.double_counted}",
                f"  next card hits  {self.probability_next_card():.1%}  "
                f"({len(self.union)} of {self.unseen} unseen)",
            ]
        )


def draw_analysis(
    hole: Sequence[Card], board: Sequence[Card], *, opponents: int = 0
) -> DrawAnalysis:
    """Decompose flush and straight outs into sets and apply inclusion-exclusion.

    ``opponents`` only adjusts the count of unseen cards. Cards in an
    opponent's hand are unknown to you, so from your seat they are still
    unseen — which is why the standard calculation divides by 47 on the flop
    rather than by 45 at a three-handed table.
    """
    flush = frozenset(flush_completing_cards(hole, board))
    straight = frozenset(straight_completing_cards(hole, board))
    known = len(hole) + len(board)

    return DrawAnalysis(
        flush_outs=flush,
        straight_outs=straight,
        both=flush & straight,
        union=flush | straight,
        unseen=len(FULL_DECK) - known,
    )
