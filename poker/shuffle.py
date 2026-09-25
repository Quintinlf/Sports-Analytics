"""Physical shuffle models: what a dealer's hands actually do to a deck.

``cards.Deck`` uses ``random.shuffle`` — a Fisher-Yates pass that is uniform by
construction. That is the right choice for a replayable learning tool, and it is
nothing like what happens at a casino table.

This module models the real operations instead: the riffle, the strip, the cut,
the overhand, and the wash. They are individually terrible at randomising a
deck. The interesting question — the one ``shuffle_analysis`` answers with an
exact number rather than a simulation — is how many of them you need before the
deck is genuinely mixed.

CONVENTION
----------
Index 0 is the **top** of the deck: the next card that would be dealt. Note
that ``cards.Deck`` deals from the *end* of its internal list, so anything
bridging the two must reverse. Nothing here imports ``cards``; these functions
operate on any sequence of hashable items, which is what makes them testable
against plain integers.

MODEL FIDELITY
--------------
``riffle`` implements the Gilbert-Shannon-Reeds model, the standard
mathematical description of a riffle shuffle and the one the seven-shuffle
result is proved against. It is a genuine model of human riffling, validated
against real dealers in the original literature.

``wash`` is the weak link: a table wash (cards spread face-down and swirled) is
*assumed* near-uniform here. That assumption is why casinos wash on every deck
change, and it is the one operation in this module that is modelled by fiat
rather than derived. Treat washed decks as a best case, not a measured one.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, TypeVar

T = TypeVar("T")

__all__ = [
    "riffle",
    "overhand",
    "strip",
    "cut",
    "wash",
    "Procedure",
    "Step",
    "RIFFLE",
    "STRIP",
    "CUT",
    "OVERHAND",
    "WASH",
    "CASINO_STANDARD",
    "CASINO_DECK_CHANGE",
    "HOME_GAME",
    "LAZY_DEALER",
    "SINGLE_RIFFLE",
    "PROCEDURES",
]


def _binomial(n: int, rng: random.Random) -> int:
    """Number of successes in n fair coin flips.

    Summing Bernoullis is O(n), which for a 52-card deck is irrelevant and
    keeps the model exactly the one the theory describes.
    """
    return sum(1 for _ in range(n) if rng.random() < 0.5)


def riffle(cards: Sequence[T], rng: random.Random) -> List[T]:
    """One Gilbert-Shannon-Reeds riffle shuffle.

    The deck is cut binomially — a human cut is near but not exactly half — and
    the two packets are interleaved by dropping from whichever packet has more
    cards left, with probability proportional to the remaining sizes. That is
    exactly the GSR model: it captures that a dealer drops roughly one card at
    a time but clumps when one packet is much thicker.

    The defining property is that the result can always be split into at most
    **two rising sequences** — two interleaved runs of the original order. That
    is what makes a single riffle so weak, and it is asserted in the tests.
    """
    count = len(cards)
    if count < 2:
        return list(cards)

    split = _binomial(count, rng)
    left = deque(cards[:split])
    right = deque(cards[split:])
    out: List[T] = []

    while left and right:
        if rng.random() < len(left) / (len(left) + len(right)):
            out.append(left.popleft())
        else:
            out.append(right.popleft())

    out.extend(left)
    out.extend(right)
    return out


def overhand(
    cards: Sequence[T], rng: random.Random, *, packets: int = 8
) -> List[T]:
    """An overhand shuffle: small packets peeled off the top, stacked in reverse.

    The shuffle most home players use, and the worst in common use. It barely
    moves cards relative to their neighbours — packets stay internally ordered
    and only their order is reversed, so information about the original
    sequence survives an enormous number of repetitions.
    """
    count = len(cards)
    if count < 2:
        return list(cards)

    remaining = list(cards)
    out: List[T] = []
    mean_size = max(1, count // max(packets, 1))

    # Peel until the deck is exhausted. Stopping after a fixed number of
    # packets would leave a large intact block on top, which is not what a
    # hand doing this actually produces.
    while remaining:
        size = min(len(remaining), max(1, round(rng.expovariate(1 / mean_size))))
        out = remaining[:size] + out
        remaining = remaining[size:]
    return out


def strip(cards: Sequence[T], rng: random.Random, *, packets: int = 5) -> List[T]:
    """A strip (or "run"): chunks pulled off the top and restacked in reverse.

    Standard in poker rooms between riffles. It is a coarse overhand: it
    reverses the order of a handful of large blocks and does nothing at all
    within them. Its real job is to break up clumps the riffles left, not to
    randomise.
    """
    count = len(cards)
    if count < 2 or packets < 2:
        return list(cards)

    boundaries = sorted(rng.sample(range(1, count), min(packets - 1, count - 1)))
    chunks: List[List[T]] = []
    previous = 0
    for boundary in boundaries:
        chunks.append(list(cards[previous:boundary]))
        previous = boundary
    chunks.append(list(cards[previous:]))

    out: List[T] = []
    for chunk in chunks:
        out = chunk + out
    return out


def cut(
    cards: Sequence[T], rng: random.Random, *, position: Optional[int] = None
) -> List[T]:
    """A single cut, by default near the middle as a player would make it.

    A cut is a rotation. It changes *no* card's position relative to any other,
    so it adds exactly zero randomness by every measure in
    ``shuffle_analysis``. Its purpose at a table is procedural: it denies the
    dealer knowledge of where the top card is, which matters against cheating
    rather than against statistics.
    """
    count = len(cards)
    if count < 2:
        return list(cards)

    if position is None:
        centre = count / 2
        position = int(min(count - 1, max(1, round(rng.gauss(centre, count / 10)))))
    position %= count
    return list(cards[position:]) + list(cards[:position])


def wash(cards: Sequence[T], rng: random.Random) -> List[T]:
    """A table wash: cards spread face-down and swirled.

    Modelled as a uniform random permutation. This is an assumption rather than
    a derivation — see the module docstring. It is why casinos wash whenever a
    new deck is introduced: a fresh deck arrives in suit order, and no realistic
    number of riffles will fix a perfectly ordered deck as reliably as scrambling
    it on the felt first.
    """
    out = list(cards)
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# Procedures: the sequences dealers actually perform.
# ---------------------------------------------------------------------------

Operation = Callable[[Sequence[T], random.Random], List[T]]


@dataclass(frozen=True)
class Step:
    """One named operation in a procedure."""

    name: str
    apply: Operation

    def __str__(self) -> str:
        return self.name


RIFFLE = Step("riffle", riffle)
STRIP = Step("strip", strip)
CUT = Step("cut", cut)
OVERHAND = Step("overhand", overhand)
WASH = Step("wash", wash)


@dataclass(frozen=True)
class Procedure:
    """A named sequence of shuffle operations."""

    name: str
    steps: tuple[Step, ...]
    description: str = ""

    @property
    def riffle_count(self) -> int:
        return sum(1 for step in self.steps if step is RIFFLE)

    def apply(self, cards: Sequence[T], rng: random.Random) -> List[T]:
        result = list(cards)
        for step in self.steps:
            result = step.apply(result, rng)
        return result

    def __str__(self) -> str:
        return f"{self.name}: {' -> '.join(str(s) for s in self.steps)}"


#: The standard poker-room hand shuffle between hands.
CASINO_STANDARD = Procedure(
    "casino standard",
    (RIFFLE, RIFFLE, STRIP, RIFFLE, CUT),
    "Riffle, riffle, strip, riffle, cut - the sequence most poker rooms "
    "require. Three riffles, which is fewer than the theory wants.",
)

#: What happens when a fresh deck is introduced.
CASINO_DECK_CHANGE = Procedure(
    "casino deck change",
    (WASH, RIFFLE, RIFFLE, STRIP, RIFFLE, CUT),
    "A wash first, because a new deck arrives in suit order and riffles alone "
    "cannot recover from a perfectly ordered start.",
)

HOME_GAME = Procedure(
    "home game",
    (OVERHAND, OVERHAND, RIFFLE, CUT),
    "Overhand shuffling with a token riffle. Substantially non-random.",
)

LAZY_DEALER = Procedure(
    "lazy dealer",
    (RIFFLE, RIFFLE, CUT),
    "Two riffles and a cut. Fast, and badly under-mixed.",
)

SINGLE_RIFFLE = Procedure(
    "single riffle",
    (RIFFLE, CUT),
    "The worst case worth naming: at most two rising sequences survive, so the "
    "original order is almost entirely recoverable.",
)


def _riffles(count: int) -> Procedure:
    return Procedure(
        f"{count}x riffle",
        tuple(RIFFLE for _ in range(count)),
        f"{count} riffle shuffles, nothing else.",
    )


PROCEDURES: dict[str, Procedure] = {
    p.name: p
    for p in (
        SINGLE_RIFFLE,
        LAZY_DEALER,
        CASINO_STANDARD,
        CASINO_DECK_CHANGE,
        HOME_GAME,
        *(_riffles(n) for n in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12)),
    )
}
