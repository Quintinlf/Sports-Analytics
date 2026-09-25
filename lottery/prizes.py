"""Per-draw prize breakdowns: how many tickets won at each tier.

This is the dataset that turns ``popularity.py`` from a documented guess into a
measurement. The open-data portals publish winning numbers only; the Multi-State
Lottery Association publishes the winner counts on the official draw-result
pages, and those counts are the fingerprint of how the crowd actually picked.

WHY WINNER COUNTS REVEAL PLAYER BEHAVIOUR
-----------------------------------------
Consider two draws with identical ticket sales. One comes up 3-7-12-21-28, all
of them dates. The other comes up 38-43-52-61-67, none of them. The jackpot odds
are identical, but far more tickets will match three numbers in the first draw,
because far more tickets contain low numbers.

So the ratio of observed lower-tier winners to the number expected under uniform
picking is a direct measurement of how popular that draw's numbers were.

INFERRING TICKET SALES
----------------------
Sales are not published per draw, but they do not need to be. Matching *only*
the Powerball is independent of which white balls a player chose, so the
"+ Powerball" tiers are a clean sales gauge:

    tickets_sold ~= (winners at 0+PB) * special_max

That estimate is used as the denominator, which is what makes the white-ball
ratios comparable across draws of wildly different jackpot sizes.

SCRAPING ETIQUETTE
------------------
``powerball.com/robots.txt`` sets no restrictions, but this module still fetches
one page per draw. It rate-limits by default and caches everything, so a history
is pulled once and never again. Do not remove the delay.
"""
from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass
from datetime import date
from typing import Iterator, Optional, Sequence

import requests

from lottery.games import Game

__all__ = [
    "PrizeBreakdown",
    "TierResult",
    "fetch_breakdown",
    "fetch_breakdowns",
    "parse_breakdown",
    "PrizeSourceError",
    "DRAW_RESULT_URL",
]

DRAW_RESULT_URL = "https://www.powerball.com/draw-result"

#: Order of rows in the official table, matching ``games.POWERBALL.prize_tiers``.
TIER_ORDER: tuple[tuple[int, bool], ...] = (
    (5, True),
    (5, False),
    (4, True),
    (4, False),
    (3, True),
    (3, False),
    (2, True),
    (1, True),
    (0, True),
)

#: Fixed prizes used to verify the table did not shift under us.
EXPECTED_PRIZES: tuple[Optional[int], ...] = (
    None,
    1_000_000,
    50_000,
    100,
    100,
    7,
    7,
    4,
    4,
)

DEFAULT_DELAY = 1.0
DEFAULT_TIMEOUT = 30
_USER_AGENT = "Mozilla/5.0 (compatible; sports-analytics lottery research)"


class PrizeSourceError(RuntimeError):
    """A draw-result page could not be read or understood."""


@dataclass(frozen=True)
class TierResult:
    """Winners and prize at one tier for one draw."""

    matched_white: int
    matched_special: bool
    winners: int
    prize: Optional[int]

    def label(self) -> str:
        base = f"{self.matched_white} white"
        return f"{base} + PB" if self.matched_special else base


@dataclass(frozen=True)
class PrizeBreakdown:
    """Every tier's winner count for one draw."""

    game_key: str
    draw_date: date
    tiers: tuple[TierResult, ...]

    def winners(self, matched_white: int, matched_special: bool) -> int:
        for tier in self.tiers:
            if (tier.matched_white, tier.matched_special) == (
                matched_white,
                matched_special,
            ):
                return tier.winners
        raise KeyError(f"no tier {matched_white}/{matched_special}")

    @property
    def special_matching_winners(self) -> int:
        """Every ticket that matched the special ball, at any white count."""
        return sum(tier.winners for tier in self.tiers if tier.matched_special)

    def estimated_tickets_sold(self, matrix) -> int:
        """Tickets sold, inferred from the tiers that matched the special ball.

        Every ticket matches the special with probability exactly
        ``1/special_max`` no matter which white balls its holder chose, and
        conditional on matching it the ticket falls into exactly one of the
        published "+ special" tiers (0 through 5 whites). Summing them therefore
        counts special-matching tickets exactly, and scaling by ``special_max``
        recovers sales **independently of the popularity model**. No
        circularity, and no sensitivity to which numbers were drawn.

        A WARNING WORTH KEEPING
        -----------------------
        The obvious estimator - ``winners(0, True) * special_max`` - is wrong
        twice over, and both errors inflate the very effect this module exists
        to measure:

        1. The 0-white tier has probability ``1/38.32``, not ``1/26``, so it
           understates sales by about 47%.
        2. Worse, that tier is *anti*-correlated with popularity. When a draw's
           numbers are popular, more tickets overlap them, so fewer tickets
           match zero whites. Estimated sales fall exactly when the numbers are
           crowded, inflating the measured ratio precisely where the hypothesis
           predicts an effect.

        Summing across all white counts removes both problems, because the sum
        no longer depends on how the whites were distributed.
        """
        return self.special_matching_winners * matrix.special_max

    def total_winners(self) -> int:
        return sum(tier.winners for tier in self.tiers)


def _clean_text(page: str) -> list[str]:
    """Flatten HTML to the ordered list of visible text fragments."""
    text = re.sub(r"<script[^>]*>.*?</script>", " ", page, flags=re.S | re.I)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", "|", text)
    text = html.unescape(text)
    return [part.strip() for part in text.split("|") if part.strip()]


def _to_int(raw: str, field: str) -> int:
    cleaned = raw.replace(",", "").replace("$", "").strip()
    if not cleaned.isdigit():
        raise PrizeSourceError(f"expected a number for {field}, got {raw!r}")
    return int(cleaned)


def parse_breakdown(game: Game, when: date, page: str) -> PrizeBreakdown:
    """Extract the winners table from a draw-result page.

    The table is laid out as ``winners, prize, power-play winners, power-play
    prize`` per row, except the jackpot row which has no Power Play. Prizes are
    checked against :data:`EXPECTED_PRIZES` so a layout change raises instead of
    silently returning misaligned numbers.
    """
    parts = _clean_text(page)

    try:
        start = len(parts) - 1 - parts[::-1].index("Powerball Prize")
    except ValueError:
        raise PrizeSourceError(
            f"{when}: no winners table found (page layout changed?)"
        ) from None

    # Skip the two Power Play column headers that follow.
    cursor = start + 1
    while cursor < len(parts) and parts[cursor] in (
        "Power Play Winners",
        "Power Play Prize",
        "Double Play Winners",
        "Double Play Prize",
    ):
        cursor += 1

    values = parts[cursor : cursor + 34]
    if len(values) < 34:
        raise PrizeSourceError(f"{when}: winners table truncated")

    rows: list[tuple[str, str]] = [(values[0], values[1])]
    for index in range(8):
        offset = 2 + index * 4
        rows.append((values[offset], values[offset + 1]))

    tiers: list[TierResult] = []
    for (matched_white, matched_special), (winners_raw, prize_raw), expected in zip(
        TIER_ORDER, rows, EXPECTED_PRIZES
    ):
        winners = _to_int(winners_raw, f"{matched_white} white winners")

        if expected is None:
            prize = None
        else:
            prize = _to_int(prize_raw, f"{matched_white} white prize")
            if prize != expected:
                raise PrizeSourceError(
                    f"{when}: tier {matched_white}/{matched_special} shows prize "
                    f"${prize:,} but ${expected:,} was expected - the table has "
                    "shifted and the parse cannot be trusted"
                )

        tiers.append(
            TierResult(
                matched_white=matched_white,
                matched_special=matched_special,
                winners=winners,
                prize=prize,
            )
        )

    return PrizeBreakdown(game.key, when, tuple(tiers))


def fetch_breakdown(
    game: Game,
    when: date,
    *,
    session: Optional[requests.Session] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> PrizeBreakdown:
    """Fetch and parse one draw's prize breakdown."""
    http = session or requests.Session()
    try:
        response = http.get(
            DRAW_RESULT_URL,
            params={"gc": game.key, "date": when.isoformat()},
            headers={"User-Agent": _USER_AGENT},
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise PrizeSourceError(f"{when}: fetch failed: {exc}") from exc

    return parse_breakdown(game, when, response.text)


def fetch_breakdowns(
    game: Game,
    dates: Sequence[date],
    *,
    delay: float = DEFAULT_DELAY,
    on_error: str = "skip",
) -> Iterator[PrizeBreakdown]:
    """Fetch many breakdowns, politely.

    Sleeps ``delay`` seconds between requests. ``on_error='skip'`` keeps going
    past individual failures, which matters over hundreds of pages; pass
    ``'raise'`` when a gap would invalidate the analysis.
    """
    session = requests.Session()
    for index, when in enumerate(dates):
        if index:
            time.sleep(delay)
        try:
            yield fetch_breakdown(game, when, session=session)
        except PrizeSourceError:
            if on_error == "raise":
                raise
