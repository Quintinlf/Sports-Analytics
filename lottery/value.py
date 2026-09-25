"""Expected value of a ticket, with the deductions people usually omit.

The "the jackpot is $1.8 billion so a $2 ticket is +EV" argument appears every
time a jackpot rolls over. It is almost always wrong, and it is wrong for four
compounding reasons that this module makes explicit:

1. **The advertised jackpot is an annuity** paid over 30 years. Taking the
   money now yields roughly half of it.
2. **Tax.** A jackpot lands in the top federal bracket, plus state tax in most
   places.
3. **Splitting.** High jackpots sell more tickets, which raises the chance of
   sharing. Rollovers are self-limiting for exactly this reason.
4. **Fixed tiers dilute.** The lower prizes are constant, so as the jackpot
   grows they become a smaller share of a still-tiny expected return.

Even at a record jackpot, the honest answer is usually "still negative, just
less negative". The point of computing it is to see where the money actually
goes, not to find a green light.

This is arithmetic on inputs you supply. It is not financial advice, and a
ticket that is marginally +EV in expectation is still a near-certain loss for
any individual buyer: the median outcome of buying one is losing the price.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from lottery.games import Era, Game
from lottery.popularity import expected_cowinners, expected_share

__all__ = ["TierOutcome", "ValueResult", "ticket_ev", "breakeven_jackpot"]


#: Cash payout as a fraction of the advertised annuity. Moves with interest
#: rates; roughly half in recent years. Override with the real figure.
DEFAULT_CASH_RATIO = 0.50

#: Top federal marginal rate. State tax varies from 0% to over 10%.
DEFAULT_FEDERAL_RATE = 0.37


@dataclass(frozen=True)
class TierOutcome:
    """One prize tier's contribution to expected value."""

    label: str
    probability: float
    gross_prize: float
    net_prize: float
    contribution: float
    is_jackpot: bool = False

    def __str__(self) -> str:
        return (
            f"{self.label:22} 1 in {1/self.probability:>14,.0f}  "
            f"net ${self.net_prize:>16,.0f}  EV ${self.contribution:>8,.4f}"
        )


@dataclass(frozen=True)
class ValueResult:
    """Full expected-value breakdown for one ticket."""

    game: str
    ticket_price: float
    advertised_jackpot: float
    cash_value: float
    net_jackpot: float
    expected_cowinners: float
    expected_share: float
    tiers: Sequence[TierOutcome]
    expected_return: float

    @property
    def expected_profit(self) -> float:
        return self.expected_return - self.ticket_price

    @property
    def return_per_dollar(self) -> float:
        return self.expected_return / self.ticket_price

    def summary(self) -> str:
        lines = [
            f"{self.game} - ${self.ticket_price:.2f} ticket",
            f"  advertised jackpot   ${self.advertised_jackpot:>18,.0f}",
            f"  cash option          ${self.cash_value:>18,.0f}",
            f"  after tax            ${self.net_jackpot:>18,.0f}",
            f"  expected co-winners  {self.expected_cowinners:>19,.2f}"
            f"  -> you keep {self.expected_share:.1%}",
            "",
        ]
        lines.extend(f"  {tier}" for tier in self.tiers)
        lines.extend(
            [
                "",
                f"  expected return      ${self.expected_return:>18,.4f}"
                f"  ({self.return_per_dollar:.1%} of price)",
                f"  expected profit      ${self.expected_profit:>18,.4f}",
            ]
        )
        verdict = (
            "positive in expectation - still a near-certain individual loss"
            if self.expected_profit > 0
            else "negative in expectation, as usual"
        )
        lines.append(f"  verdict: {verdict}")
        return "\n".join(lines)


def ticket_ev(
    game: Game,
    advertised_jackpot: float,
    *,
    tickets_sold: int,
    crowd_score: float = 1.0,
    cash_ratio: float = DEFAULT_CASH_RATIO,
    tax_rate: float = DEFAULT_FEDERAL_RATE,
    era: Optional[Era] = None,
) -> ValueResult:
    """Expected return of one ticket.

    ``crowd_score`` comes from :class:`~lottery.popularity.PopularityModel` and
    is the only input a player actually controls. Everything else is set by the
    lottery and the tax code.

    Fixed lower tiers are taxed at the same rate for simplicity, which slightly
    understates their value for small prizes that fall in lower brackets. Their
    total contribution is a fraction of a dollar either way.
    """
    if advertised_jackpot < 0:
        raise ValueError("jackpot cannot be negative")
    if not 0 <= cash_ratio <= 1:
        raise ValueError("cash_ratio must be between 0 and 1")
    if not 0 <= tax_rate < 1:
        raise ValueError("tax_rate must be between 0 and 1")

    era = era or game.current_era
    matrix = era.matrix

    cash_value = advertised_jackpot * cash_ratio
    net_jackpot = cash_value * (1 - tax_rate)

    cowinners = expected_cowinners(crowd_score, tickets_sold, matrix)
    share = expected_share(cowinners)

    tiers: list[TierOutcome] = []
    total = 0.0

    for tier in game.prize_tiers:
        probability = matrix.tier_probability(tier.matched_white, tier.matched_special)

        if tier.is_jackpot:
            net = net_jackpot * share
        else:
            net = tier.prize * (1 - tax_rate)

        contribution = probability * net
        total += contribution

        tiers.append(
            TierOutcome(
                label=tier.label(),
                probability=probability,
                gross_prize=cash_value if tier.is_jackpot else float(tier.prize),
                net_prize=net,
                contribution=contribution,
                is_jackpot=tier.is_jackpot,
            )
        )

    return ValueResult(
        game=game.name,
        ticket_price=era.ticket_price,
        advertised_jackpot=advertised_jackpot,
        cash_value=cash_value,
        net_jackpot=net_jackpot,
        expected_cowinners=cowinners,
        expected_share=share,
        tiers=tiers,
        expected_return=total,
        )


def breakeven_jackpot(
    game: Game,
    *,
    tickets_sold: int,
    crowd_score: float = 1.0,
    cash_ratio: float = DEFAULT_CASH_RATIO,
    tax_rate: float = DEFAULT_FEDERAL_RATE,
    era: Optional[Era] = None,
    upper: float = 100e9,
) -> Optional[float]:
    """Smallest advertised jackpot at which a ticket breaks even, if any.

    Returns ``None`` when no jackpot suffices — which happens whenever ticket
    sales scale with the jackpot, because the extra tickets split it back down.
    That self-limiting behaviour is the reason a lottery can safely advertise
    unbounded rollovers.

    Bisection: expected return rises monotonically with the jackpot when
    ``tickets_sold`` is held fixed.
    """
    era = era or game.current_era

    def profit(jackpot: float) -> float:
        return (
            ticket_ev(
                game,
                jackpot,
                tickets_sold=tickets_sold,
                crowd_score=crowd_score,
                cash_ratio=cash_ratio,
                tax_rate=tax_rate,
                era=era,
            ).expected_profit
        )

    if profit(upper) < 0:
        return None

    low, high = 0.0, upper
    for _ in range(200):
        mid = (low + high) / 2
        if profit(mid) < 0:
            low = mid
        else:
            high = mid
    return high
