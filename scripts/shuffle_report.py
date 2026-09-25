"""How random is a shuffled deck? Exact theory and empirical measurement.

    python scripts/shuffle_report.py                 # the seven-shuffle table
    python scripts/shuffle_report.py --procedures    # measure real procedures
    python scripts/shuffle_report.py --all
"""
from __future__ import annotations

import argparse
import sys
from math import factorial
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker.shuffle import (  # noqa: E402
    CASINO_DECK_CHANGE,
    CASINO_STANDARD,
    HOME_GAME,
    LAZY_DEALER,
    PROCEDURES,
    SINGLE_RIFFLE,
)
from poker.shuffle_analysis import (  # noqa: E402
    assess,
    reachable_deck_orders,
    riffles_needed,
    tv_distance_after_riffles,
)

SHOWCASE = (
    SINGLE_RIFFLE,
    LAZY_DEALER,
    HOME_GAME,
    CASINO_STANDARD,
    CASINO_DECK_CHANGE,
    PROCEDURES["5x riffle"],
    PROCEDURES["7x riffle"],
    PROCEDURES["10x riffle"],
)


def print_tv_table() -> None:
    print("Total variation distance from a perfectly shuffled 52-card deck.")
    print("Exact arithmetic via Bayer & Diaconis (1992), not simulation.")
    print()
    print("  riffles   distance   interpretation")
    for m in range(1, 13):
        distance = tv_distance_after_riffles(m)
        if distance > 0.99:
            note = "not shuffled in any meaningful sense"
        elif distance > 0.5:
            note = "an observer still has a large edge"
        elif distance > 0.2:
            note = "the conventional threshold is crossed here"
        elif distance > 0.05:
            note = "close to random"
        else:
            note = "indistinguishable in practice"
        marker = "  <-- seven" if m == 7 else ""
        print(f"  {m:>7}   {distance:8.3f}   {note}{marker}")
    print()
    print(f"  Riffles needed to get below 0.5: {riffles_needed(0.5)}")
    print(
        "  The drop is abrupt, not gradual: 5 riffles leaves 0.924 and 8 leaves "
        "0.167.\n  That is why 'a few shuffles' is not a matter of opinion."
    )


def print_procedures(trials: int) -> None:
    print("Measured procedures. 'Flags' is the share of independent runs whose")
    print("position test comes out significant; a fair procedure sits near 5%.")
    print()
    for procedure in SHOWCASE:
        print(assess(procedure, trials=trials).summary())
        print()


def print_seeding() -> None:
    reachable, total = reachable_deck_orders(63)
    print("Seeding")
    print("-------")
    print(f"  distinct 52-card orders      {total:.3e}")
    print(f"  reachable from a 63-bit seed {reachable:.3e}")
    print(f"  fraction                     {reachable / total:.3e}")
    print()
    print(
        "  poker/cards.py seeds from randrange(2**63) so that every hand is\n"
        "  replayable, which is the right call for a learning tool. It does mean\n"
        "  the deck order is drawn from a vanishing slice of the possible orders.\n"
        "  Nothing a player could exploit; everything a regulator would care about."
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--procedures", action="store_true", help="measure procedures")
    parser.add_argument("--seeding", action="store_true", help="seed-entropy note")
    parser.add_argument("--all", action="store_true", help="everything")
    parser.add_argument("--trials", type=int, default=1500)
    args = parser.parse_args(argv)

    show_table = args.all or not (args.procedures or args.seeding)

    if show_table:
        print_tv_table()
        print()
    if args.procedures or args.all:
        print_procedures(args.trials)
    if args.seeding or args.all:
        print_seeding()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
