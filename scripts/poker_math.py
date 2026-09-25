"""Poker as set theory, and cutting as Fourier analysis.

    python scripts/poker_math.py --ranges
    python scripts/poker_math.py --outs "Jh Th" "9h 8s 2h"
    python scripts/poker_math.py --cuts
    python scripts/poker_math.py --all
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker.cards import parse_cards  # noqa: E402
from poker.fourier import analyse_cuts, cuts_never_mix  # noqa: E402
from poker.ranges import ALL_COMBOS, Range, draw_analysis, parse_range  # noqa: E402
from poker.shuffle import cut  # noqa: E402


def show_ranges() -> None:
    print("Ranges are sets of combos. There are C(52,2) =", len(ALL_COMBOS), "combos.")
    print()
    print("  notation   combos   why")
    for text, why in (
        ("AA", "a pair: C(4,2) = 6 ways to choose 2 of 4 aces"),
        ("AKs", "suited: one combo per suit"),
        ("AKo", "offsuit: 4 x 4 minus the 4 suited ones"),
        ("AK", "both forms"),
        ("JJ+", "four pair ranks x 6"),
        ("ATs+", "four suited kickers x 4"),
    ):
        print(f"  {text:<10} {len(parse_range(text)):>6}   {why}")

    print()
    print("Blockers are set difference. Holding one ace:")
    aces = parse_range("AA")
    print(f"  AA before      {len(aces)} combos")
    print(f"  AA after As    {len(aces.remove_dead(parse_cards('As')))} combos  <- halved")
    print(f"  KK after As    {len(parse_range('KK').remove_dead(parse_cards('As')))} combos  <- untouched")

    print()
    print("Set identities hold because these really are sets:")
    a, b = parse_range("AA, KK"), parse_range("AKs, QQ")
    print(f"  ~(A | B) == ~A & ~B   {(~(a | b)) == ((~a) & (~b))}")
    print(f"  A ^ B has {len(a ^ b)} combos; A & B has {len(a & b)}")
    print(f"  |A| + |~A| = {len(a)} + {len(~a)} = {len(a) + len(~a)}")


def show_outs(hole_text: str, board_text: str) -> None:
    hole, board = parse_cards(hole_text), parse_cards(board_text)
    print(f"Hole {hole_text}   Board {board_text}")
    print()
    print(draw_analysis(hole, board).summary())
    print()
    analysis = draw_analysis(hole, board)
    print("  Adding the two draws together double-counts every card belonging")
    print("  to both sets. Inclusion-exclusion is the correction: it is the")
    print(
        f"  difference between the {analysis.naive_sum}-out hand that does not "
        f"exist and the {len(analysis.union)}-out hand that does."
    )


def show_cuts(trials: int = 100_000) -> None:
    print("A cut is a rotation, so repeated cutting is a random walk on Z_52.")
    print("Its Fourier coefficients are roots of unity, and |P(m)|^2 = P(m) * conj(P(m)).")
    print()

    rng = random.Random(11)
    deck = list(range(52))
    counts: Counter[int] = Counter()
    for _ in range(trials):
        counts[cut(deck, rng)[0]] += 1
    distribution = [counts.get(j, 0) / trials for j in range(52)]

    print("  Measured from poker.shuffle.cut over", f"{trials:,}", "cuts.")
    print()
    for analysis in analyse_cuts(distribution):
        print(analysis.summary())

    print()
    print(
        f"  The right-hand column never moves. Cutting reaches only 52 of the\n"
        f"  52! arrangements, so distance from a shuffled deck is fixed at\n"
        f"  1 - 52/52! = {cuts_never_mix(52)!r}. Twenty-five cuts leave the deck\n"
        f"  uniformly *rotated* and perfectly ordered."
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--ranges", action="store_true")
    parser.add_argument("--outs", nargs=2, metavar=("HOLE", "BOARD"))
    parser.add_argument("--cuts", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)

    nothing_chosen = not (args.ranges or args.outs or args.cuts)

    if args.ranges or args.all or nothing_chosen:
        show_ranges()
        print()
    if args.outs:
        show_outs(*args.outs)
        print()
    elif args.all or nothing_chosen:
        show_outs("Jh Th", "9h 8s 2h")
        print()
    if args.cuts or args.all or nothing_chosen:
        show_cuts()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
