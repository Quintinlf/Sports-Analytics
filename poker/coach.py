"""The coach: what is happening in this hand, in plain words, and how good each choice is.

The table's maths panel shows formulas; the coach instead stops the player at the moments that
matter and says what is going on. Its numbers follow the laboratory's rule -- nothing is invented:

* **The price** (pot odds) is exact arithmetic on the pot and the bet.
* **Your chances** are a Monte Carlo estimate against the opponent's *modelled range*: every
  two-card hand the opponent could hold, weighted by how likely this opponent's own policy
  (:mod:`poker.opponents.heuristic`) was to take the actions it has taken this hand. The number
  of simulated run-outs is reported alongside the estimate.
* **How often they fold** to a bet comes from the same policy, applied to the same range.
* **The value of each choice** is a stated simple model: the hand goes to showdown, and the
  opponent's re-raises are treated as calls. It ranks choices; it is not a solver.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from random import Random
from typing import Dict, List, Optional, Sequence, Tuple

from poker.cards import FULL_DECK, Card, parse_cards
from poker.engine.rules import LegalActions
from poker.engine.state import ActionRecord, ActionType, HandState, Player, Street
from poker.evaluator import HandCategory, evaluate
from poker.opponents.base import OpponentProfile
from poker.opponents.heuristic import defend_chance, raise_band
from poker.ranges import draw_analysis
from poker.strength import hand_strength, preflop_cutoff, preflop_strength

__all__ = ["coach_decision", "grade_of", "hand_recap", "OPPONENT_LESSONS", "TRIALS"]

#: Simulated run-outs per decision. The standard error of an equity estimate is at most
#: sqrt(0.25 / TRIALS), about 1.6 percentage points here.
TRIALS = 1000
#: The size the coach tests for a bet or raise: three quarters of the pot.
BET_FRACTION = 0.75

Combo = Tuple[Card, Card]


# ---------------------------------------------------------------------------
# The opponent's range, from its own policy
# ---------------------------------------------------------------------------

_p_defends = defend_chance


def _preflop_raise(profile: OpponentProfile, strength: float) -> float:
    return profile.aggression if strength >= preflop_cutoff(raise_band(profile)) else 0.0


def _action_likelihood(profile: OpponentProfile, rec: ActionRecord, strength: float) -> float:
    """P(this action | the opponent held a hand of this strength), mirroring heuristic.py."""
    kind = rec.action.type
    facing = rec.to_call_before > 0
    aggressive = kind in (ActionType.BET, ActionType.RAISE)
    if rec.street is Street.PREFLOP:
        p_raise = _preflop_raise(profile, strength)
        if aggressive:
            return p_raise
        if kind is ActionType.CHECK:
            return 1.0 - p_raise
        if kind is ActionType.CALL:
            plays = strength >= preflop_cutoff(profile.vpip)
            return (1.0 - p_raise) * (1.0 if plays else 0.0) * _p_defends(
                profile, strength, rec.to_call_before, rec.pot_before)
        return 1.0
    if not facing:
        p_bet = (profile.aggression if strength >= 0.55 else 0.0) + (
            profile.bluff_frequency if strength < 0.30 else 0.0)
        if aggressive:
            return p_bet
        if kind is ActionType.CHECK:
            return 1.0 - p_bet
        return 1.0
    p_raise = profile.aggression if strength >= 0.72 else 0.0
    if aggressive:
        return p_raise
    if kind is ActionType.CALL:
        return (1.0 - p_raise) * _p_defends(profile, strength, rec.to_call_before, rec.pot_before)
    return 1.0


def _continue_chance(profile: OpponentProfile, street: Street, strength: float,
                     to_call: int, pot_before: int) -> float:
    """Chance the opponent calls or raises (rather than folds) when facing a bet of ``to_call``."""
    if street is Street.PREFLOP:
        p_raise = _preflop_raise(profile, strength)
        plays = strength >= preflop_cutoff(profile.vpip)
        return p_raise + (1.0 - p_raise) * (1.0 if plays else 0.0) * _p_defends(profile, strength, to_call, pot_before)
    p_raise = profile.aggression if strength >= 0.72 else 0.0
    return p_raise + (1.0 - p_raise) * _p_defends(profile, strength, to_call, pot_before)


@lru_cache(maxsize=32)
def strength_table(board_key: Tuple[str, ...]) -> Dict[Combo, float]:
    """The opponent's heuristic strength for every two-card hand on this board (cached per board)."""
    board = parse_cards(list(board_key)) if board_key else []
    blocked = set(board)
    return {c: hand_strength(list(c), board) for c in combinations(FULL_DECK, 2)
            if c[0] not in blocked and c[1] not in blocked}


def modelled_range(state: HandState, villain: Player, profile: OpponentProfile,
                   dead: Sequence[Card]) -> Dict[Combo, float]:
    """Every hand the opponent could hold, weighted by its actions so far this hand."""
    dead_set = set(dead)
    live = [c for c in FULL_DECK if c not in dead_set]
    combos: List[Combo] = list(combinations(live, 2))
    weights = {c: 1.0 for c in combos}
    for rec in state.action_log:
        if rec.player_id != villain.player_id or rec.action.type is ActionType.POST_BLIND:
            continue
        table = strength_table(tuple(rec.board))
        for c in combos:
            if weights[c]:
                weights[c] *= _action_likelihood(profile, rec, table[c])
    if sum(weights.values()) <= 0:        # it acted off-model (a size jitter, an all-in): no information
        weights = {c: 1.0 for c in combos}
    return {c: w for c, w in weights.items() if w > 0}


# ---------------------------------------------------------------------------
# Equity and fold chances
# ---------------------------------------------------------------------------

@dataclass
class Simulation:
    equity: float               # share of pots won (ties count half) against the range
    fold_chance: float          # share of the range that folds to the coach's test bet
    equity_when_called: Optional[float]
    trials: int
    exact: bool


def simulate(hero: Sequence[Card], board: Sequence[Card], weights: Dict[Combo, float],
             continue_of: Dict[Combo, float], rng: Random, trials: int = TRIALS) -> Simulation:
    combos = list(weights)
    w = [weights[c] for c in combos]
    need = 5 - len(board)
    results: List[Tuple[float, float, float]] = []      # (weight, result, continue chance)
    if need == 0:
        mine = evaluate(list(hero) + list(board))
        for c, wt in zip(combos, w):
            theirs = evaluate(list(c) + list(board))
            results.append((wt, 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0, continue_of[c]))
        exact = True
    else:
        known = set(hero) | set(board)
        deck = [c for c in FULL_DECK if c not in known]
        for c in rng.choices(combos, weights=w, k=trials):
            rest = [d for d in deck if d != c[0] and d != c[1]]
            full = list(board) + rng.sample(rest, need)
            mine, theirs = evaluate(list(hero) + full), evaluate(list(c) + full)
            results.append((1.0, 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0, continue_of[c]))
        exact = False
    total = sum(r[0] for r in results)
    equity = sum(r[0] * r[1] for r in results) / total
    cont = sum(r[0] * r[2] for r in results)
    return Simulation(
        equity=equity,
        fold_chance=1.0 - cont / total,
        equity_when_called=(sum(r[0] * r[2] * r[1] for r in results) / cont) if cont > 0 else None,
        trials=len(results) if exact else trials,
        exact=exact,
    )


# ---------------------------------------------------------------------------
# Words
# ---------------------------------------------------------------------------

def _pct(x: float) -> str:
    return f"{round(100 * x)}%"


@lru_cache(maxsize=1)
def _preflop_ladder() -> List[float]:
    return sorted(preflop_strength(list(c)) for c in combinations(FULL_DECK, 2))


def describe_hole(cards: Sequence[Card]) -> str:
    a, b = sorted(cards, key=lambda c: int(c.rank), reverse=True)
    shown = f"{a.display} {b.display}"
    if a.rank == b.rank:
        what = f"a pair of {a.rank.name_plural} in your hand (a \"pocket pair\")"
    else:
        suited = " of the same suit" if a.suit == b.suit else ""
        what = f"{_article(a.rank.name_singular)} {a.rank.name_singular} and {_article(b.rank.name_singular)} " \
               f"{b.rank.name_singular}{suited}"
    ladder = _preflop_ladder()
    better_than = bisect_left(ladder, preflop_strength(list(cards))) / len(ladder)
    return (f"{shown}: {what}. By the Chen starting-hand formula it ranks above about "
            f"{_pct(better_than)} of all two-card hands.")


def _article(word: str) -> str:
    return "an" if word[0] in "AEIO" else "a"


def _board_words(hero: Sequence[Card], board: Sequence[Card]) -> List[str]:
    if not board:
        return []
    value = evaluate(list(hero) + list(board))
    uses_mine = any(c in value.cards for c in hero)
    lines = [f"The table shows {' '.join(c.display for c in board)}. Your best five cards make "
             f"{value.describe().lower()}{'' if uses_mine else ', but that is all on the table: they have it too'}."]
    if len(board) < 5 and value.category < HandCategory.STRAIGHT:
        draws = draw_analysis(hero, board)
        if draws.union:
            kinds = [k for k, s in (("flush", draws.flush_outs), ("straight", draws.straight_outs)) if s]
            lines.append(f"You're drawing to a {' or '.join(kinds)}: {len(draws.union)} of the "
                         f"{draws.unseen} unseen cards complete it on the next card "
                         f"({_pct(draws.probability_next_card())}).")
    return lines


# ---------------------------------------------------------------------------
# Opponent lessons (the numbers are the profile's own)
# ---------------------------------------------------------------------------

OPPONENT_LESSONS: Dict[str, Dict[str, object]] = {
    "balanced": {
        "leak": "No big leak. It plays a fair range and bets its good hands.",
        "plan": ["Play good starting hands and fold the junk.",
                 "Bet when you have a strong hand; check or fold when you don't.",
                 "Use the price: call only when you win often enough to pay for it."],
    },
    "tight_passive": {
        "leak": "Folds too much and almost never bluffs. Its bets are honest.",
        "plan": ["Bet and raise small pots often: it gives up on most hands.",
                 "When it bets or raises, believe it and let medium hands go.",
                 "Don't pay it off with one pair when it suddenly gets aggressive."],
    },
    "loose_aggressive": {
        "leak": "Plays too many hands and bluffs too often.",
        "plan": ["Call its bets more often with decent pairs: most of its bets are weak hands.",
                 "Re-raise your strong hands; it will keep putting chips in.",
                 "Bluff it only now and then: it folds about half the time, so it's a coin flip."],
    },
    "calling_station": {
        "leak": "Calls far too often and almost never folds or raises.",
        "plan": ["Never bluff it: it calls.",
                 "Bet your good hands every street, and bet bigger: it pays you.",
                 "When it finally raises, it has something; fold one pair."],
    },
    "maniac": {
        "leak": "Raises with anything and bluffs half the time.",
        "plan": ["Wait for a real hand, then let it bet for you: check and call.",
                 "Fold the weak hands you'd play against anyone else; the pots get big fast.",
                 "Don't bluff; it doesn't fold."],
    },
}


@lru_cache(maxsize=16)
def opponent_stats(key: str, samples: int = 3000) -> Dict[str, float]:
    """How an opponent really plays, measured from its own rules.

    Preflop shares are exact over all 1326 starting hands. Flop numbers average over ``samples``
    random hands and flops (seeded, so the lesson text never changes between visits): how often it
    folds to a three-quarter-pot bet, how often it bets when checked to, and how many of those bets
    are bluffs (weak hands).
    """
    from poker.opponents.base import get_profile
    profile = get_profile(key)
    pre = [preflop_strength(list(c)) for c in combinations(FULL_DECK, 2)]
    plays = sum(s >= preflop_cutoff(profile.vpip) for s in pre) / len(pre)
    raises = sum(_preflop_raise(profile, s) for s in pre) / len(pre)
    rng = Random(f"stats:{key}")
    folds = bets = bluffs = 0.0
    for _ in range(samples):
        cards = rng.sample(FULL_DECK, 5)
        s = hand_strength(cards[:2], cards[2:])
        folds += 1.0 - _continue_chance(profile, Street.FLOP, s, 6, 14)     # a 6-chip bet into 8
        bluff = profile.bluff_frequency if s < 0.30 else 0.0
        bets += (profile.aggression if s >= 0.55 else 0.0) + bluff
        bluffs += bluff
    return {"plays_preflop": round(plays, 3), "raises_preflop": round(raises, 3),
            "folds_to_flop_bet": round(folds / samples, 3), "bets_when_checked_to": round(bets / samples, 3),
            "bluff_share_of_bets": round(bluffs / bets, 3) if bets else 0.0}


def _opponent_tip(profile: OpponentProfile, facing: bool, equity: float, fold_chance: float) -> str:
    name = profile.name
    key = profile.key
    if key == "calling_station":
        return (f"{name} rarely folds, so bet your good hands and never bluff."
                if equity >= 0.5 else f"{name} rarely folds: a bluff here gets called.")
    if key == "tight_passive":
        return (f"{name} almost never bluffs: its bet usually means a real hand."
                if facing else f"{name} folds a lot: small bets win a lot of small pots.")
    if key == "maniac":
        return (f"{name} bets with anything, so your hand is better than it looks."
                if facing else f"{name} will bet for you: with a strong hand you can just check and call.")
    if key == "loose_aggressive":
        return (f"{name} bluffs often: decent hands can call its bets."
                if facing else f"{name} plays many weak hands: bet your good ones.")
    return f"{name} plays a fair game: follow the price and your hand strength."


# ---------------------------------------------------------------------------
# The coach's view of the decision
# ---------------------------------------------------------------------------

def coach_decision(state: HandState, hero: Player, villain: Player, profile: OpponentProfile,
                   options: LegalActions, trials: int = TRIALS) -> Dict[str, object]:
    """Plain-words situation, each legal choice graded, and whether this is a moment to stop."""
    board = list(state.board)
    hole = list(hero.hole_cards)
    pot = state.pot_total
    to_call = options.call_amount if options.can_call else 0
    facing = to_call > 0
    rng = Random(f"{state.hand_id}:{len(state.action_log)}")

    weights = modelled_range(state, villain, profile, dead=hole + board)
    # The tested bet or raise: three quarters of the pot on top of any call, kept within the rules.
    aggressive = "raise" if options.can_raise else "bet" if options.can_bet else None
    raise_to = hero.committed_street + to_call + round(BET_FRACTION * (pot + to_call))
    if aggressive:
        raise_to = min(max(raise_to, options.min_raise_to), options.max_raise_to)
    extra = raise_to - hero.committed_street                         # chips the tested bet/raise puts in
    villain_to_call = max(extra - to_call, 0)
    board_strength = strength_table(tuple(str(c) for c in board))
    continue_of = {c: _continue_chance(profile, state.street, board_strength[c], villain_to_call, pot + extra)
                   for c in weights}
    sim = simulate(hole, board, weights, continue_of, rng, trials)
    e, f = sim.equity, sim.fold_chance
    e_call = sim.equity_when_called if sim.equity_when_called is not None else e

    # Chips each choice is worth on average, relative to folding now (the stated simple model).
    ev: Dict[str, float] = {}
    if options.can_fold and facing:
        ev["fold"] = 0.0
    if options.can_check:
        ev["check"] = e * pot
    if options.can_call:
        ev["call"] = e * (pot + to_call) - to_call
    if aggressive and villain_to_call > 0:
        ev[aggressive] = f * pot + (1.0 - f) * (e_call * (pot + extra + villain_to_call) - extra)

    bb = state.config.big_blind
    slack = 0.25 * bb + 0.03 * pot          # differences smaller than this are a matter of taste
    ok_band = 0.75 * bb + 0.12 * pot        # beyond this, a choice gives up real chips
    need = to_call / (pot + to_call) if facing else None
    name = profile.name

    def why(action: str) -> str:
        if action == "fold":
            if not facing:
                return "Checking is free, so folding only throws the hand away."
            return (f"Folding saves the {to_call} chips: you'd need to win {_pct(need)} of the time and you win "
                    f"about {_pct(e)}." if e < need else
                    f"Folding gives up a hand that wins about {_pct(e)} when you only need {_pct(need)}.")
        if action == "check":
            return ("Checking is fine: your hand isn't strong enough to want a big pot."
                    if e < 0.55 else
                    f"Checking is safe, but you win about {_pct(e)}: betting would get more chips in.")
        if action == "call":
            return (f"Calling costs {to_call} to win {pot + to_call}: you need to win {_pct(need)} of the time "
                    f"and you win about {_pct(e)} against the hands {name} plays like this.")
        sized = f"raise to {raise_to}" if action == "raise" else f"bet {extra}"
        return (f"If you {sized}, {name} folds about {_pct(f)} of the hands it could have here; "
                f"when it calls, you win about {_pct(e_call)}.")

    def grade(action: str) -> str:
        # Calling and folding are judged against the price; betting against the other choices.
        if action == "call":
            return "good" if e >= need + 0.03 else "ok" if e >= need - 0.05 else "mistake"
        if action == "fold":
            return "mistake" if e >= need + 0.10 else "ok" if e >= need - 0.03 else "good"
        if action == "check":
            return "good" if ev.get(aggressive, -1e9) - ev["check"] <= ok_band else "ok"
        rival = max(v for a, v in ev.items() if a != action)
        gap = rival - ev[action]
        return "good" if gap <= slack else "ok" if gap <= ok_band else "mistake"

    verdicts = {a: {"grade": grade(a), "why": why(a), "chips": round(ev[a], 1)} for a in ev}
    if options.can_fold and not facing:
        verdicts["fold"] = {"grade": "mistake", "why": why("fold"), "chips": None}
    rank = {"good": 2, "ok": 1, "mistake": 0}
    top = max(ev, key=lambda a: (rank[verdicts[a]["grade"]], ev[a], a != "fold"))

    lines = [describe_hole(hole)] + _board_words(hole, board)
    they_bet = any(r.player_id == villain.player_id and r.street is state.street
                   and r.action.type in (ActionType.BET, ActionType.RAISE) for r in state.action_log)
    if facing:
        what = (f"{name} {'raised' if options.can_raise and state.street is Street.PREFLOP else 'bet'}"
                if they_bet else f"You posted the small blind and {name} the big blind")
        lines.append(f"{what}: it costs you {to_call} to call, and the pot would be {pot + to_call}. "
                     f"To make calling pay, you need to win at least {_pct(need)} of the time.")
    else:
        lines.append(f"Nobody has bet yet this round. The pot is {pot}; you can check for free or bet.")
    chances = (f"You win about {_pct(e)} of the time against the hands {name} could have here"
               + (" (counted exactly for this last card)." if sim.exact else
                  f" ({sim.trials:,} simulated deals of the cards still to come)."))
    lines.append(chances)

    big = (facing and to_call >= state.config.big_blind * 2) or e >= 0.65 or \
        (facing and abs(e - (need or 0)) < 0.08)
    first_choice = not any(r.player_id == hero.player_id and r.action.type is not ActionType.POST_BLIND
                           for r in state.action_log)
    return {
        "street": state.street.name,
        "lines": lines,
        "advice": f"Best here: {top}. {why(top)}",
        "opponent_tip": _opponent_tip(profile, facing, e, f),
        "equity": round(e, 3),
        "equity_when_called": round(e_call, 3),
        "fold_chance": round(f, 3),
        "need": round(need, 3) if need is not None else None,
        "trials": sim.trials,
        "exact": sim.exact,
        "test_bet": extra if aggressive else None,
        "verdicts": verdicts,
        "best": top,
        "big_moment": bool(big),
        "first_choice": first_choice,
        "model": ("Chances are simulated against every hand the opponent could hold, weighted by how its own "
                  "rules would have played each one so far. Chip values assume the hand reaches showdown."),
    }


def grade_of(coach: Dict[str, object], action: str) -> Dict[str, str]:
    verdicts = coach.get("verdicts") or {}
    return verdicts.get(action) or {"grade": "ok", "why": ""}


def hand_recap(result_summary: str, net: int, decisions: List[Dict[str, str]],
               villain_cards: List[str], showdown: bool, profile: OpponentProfile) -> Dict[str, object]:
    """What happened, how each of your choices graded, and the one thing to take away."""
    lines = [result_summary]
    if showdown and villain_cards:
        lines.append(f"{profile.name} had {' '.join(villain_cards)}.")
    mistakes = [d for d in decisions if d["grade"] == "mistake"]
    if mistakes:
        take = f"On the {mistakes[0]['street'].lower()}, you chose {mistakes[0]['action']}: {mistakes[0]['why']}"
    elif decisions and net < 0:
        take = ("Every choice you made was right, and you still lost: that's the luck in poker. Good choices "
                "win over many hands, not every hand; judge the choice, not the result.")
    elif decisions:
        take = "Every choice you made this hand was one the coach agreed with."
    else:
        take = "You didn't have to decide anything this hand."
    return {"net": net, "lines": lines, "decisions": decisions, "takeaway": take}
