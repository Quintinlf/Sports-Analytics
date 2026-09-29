"""How you play: patterns across every reviewed game.

Built only from what the reviews stored -- results, openings, centipawns lost per phase, the
principles each game broke, and habits (castling, early queen, checks and captures) -- so it
grows with every game and says nothing it can't count.
"""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Iterable

from .principles import BY_KEY, PHASES

STYLE_MIN_MOVES = 40    # about two full games of your moves before calling how sharply you play


def _score(outcome: str) -> float:
    return {"win": 1.0, "draw": 0.5, "loss": 0.0}.get(outcome, 0.0)


def build_profile(reviews: Iterable[dict]) -> dict:
    """``reviews`` are stored review dicts, oldest first."""
    games = [r for r in reviews if r.get("outcome") != "unfinished"]
    n = len(games)
    record = Counter(r["outcome"] for r in games)
    if not n:
        return {"games": 0, "record": {}, "summary": ["Play a game to start your profile."]}

    by_color = {}
    for color in ("white", "black"):
        mine = [r for r in games if r["player_color"] == color]
        if mine:
            by_color[color] = {"games": len(mine), "score": round(mean(_score(r["outcome"]) for r in mine), 2)}

    openings: dict[str, list[float]] = {}
    for r in games:
        name = (r.get("opening") or {}).get("name")
        if name:
            openings.setdefault(f"{r['player_color']}: {name.split(':')[0]}", []).append(_score(r["outcome"]))
    top_openings = sorted(({"line": k, "games": len(v), "score": round(mean(v), 2)} for k, v in openings.items()),
                          key=lambda o: (-o["games"], -o["score"]))[:6]

    acpl = {}
    for ph in (*PHASES, "all"):
        vals = [r["acpl"][ph] for r in games if r.get("acpl", {}).get(ph) is not None]
        acpl[ph] = round(mean(vals), 1) if vals else None
    trend = [r["acpl"]["all"] for r in games if r.get("acpl", {}).get("all") is not None]
    improving = None
    if len(trend) >= 6:
        recent, before = mean(trend[-3:]), mean(trend[:-3])
        improving = {"recent_acpl": round(recent, 1), "earlier_acpl": round(before, 1), "better": recent < before}

    habits = [r.get("habits", {}) for r in games]
    castled = [h for h in habits if h.get("moves", 0) >= 10]
    castle_rate = mean(1.0 if h.get("castled") else 0.0 for h in castled) if castled else None
    castle_moves = [h["castle_move"] for h in castled if h.get("castle_move")]
    early_queen = mean(1.0 if (h.get("first_queen_move") or 99) <= 5 else 0.0 for h in habits)
    # Checks and captures, pooled over every move, next to the opponents' share in the same games.
    own_moves = sum(h.get("moves", 0) for h in habits)
    opp_moves = sum(h.get("opp_moves", 0) for h in habits)
    forcing = sum(h.get("checks", 0) + h.get("captures", 0) for h in habits) / own_moves if own_moves else None
    opp_forcing = sum(h.get("opp_forcing", 0) for h in habits) / opp_moves if opp_moves else None
    blunders = mean(r.get("counts", {}).get("blunder", 0) for r in games)

    broken = Counter(k for r in games for k in r.get("broken", []))
    recurring = [{"key": k, "title": BY_KEY[k].title, "phase": BY_KEY[k].phase, "games": c}
                 for k, c in broken.most_common(4)]

    worst = max((ph for ph in PHASES if acpl.get(ph) is not None), key=lambda ph: acpl[ph], default=None)
    summary = []
    if worst:
        summary.append(f"You lose the most per move in the {worst} ({acpl[worst]:.0f} centipawns on average).")
    if recurring:
        summary.append(f"Your most repeated habit to fix: {recurring[0]['title'].lower()} "
                       f"({recurring[0]['games']} of {n} games).")
    if castle_rate is not None:
        when = f", usually by move {round(mean(castle_moves))}" if castle_moves else ""
        summary.append(f"You castle in {castle_rate:.0%} of games{when}.")
    if forcing is not None and opp_forcing is not None and own_moves >= STYLE_MIN_MOVES:
        gap = forcing - opp_forcing
        compared = ("you play the sharper game" if gap >= 0.05 else
                    "you play the quieter game" if gap <= -0.05 else "about the same as they do")
        summary.append(f"You check or capture on {forcing:.0%} of your moves; your opponents did on "
                       f"{opp_forcing:.0%} in the same games, so {compared}.")
    if improving:
        summary.append("You're losing fewer centipawns per move in your last three games." if improving["better"]
                       else "Your last three games were rougher than the ones before; slow down on each move.")

    return {
        "games": n,
        "record": {"win": record.get("win", 0), "draw": record.get("draw", 0), "loss": record.get("loss", 0)},
        "by_color": by_color,
        "openings": top_openings,
        "acpl": acpl,
        "acpl_trend": trend[-20:],
        "improving": improving,
        "habits": {"castle_rate": castle_rate, "avg_castle_move": round(mean(castle_moves), 1) if castle_moves else None,
                   "early_queen_rate": round(early_queen, 2),
                   "forcing_share": round(forcing, 2) if forcing is not None else None,
                   "opponent_forcing_share": round(opp_forcing, 2) if opp_forcing is not None else None,
                   "blunders_per_game": round(blunders, 2)},
        "recurring": recurring,
        "summary": summary,
    }
