"""Shuffle & Fourier laboratory — API router.

Endpoints under /api/poker/lab/. Everything here is either exact arithmetic
computed per request (the riffle table, the cut transform, a single shuffle) or
read from ``poker/data/shuffle_lab.json``, which ``scripts/build_shuffle_lab.py``
writes from seeded simulations. The simulations take seconds per procedure, too
slow for a request on the free web tier, so they are run once and stored rather
than approximated with fewer trials.
"""
from __future__ import annotations

import json
import math
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from poker.fourier import analyse_cuts, characters_orthogonal, cuts_never_mix, dft
from poker.shuffle import PROCEDURES
from poker.shuffle_analysis import (
    DECK_SIZE,
    reachable_deck_orders,
    riffles_needed,
    rising_sequences,
    tv_distance_after_riffles,
)

router = APIRouter(prefix="/api/poker/lab", tags=["poker-laboratory"])

LAB_DATA = Path(__file__).resolve().parent.parent.parent / "poker" / "data" / "shuffle_lab.json"


@lru_cache(maxsize=1)
def _stored_reports() -> Dict[str, Any]:
    try:
        return json.loads(LAB_DATA.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"reports": {}, "generated_by": None}


@lru_cache(maxsize=4)
def _riffle_table(max_riffles: int) -> List[Dict[str, float]]:
    return [{"riffles": m, "tv_distance": tv_distance_after_riffles(m)}
            for m in range(max_riffles + 1)]


def cut_distribution(n: int = DECK_SIZE, sd: float = DECK_SIZE / 10) -> List[float]:
    """Exact distribution of ``poker.shuffle.cut``'s cut point.

    ``cut`` draws ``round(gauss(n/2, sd))`` and clamps it to 1..n-1, so the
    probability of each cut point is a normal probability over a unit-wide bin,
    with the two tails folded into the end bins. No sampling involved.
    """
    if n < 3:
        raise ValueError("deck must have at least three cards")
    centre = n / 2

    def below(x: float) -> float:
        return 0.5 * (1 + math.erf((x - centre) / (sd * math.sqrt(2))))

    dist = [0.0] * n
    for k in range(1, n):
        low = below(k - 0.5) if k > 1 else 0.0
        high = below(k + 0.5) if k < n - 1 else 1.0
        dist[k] = high - low
    total = math.fsum(dist)
    return [p / total for p in dist]


@router.get("/riffles")
def riffle_table(max_riffles: int = Query(default=12, ge=1, le=20)) -> Dict[str, Any]:
    """Exact Bayer-Diaconis distance from random after m riffles."""
    return {
        "deck_size": DECK_SIZE,
        "table": _riffle_table(max_riffles),
        "riffles_needed_half": riffles_needed(0.5),
        "formula": "P(arrangement with r rising sequences) = C(2^m + n - r, n) / 2^(mn)",
    }


@router.get("/procedures")
def list_procedures() -> Dict[str, Any]:
    stored = _stored_reports()
    procedures = []
    for key, procedure in PROCEDURES.items():
        procedures.append({
            "key": key,
            "steps": [step.name for step in procedure.steps],
            "description": procedure.description,
            "riffle_count": procedure.riffle_count,
            "report": stored["reports"].get(key),
        })
    return {
        "procedures": procedures,
        "report_settings": stored.get("settings"),
        "generated_by": stored.get("generated_by"),
    }


@router.get("/shuffle")
def shuffle_once(
    procedure: str = Query(default="casino standard"),
    seed: Optional[int] = Query(default=None, ge=0, le=2**63 - 1),
) -> Dict[str, Any]:
    """One run of a procedure on a new-deck-order deck, step by step.

    ``arrangement[i]`` is the original position of the card now at position
    ``i`` (0 = top), the convention ``shuffle_analysis`` measures.
    """
    chosen = PROCEDURES.get(procedure)
    if chosen is None:
        raise HTTPException(404, f"Unknown procedure {procedure!r}")
    if seed is None:
        seed = random.randrange(2**31)
    rng = random.Random(seed)

    deck = list(range(DECK_SIZE))
    frames = [{"step": "new deck", "arrangement": deck, "rising_sequences": 1}]
    for step in chosen.steps:
        deck = step.apply(deck, rng)
        frames.append({
            "step": step.name,
            "arrangement": deck,
            "rising_sequences": rising_sequences(deck),
        })
    return {"procedure": procedure, "seed": seed, "frames": frames,
            "uniform_mean_rising_sequences": (DECK_SIZE + 1) / 2}


@router.get("/cuts")
def cut_fourier(
    sd: float = Query(default=DECK_SIZE / 10, ge=0.3, le=40.0),
    max_cuts: int = Query(default=25, ge=1, le=60),
) -> Dict[str, Any]:
    """Repeated cutting as a random walk on Z_52, analysed with characters.

    ``coefficients[m]`` is the Fourier coefficient P_hat(m) = sum_j P(j) omega^(jm);
    after k cuts it becomes P_hat(m)^k, so its modulus is what has to shrink.
    """
    dist = cut_distribution(DECK_SIZE, sd)
    coefficients = dft(dist)
    steps = sorted({1, 2, 3, 5, 10, max_cuts})
    return {
        "deck_size": DECK_SIZE,
        "sd": sd,
        "distribution": dist,
        "coefficients": [{"m": m, "re": c.real, "im": c.imag, "modulus": abs(c)}
                         for m, c in enumerate(coefficients)],
        "walk": [
            {"cuts": a.steps, "tv_on_rotations": a.exact_on_cyclic,
             "upper_bound": a.bound_on_cyclic, "tv_from_shuffled": a.distance_from_shuffled}
            for a in analyse_cuts(dist, steps)
        ],
        "never_mix": cuts_never_mix(DECK_SIZE),
        "orthogonality_check": {
            "same": abs(characters_orthogonal(DECK_SIZE, 5, 5)),
            "different": abs(characters_orthogonal(DECK_SIZE, 5, 6)),
        },
    }


@router.get("/seeding")
def seeding() -> Dict[str, Any]:
    reachable, total = reachable_deck_orders(63)
    return {
        "seed_bits": 63,
        "reachable_log10": math.log10(reachable),
        "total_log10": math.log10(total),
    }
