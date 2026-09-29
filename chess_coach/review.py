"""Review one game: every move explained, the mistakes found, and the lessons it earned.

The browser plays and analyses with Stockfish and sends, for every position, the engine's
evaluation (white's point of view, centipawns, mates capped at +/-1000) and its best move. From
that this module works out, for the player's side only, the centipawns lost on each move, the
average loss per phase, the worst moments with a coach's explanation (the explainer from
hobbies/chess), which principles the game broke, and the player's habits for the style profile.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Optional

import chess

from .explainer import MoveExplainer
from .principles import BY_KEY, BIG_LOSS, Ply, check, phase_of

EVAL_CAP = 1000
# Bump whenever a change here or in principles.py would change a review: stored games are
# re-reviewed from their saved engine data the next time they are read.
COACH_VERSION = 2
OPENINGS = Path(__file__).with_name("data") / "openings.tsv"


@lru_cache(maxsize=1)
def _openings() -> dict[tuple[str, ...], tuple[str, str]]:
    table: dict[tuple[str, ...], tuple[str, str]] = {}
    with OPENINGS.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            table.setdefault(tuple(row["uci"].split()), (row["eco"], row["name"]))
    return table


@lru_cache(maxsize=1)
def _book() -> frozenset[tuple[str, ...]]:
    """Every move sequence that is a named line or the start of one: the opening book."""
    return frozenset(line[:k] for line in _openings() for k in range(1, len(line) + 1))


def identify_opening(ucis: list[str]) -> Optional[dict]:
    """The longest named line (lichess chess-openings, CC0) the game followed."""
    table = _openings()
    for k in range(min(len(ucis), 40), 0, -1):
        hit = table.get(tuple(ucis[:k]))
        if hit:
            return {"eco": hit[0], "name": hit[1], "plies": k}
    return None


def _cap(v: Optional[float]) -> Optional[int]:
    if v is None:
        return None
    return int(max(-EVAL_CAP, min(EVAL_CAP, round(v))))


def result_for(result: str, color: chess.Color) -> str:
    if result == "1/2-1/2":
        return "draw"
    if result not in ("1-0", "0-1"):
        return "unfinished"
    won = (result == "1-0") == (color == chess.WHITE)
    return "win" if won else "loss"


def review_game(moves_uci: list[str], evals: list[Optional[float]], best_moves: list[Optional[str]],
                player_color: str, result: str) -> dict:
    """Everything the page shows after a game. ``evals`` has one entry per position (moves + 1)."""
    color = chess.WHITE if player_color == "white" else chess.BLACK
    if len(evals) != len(moves_uci) + 1:
        raise ValueError(f"need {len(moves_uci) + 1} evaluations (one per position), got {len(evals)}")
    evals = [_cap(v) for v in evals]
    book = _book()
    board = chess.Board()
    plies: list[Ply] = []
    in_book: list[bool] = []
    for i, uci in enumerate(moves_uci):
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"move {i + 1} ({uci}) is not legal")
        mover = board.turn
        sign = 1 if mover == chess.WHITE else -1
        before, after = evals[i], evals[i + 1]
        best = best_moves[i] if i < len(best_moves) else None
        in_book.append(tuple(moves_uci[:i + 1]) in book)
        # A book move or the engine's own choice loses nothing; any drop there is search noise
        # from the browser's shallow search, not the player's error.
        if before is None or after is None:
            loss = None
        elif in_book[-1] or best == uci:
            loss = 0
        else:
            loss = max(0, sign * (before - after))
        plies.append(Ply(index=i, board_before=board.copy(stack=False), move=move, san=board.san(move),
                         mover=mover, phase=phase_of(board), cp_loss=loss,
                         best=chess.Move.from_uci(best) if best else None))
        board.push(move)

    for p, nxt in zip(plies, plies[1:] + [None]):
        p.reply = nxt.best if nxt is not None else None
    explainer = MoveExplainer()
    explained = []
    for p, booked in zip(plies, in_book):
        sign = 1 if p.mover == chess.WHITE else -1
        before = sign * evals[p.index] if p.cp_loss is not None else None
        after = before - p.cp_loss if before is not None else None
        ex = explainer.explain_move(p.board_before, p.move, before, after, None if booked else p.best)
        if booked:
            ex.quality = "book"
        explained.append(ex)
    outcome = result_for(result, color)

    per_phase: dict[str, list[int]] = {"opening": [], "middlegame": [], "endgame": []}
    counts = {"blunder": 0, "mistake": 0, "inaccuracy": 0}
    move_rows = []
    for p, ex in zip(plies, explained):
        mine = p.mover == color
        if mine and p.cp_loss is not None:
            per_phase[p.phase].append(p.cp_loss)
            if ex.quality in counts:
                counts[ex.quality] += 1
        headline = ex.headline
        if p.cp_loss is not None and p.cp_loss >= BIG_LOSS:
            headline = _what_went_wrong(p, ex.better_move_san)
        move_rows.append({"ply": p.index + 1, "san": p.san, "uci": p.move.uci(), "mine": mine,
                          "phase": p.phase, "quality": ex.quality, "cp_loss": p.cp_loss,
                          "headline": headline, "details": ex.details, "better": ex.better_move_san})

    mine_scored = [r for r in move_rows if r["mine"] and r["cp_loss"] is not None]
    acpl = {ph: round(sum(v) / len(v), 1) if v else None for ph, v in per_phase.items()}
    acpl["all"] = round(sum(r["cp_loss"] for r in mine_scored) / len(mine_scored), 1) if mine_scored else None
    key_moments = sorted((r for r in mine_scored if r["cp_loss"] >= BIG_LOSS), key=lambda r: -r["cp_loss"])[:3]

    findings = check(plies, color, outcome)
    lessons = [{**vars(BY_KEY[f.key]), "evidence": f.evidence} for f in findings[:3]]
    worst_phase = max((ph for ph in per_phase if per_phase[ph]), key=lambda ph: acpl[ph], default=None)

    return {
        "coach_version": COACH_VERSION,
        "player_color": player_color,
        "result": result,
        "outcome": outcome,
        "opening": identify_opening(moves_uci),
        "acpl": acpl,
        "counts": counts,
        "worst_phase": worst_phase,
        "key_moments": key_moments,
        "lessons": lessons,
        "broken": [f.key for f in findings],
        "habits": _habits(plies, color),
        "moves": move_rows,
    }


def _what_went_wrong(p: Ply, better: Optional[str]) -> str:
    """Name the concrete reason a big loss happened, instead of a generic move description."""
    after = p.board_before.copy()
    after.push(p.move)
    alt = f" {better} was the move." if better else ""
    for reply in after.legal_moves:
        after.push(reply)
        mate = after.is_checkmate()
        after.pop()
        if mate:
            return f"{p.san} allows checkmate with {after.san(reply)}.{alt}"
    reply = p.reply
    if reply is not None and reply in after.legal_moves:
        if after.is_capture(reply):
            victim = after.piece_at(reply.to_square)
            name = chess.piece_name(victim.piece_type) if victim else "pawn"
            return f"{p.san} lets your opponent win material with {after.san(reply)}, taking your {name}.{alt}"
        if after.gives_check(reply):
            return f"{p.san} walks into {after.san(reply)}, a check that wins something.{alt}"
    if p.best is not None and p.best != p.move:
        b = p.board_before
        if b.is_capture(p.best):
            return f"{p.san} misses {b.san(p.best)}, which wins material."
        if b.gives_check(p.best):
            return f"{p.san} misses {b.san(p.best)}, a strong check."
    pawns = (p.cp_loss or 0) / 100
    return f"{p.san} costs about {pawns:.1f} pawns of advantage.{alt}"


def _habits(plies: list[Ply], color: chess.Color) -> dict:
    own = [p for p in plies if p.mover == color]
    theirs = [p for p in plies if p.mover != color]
    castle_move = next((p.board_before.fullmove_number for p in own if p.board_before.is_castling(p.move)), None)
    queen_move = next((p.board_before.fullmove_number for p in own
                       if p.board_before.piece_type_at(p.move.from_square) == chess.QUEEN), None)
    return {
        "moves": len(own),
        "castled": castle_move is not None,
        "castle_move": castle_move,
        "first_queen_move": queen_move,
        "checks": sum(1 for p in own if p.board_before.gives_check(p.move)),
        "captures": sum(1 for p in own if p.board_before.is_capture(p.move)),
        "opp_moves": len(theirs),
        "opp_forcing": sum(1 for p in theirs if p.board_before.is_capture(p.move) or p.board_before.gives_check(p.move)),
        "queens_traded_by_move": next((p.board_before.fullmove_number for p in plies
                                       if not p.board_before.pieces(chess.QUEEN, chess.WHITE)
                                       and not p.board_before.pieces(chess.QUEEN, chess.BLACK)), None),
    }
