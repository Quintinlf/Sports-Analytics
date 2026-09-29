"""Opening, middlegame and endgame principles, each with a check against a real game.

A principle is teaching text (what to do and why) plus, where it can be judged from the moves
and the engine's evaluations, a detector that says whether *this* game broke it and where. The
review turns the detectors that fired into the game's lessons, so the tips a player sees are the
ones their own moves earned.

The principles are the standard ones every chess teacher gives; the thresholds (move 10 for
development, 150 centipawns for a real mistake) are coaching conventions, not laws.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import chess

PHASES = ("opening", "middlegame", "endgame")
OPENING_END_MOVE = 15
MIDDLEGAME_END_MOVE = 40
BIG_LOSS = 150          # centipawns: a mistake or worse

VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


@dataclass(frozen=True)
class Principle:
    key: str
    phase: str
    title: str
    why: str
    practice: str


PRINCIPLES: tuple[Principle, ...] = (
    # ---- opening -------------------------------------------------------------
    Principle("center", "opening", "Fight for the center",
              "Pawns and pieces on e4, d4, e5 and d5 control the most squares and give your pieces room to come out.",
              "Start with 1.e4 or 1.d4 as White; as Black answer with ...e5, ...d5 or ...c5."),
    Principle("develop", "opening", "Develop knights and bishops first",
              "Every move a knight or bishop sits at home, you are playing with fewer pieces than your opponent.",
              "Aim to have all four minor pieces off the back rank by about move 10."),
    Principle("early_queen", "opening", "Don't bring the queen out early",
              "An early queen gets chased: your opponent develops pieces while attacking her, gaining moves.",
              "Bring out knights and bishops first; the queen can wait until they are out."),
    Principle("one_move_each", "opening", "Move each piece once in the opening",
              "Moving the same piece twice gives your opponent a free developing move.",
              "Only move a piece again to capture, to escape an attack, or to make a real threat."),
    Principle("castle", "opening", "Castle early",
              "The king is safest tucked behind pawns, and castling brings a rook toward the center.",
              "Castle by about move 10, before the center opens."),
    Principle("king_pawns", "opening", "Keep the pawns in front of your king",
              "Moving the f-pawn early opens a line to your king; f7 and f2 are guarded only by the king at the start.",
              "Leave the f-pawn at home until your king is castled."),
    # ---- middlegame ----------------------------------------------------------
    Principle("their_threat", "middlegame", "Ask what your opponent's last move threatens",
              "The most common way to lose a game is to answer a threat you never looked for.",
              "Before your own plan, ask: what does their last move attack, and can they check or capture next?"),
    Principle("cct", "middlegame", "Checks, captures, threats -- before every move",
              "Most games between developing players are decided by a tactic one side didn't look for.",
              "Before each move, list your opponent's checks, captures and threats, then your own."),
    Principle("loose_pieces", "middlegame", "Loose pieces drop off",
              "An undefended piece is the target of every fork and double attack.",
              "After choosing a move, glance at each of your pieces: is it defended?"),
    Principle("convert", "middlegame", "When ahead, trade pieces; when behind, avoid trades",
              "With fewer pieces left, an extra pawn or piece counts for more and counterplay dies.",
              "Up material? Offer trades of pieces, keep your pawns, head for a simpler position."),
    Principle("open_files", "middlegame", "Put rooks on open files",
              "Rooks need open lines; a rook behind its own pawns does nothing.",
              "Move a rook to a file with no pawns, or to one you are about to open."),
    Principle("worst_piece", "middlegame", "Improve your worst piece",
              "When there is no tactic, the quietest good move is to give your least active piece a job.",
              "Find the piece doing least and move it to a square where it attacks or defends something."),
    # ---- endgame -------------------------------------------------------------
    Principle("king_active", "endgame", "Activate your king",
              "With the queens off, the king is a strong fighting piece; hiding it wastes one of your best pieces.",
              "Once the queens are traded, walk your king toward the center and the pawns."),
    Principle("passed_pawns", "endgame", "Passed pawns must be pushed",
              "A passed pawn ties down the enemy pieces that have to stop it; standing still gives them time.",
              "Push it with your king or a rook supporting it."),
    Principle("rook_behind", "endgame", "Rooks belong behind passed pawns",
              "From behind, a rook gains power as the pawn advances; in front, it blocks its own pawn.",
              "Put your rook behind your passed pawn, or behind your opponent's to stop it."),
    Principle("opposition", "endgame", "Know the opposition",
              "In king-and-pawn endings, whoever holds the opposition (kings facing with one square between, "
              "the other side to move) decides whether a pawn can queen.",
              "Practise king and pawn against king until you can win or draw it without thinking."),
    Principle("rook_endings", "endgame", "Learn Lucena and Philidor",
              "Rook endings are the most common endings, and these two positions decide most of them.",
              "Learn the Lucena (how to win) and the Philidor (how to draw) rook-and-pawn positions."),
)
BY_KEY = {p.key: p for p in PRINCIPLES}


def phase_of(board: chess.Board) -> str:
    """The same rule as hobbies/chess board_manager.detect_phase."""
    queens = board.pieces(chess.QUEEN, chess.WHITE) or board.pieces(chess.QUEEN, chess.BLACK)
    if not queens and len(board.piece_map()) <= 12:
        return "endgame"
    if board.fullmove_number <= OPENING_END_MOVE:
        return "opening"
    if board.fullmove_number <= MIDDLEGAME_END_MOVE:
        return "middlegame"
    return "endgame"


def material(board: chess.Board, color: chess.Color) -> int:
    return sum(VALUES[p.piece_type] for p in board.piece_map().values() if p.color == color)


@dataclass
class Ply:
    """One half-move with everything the detectors need."""
    index: int
    board_before: chess.Board
    move: chess.Move
    san: str
    mover: chess.Color
    phase: str
    cp_loss: Optional[int]
    best: Optional[chess.Move]
    reply: Optional[chess.Move] = None     # the engine's best answer to this move


def num(p: "Ply") -> str:
    """Move-number label: '3.' for White, '3...' for Black."""
    n = p.board_before.fullmove_number
    return f"{n}." if p.mover == chess.WHITE else f"{n}..."


@dataclass
class Finding:
    key: str
    severity: float
    evidence: str


def _player_moves(plies: list[Ply], color: chess.Color) -> list[Ply]:
    return [p for p in plies if p.mover == color]


def _center(plies, color) -> Optional[Finding]:
    own = _player_moves(plies, color)[:3]
    if len(own) < 3:
        return None
    targets = {chess.WHITE: {chess.E4, chess.D4, chess.C4}, chess.BLACK: {chess.E5, chess.D5, chess.C5}}[color]
    for p in own:
        piece = p.board_before.piece_at(p.move.from_square)
        if piece and piece.piece_type == chess.PAWN and p.move.to_square in targets:
            return None
    return Finding("center", 1, "no pawn to the center in your first three moves")


def _minors_home(board: chess.Board, color: chess.Color) -> int:
    home = {chess.WHITE: {chess.B1: chess.KNIGHT, chess.G1: chess.KNIGHT, chess.C1: chess.BISHOP, chess.F1: chess.BISHOP},
            chess.BLACK: {chess.B8: chess.KNIGHT, chess.G8: chess.KNIGHT, chess.C8: chess.BISHOP, chess.F8: chess.BISHOP}}[color]
    return sum(1 for sq, pt in home.items() if (pc := board.piece_at(sq)) and pc.color == color and pc.piece_type == pt)


def _develop(plies, color) -> Optional[Finding]:
    own = _player_moves(plies, color)
    if len(own) < 10:
        return None
    board = own[9].board_before.copy()
    board.push(own[9].move)
    home = _minors_home(board, color)
    if home >= 2:
        return Finding("develop", home, f"{home} knights or bishops still at home after your 10th move")
    return None


def _early_queen(plies, color) -> Optional[Finding]:
    for p in _player_moves(plies, color)[:5]:
        piece = p.board_before.piece_at(p.move.from_square)
        if piece and piece.piece_type == chess.QUEEN and _minors_home(p.board_before, color) >= 3:
            return Finding("early_queen", 2, f"queen moved on move {p.board_before.fullmove_number} ({p.san}) "
                                              "before your knights and bishops")
    return None


def _one_move_each(plies, color) -> Optional[Finding]:
    moved: dict[int, int] = {}
    repeats = []
    for p in _player_moves(plies, color)[:8]:
        piece = p.board_before.piece_at(p.move.from_square)
        if not piece or piece.piece_type in (chess.PAWN, chess.KING):
            continue
        attacked = p.board_before.is_attacked_by(not color, p.move.from_square)
        if p.move.from_square in moved and not p.board_before.is_capture(p.move) and not attacked \
                and _minors_home(p.board_before, color) >= 1:
            repeats.append(f"{num(p)}{p.san}")
        moved[p.move.to_square] = moved.pop(p.move.from_square, 0) + 1
    if repeats:
        return Finding("one_move_each", len(repeats), "moved the same piece again with others undeveloped: " + ", ".join(repeats))
    return None


def _castle(plies, color) -> Optional[Finding]:
    own = _player_moves(plies, color)
    if len(own) < 12:
        return None
    for p in own[:12]:
        if p.board_before.is_castling(p.move):
            return None
    board = own[11].board_before.copy()
    board.push(own[11].move)
    king = board.king(color)
    start = chess.E1 if color == chess.WHITE else chess.E8
    queens_on = bool(board.pieces(chess.QUEEN, chess.WHITE) or board.pieces(chess.QUEEN, chess.BLACK))
    if king == start and queens_on:
        return Finding("castle", 2, "not castled by move 12, king still in the middle with queens on")
    return None


def _king_pawns(plies, color) -> Optional[Finding]:
    for p in _player_moves(plies, color)[:6]:
        piece = p.board_before.piece_at(p.move.from_square)
        castled = p.board_before.king(color) not in (chess.E1, chess.E8)
        if piece and piece.piece_type == chess.PAWN and chess.square_file(p.move.from_square) == 5 and not castled:
            return Finding("king_pawns", 1, f"f-pawn moved on move {p.board_before.fullmove_number} ({p.san}) "
                                             "before castling")
    return None


def _cct(plies, color) -> Optional[Finding]:
    missed = []
    for p in _player_moves(plies, color):
        if p.cp_loss is None or p.cp_loss < BIG_LOSS or p.best is None or p.best == p.move:
            continue
        b = p.board_before
        if b.is_capture(p.best) or b.gives_check(p.best):
            missed.append(f"{num(p)}{b.san(p.best)} instead of {p.san}")
    if missed:
        return Finding("cct", 3 * len(missed), "missed a check or capture: " + "; ".join(missed[:3]))
    return None


def _their_threat(plies, color) -> Optional[Finding]:
    """A big loss where the opponent's best answer was a check, a capture or mate."""
    hit = []
    for p in _player_moves(plies, color):
        if p.cp_loss is None or p.cp_loss < BIG_LOSS or p.index + 1 >= len(plies):
            continue
        reply = plies[p.index + 1].best
        if reply is None:
            continue
        after = p.board_before.copy()
        after.push(p.move)
        if reply in after.legal_moves and (after.is_capture(reply) or after.gives_check(reply)):
            hit.append(f"{num(p)}{p.san} allowed {after.san(reply)}")
    if hit:
        return Finding("their_threat", 3 * len(hit), "; ".join(hit[:3]))
    return None


def _loose_pieces(plies, color) -> Optional[Finding]:
    dropped = []
    for p in _player_moves(plies, color):
        if p.cp_loss is None or p.cp_loss < BIG_LOSS:
            continue
        after = p.board_before.copy()
        after.push(p.move)
        for sq, piece in after.piece_map().items():
            if piece.color == color and VALUES[piece.piece_type] >= 3 \
                    and after.is_attacked_by(not color, sq) and not after.is_attacked_by(color, sq):
                dropped.append(f"{num(p)}{p.san} left the {chess.piece_name(piece.piece_type)} "
                               f"on {chess.square_name(sq)} undefended")
                break
    if dropped:
        return Finding("loose_pieces", 3 * len(dropped), "; ".join(dropped[:3]))
    return None


def _convert(plies, color, result: str) -> Optional[Finding]:
    ahead = 0
    for p in plies:
        if p.phase == "opening":
            continue
        diff = material(p.board_before, color) - material(p.board_before, not color)
        ahead = ahead + 1 if diff >= 3 else 0
        if ahead >= 6 and result in ("loss", "draw"):
            return Finding("convert", 4, f"up at least three points of material for several moves and still {result}")
    return None


def _king_active(plies, color) -> Optional[Finding]:
    own = [p for p in _player_moves(plies, color) if p.phase == "endgame"]
    if len(own) < 10:
        return None
    king_moves = sum(1 for p in own[:10] if p.board_before.piece_type_at(p.move.from_square) == chess.KING)
    if king_moves < 2:
        return Finding("king_active", 2, f"only {king_moves} king move{'s' if king_moves != 1 else ''} "
                                          "in your first ten endgame moves")
    return None


def _passed(board: chess.Board, color: chess.Color) -> list[int]:
    out = []
    for sq in board.pieces(chess.PAWN, color):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        ahead = range(r + 1, 8) if color == chess.WHITE else range(0, r)
        blocked = any(board.piece_at(chess.square(ff, rr)) == chess.Piece(chess.PAWN, not color)
                      for ff in (f - 1, f, f + 1) if 0 <= ff < 8 for rr in ahead)
        if not blocked:
            out.append(sq)
    return out


def _passed_pawns(plies, color) -> Optional[Finding]:
    own = [p for p in _player_moves(plies, color) if p.phase == "endgame"]
    idle = 0
    for p in own:
        passers = _passed(p.board_before, color)
        if not passers:
            idle = 0
            continue
        idle = 0 if p.move.from_square in passers else idle + 1
        if idle >= 8:
            return Finding("passed_pawns", 2, "had a passed pawn for eight moves without pushing it")
    return None


def _endgame_study(plies, color) -> Optional[Finding]:
    own = [p for p in _player_moves(plies, color) if p.phase == "endgame" and p.cp_loss is not None]
    if len(own) < 5:
        return None
    acpl = sum(p.cp_loss for p in own) / len(own)
    if acpl < 60:
        return None
    last = own[-1].board_before
    pieces = {pc.piece_type for pc in last.piece_map().values()} - {chess.KING}
    if pieces == {chess.PAWN}:
        return Finding("opposition", acpl / 30, f"king-and-pawn ending played at {acpl:.0f} centipawns lost per move")
    if chess.ROOK in pieces and not pieces & {chess.QUEEN, chess.KNIGHT, chess.BISHOP}:
        return Finding("rook_endings", acpl / 30, f"rook ending played at {acpl:.0f} centipawns lost per move")
    return None


DETECTORS: tuple[Callable, ...] = (_center, _develop, _early_queen, _one_move_each, _castle, _king_pawns,
                                   _their_threat, _cct, _loose_pieces, _king_active, _passed_pawns, _endgame_study)


def check(plies: list[Ply], color: chess.Color, result: str) -> list[Finding]:
    """Every principle this game broke, worst first."""
    found = [f for d in DETECTORS if (f := d(plies, color)) is not None]
    if (c := _convert(plies, color, result)) is not None:
        found.append(c)
    return sorted(found, key=lambda f: -f.severity)
