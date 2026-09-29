"""Coach-style move explanation engine for the chess training system.

Copied unchanged from My_folder/hobbies/chess/chess_training/analysis/explainer.py so the web
app can explain moves without the rest of that package (its engine wrapper needs a local
Stockfish binary, which Render does not have)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Tuple

import chess

# Piece values in pawn units (used only in explanations, not centipawns)
_PIECE_VALUES: Dict[int, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

_CENTER_SQUARES: Set[int] = {chess.D4, chess.D5, chess.E4, chess.E5}

_EXTENDED_CENTER: Set[int] = {
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.D4, chess.E4, chess.F4,
    chess.C5, chess.D5, chess.E5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
}

_PIECE_NAMES: Dict[int, str] = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

MoveQuality = Literal["excellent", "good", "inaccuracy", "mistake", "blunder", "unknown"]


@dataclass
class MoveExplanation:
    """Structured coaching explanation for a single chess move."""

    move: chess.Move
    move_san: str                    # e.g. "Nf3"
    ply: int                         # 1-indexed half-move number

    # Primary explanation (always present)
    headline: str                    # 1-sentence coach verdict

    # Supporting detail (always present)
    details: List[str]               # 2–4 bullet-point coaching observations

    # Classification
    quality: MoveQuality

    # Optional engine context
    centipawn_loss: Optional[int]    # positive = loss from player's perspective
    had_better: bool                 # True if engine best move differs from played move
    better_move_san: Optional[str]   # SAN of engine best move, if different


class MoveExplainer:
    """Generates coach-style teaching explanations for individual chess moves."""

    def explain_move(
        self,
        board_before: chess.Board,
        move: chess.Move,
        engine_eval_before: Optional[int] = None,
        engine_eval_after: Optional[int] = None,
        best_move: Optional[chess.Move] = None,
    ) -> MoveExplanation:
        """
        Analyze a single move and return a structured explanation.
        board_before is the position BEFORE the move is made.
        engine_eval_before/after are centipawns from the mover's perspective.
        """
        board_after = board_before.copy()
        board_after.push(move)

        ply = board_before.ply() + 1
        mover_color = board_before.turn
        move_san = board_before.san(move)

        # Centipawn loss (positive = bad for the mover)
        cp_loss: Optional[int] = None
        if engine_eval_before is not None and engine_eval_after is not None:
            cp_loss = max(0, engine_eval_before - engine_eval_after)

        # Engine best move metadata
        had_better = False
        better_move_san: Optional[str] = None
        if best_move is not None and best_move != move:
            had_better = True
            try:
                better_move_san = board_before.san(best_move)
            except Exception:
                better_move_san = best_move.uci()

        signals = self._analyze_signals(board_before, move, board_after, ply, mover_color)
        quality = self._classify_quality(cp_loss, signals, had_better)
        headline = self._build_headline(quality, signals, move_san, board_before, move, board_after, mover_color)
        details = self._build_details(signals, board_before, move, board_after, ply, mover_color)

        return MoveExplanation(
            move=move,
            move_san=move_san,
            ply=ply,
            headline=headline,
            details=details,
            quality=quality,
            centipawn_loss=cp_loss,
            had_better=had_better,
            better_move_san=better_move_san,
        )

    def explain_game(
        self,
        moves: List[chess.Move],
        engine_evals: Optional[List[int]] = None,
        best_moves: Optional[List[chess.Move]] = None,
    ) -> List[MoveExplanation]:
        """Explain every move in a game. Returns one MoveExplanation per move.

        engine_evals, if supplied, should be white-centric centipawns with one
        value per position (len(moves) + 1 entries: before move 0, after move 0,
        …, after last move).  The method converts them to mover-perspective
        automatically, so callers can pass EngineAnalyzer output directly.
        """
        board = chess.Board()
        explanations: List[MoveExplanation] = []

        for i, move in enumerate(moves):
            mover = board.turn

            # Convert white-centric eval → mover-perspective
            def _to_mover(raw: Optional[int], color: chess.Color) -> Optional[int]:
                if raw is None:
                    return None
                return raw if color == chess.WHITE else -raw

            eval_before: Optional[int] = None
            eval_after: Optional[int] = None

            if engine_evals is not None:
                raw_b = engine_evals[i] if i < len(engine_evals) else None
                raw_a = engine_evals[i + 1] if (i + 1) < len(engine_evals) else None
                eval_before = _to_mover(raw_b, mover)
                eval_after = _to_mover(raw_a, mover)

            best = best_moves[i] if best_moves and i < len(best_moves) else None

            exp = self.explain_move(
                board_before=board.copy(),
                move=move,
                engine_eval_before=eval_before,
                engine_eval_after=eval_after,
                best_move=best,
            )
            explanations.append(exp)
            board.push(move)

        return explanations

    # ------------------------------------------------------------------ #
    # Signal analysis
    # ------------------------------------------------------------------ #

    def _analyze_signals(
        self,
        board_before: chess.Board,
        move: chess.Move,
        board_after: chess.Board,
        ply: int,
        mover_color: chess.Color,
    ) -> Dict:
        """Compute all analysis signals and return as a flat dictionary."""
        s: Dict = {}
        enemy_color = not mover_color
        moved_piece = board_before.piece_at(move.from_square)

        # ── Material ─────────────────────────────────────────────────────
        is_en_passant = board_before.is_en_passant(move)
        captured_piece = board_before.piece_at(move.to_square)
        if is_en_passant:
            captured_piece = chess.Piece(chess.PAWN, enemy_color)

        s["is_capture"] = captured_piece is not None
        s["captured_piece"] = captured_piece
        s["captured_value"] = _PIECE_VALUES.get(captured_piece.piece_type, 0) if captured_piece else 0
        s["moved_piece"] = moved_piece
        s["moved_piece_type"] = moved_piece.piece_type if moved_piece else None
        s["moved_piece_value"] = _PIECE_VALUES.get(moved_piece.piece_type, 0) if moved_piece else 0

        if captured_piece and moved_piece:
            s["trade_diff"] = s["captured_value"] - s["moved_piece_value"]
        else:
            s["trade_diff"] = 0

        # ── Piece safety after move ───────────────────────────────────────
        # Is the moved piece hanging?
        moved_to = move.to_square
        s["moved_piece_hanging"] = (
            moved_piece is not None
            and moved_piece.piece_type != chess.KING
            and board_after.is_attacked_by(enemy_color, moved_to)
            and not board_after.is_attacked_by(mover_color, moved_to)
        )

        # Is the moved piece attacked by a lower-value enemy piece?
        s["moved_piece_attacked_by_lower"] = (
            moved_piece is not None
            and board_after.is_attacked_by(enemy_color, moved_to)
            and self._attacked_by_lower_value(board_after, moved_to, moved_piece, enemy_color)
        )

        # All own hanging pieces after the move
        hanging: List[Tuple[int, chess.Piece]] = []
        for sq, piece in board_after.piece_map().items():
            if (piece.color == mover_color
                    and piece.piece_type != chess.KING
                    and board_after.is_attacked_by(enemy_color, sq)
                    and not board_after.is_attacked_by(mover_color, sq)):
                hanging.append((sq, piece))
        s["hanging_pieces"] = hanging
        s["any_hanging"] = len(hanging) > 0

        # ── Center control ────────────────────────────────────────────────
        s["lands_on_center"] = move.to_square in _CENTER_SQUARES
        s["lands_on_ext_center"] = move.to_square in _EXTENDED_CENTER
        s["vacates_center"] = move.from_square in _CENTER_SQUARES
        s["attacks_center_after"] = (
            moved_piece is not None
            and any(sq in _CENTER_SQUARES for sq in board_after.attacks(move.to_square))
        )

        # ── Mobility ─────────────────────────────────────────────────────
        s["mobility_before_mover"] = board_before.legal_moves.count()
        s["mobility_after_opp"] = board_after.legal_moves.count()

        # ── Tactical signals ──────────────────────────────────────────────
        s["gives_check"] = board_after.is_check()
        s["is_checkmate"] = board_after.is_checkmate()
        s["threatens_mate"] = (
            not board_after.is_game_over()
            and self._threatens_mate_in_one(board_after, mover_color)
        )
        s["is_fork"] = self._detect_fork(board_after, move, mover_color)
        s["creates_pin"] = self._detect_pin(board_after, enemy_color)

        # ── Development (opening only) ────────────────────────────────────
        s["is_opening"] = ply <= 20
        s["is_castling"] = board_before.is_castling(move)
        if s["is_opening"]:
            s["is_first_move_for_piece"] = not self._has_piece_moved_before(board_before, move.from_square)
            s["moves_same_piece_twice"] = self._has_piece_moved_before(board_before, move.from_square)
            s["minor_pieces_undeveloped"] = self._count_undeveloped_minor_pieces(board_after, mover_color)
        else:
            s["is_first_move_for_piece"] = False
            s["moves_same_piece_twice"] = False
            s["minor_pieces_undeveloped"] = 0

        return s

    # ------------------------------------------------------------------ #
    # Quality classification
    # ------------------------------------------------------------------ #

    def _classify_quality(
        self,
        cp_loss: Optional[int],
        signals: Dict,
        had_better: bool,
    ) -> MoveQuality:
        """Classify move quality using centipawn loss when available, else heuristics."""
        if cp_loss is not None:
            if cp_loss <= 10:
                return "excellent"
            if cp_loss <= 30:
                return "good"
            if cp_loss <= 70:
                return "inaccuracy"
            if cp_loss <= 150:
                return "mistake"
            return "blunder"

        # Heuristic path (no engine data)
        if signals.get("is_checkmate"):
            return "excellent"
        if signals.get("threatens_mate") or signals.get("is_fork"):
            return "excellent"
        if signals.get("gives_check") and signals.get("is_fork"):
            return "excellent"

        # Hanging the moved piece is very bad
        if signals.get("moved_piece_hanging"):
            moved_val = signals.get("moved_piece_value", 0)
            if moved_val >= 3:
                return "blunder"
            return "mistake"

        # Any other piece hanging = at least inaccuracy
        if signals.get("any_hanging"):
            # If it was a capture that left something else hanging, it might still be bad
            if signals.get("is_capture") and signals.get("trade_diff", 0) > 0:
                return "inaccuracy"
            return "blunder"

        if signals.get("is_capture"):
            td = signals.get("trade_diff", 0)
            if td > 0:
                return "excellent"
            if td == 0:
                return "good"
            return "mistake"

        if signals.get("gives_check"):
            return "good"

        if signals.get("is_castling"):
            return "good"

        if signals.get("is_opening"):
            if signals.get("is_first_move_for_piece") and not signals.get("any_hanging"):
                if signals.get("lands_on_center") or signals.get("attacks_center_after"):
                    return "good"
                return "good"
            if signals.get("moves_same_piece_twice"):
                return "inaccuracy"

        if signals.get("lands_on_center") or signals.get("attacks_center_after"):
            return "good"

        return "unknown"

    # ------------------------------------------------------------------ #
    # Headline generation
    # ------------------------------------------------------------------ #

    def _build_headline(
        self,
        quality: MoveQuality,
        signals: Dict,
        move_san: str,
        board_before: chess.Board,
        move: chess.Move,
        board_after: chess.Board,
        mover_color: chess.Color,
    ) -> str:
        """Generate a single-sentence coach-style headline for this move."""
        # Special cases in priority order
        if signals.get("is_checkmate"):
            return "Checkmate — a decisive finish. Well played."

        if signals.get("threatens_mate"):
            return "Powerful move — you're threatening checkmate on the very next turn."

        if signals.get("is_fork"):
            piece_nm = self._pname(signals.get("moved_piece_type"))
            return f"Sharp tactic — your {piece_nm} forks two pieces at once."

        if signals.get("creates_pin"):
            return "Nice positional idea — you've pinned an opponent piece to its king."

        if signals.get("gives_check"):
            if quality in ("excellent", "good"):
                return "You deliver check and keep the pressure on."
            return "A checking move, though the follow-up will need care."

        if signals.get("is_castling"):
            return "Perfect timing on the castle — your king is safely tucked away."

        # Piece safety issues
        if signals.get("moved_piece_hanging"):
            piece_nm = self._pname(signals.get("moved_piece_type"))
            return (
                f"This move leaves your {piece_nm} undefended — "
                f"your opponent can win material here."
            )

        if signals.get("any_hanging") and not signals.get("moved_piece_hanging"):
            hp_sq, hp = signals["hanging_pieces"][0]
            hp_nm = self._pname(hp.piece_type)
            return (
                f"After this move your {hp_nm} on {chess.square_name(hp_sq)} "
                f"is left undefended — a detail worth watching."
            )

        # Captures
        if signals.get("is_capture"):
            td = signals.get("trade_diff", 0)
            cap = signals.get("captured_piece")
            if td > 0:
                cap_nm = self._pname(cap.piece_type) if cap else "piece"
                return f"Good capture — you win material by taking the {cap_nm}."
            if td == 0:
                return "Good recapture — you maintain material balance."
            return "This capture gives up more than it gains — think twice before committing."

        # Development / opening moves
        if signals.get("is_opening") and signals.get("is_first_move_for_piece"):
            piece_nm = self._pname(signals.get("moved_piece_type"))
            if signals.get("lands_on_center") or signals.get("attacks_center_after"):
                return (
                    f"Good development — your {piece_nm} enters the game "
                    f"and targets the center."
                )
            return f"Solid developing move that brings your {piece_nm} into play."

        if signals.get("moves_same_piece_twice") and signals.get("is_opening"):
            piece_nm = self._pname(signals.get("moved_piece_type"))
            return (
                f"Moving the same {piece_nm} twice in the opening costs you a tempo — "
                f"consider developing another piece instead."
            )

        if signals.get("lands_on_center"):
            return "Useful pawn advance — you claim space in the heart of the board."

        if signals.get("attacks_center_after"):
            return "Solid move that keeps pressure on the central squares."

        # Generic fallbacks keyed on quality
        if quality == "excellent":
            return "Well-timed move that improves your position significantly."
        if quality == "good":
            return "Solid move that keeps the position flexible and your options open."
        if quality == "inaccuracy":
            return "Reasonable move, but a more active option was available here."
        if quality == "mistake":
            return "This move gives your opponent a real chance to seize control."
        if quality == "blunder":
            return "After this, your opponent has a strong response that's hard to counter."
        return "A quiet, consolidating move — the position stays roughly balanced."

    # ------------------------------------------------------------------ #
    # Detail bullets
    # ------------------------------------------------------------------ #

    def _build_details(
        self,
        signals: Dict,
        board_before: chess.Board,
        move: chess.Move,
        board_after: chess.Board,
        ply: int,
        mover_color: chess.Color,
    ) -> List[str]:
        """Build 2–4 coaching observations as bullet-point strings."""
        details: List[str] = []

        # Material observation
        if signals.get("is_capture"):
            cap = signals.get("captured_piece")
            mv_type = signals.get("moved_piece_type")
            mv_val = signals.get("moved_piece_value", 0)
            cap_val = signals.get("captured_value", 0)
            td = signals.get("trade_diff", 0)
            cap_nm = self._pname(cap.piece_type) if cap else "piece"
            mv_nm = self._pname(mv_type)
            if td > 0:
                details.append(
                    f"You take the {cap_nm} (worth {cap_val}) with your {mv_nm} "
                    f"(worth {mv_val}) — a material gain of {td} pawn unit{'s' if td != 1 else ''}."
                )
            elif td == 0:
                details.append(
                    f"You trade your {mv_nm} for the opponent's {cap_nm} — an equal exchange."
                )
            else:
                details.append(
                    f"You give up your {mv_nm} ({mv_val}) for the {cap_nm} ({cap_val}) — "
                    f"you lose {abs(td)} pawn unit{'s' if abs(td) != 1 else ''} of material."
                )

        # Piece safety
        if signals.get("moved_piece_hanging"):
            piece_nm = self._pname(signals.get("moved_piece_type"))
            details.append(
                f"Your {piece_nm} lands on an undefended square — "
                f"it can be captured for free on the next move."
            )
        elif signals.get("moved_piece_attacked_by_lower"):
            piece_nm = self._pname(signals.get("moved_piece_type"))
            details.append(
                f"Your {piece_nm} is now attacked by a less valuable piece — "
                f"you may need to move it again."
            )
        elif signals.get("any_hanging"):
            hp_sq, hp = signals["hanging_pieces"][0]
            hp_nm = self._pname(hp.piece_type)
            details.append(
                f"Your {hp_nm} on {chess.square_name(hp_sq)} is now undefended — "
                f"keep an eye on it."
            )

        # Tactical highlights
        if signals.get("gives_check"):
            details.append(
                "This move delivers check, forcing your opponent to respond immediately "
                "rather than pursuing their own plans."
            )

        if signals.get("is_fork"):
            details.append(
                "The fork attacks two enemy pieces at once — "
                "your opponent can only save one of them."
            )

        if signals.get("creates_pin"):
            details.append(
                "A pin is created: an enemy piece can no longer move safely "
                "without exposing a more valuable piece behind it."
            )

        if signals.get("threatens_mate"):
            details.append(
                "You now have a checkmate threat on the next move — "
                "your opponent must find the only defensive reply."
            )

        # Center control
        if signals.get("lands_on_center"):
            details.append(
                "Placing a piece on a central square gives you influence over "
                "the most important region of the board."
            )
        elif signals.get("attacks_center_after"):
            details.append(
                "This piece now pressures central squares, limiting your opponent's "
                "ability to place pieces there comfortably."
            )
        elif signals.get("vacates_center") and not signals.get("is_capture"):
            details.append(
                "Moving away from the center can cede space — "
                "watch for your opponent to occupy those squares."
            )

        # Development (opening only)
        if signals.get("is_opening"):
            if signals.get("is_castling"):
                details.append(
                    "Castling connects your rooks and shields your king behind the pawn structure."
                )
            elif signals.get("is_first_move_for_piece"):
                undevel = signals.get("minor_pieces_undeveloped", 0)
                if undevel > 0:
                    details.append(
                        f"You still have {undevel} minor piece"
                        f"{'s' if undevel > 1 else ''} on their starting squares — "
                        f"aim to develop them before launching an attack."
                    )
            elif signals.get("moves_same_piece_twice"):
                details.append(
                    "Repeating a move in the opening lets your opponent gain "
                    "a tempo — try to develop fresh pieces instead."
                )

        # Mobility note
        opp_mob = signals.get("mobility_after_opp", 0)
        if opp_mob <= 5 and not signals.get("is_checkmate") and not signals.get("gives_check"):
            details.append(
                "Your opponent has very few legal replies — "
                "you've greatly limited their options."
            )
        elif opp_mob >= 40 and not signals.get("is_capture"):
            details.append(
                "Your opponent retains many possible responses — "
                "the position is still wide open."
            )

        # Safety net: always return at least one bullet
        if not details:
            if signals.get("lands_on_ext_center"):
                details.append(
                    "The piece takes a good spot in the extended center, "
                    "keeping options flexible."
                )
            else:
                details.append(
                    "No immediate threats or weaknesses are introduced — "
                    "the position stays steady."
                )

        return details[:4]

    # ------------------------------------------------------------------ #
    # Tactical helpers
    # ------------------------------------------------------------------ #

    def _threatens_mate_in_one(
        self,
        board_after: chess.Board,
        mover_color: chess.Color,
    ) -> bool:
        """Check if the mover has a latent checkmate threat after their move."""
        try:
            temp = board_after.copy()
            # Give the turn back to the mover via a null move (opponent "passes")
            temp.push(chess.Move.null())
            for m in temp.legal_moves:
                b2 = temp.copy()
                b2.push(m)
                if b2.is_checkmate():
                    return True
            return False
        except Exception:
            return False

    def _detect_fork(
        self,
        board_after: chess.Board,
        move: chess.Move,
        mover_color: chess.Color,
    ) -> bool:
        """Return True if the moved piece attacks two or more valuable enemy targets."""
        piece = board_after.piece_at(move.to_square)
        if piece is None:
            return False

        enemy_color = not mover_color
        attacked_valuable: List[int] = []
        for sq in board_after.attacks(move.to_square):
            target = board_after.piece_at(sq)
            if target is not None and target.color == enemy_color:
                if target.piece_type in (
                    chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING
                ):
                    attacked_valuable.append(sq)

        return len(attacked_valuable) >= 2

    def _detect_pin(
        self,
        board_after: chess.Board,
        enemy_color: chess.Color,
    ) -> bool:
        """Return True if any enemy piece is now pinned to its king."""
        for sq, piece in board_after.piece_map().items():
            if piece.color == enemy_color and piece.piece_type != chess.KING:
                if board_after.is_pinned(enemy_color, sq):
                    return True
        return False

    def _attacked_by_lower_value(
        self,
        board: chess.Board,
        square: int,
        piece: chess.Piece,
        enemy_color: chess.Color,
    ) -> bool:
        """Return True if `square` is attacked by any enemy piece cheaper than `piece`."""
        own_value = _PIECE_VALUES.get(piece.piece_type, 0)
        for sq, ep in board.piece_map().items():
            if ep.color == enemy_color:
                attacker_val = _PIECE_VALUES.get(ep.piece_type, 0)
                if attacker_val < own_value and board.is_attacked_by(enemy_color, square):
                    # Confirm this specific piece attacks the square
                    if square in board.attacks(sq):
                        return True
        return False

    # ------------------------------------------------------------------ #
    # Development helpers
    # ------------------------------------------------------------------ #

    def _has_piece_moved_before(self, board: chess.Board, square: int) -> bool:
        """Return True if the piece currently at `square` has moved before in the game."""
        piece = board.piece_at(square)
        if piece is None:
            return False

        # If not on its original starting square, it must have moved at some point
        start_squares = self._starting_squares(piece)
        if square not in start_squares:
            return True

        # On starting square — check if any earlier move landed here
        for past_move in board.move_stack:
            if past_move.to_square == square:
                return True

        return False

    def _starting_squares(self, piece: chess.Piece) -> Set[int]:
        """Return the set of expected starting squares for a given piece."""
        rank = 0 if piece.color == chess.WHITE else 7
        pawn_rank = 1 if piece.color == chess.WHITE else 6

        mapping: Dict[int, Set[int]] = {
            chess.PAWN: {chess.square(f, pawn_rank) for f in range(8)},
            chess.ROOK: {chess.square(0, rank), chess.square(7, rank)},
            chess.KNIGHT: {chess.square(1, rank), chess.square(6, rank)},
            chess.BISHOP: {chess.square(2, rank), chess.square(5, rank)},
            chess.QUEEN: {chess.square(3, rank)},
            chess.KING: {chess.square(4, rank)},
        }
        return mapping.get(piece.piece_type, set())

    def _count_undeveloped_minor_pieces(
        self,
        board: chess.Board,
        color: chess.Color,
    ) -> int:
        """Count knights and bishops still sitting on their starting squares."""
        rank = 0 if color == chess.WHITE else 7
        starting = {
            chess.square(1, rank), chess.square(6, rank),  # knights
            chess.square(2, rank), chess.square(5, rank),  # bishops
        }
        count = 0
        for sq in starting:
            p = board.piece_at(sq)
            if p is not None and p.color == color and p.piece_type in (chess.KNIGHT, chess.BISHOP):
                count += 1
        return count

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pname(piece_type: Optional[int]) -> str:
        """Return a human-readable piece name."""
        return _PIECE_NAMES.get(piece_type, "piece") if piece_type is not None else "piece"
