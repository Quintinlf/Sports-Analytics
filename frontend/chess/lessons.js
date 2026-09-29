// What the Learn tab teaches: the rules people miss (each one a position to solve on the board),
// and the opponent styles, each with how it plays and how to beat it.

// goal: {type: "moves", accept: [uci...], otherwise} | {type: "mate"} | {type: "notStalemate"} | {type: "any"}
export const RULE_LESSONS = [
  {
    key: "pieces",
    title: "The pieces and the board",
    fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    text: [
      "White always moves first. Set the board so each player has a light square in the right-hand corner, and the queen starts on her own colour.",
      "King: one square any way. Queen: any distance, straight or diagonal. Rook: any distance, straight. Bishop: any distance, diagonal, so it stays on one colour. Knight: an L (two then one), and it jumps over pieces. Pawn: straight ahead one square, or two from its starting square.",
      "Rough values, for deciding trades: pawn 1, knight 3, bishop 3, rook 5, queen 9. The king can't be traded: lose it and you lose the game.",
    ],
    goal: null,
  },
  {
    key: "pawn-capture",
    title: "Pawns capture diagonally",
    fen: "4k3/8/8/3np3/4P3/8/8/4K3 w - - 0 1",
    text: [
      "Pawns move straight ahead but capture one square diagonally forward. They can never capture straight ahead: the black pawn on e5 simply blocks yours.",
      "Capture the knight.",
    ],
    goal: { type: "moves", accept: ["e4d5"] },
    success: "Right: a diagonal step, only when there's something to take.",
  },
  {
    key: "en-passant",
    title: "En passant",
    fen: "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    text: [
      "Black just pushed the d-pawn two squares, from d7 to d5, so it landed right beside your pawn on e5.",
      "On this move only, you may capture it as if it had moved just one square: your pawn goes to d6 and the black pawn is removed. That's en passant (\"in passing\"). Wait one move and the chance is gone.",
      "Capture en passant.",
    ],
    goal: { type: "moves", accept: ["e5d6"], otherwise: "That's legal, but the lesson is the capture: move your e5 pawn diagonally to d6." },
    success: "That's en passant: the pawn you took was on d5, not d6.",
  },
  {
    key: "promotion",
    title: "Promotion",
    fen: "8/1P2k3/8/8/8/8/8/4K3 w - - 0 1",
    text: [
      "A pawn that reaches the far end of the board must become a queen, rook, bishop or knight, your choice. You can have two queens (or more).",
      "Push the pawn to b8 and pick a piece.",
    ],
    goal: { type: "moves", accept: ["b7b8q", "b7b8r", "b7b8b", "b7b8n"] },
    success: "Promoted. A queen is almost always right; the rare exception is a knight when its check wins something.",
  },
  {
    key: "castling",
    title: "When you can't castle",
    fen: "4k3/8/8/8/2b5/8/8/R3K2R w KQ - 0 1",
    text: [
      "Castling moves your king two squares toward a rook, and the rook jumps over to the other side of the king. You do it by moving the king two squares.",
      "You can't castle if: the king or that rook has already moved, the king is in check, the king would pass over or land on a square an enemy piece attacks, or anything stands in between.",
      "Here the black bishop watches f1, so castling kingside is illegal. Castle the other way: click your king, then c1.",
    ],
    goal: { type: "moves", accept: ["e1c1"], otherwise: "Legal, but this lesson is castling: click the king, then c1." },
    success: "Queenside castling: the king went to c1 and the rook jumped to d1.",
  },
  {
    key: "check",
    title: "Getting out of check",
    fen: "4k3/8/8/8/8/1N6/8/r3K3 w - - 0 1",
    text: [
      "Your king is in check from the rook on a1. When you're in check you must get out of it on this move, in one of three ways: move the king, put a piece in between, or capture the attacking piece.",
      "Find the best way out here.",
    ],
    goal: { type: "moves", accept: ["b3a1"], otherwise: "That does get you out of check. But look: one of your pieces can capture the rook that's attacking you." },
    success: "Capturing the attacker: out of check and a rook ahead.",
  },
  {
    key: "mate",
    title: "Checkmate",
    fen: "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",
    text: [
      "Checkmate is check with no way out: the king can't move to safety, nothing can block, and nothing can capture the attacker. It ends the game at once.",
      "Find the checkmate in one move.",
    ],
    goal: { type: "mate", otherwise: "Not mate yet. The black king is boxed in by its own pawns: attack it along the back row." },
    success: "Back-rank mate: the king's own pawns trapped it.",
  },
  {
    key: "stalemate",
    title: "Stalemate is a draw",
    fen: "k7/8/8/1Q6/8/8/8/K7 w - - 0 1",
    text: [
      "If the player to move has no legal move and is not in check, the game is a draw: stalemate. It's the classic way to throw away a won game.",
      "You're a queen up. Make any move that still leaves Black a legal move.",
    ],
    goal: { type: "notStalemate", otherwise: "Stalemate: a draw! Black's king has no square to go to and isn't in check. Try again." },
    success: "Good: Black can still move. Next you'd walk your king up and checkmate together.",
  },
  {
    key: "pin",
    title: "Pinned pieces",
    fen: "4k3/4r3/8/8/8/8/4N3/4K3 w - - 0 1",
    text: [
      "You may never make a move that leaves your own king in check. So your knight on e2 is pinned: if it moved, the rook on e7 would be attacking your king.",
      "Click the knight: it has no moves. Then make any other move.",
    ],
    goal: { type: "any" },
    success: "Right. A pinned piece also can't defend anything off its line, which is why pins win games.",
  },
  {
    key: "draws",
    title: "Other draws and table manners",
    fen: "8/8/4k3/8/8/3NK3/8/8 w - - 0 1",
    text: [
      "Draws also happen by threefold repetition (the same position three times, same player to move), by the 50-move rule (fifty moves each with no capture and no pawn move), by insufficient material (king against king, or king and one knight or one bishop against a king, like this board: nobody can ever mate), or by agreement.",
      "At a real board: if you touch one of your pieces you must move it (touch-move), you finish a move when you let go of the piece, and you can resign at any time by tipping over your king.",
    ],
    goal: null,
  },
];

// The opponent styles. Each chooses among Stockfish's top five moves, within a margin set by the
// level, preferring the moves its style likes; the weaknesses below are what that preference costs.
export const STYLES = [
  {
    key: "stockfish",
    name: "Stockfish",
    how: "Plays the engine's own choices, weakened by the level you pick.",
    beat: ["Develop your pieces, castle, and before every move ask what its last move threatens."],
  },
  {
    key: "rusher",
    name: "The Queen Rusher",
    how: "Brings its queen out in the first few moves and aims at f7 (f2 when you're White) for a four-move checkmate, the Scholar's mate.",
    beat: [
      "f7 (f2 for White) is your weakest square: only your king guards it. Once its queen and bishop both aim there, guard it: ...g6 (which also hits the queen) or ...Qe7.",
      "Beware ...Nf6: it attacks the queen on h5 but doesn't guard f7, so Qxf7 is checkmate. It's the most common way to fall for this.",
      "Then attack its queen with your knights and bishops. Every time she has to run, you get a free developing move.",
      "Don't send your own queen out to chase hers: develop and castle, and its early queen becomes a target.",
    ],
    book: { w: ["e2e4", "d1h5", "f1c4", "h5f7"], b: ["e7e5", "f8c5", "d8f6", "f6f2"] },
    extra: 60,
  },
  {
    key: "grabber",
    name: "The Grabber",
    how: "Takes almost anything you leave, even when grabbing loses time or opens its own king.",
    beat: [
      "Offer a pawn that opens lines for your pieces (a gambit), then develop with threats while it grabs.",
      "Keep your real pieces defended: it will take anything loose.",
      "Count what it gave up to grab: usually moves, and often its king's safety.",
    ],
    extra: 150,
  },
  {
    key: "attacker",
    name: "The Attacker",
    how: "Goes straight for your king: checks, sacrifices, pieces piling toward your side of the board.",
    beat: [
      "Castle early and leave the pawns in front of your castled king alone.",
      "Near your king, count its attackers against your defenders; trade off its attacking pieces, especially the queen.",
      "Attacks cost material. Survive the storm and you'll usually be the one ahead.",
    ],
    extra: 100,
  },
  {
    key: "turtle",
    name: "The Turtle",
    how: "Stays home and defends, rarely crosses the middle, avoids trades and fights.",
    beat: [
      "Take space: push your center pawns and put your pieces on active squares.",
      "Don't rush. Build up, open a file for your rooks, then break through.",
      "When you're ahead, trade pieces: an endgame a pawn up is easy to win.",
    ],
    extra: 80,
  },
];
