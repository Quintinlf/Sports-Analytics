// The coach's eyes: threats, loose pieces, missed mates and blunders, found with the rules alone
// (chess.js), so it is instant and never depends on the engine's noisy shallow search.
//
// "Wins material" is a two-move check: a capture wins if the captured piece is worth more than the
// capturer, or if nothing can take back. Crude next to an engine, but it is exactly the question the
// Tips tab teaches: what does their move attack, and is anything of mine loose?
import { Chess } from "https://cdn.jsdelivr.net/npm/chess.js@1.0.0/dist/esm/chess.js";

export const VALUE = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };
export const NAME = { p: "pawn", n: "knight", b: "bishop", r: "rook", q: "queen", k: "king" };
const WORTH_STOPPING = 2;            // a minor piece for a pawn, or better

function copy(fen) {
  try { return new Chess(fen); } catch { return null; }
}

// The same position with the other side to move: "if it were their turn, what could they do?"
export function passTurn(fen) {
  const f = fen.split(" ");
  f[1] = f[1] === "w" ? "b" : "w";
  f[3] = "-";
  return copy(f.join(" "));
}

export function mateInOne(fen) {
  const g = copy(fen);
  if (!g) return null;
  for (const m of g.moves({ verbose: true })) {
    g.move(m);
    const mate = g.isCheckmate();
    g.undo();
    if (mate) return m;
  }
  return null;
}

// Best capture for the side to move, judged two moves deep. Returns {move, net} or null.
export function bestCapture(fen) {
  const g = copy(fen);
  if (!g) return null;
  let best = null;
  for (const m of g.moves({ verbose: true })) {
    if (!m.captured) continue;
    const gain = VALUE[m.captured] + (m.promotion ? VALUE[m.promotion] - 1 : 0);
    g.move(m);
    const recapture = g.moves({ verbose: true }).some((r) => r.to === m.to);
    g.undo();
    const net = recapture ? gain - VALUE[m.piece] : gain;
    if (!best || net > best.net) best = { move: m, net };
  }
  return best;
}

function kingSquare(g, color) {
  for (const row of g.board()) for (const cell of row) if (cell && cell.type === "k" && cell.color === color) return cell.square;
  return null;
}

function checkers(g) {
  const color = g.turn();
  return g.attackers(kingSquare(g, color), color === "w" ? "b" : "w");
}

// Before your move: the one thing most worth stopping for, or null.
export function beforeYourMove(fen, { checksExplained = 0 } = {}) {
  const g = copy(fen);
  if (!g || g.isGameOver()) return null;
  const me = g.turn();
  if (g.inCheck()) {
    if (checksExplained >= 2) return null;
    const from = checkers(g);
    const piece = from.length ? g.get(from[0]) : null;
    return {
      kind: "check",
      title: "You're in check",
      text: `Your king is attacked${piece ? ` by the ${NAME[piece.type]} on ${from[0]}` : ""}. You must deal with it right now, in one of three ways: move the king, put a piece in the way, or capture the attacker.`,
      danger: [kingSquare(g, me), ...from],
    };
  }
  const mine = mateInOne(fen);
  if (mine) {
    return {
      kind: "your-mate",
      title: "You have checkmate in one",
      text: "One of your moves ends the game right now. Look at every check you can give.",
      reveal: `${mine.san} is checkmate.`,
      hint: [mine.from, mine.to],
    };
  }
  const them = passTurn(fen);
  const threat = them && mateInOne(them.fen());
  if (threat) {
    return {
      kind: "mate-threat",
      title: "They threaten checkmate",
      text: `If you don't stop it, ${threat.san} is checkmate next move. Guard the square, block the line, or capture the attacker.`,
      danger: [threat.from, threat.to],
    };
  }
  const grab = them && bestCapture(them.fen());
  if (grab && grab.net >= WORTH_STOPPING) {
    const m = grab.move;
    return {
      kind: "hanging",
      title: `Your ${NAME[m.captured]} is in danger`,
      text: `Their ${NAME[m.piece]} on ${m.from} can take your ${NAME[m.captured]} on ${m.to}${grab.net >= VALUE[m.captured] ? ", and nothing of yours can take back" : " and come out ahead"}. Move it, defend it, or make a bigger threat.`,
      danger: [m.from, m.to],
    };
  }
  const win = bestCapture(fen);
  if (win && win.net >= WORTH_STOPPING) {
    return {
      kind: "free-material",
      title: "Something of theirs is loose",
      text: "You can win material with a capture here. Look at each of their pieces: is it defended?",
      reveal: `${win.move.san} wins their ${NAME[win.move.captured]}.`,
      hint: [win.move.from, win.move.to],
    };
  }
  return null;
}

// After you pick a move, before it's played: is it a blunder? Returns null if it looks fine.
export function checkYourMove(fen, move) {
  const g = copy(fen);
  if (!g) return null;
  const hadMate = mateInOne(fen);
  const played = g.move(move);
  if (!played) return null;
  if (g.isCheckmate()) return null;
  if (hadMate) {
    return { kind: "missed-mate", title: "Wait: you have checkmate", text: "There's a move that wins the game on the spot. Look again at your checks.", reveal: `${hadMate.san} is checkmate.` };
  }
  const reply = mateInOne(g.fen());
  if (reply) {
    return { kind: "allows-mate", title: "Hold on: this loses", text: `After ${played.san}, they have ${reply.san}, which is checkmate.`, danger: [reply.from, reply.to] };
  }
  const gained = (played.captured ? VALUE[played.captured] : 0) + (played.promotion ? VALUE[played.promotion] - 1 : 0);
  const answer = bestCapture(g.fen());
  if (answer && answer.net - gained >= WORTH_STOPPING) {
    const m = answer.move;
    return {
      kind: "loses-material",
      title: `Hold on: this loses your ${NAME[m.captured]}`,
      text: `After ${played.san}, their ${NAME[m.piece]} takes your ${NAME[m.captured]} on ${m.to} (${m.san})${gained ? `, which is worth more than what you took` : ""}.`,
      danger: [m.from, m.to],
    };
  }
  return null;
}

// A description of a candidate move, for the opponent styles to score.
export function moveFacts(fen, uci) {
  const g = copy(fen);
  if (!g) return null;
  const color = g.turn();
  const m = g.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] });
  if (!m) return null;
  const enemyKing = kingSquare(g, color === "w" ? "b" : "w");
  const dist = enemyKing ? Math.max(Math.abs(enemyKing.charCodeAt(0) - m.to.charCodeAt(0)), Math.abs(Number(enemyKing[1]) - Number(m.to[1]))) : 7;
  const rank = Number(m.to[1]);
  const fromRank = Number(m.from[1]);
  const forward = color === "w" ? rank - fromRank : fromRank - rank;
  return {
    san: m.san,
    piece: m.piece,
    captured: m.captured || null,
    check: g.inCheck(),
    mate: g.isCheckmate(),
    kingDistance: dist,
    ownHalf: color === "w" ? rank <= 4 : rank >= 5,
    forward,
    castle: m.flags.includes("k") || m.flags.includes("q"),
    attacksWeakSquare: ["f7", "f2"].includes(m.to) || (m.piece === "q" && ["h5", "h4", "f3", "f6"].includes(m.to)),
  };
}
