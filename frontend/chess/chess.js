// Chess: play Stockfish (in the browser), then review the game and build your profile.
// Move rules: chess.js. Engine: Stockfish 10 compiled to JavaScript (lichess's build), run in a
// worker. The server stores each game and returns the coach's review (chess_coach/).
import { Chess } from "https://cdn.jsdelivr.net/npm/chess.js@1.0.0/dist/esm/chess.js";
import { VALUE, beforeYourMove, checkYourMove, moveFacts } from "/chess/coach.js?v=coach1";
import { RULE_LESSONS, STYLES } from "/chess/lessons.js?v=coach1";

const ENGINE_URL = "https://cdn.jsdelivr.net/npm/stockfish.js@10.0.2/stockfish.js";
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const REVIEWER = params.get("reviewer_id") || "";
if (params.get("embed") === "1") document.body.classList.add("embed");

const GLYPH = { p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚" };
// Skill Level is Stockfish's own 0-20 handicap; think time grows with it.
const LEVELS = [
  { skill: 0, ms: 150, hint: "Beginner: hangs pieces often." },
  { skill: 3, ms: 200, hint: "Novice: misses simple tactics." },
  { skill: 6, ms: 300, hint: "Casual: makes real mistakes." },
  { skill: 9, ms: 400, hint: "Club player: punishes loose pieces." },
  { skill: 12, ms: 500, hint: "Strong club player." },
  { skill: 15, ms: 700, hint: "Expert: few mistakes." },
  { skill: 18, ms: 900, hint: "Master strength." },
  { skill: 20, ms: 1200, hint: "Full strength. Good luck." },
];
// Review search per position: this depth, unless the time share runs out first (slow phones).
const ANALYSIS_DEPTH = 12;
const ANALYSIS_BUDGET_MS = 45000;           // the whole game's review, split across its positions
const ANALYSIS_MS = [250, 1500];            // floor and ceiling per position
const MATE = 1000;
// Styled opponents pick among Stockfish's top moves within this many centipawns of its best
// (plus the style's own margin), so a lower level means bigger, more characteristic mistakes.
const STYLE_MARGIN = [320, 240, 180, 130, 90, 60, 35, 15];
const STYLE_BY_KEY = Object.fromEntries(STYLES.map((s) => [s.key, s]));

// ---------------------------------------------------------------- state
let game = new Chess();
let playerColor = "w";
let level = 3;
let state = "idle";               // idle | playing | thinking | analysing | review
let selected = null;
let flipped = false;
let viewPly = null;               // null = live position; otherwise index into positions
let pendingPromotion = null;
let reviewData = null;            // the server's review of the game on the board
let positions = [new Chess().fen()];
let history = [];                 // verbose moves of the game on the board
let style = STYLE_BY_KEY.stockfish;
let coachMode = "threats";        // threats | mistakes | off (remembered per browser)
try { coachMode = localStorage.getItem("chess-coach") || "threats"; } catch { /* storage blocked */ }
let checksExplained = 0;          // "you're in check" is explained twice per game, then trusted
let marks = { danger: [], hint: [] };
let lesson = null;                // {index, done} while a Learn lesson is on the board

// ---------------------------------------------------------------- engine
class Engine {
  constructor() {
    this.listeners = new Set();
    this.queue = Promise.resolve();
    this.multipv = 1;
    try {
      const boot = `importScripts(${JSON.stringify(ENGINE_URL)});`;
      this.worker = new Worker(URL.createObjectURL(new Blob([boot], { type: "application/javascript" })));
      this.worker.onmessage = (e) => { const line = String(e.data); this.listeners.forEach((fn) => fn(line)); };
      this.worker.onerror = () => { this.failed = true; };
      this.ready = this.until("uci", (l) => l === "uciok").then(() => this.until("isready", (l) => l === "readyok"));
    } catch (err) {
      this.failed = true;
      this.ready = Promise.reject(err);
    }
  }
  until(cmd, done) {
    return new Promise((resolve) => {
      const fn = (line) => { if (done(line)) { this.listeners.delete(fn); resolve(line); } };
      this.listeners.add(fn);
      this.worker.postMessage(cmd);
    });
  }
  send(cmd) { this.worker.postMessage(cmd); }
  // One search at a time: one asked for mid-search waits its turn, so bestmoves never cross.
  search(moves, go, multipv = 1) {
    const run = this.queue.then(() => this.#search(moves, go, multipv));
    this.queue = run.catch(() => {});
    return run;
  }
  idle() { return this.queue; }
  // Score from the side to move's view, the best move, and with multipv > 1 the top lines.
  async #search(moves, go, multipv) {
    await this.ready;
    if (multipv !== this.multipv) {
      this.send(`setoption name MultiPV value ${multipv}`);
      this.multipv = multipv;
    }
    let score = null;
    const lines = new Map();
    const watch = (line) => {
      const m = line.match(/score (cp|mate) (-?\d+)/);
      if (!m) return;
      const s = m[1] === "cp" ? Number(m[2]) : (Number(m[2]) > 0 ? MATE : -MATE);
      const k = Number((line.match(/ multipv (\d+)/) || [0, 1])[1]);
      const pv = (line.match(/ pv (\S+)/) || [])[1];
      if (k === 1) score = s;
      if (pv && !/(upper|lower)bound/.test(line)) lines.set(k, { move: pv, score: s });
    };
    this.listeners.add(watch);
    this.send(`position startpos${moves.length ? " moves " + moves.join(" ") : ""}`);
    const line = await this.until(go, (l) => l.startsWith("bestmove"));
    this.listeners.delete(watch);
    const best = line.split(" ")[1];
    return { score, best: best && best !== "(none)" ? best : null, lines: [...lines.values()] };
  }
}
const engine = new Engine();
engine.ready.then(() => { $("engine-status").textContent = "Stockfish ready."; })
  .catch(() => { $("engine-status").textContent = "Stockfish couldn't load, so the opponent and the review are off. You can still move both sides."; });

// ---------------------------------------------------------------- board
const FILES = "abcdefgh";
function squares() {
  const out = [];
  for (let r = 0; r < 8; r++) for (let f = 0; f < 8; f++) {
    const rank = flipped ? r + 1 : 8 - r;
    const file = flipped ? 7 - f : f;
    out.push(FILES[file] + rank);
  }
  return out;
}

function shownGame() {
  if (viewPly === null) return game;
  return new Chess(positions[viewPly]);
}

function render() {
  const g = shownGame();
  const board = $("board");
  board.innerHTML = "";
  const legal = selected && viewPly === null ? game.moves({ square: selected, verbose: true }) : [];
  const targets = new Map(legal.map((m) => [m.to, m]));
  const last = viewPly === null ? history[history.length - 1] : history[viewPly - 1];
  const checkSq = g.inCheck() ? findKing(g, g.turn()) : null;
  for (const sq of squares()) {
    const fileIdx = FILES.indexOf(sq[0]);
    const rankIdx = Number(sq[1]);
    const div = document.createElement("div");
    div.className = `sq ${(fileIdx + rankIdx) % 2 === 1 ? "light" : "dark"}`;
    if (last && (last.from === sq || last.to === sq)) div.classList.add("last");
    if (selected === sq) div.classList.add("sel");
    if (checkSq === sq) div.classList.add("check");
    if (viewPly === null && marks.danger.includes(sq)) div.classList.add("danger");
    if (viewPly === null && marks.hint.includes(sq)) div.classList.add("hint-sq");
    div.dataset.square = sq;
    const piece = g.get(sq);
    if (piece) {
      const span = document.createElement("span");
      span.className = `piece ${piece.color}`;
      span.textContent = GLYPH[piece.type];
      div.appendChild(span);
    }
    if (targets.has(sq)) {
      const mark = document.createElement("span");
      mark.className = piece ? "ring" : "dot";
      div.appendChild(mark);
    }
    const bottomRank = flipped ? "8" : "1";
    const leftFile = flipped ? "h" : "a";
    if (sq[1] === bottomRank) div.insertAdjacentHTML("beforeend", `<span class="coord file">${sq[0]}</span>`);
    if (sq[0] === leftFile) div.insertAdjacentHTML("beforeend", `<span class="coord rank">${sq[1]}</span>`);
    board.appendChild(div);
  }
  renderMoves();
  renderEval();
}

function findKing(g, color) {
  for (const row of g.board()) for (const cell of row) if (cell && cell.type === "k" && cell.color === color) return cell.square;
  return null;
}

function renderMoves() {
  const list = $("movelist");
  list.innerHTML = "";
  const quality = reviewData ? reviewData.moves.map((m) => m.quality) : [];
  history.forEach((mv, i) => {
    if (i % 2 === 0) list.insertAdjacentHTML("beforeend", `<li class="num">${i / 2 + 1}.</li>`);
    const li = document.createElement("li");
    const q = quality[i];
    li.className = `mv ${q ? "q-" + q : ""} ${viewPly === i + 1 ? "cur" : ""}`;
    li.textContent = mv.san;
    li.addEventListener("click", () => { viewPly = i + 1; selected = null; render(); });
    list.appendChild(li);
  });
  if (history.length % 2 === 1) list.insertAdjacentHTML("beforeend", "<li></li>");
}

function renderEval() {
  const evals = reviewData && reviewData._evals;
  const idx = viewPly === null ? positions.length - 1 : viewPly;
  const cp = evals ? evals[idx] : null;
  const white = cp === null || cp === undefined ? 50 : 50 + 50 * Math.tanh(cp / 400);
  $("eval-fill").style.height = `${white.toFixed(1)}%`;
  $("eval-bar").title = cp === null || cp === undefined ? "Evaluation appears after the review"
    : `Evaluation ${(cp / 100).toFixed(1)} (White's side)`;
}

function setStatus(text) { $("status").textContent = text; }

// ---------------------------------------------------------------- moving
$("board").addEventListener("click", (e) => {
  const cell = e.target.closest(".sq");
  if (!cell) return;
  const sq = cell.dataset.square;
  if (viewPly !== null) { viewPly = null; selected = null; render(); return; }
  if (!["playing", "idle", "lesson"].includes(state)) return;
  if (state === "lesson" && (!lesson || lesson.done)) return;
  if (state === "playing" && game.turn() !== playerColor && !engine.failed) return;
  if (selected) {
    const move = game.moves({ square: selected, verbose: true }).find((m) => m.to === sq);
    if (move) {
      if (move.flags.includes("p")) { pendingPromotion = { from: selected, to: sq }; $("promo").hidden = false; return; }
      userMove({ from: selected, to: sq });
      return;
    }
  }
  const piece = game.get(sq);
  selected = piece && piece.color === game.turn() ? sq : null;
  render();
});

$("promo").addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (!btn || !pendingPromotion) return;
  $("promo").hidden = true;
  const move = { ...pendingPromotion, promotion: btn.dataset.piece };
  pendingPromotion = null;
  userMove(move);
});

// A move you chose on the board: a lesson answer, or (in a game) checked by the coach first.
function userMove(move) {
  if (state === "lesson") { lessonTry(move); return; }
  const verdict = state === "playing" && !engine.failed && coachMode !== "off" ? checkYourMove(game.fen(), move) : null;
  if (!verdict) { marks = { danger: [], hint: [] }; play(move); return; }
  selected = null;
  marks = { danger: verdict.danger || [], hint: [] };
  render();
  openCoach("Coach", verdict.title);
  para(verdict.text, "coach-advice");
  if (verdict.reveal) {
    const reveal = coachButton("Show me", "btn-ghost", () => { para(verdict.reveal, "coach-tip"); reveal.remove(); });
  }
  coachButton("Take it back", "btn-primary", () => { closeCoach(); marks = { danger: [], hint: [] }; render(); }).focus();
  coachButton("Play it anyway", "btn-ghost", () => { closeCoach(); marks = { danger: [], hint: [] }; play(move); });
}

function play(move) {
  const done = game.move(move);
  history.push(done);
  positions.push(game.fen());
  selected = null;
  if (state === "idle") state = "playing";
  $("btn-undo").disabled = history.length === 0;
  $("btn-resign").disabled = false;
  render();
  if (game.isGameOver()) { finish(resultOf()); return; }
  if (!engine.failed && game.turn() !== playerColor) engineMove();
  else setStatus(game.turn() === "w" ? "White to move." : "Black to move.");
}

async function engineMove() {
  state = "thinking";
  setStatus(`${style.name} is thinking…`);
  const best = style.key === "stockfish"
    ? (await engine.search(history.map(uci), `go movetime ${LEVELS[level - 1].ms}`)).best
    : await styledMove();
  if (state !== "thinking") return;         // resigned or restarted while it thought
  state = "playing";
  if (!best) return;
  const mv = { from: best.slice(0, 2), to: best.slice(2, 4) };
  if (best.length > 4) mv.promotion = best[4];
  play(mv);
  if (state === "playing") { setStatus("Your move."); coachBefore(); }
}

// ---------------------------------------------------------------- opponent styles
let bookOn = true;                          // a style's signature opening, until it stops being safe

// How much each style likes a candidate move (centipawn-like units, weighed against the engine).
function styleBonus(key, f, ply) {
  const won = f.captured ? VALUE[f.captured] : 0;
  const mate = f.mate ? 1000 : 0;
  switch (key) {
    case "rusher": return (f.piece === "q" && ply < 24 ? 140 : 0) + (f.attacksWeakSquare ? 80 : 0) + (f.check ? 60 : 0) + mate;
    case "grabber": return 130 * won + mate;
    case "attacker": return (f.check ? 140 : 0) + 22 * (7 - f.kingDistance) + 25 * won + mate;
    case "turtle": return (f.ownHalf ? 70 : -60) + (f.captured ? -40 : 0) + (f.forward < 0 ? 30 : 0) + (f.castle ? 80 : 0) + mate;
    default: return mate;
  }
}

// A styled opponent: its signature opening while that's safe, then its favourite among Stockfish's
// top five moves, as long as that move is within the level's margin of the best one.
async function styledMove() {
  const fen = game.fen();
  const side = game.turn();
  const book = style.book && style.book[side];
  const played = history.filter((m) => m.color === side).length;
  if (book && bookOn && played < book.length) {
    const u = book[played];
    const facts = moveFacts(fen, u);
    const move = { from: u.slice(0, 2), to: u.slice(2, 4) };
    if (facts && (played < book.length - 1 || facts.mate) && !checkYourMove(fen, move)) return u;
    bookOn = false;
  }
  const { lines } = await engine.search(history.map(uci), `go movetime ${LEVELS[level - 1].ms + 250}`, 5);
  if (!lines.length) return null;
  const top = Math.max(...lines.map((l) => l.score));
  const margin = STYLE_MARGIN[level - 1] + (style.extra || 0);
  let pick = null;
  let bestValue = -Infinity;
  for (const l of lines) {
    if (l.score < top - margin) continue;
    const f = moveFacts(fen, l.move);
    if (!f) continue;
    const value = styleBonus(style.key, f, history.length) + 0.5 * (l.score - top) + Math.random() * 15;
    if (value > bestValue) { bestValue = value; pick = l.move; }
  }
  return pick || lines[0].move;
}

// ---------------------------------------------------------------- the coach
function openCoach(kicker, title) {
  $("coach-kicker").textContent = kicker;
  $("coach-title").textContent = title;
  $("coach-body").innerHTML = "";
  $("coach-actions").innerHTML = "";
  $("coach-backdrop").hidden = false;
}
function closeCoach() { $("coach-backdrop").hidden = true; }
function para(text, className) {
  const p = document.createElement("p");
  if (className) p.className = className;
  p.textContent = text;
  $("coach-body").appendChild(p);
  return p;
}
function coachButton(label, className, handler) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = `btn ${className}`;
  b.textContent = label;
  b.addEventListener("click", handler);
  $("coach-actions").appendChild(b);
  return b;
}
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("coach-backdrop").hidden) closeCoach(); });

// Your turn: stop for a check, a mate threat, a loose piece, or a chance to win something.
function coachBefore() {
  if (coachMode !== "threats" || state !== "playing" || game.turn() !== playerColor) return;
  const note = beforeYourMove(game.fen(), { checksExplained });
  marks = { danger: note ? note.danger || [] : [], hint: [] };
  render();
  if (!note) return;
  if (note.kind === "check") checksExplained += 1;
  openCoach("Coach", note.title);
  para(note.text, "coach-advice");
  if (note.reveal) {
    const show = coachButton("Show me", "btn-ghost", () => {
      para(note.reveal, "coach-tip");
      marks = { danger: marks.danger, hint: note.hint || [] };
      render();
      show.remove();
    });
  }
  coachButton("Got it", "btn-primary", closeCoach).focus();
}

function introduceStyle() {
  if (style.key === "stockfish" || coachMode === "off") return;
  openCoach("Coach · your opponent", `You're playing ${style.name}`);
  para(style.how);
  para("How to beat it:", "coach-advice");
  const ul = document.createElement("ul");
  ul.className = "coach-lines";
  style.beat.forEach((b) => { const li = document.createElement("li"); li.textContent = b; ul.appendChild(li); });
  $("coach-body").appendChild(ul);
  coachButton("Let's play", "btn-primary", closeCoach).focus();
}

function uci(m) { return m.from + m.to + (m.promotion || ""); }

function resultOf() {
  if (game.isCheckmate()) return game.turn() === "w" ? "0-1" : "1-0";
  return "1/2-1/2";
}

function describe(result) {
  if (game.isCheckmate()) return "Checkmate.";
  if (game.isStalemate()) return "Stalemate: a draw.";
  if (game.isThreefoldRepetition()) return "Draw by repetition.";
  if (game.isInsufficientMaterial()) return "Draw: not enough material to mate.";
  if (result === "1/2-1/2") return "Draw.";
  return "Game over.";
}

// ---------------------------------------------------------------- game controls
$("level").addEventListener("input", () => {
  level = Number($("level").value);
  $("level-out").textContent = level;
  $("level-hint").textContent = LEVELS[level - 1].hint;
});

$("btn-start").addEventListener("click", async () => {
  if (state === "analysing") return;
  const side = $("side").value;
  playerColor = side === "r" ? (Math.random() < 0.5 ? "w" : "b") : side;
  style = STYLE_BY_KEY[$("opponent").value] || STYLE_BY_KEY.stockfish;
  game = new Chess();
  history = [];
  positions = [game.fen()];
  reviewData = null;
  viewPly = null;
  selected = null;
  lesson = null;
  bookOn = true;
  checksExplained = 0;
  marks = { danger: [], hint: [] };
  flipped = playerColor === "b";
  state = "playing";
  $("btn-undo").disabled = true;
  $("btn-resign").disabled = false;
  showTab("play");
  render();
  if (!engine.failed) {
    try {
      await engine.ready;
      await engine.idle();
      engine.send("ucinewgame");
      // A style judges Stockfish's real top moves itself, so the engine plays at full strength.
      engine.send(`setoption name Skill Level value ${style.key === "stockfish" ? LEVELS[level - 1].skill : 20}`);
    } catch { /* engine missing: two-player mode */ }
  }
  setStatus(playerColor === "w" ? "You have White. Your move." : "You have Black.");
  introduceStyle();
  if (!engine.failed && playerColor === "b") engineMove();
});

$("btn-undo").addEventListener("click", () => {
  if (state !== "playing" || history.length === 0) return;
  const back = engine.failed ? 1 : (game.turn() === playerColor ? 2 : 1);
  for (let i = 0; i < back && history.length; i++) { game.undo(); history.pop(); positions.pop(); }
  selected = null;
  marks = { danger: [], hint: [] };
  $("btn-undo").disabled = history.length === 0;
  render();
});

$("btn-resign").addEventListener("click", () => {
  if (!["playing", "thinking"].includes(state) || history.length === 0) return;
  finish(playerColor === "w" ? "0-1" : "1-0", "You resigned.");
});

$("btn-flip").addEventListener("click", () => { flipped = !flipped; render(); });
$("btn-first").addEventListener("click", () => { viewPly = 0; render(); });
$("btn-prev").addEventListener("click", () => { viewPly = Math.max(0, (viewPly ?? positions.length - 1) - 1); render(); });
$("btn-next").addEventListener("click", () => {
  const next = (viewPly ?? positions.length - 1) + 1;
  viewPly = next >= positions.length - 1 ? null : next;
  render();
});
$("btn-last").addEventListener("click", () => { viewPly = null; render(); });
document.addEventListener("keydown", (e) => {
  if (e.target.closest("input, select")) return;
  if (e.key === "ArrowLeft") $("btn-prev").click();
  if (e.key === "ArrowRight") $("btn-next").click();
});

// ---------------------------------------------------------------- after the game
async function finish(result, why) {
  state = "analysing";
  $("btn-undo").disabled = true;
  $("btn-resign").disabled = true;
  setStatus(`${why || describe(result)} ${result === "1/2-1/2" ? "½-½" : result}`);
  showTab("review");
  const { evals, best } = await analyse();
  const body = {
    moves: history.map(uci), evals, best_moves: best,
    player_color: playerColor === "w" ? "white" : "black", result,
    opponent: engine.failed ? "local" : `${style.key}-${level}`,
  };
  if (REVIEWER) body.reviewer_id = REVIEWER;
  try {
    const res = await fetch("/api/chess/games", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const data = await res.json();
    reviewData = { ...data.review, _evals: evals };
    $("games-count").textContent = `Saved. ${data.games_played} game${data.games_played === 1 ? "" : "s"} played in total.`;
    showReview(reviewData);
  } catch (err) {
    $("review-body").innerHTML = `<p class="hint">Couldn't save the game: ${escapeHtml(err.message)}</p>`;
  }
  state = "review";
  render();
}

async function analyse() {
  const n = history.length;
  const evals = new Array(n + 1).fill(null);
  const best = new Array(n).fill(null);
  if (engine.failed) return { evals, best };
  $("analysis-progress").hidden = false;
  $("btn-start").disabled = true;
  await engine.idle();                      // let a move it was still thinking about finish
  engine.send("setoption name Skill Level value 20");
  const [floor, ceiling] = ANALYSIS_MS;
  const ms = Math.round(Math.min(ceiling, Math.max(floor, ANALYSIS_BUDGET_MS / (n + 1))));
  const replay = new Chess();
  const moves = history.map(uci);
  for (let k = 0; k <= n; k++) {
    $("analysis-label").textContent = `Analysing position ${k + 1} of ${n + 1}…`;
    $("analysis-fill").style.width = `${(100 * k) / (n + 1)}%`;
    if (k > 0) replay.move(history[k - 1].san);
    if (replay.isCheckmate()) { evals[k] = replay.turn() === "w" ? -MATE : MATE; continue; }
    if (replay.isDraw() || replay.isStalemate()) { evals[k] = 0; continue; }
    const { score, best: mv } = await engine.search(moves.slice(0, k), `go depth ${ANALYSIS_DEPTH} movetime ${ms}`);
    evals[k] = score === null ? null : (replay.turn() === "w" ? score : -score);
    if (k < n) best[k] = mv;
  }
  $("analysis-fill").style.width = "100%";
  $("analysis-progress").hidden = true;
  $("btn-start").disabled = false;
  return { evals, best };
}

// ---------------------------------------------------------------- review view
const PHASE_COLOURS = { opening: "#3b82f6", middlegame: "#f59e0b", endgame: "#22c55e" };

function showReview(r) {
  const acpl = r.acpl || {};
  const maxLoss = Math.max(60, ...["opening", "middlegame", "endgame"].map((p) => acpl[p] || 0));
  const bars = ["opening", "middlegame", "endgame"].filter((p) => acpl[p] !== null && acpl[p] !== undefined).map((p) => `
    <div class="bar-row"><span>${p}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${(100 * acpl[p]) / maxLoss}%;background:${PHASE_COLOURS[p]}"></div></div>
      <span class="num">${acpl[p].toFixed(0)}</span></div>`).join("");
  const moments = (r.key_moments || []).map((m) => `
    <div class="moment" data-ply="${m.ply}"><h4>Move ${Math.ceil(m.ply / 2)}${m.ply % 2 ? "." : "…"} ${escapeHtml(m.san)}
      <span class="q-${m.quality}">(${m.quality}, −${(m.cp_loss / 100).toFixed(1)})</span></h4>
      <p>${escapeHtml(m.headline)}</p></div>`).join("");
  const lessons = (r.lessons || []).map((l) => `
    <div class="lesson"><h4>${escapeHtml(l.title)}</h4>
      <p class="evidence">${escapeHtml(l.evidence)}</p><p>${escapeHtml(l.why)} ${escapeHtml(l.practice)}</p></div>`).join("");
  const outcomeWord = { win: "You won", loss: "You lost", draw: "Draw", unfinished: "Unfinished" }[r.outcome];
  $("review-body").innerHTML = `
    <div class="stat-row">
      <div class="stat"><div class="label">Result</div><div class="value ${r.outcome}">${outcomeWord}</div></div>
      <div class="stat"><div class="label">Avg loss / move</div><div class="value">${acpl.all === null || acpl.all === undefined ? "—" : acpl.all.toFixed(0)}</div></div>
      <div class="stat"><div class="label">Blunders</div><div class="value">${r.counts.blunder}</div></div>
      <div class="stat"><div class="label">Mistakes</div><div class="value">${r.counts.mistake}</div></div>
    </div>
    <p class="hint">${r.opening ? `Opening: ${escapeHtml(r.opening.name)} (${r.opening.eco})` : "Opening not in the lichess list."}
      Loss is in centipawns (100 = one pawn); lower is better. Book moves (grey in the move list) count as zero.</p>
    ${bars ? `<p class="section-title">Where you lost ground</p><div class="bars">${bars}</div>` : ""}
    <p class="section-title">Lessons from this game</p>
    ${lessons || '<p class="hint">No principle broken this game. Nice.</p>'}
    <p class="section-title">Key moments</p>
    ${moments || '<p class="hint">No move of yours cost more than a pawn and a half.</p>'}`;
  document.querySelectorAll(".moment").forEach((el) => el.addEventListener("click", () => {
    viewPly = Number(el.dataset.ply) - 1;
    render();
    setStatus(`Before your move ${el.querySelector("h4").textContent.split(" (")[0].replace("Move ", "")}. Try to find the better move.`);
  }));
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// The server stores UTC; show the day it was where you played.
function localDate(utc) {
  const d = new Date(String(utc).slice(0, 19).replace(" ", "T") + "Z");
  return Number.isNaN(d.getTime()) ? String(utc).slice(0, 10) : d.toLocaleDateString();
}

// ---------------------------------------------------------------- tabs and other panels
function showTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  ["play", "learn", "review", "games", "profile", "tips"].forEach((p) => { $(`panel-${p}`).hidden = p !== name; });
  if (name === "learn" && !lesson) renderLearnIndex();
  if (name === "games") loadGames();
  if (name === "profile") loadProfile();
  if (name === "tips") loadTips();
}
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));

const q = REVIEWER ? `?reviewer_id=${encodeURIComponent(REVIEWER)}` : "";

async function loadGames() {
  const data = await (await fetch(`/api/chess/games${q}`)).json();
  if (!data.games.length) { $("games-body").innerHTML = '<p class="hint">No games yet. Play one!</p>'; return; }
  const rows = data.games.map((g) => `
    <tr class="clickable" data-id="${g.game_id}">
      <td>${localDate(g.played_at)}</td><td>${g.color}</td><td class="${g.outcome}">${g.outcome}</td>
      <td>${escapeHtml(g.opening || "—")}</td><td>${g.acpl === null ? "—" : g.acpl.toFixed(0)}</td>
      <td>${escapeHtml(g.lessons.slice(0, 2).join("; "))}</td></tr>`).join("");
  $("games-body").innerHTML = `<p class="hint">${data.total} game${data.total === 1 ? "" : "s"} played.</p>
    <table><thead><tr><th>Date</th><th>Side</th><th>Result</th><th>Opening</th><th>Loss</th><th>Lessons</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
  document.querySelectorAll("tr[data-id]").forEach((tr) => tr.addEventListener("click", () => openGame(tr.dataset.id)));
}

async function openGame(id) {
  if (state === "analysing") return;        // the finished game is still being reviewed and saved
  if (["playing", "thinking"].includes(state) && history.length && !confirm("Leave the game in progress?")) return;
  const data = await (await fetch(`/api/chess/games/${id}`)).json();
  game = new Chess();
  history = [];
  positions = [game.fen()];
  for (const m of data.moves) {
    const mv = game.move({ from: m.slice(0, 2), to: m.slice(2, 4), promotion: m[4] });
    history.push(mv);
    positions.push(game.fen());
  }
  playerColor = data.review.player_color === "white" ? "w" : "b";
  flipped = playerColor === "b";
  reviewData = { ...data.review, _evals: data.evals };
  state = "review";
  viewPly = 0;
  showTab("review");
  showReview(reviewData);
  render();
  setStatus("Step through the game with ◀ ▶ or the arrow keys.");
}

async function loadProfile() {
  const p = await (await fetch(`/api/chess/profile${q}`)).json();
  if (!p.games) { $("profile-body").innerHTML = `<p class="hint">${escapeHtml(p.summary[0])}</p>`; return; }
  const summary = p.summary.map((s) => `<li>${escapeHtml(s)}</li>`).join("");
  const openings = (p.openings || []).map((o) => `<tr><td>${escapeHtml(o.line)}</td><td>${o.games}</td><td>${Math.round(o.score * 100)}%</td></tr>`).join("");
  const recurring = (p.recurring || []).map((r) => `<tr><td>${escapeHtml(r.title)}</td><td>${r.phase}</td><td>${r.games}</td></tr>`).join("");
  const vs = (p.by_opponent || []).map((o) => `<tr><td>${escapeHtml((STYLE_BY_KEY[o.opponent] || { name: o.opponent }).name)}</td>
    <td>${o.games}</td><td class="win">${o.win}</td><td class="draw">${o.draw}</td><td class="loss">${o.loss}</td></tr>`).join("");
  const maxLoss = Math.max(60, ...["opening", "middlegame", "endgame"].map((x) => p.acpl[x] || 0));
  const bars = ["opening", "middlegame", "endgame"].filter((x) => p.acpl[x] !== null).map((x) => `
    <div class="bar-row"><span>${x}</span><div class="bar-track"><div class="bar-fill" style="width:${(100 * p.acpl[x]) / maxLoss}%;background:${PHASE_COLOURS[x]}"></div></div>
    <span class="num">${p.acpl[x].toFixed(0)}</span></div>`).join("");
  $("profile-body").innerHTML = `
    <div class="stat-row">
      <div class="stat"><div class="label">Games</div><div class="value">${p.games}</div></div>
      <div class="stat"><div class="label">Won</div><div class="value win">${p.record.win}</div></div>
      <div class="stat"><div class="label">Drawn</div><div class="value draw">${p.record.draw}</div></div>
      <div class="stat"><div class="label">Lost</div><div class="value loss">${p.record.loss}</div></div>
    </div>
    <ul class="summary">${summary}</ul>
    ${bars ? `<p class="section-title">Average loss per move, by phase</p><div class="bars">${bars}</div>` : ""}
    ${p.acpl_trend && p.acpl_trend.length > 1 ? `<p class="section-title">Loss per move, game by game</p>${spark(p.acpl_trend)}` : ""}
    ${vs ? `<p class="section-title">Against each opponent</p><table><thead><tr><th>Opponent</th><th>Games</th><th>Won</th><th>Drawn</th><th>Lost</th></tr></thead><tbody>${vs}</tbody></table>` : ""}
    ${openings ? `<p class="section-title">Your openings</p><table><thead><tr><th>Line</th><th>Games</th><th>Score</th></tr></thead><tbody>${openings}</tbody></table>` : ""}
    ${recurring ? `<p class="section-title">Habits to fix</p><table><thead><tr><th>Principle</th><th>Phase</th><th>Games</th></tr></thead><tbody>${recurring}</tbody></table>` : ""}`;
}

function spark(values) {
  const w = 300, h = 56, max = Math.max(...values, 1);
  const pts = values.map((v, i) => `${(i * w) / (values.length - 1)},${h - 4 - ((h - 8) * v) / max}`).join(" ");
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" role="img" aria-label="Loss per move by game">
    <polyline points="${pts}" fill="none" stroke="#3b82f6" stroke-width="2"/></svg>`;
}

async function loadTips() {
  const [tips, profile] = await Promise.all([
    fetch("/api/chess/principles").then((r) => r.json()),
    fetch(`/api/chess/profile${q}`).then((r) => r.json()),
  ]);
  const mine = new Set((profile.recurring || []).map((r) => r.key));
  $("tips-body").innerHTML = ["opening", "middlegame", "endgame"].map((phase) => `
    <div class="tips-phase"><h3>${phase}</h3>
      ${tips[phase].map((t) => `<div class="tip ${mine.has(t.key) ? "yours" : ""}"><h4>${escapeHtml(t.title)}</h4>
        <p>${escapeHtml(t.why)}</p><p>${escapeHtml(t.practice)}</p></div>`).join("")}</div>`).join("");
}

// ---------------------------------------------------------------- Learn: the rules, on the board
function lessonsDone() {
  try { return new Set(JSON.parse(localStorage.getItem("chess-lessons-done")) || []); } catch { return new Set(); }
}
function markLessonDone(key) {
  const done = lessonsDone();
  done.add(key);
  try { localStorage.setItem("chess-lessons-done", JSON.stringify([...done])); } catch { /* storage blocked */ }
}

function renderLearnIndex() {
  $("lesson-view").hidden = true;
  $("lesson-index").hidden = false;
  const done = lessonsDone();
  $("lesson-list").innerHTML = RULE_LESSONS.map((l, i) => `
    <li><button type="button" class="lesson-link" data-i="${i}">${escapeHtml(l.title)}${done.has(l.key) ? ' <span class="done">✓</span>' : ""}</button></li>`).join("");
  document.querySelectorAll(".lesson-link").forEach((b) => b.addEventListener("click", () => startLesson(Number(b.dataset.i))));
  $("style-list").innerHTML = STYLES.filter((s) => s.key !== "stockfish").map((s) => `
    <div class="style-card"><h4>${escapeHtml(s.name)}</h4><p>${escapeHtml(s.how)}</p>
      <ul>${s.beat.map((b) => `<li>${escapeHtml(b)}</li>`).join("")}</ul>
      <button type="button" class="btn btn-primary style-play" data-key="${s.key}">Play ${escapeHtml(s.name)}</button></div>`).join("");
  document.querySelectorAll(".style-play").forEach((b) => b.addEventListener("click", () => {
    $("opponent").value = b.dataset.key;
    $("opponent").dispatchEvent(new Event("change"));
    showTab("play");
    setStatus("Pick your side and level, then press Start game.");
  }));
}

function startLesson(i) {
  if (state === "analysing" || state === "thinking") return;
  if (state === "playing" && history.length && !confirm("Leave the game in progress?")) return;
  const L = RULE_LESSONS[i];
  lesson = { index: i, done: !L.goal };
  game = new Chess(L.fen);
  history = [];
  positions = [game.fen()];
  reviewData = null;
  viewPly = null;
  selected = null;
  flipped = false;
  marks = { danger: [], hint: [] };
  state = "lesson";
  $("btn-undo").disabled = true;
  $("btn-resign").disabled = true;
  $("lesson-index").hidden = true;
  $("lesson-view").hidden = false;
  $("lesson-kicker").textContent = `Lesson ${i + 1} of ${RULE_LESSONS.length}`;
  $("lesson-title").textContent = L.title;
  $("lesson-text").innerHTML = L.text.map((t) => `<p>${escapeHtml(t)}</p>`).join("");
  $("lesson-feedback").textContent = "";
  $("lesson-feedback").className = "lesson-feedback";
  $("lesson-retry").hidden = !L.goal;
  $("lesson-next").hidden = i === RULE_LESSONS.length - 1;
  if (!L.goal) markLessonDone(L.key);
  setStatus(L.goal ? "Your move." : "Read, then go on to the next lesson.");
  render();
}

function lessonTry(move) {
  const L = RULE_LESSONS[lesson.index];
  const done = game.move(move);
  if (!done) return;
  history.push(done);
  positions.push(game.fen());
  selected = null;
  lesson.done = true;
  const goal = L.goal;
  const ok = goal.type === "moves" ? goal.accept.includes(uci(done))
    : goal.type === "mate" ? game.isCheckmate()
    : goal.type === "notStalemate" ? !game.isStalemate()
    : true;
  $("lesson-feedback").textContent = ok ? L.success || "Right." : goal.otherwise || "Not quite. Try again.";
  $("lesson-feedback").className = `lesson-feedback ${ok ? "ok" : "bad"}`;
  if (ok) markLessonDone(L.key);
  setStatus(ok ? "Solved." : "Press Try again.");
  render();
}

$("lesson-retry").addEventListener("click", () => startLesson(lesson ? lesson.index : 0));
$("lesson-next").addEventListener("click", () => startLesson(Math.min((lesson ? lesson.index : -1) + 1, RULE_LESSONS.length - 1)));
$("lesson-list-btn").addEventListener("click", () => { lesson = null; state = "idle"; game = new Chess(); history = []; positions = [game.fen()]; render(); renderLearnIndex(); setStatus("Pick a lesson, or go to Play."); });

// ---------------------------------------------------------------- opponent and coach settings
$("opponent").innerHTML = STYLES.map((s) => `<option value="${s.key}">${escapeHtml(s.name)}</option>`).join("");
function opponentHint() { $("opponent-hint").textContent = (STYLE_BY_KEY[$("opponent").value] || STYLES[0]).how; }
$("opponent").addEventListener("change", opponentHint);
opponentHint();
$("coach-mode").value = coachMode;
$("coach-mode").addEventListener("change", () => {
  coachMode = $("coach-mode").value;
  try { localStorage.setItem("chess-coach", coachMode); } catch { /* storage blocked */ }
  if (coachMode === "off") { marks = { danger: [], hint: [] }; render(); }
});

// ---------------------------------------------------------------- boot
render();
const countUrl = new URL("/api/chess/games", location.origin);
countUrl.searchParams.set("limit", "1");
if (REVIEWER) countUrl.searchParams.set("reviewer_id", REVIEWER);
fetch(countUrl)
  .then((r) => r.json())
  .then((d) => { $("games-count").textContent = d.total ? `${d.total} game${d.total === 1 ? "" : "s"} played so far.` : ""; })
  .catch(() => {});
