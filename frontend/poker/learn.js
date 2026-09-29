/* Learn to play: the lessons' small interactive parts. The server deals the practice hands and
   judges them with the same evaluator the table uses (/api/poker/learn/*). */

const $ = (id) => document.getElementById(id);
const SUIT_GLYPH = { s: "♠", h: "♥", d: "♦", c: "♣" };

function card(code, { small = false, highlight = false } = {}) {
  const el = document.createElement("div");
  el.className = "playing-card" + (small ? " small" : "");
  if (!code) { el.classList.add("placeholder"); return el; }
  const rank = code.slice(0, -1);
  const suit = code.slice(-1);
  if (suit === "h" || suit === "d") el.classList.add("red");
  if (highlight) el.classList.add("winning");
  const r = document.createElement("span");
  r.className = "rank";
  r.textContent = rank === "T" ? "10" : rank;
  const s = document.createElement("span");
  s.className = "suit";
  s.textContent = SUIT_GLYPH[suit];
  el.append(r, s);
  return el;
}

function cards(codes, options = {}) {
  const row = document.createElement("div");
  row.className = "card-row";
  codes.forEach((c) => row.appendChild(card(c, { small: true, ...options, highlight: (options.best || []).includes(c) })));
  return row;
}

// ── Table of contents ──────────────────────────────────────
document.querySelectorAll(".lesson").forEach((lesson) => {
  const a = document.createElement("a");
  a.href = `#${lesson.id}`;
  a.textContent = lesson.querySelector("h2").textContent;
  $("toc").appendChild(a);
});

// ── Lesson 2: a dealt example ──────────────────────────────
(function dealDemo() {
  const demo = $("deal-demo");
  const parts = [
    ["Your two cards", ["Ah", "Jh"]],
    ["The flop", ["Kh", "7c", "2h"]],
    ["The turn", ["Qs"]],
    ["The river", ["9h"]],
  ];
  parts.forEach(([label, codes]) => {
    const block = document.createElement("div");
    block.className = "deal-part";
    const p = document.createElement("p");
    p.className = "section-title";
    p.textContent = label;
    block.append(p, cards(codes));
    demo.appendChild(block);
  });
  const note = document.createElement("p");
  note.className = "hint";
  note.textContent = "Best five here: A♥ K♥ J♥ 9♥ 2♥, a flush (five hearts), using both of your cards.";
  demo.appendChild(note);
})();

// ── Lesson 3: the ranking, with an example of each ─────────
const RANKS = [
  ["Royal flush", "A, K, Q, J, 10, all one suit", ["As", "Ks", "Qs", "Js", "Ts"]],
  ["Straight flush", "Five in a row, all one suit", ["9h", "8h", "7h", "6h", "5h"]],
  ["Four of a kind", "Four cards of one rank", ["Qc", "Qd", "Qh", "Qs", "4d"]],
  ["Full house", "Three of one rank and two of another", ["8s", "8d", "8c", "Kh", "Kd"]],
  ["Flush", "Any five of one suit", ["Ad", "Jd", "9d", "6d", "3d"]],
  ["Straight", "Five in a row, mixed suits (an ace can be high or low: 5-4-3-2-A counts)", ["Tc", "9d", "8s", "7h", "6c"]],
  ["Three of a kind", "Three cards of one rank", ["7c", "7d", "7s", "Kd", "2h"]],
  ["Two pair", "Two pairs", ["Jc", "Jh", "4s", "4d", "Ac"]],
  ["One pair", "Two cards of one rank", ["Ts", "Th", "Kc", "8d", "3s"]],
  ["High card", "None of the above: your highest card plays", ["Ad", "Qc", "9s", "6h", "2d"]],
];
RANKS.forEach(([name, what, example], i) => {
  const row = document.createElement("div");
  row.className = "rank-row";
  const text = document.createElement("div");
  text.innerHTML = `<strong>${i + 1}. ${name}</strong><span>${what}</span>`;
  row.append(text, cards(example));
  $("rank-list").appendChild(row);
});

// ── Lesson 3: which hand wins? ─────────────────────────────
let quiz = null;
let answered = false;
let score = { right: 0, total: 0 };

async function newQuiz() {
  answered = false;
  $("quiz-feedback").textContent = "";
  $("quiz-prompt").textContent = "Both players share the five table cards. Tap the hand that wins, or \"It's a tie\".";
  const res = await fetch("/api/poker/learn/showdown");
  quiz = await res.json();
  const board = $("quiz-board");
  board.innerHTML = "";
  board.appendChild(cards(quiz.board));
  const hands = $("quiz-hands");
  hands.innerHTML = "";
  quiz.hands.forEach((hand, i) => {
    const button = document.createElement("button");
    button.className = "quiz-hand";
    button.type = "button";
    const label = document.createElement("span");
    label.className = "section-title";
    label.textContent = i === 0 ? "Hand A" : "Hand B";
    button.append(label, cards(hand.cards));
    button.addEventListener("click", () => answer(i));
    hands.appendChild(button);
  });
}

function answer(choice) {
  if (!quiz || answered) return;
  answered = true;
  const right = choice === quiz.winner;
  score.total += 1;
  if (right) score.right += 1;
  const names = quiz.hands.map((h, i) => `${i === 0 ? "A" : "B"} makes ${h.best.description.toLowerCase()}`);
  const verdict = quiz.winner === null ? "It's a tie: the best five cards are equal, so the pot is split."
    : `Hand ${quiz.winner === 0 ? "A" : "B"} wins.`;
  $("quiz-feedback").textContent = `${right ? "Right." : "Not quite."} ${verdict} ${names.join("; ")}.`;
  $("quiz-feedback").className = `quiz-feedback ${right ? "right" : "wrong"}`;
  $("quiz-score").textContent = `${score.right} of ${score.total} right so far.`;
  // Outline the five cards that make each hand.
  const hands = $("quiz-hands").querySelectorAll(".quiz-hand");
  quiz.hands.forEach((hand, i) => {
    hands[i].classList.toggle("won", quiz.winner === i || quiz.winner === null);
    const best = document.createElement("div");
    best.className = "best-five";
    const label = document.createElement("span");
    label.className = "hint";
    label.textContent = "Best five:";
    best.append(label, cards(hand.best.cards));
    hands[i].appendChild(best);
  });
}

$("quiz-next").addEventListener("click", newQuiz);
$("quiz-tie").addEventListener("click", () => answer(null));
newQuiz();

// ── Lesson 7: the price calculator ─────────────────────────
function price() {
  const pot = Math.max(0, Number($("pc-pot").value) || 0);
  const bet = Math.max(0, Number($("pc-bet").value) || 0);
  if (!bet) { $("pc-out").textContent = "Enter their bet."; return; }
  const total = pot + 2 * bet;
  const need = (100 * bet) / total;
  $("pc-out").textContent = `You pay ${bet} to win ${total}: you need to win at least ${need.toFixed(0)}% of the time.`;
}
$("pc-pot").addEventListener("input", price);
$("pc-bet").addEventListener("input", price);
price();

// ── Lesson 9: each opponent ────────────────────────────────
function pct(x) { return `${Math.round(100 * x)}%`; }

fetch("/api/poker/learn/opponents").then((r) => r.json()).then(({ opponents }) => {
  const box = $("opponent-lessons");
  box.innerHTML = "";
  opponents.forEach((o) => {
    const m = o.measured;
    const el = document.createElement("article");
    el.className = "opponent-lesson";
    const plan = (o.plan || []).map((p) => `<li>${p}</li>`).join("");
    el.innerHTML = `
      <h3>${o.name}</h3>
      <p>${o.description}</p>
      <div class="measured">
        <div><span>${pct(m.plays_preflop)}</span>of hands it plays</div>
        <div><span>${pct(m.raises_preflop)}</span>it raises before the flop</div>
        <div><span>${pct(m.folds_to_flop_bet)}</span>folds to a bet on the flop</div>
        <div><span>${pct(m.bluff_share_of_bets)}</span>of its flop bets are weak hands</div>
      </div>
      <p><b>Its leak:</b> ${o.leak || ""}</p>
      <p><b>How to beat it:</b></p>
      <ul>${plan}</ul>
      <a class="btn btn-primary" href="/poker?play=1&opponent=${encodeURIComponent(o.key)}">Play ${o.name} with the coach →</a>`;
    box.appendChild(el);
  });
}).catch(() => { $("opponent-lessons").innerHTML = '<p class="hint">Couldn\'t load the opponents.</p>'; });
