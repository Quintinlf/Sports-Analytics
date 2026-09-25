/* =========================================================
   Poker Laboratory — table client

   Deliberately dependency-free vanilla JS, matching the existing
   frontend/feedback surface. The server owns all game state and all
   arithmetic; this file only renders it and submits hero actions.
   ========================================================= */

const API = "/api/poker";
const STORAGE_KEY = "poker_session_id";

const SUIT_GLYPH = { s: "♠", h: "♥", d: "♦", c: "♣" };
const RED_SUITS = new Set(["h", "d"]);

let state = null;         // latest session payload from the server
let opponents = [];
let modes = [];
let mathRevealed = false; // per-decision reveal in intermediate mode

// ── Helpers ────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);

function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); }

async function api(path, options = {}) {
  const response = await fetch(API + path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let payload = null;
  try { payload = await response.json(); } catch (_) { /* no body */ }
  if (!response.ok) {
    const detail = (payload && payload.detail) || `Request failed (${response.status})`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload;
}

function cardEl(code, { small = false, faceDown = false, highlight = false } = {}) {
  const el = document.createElement("div");
  el.className = "playing-card" + (small ? " small" : "");
  if (faceDown) { el.classList.add("back"); return el; }
  if (!code) { el.classList.add("placeholder"); return el; }

  const rank = code.slice(0, -1);
  const suit = code.slice(-1);
  if (RED_SUITS.has(suit)) el.classList.add("red");
  if (highlight) el.classList.add("winning");

  const rankEl = document.createElement("span");
  rankEl.className = "rank";
  rankEl.textContent = rank === "T" ? "10" : rank;

  const suitEl = document.createElement("span");
  suitEl.className = "suit";
  suitEl.textContent = SUIT_GLYPH[suit] || suit;

  el.append(rankEl, suitEl);
  return el;
}

function renderMath(el, tex) {
  if (!tex) { el.textContent = ""; return; }
  if (window.katex) {
    try {
      window.katex.render(tex, el, { throwOnError: false, displayMode: true });
      return;
    } catch (_) { /* fall through to plain text */ }
  }
  el.textContent = tex;
}

function signed(value) { return value > 0 ? `+${value}` : `${value}`; }

// ── Setup screen ───────────────────────────────────────────

async function loadSetupOptions() {
  const [opponentData, modeData] = await Promise.all([
    api("/opponents"),
    api("/learning-modes"),
  ]);
  opponents = opponentData.opponents;
  modes = modeData.modes;

  const opponentSelect = $("opponent-select");
  clear(opponentSelect);
  opponents.forEach((profile) => {
    const option = document.createElement("option");
    option.value = profile.key;
    option.textContent = profile.name;
    opponentSelect.appendChild(option);
  });
  opponentSelect.addEventListener("change", updateOpponentHint);
  updateOpponentHint();

  [$("mode-select"), $("mode-live-select")].forEach((select) => {
    clear(select);
    modes.forEach((mode) => {
      const option = document.createElement("option");
      option.value = mode.key;
      option.textContent = mode.key.charAt(0).toUpperCase() + mode.key.slice(1);
      select.appendChild(option);
    });
  });
  $("mode-select").addEventListener("change", updateModeHint);
  updateModeHint();
}

function updateOpponentHint() {
  const profile = opponents.find((p) => p.key === $("opponent-select").value);
  $("opponent-hint").textContent = profile ? profile.description : "";
}

function updateModeHint() {
  const mode = modes.find((m) => m.key === $("mode-select").value);
  $("mode-hint").textContent = mode ? mode.description : "";
}

async function sitDown() {
  $("setup-error").textContent = "";
  $("sit-btn").disabled = true;
  try {
    const payload = await api("/sessions", {
      method: "POST",
      body: JSON.stringify({
        opponent_profile: $("opponent-select").value,
        small_blind: Number($("sb-input").value),
        big_blind: Number($("bb-input").value),
        starting_stack: Number($("stack-input").value),
        learning_mode: $("mode-select").value,
      }),
    });
    localStorage.setItem(STORAGE_KEY, payload.session_id);
    showTable(payload);
  } catch (error) {
    $("setup-error").textContent = error.message;
  } finally {
    $("sit-btn").disabled = false;
  }
}

function showTable(payload) {
  state = payload;
  mathRevealed = false;
  $("setup-screen").hidden = true;
  $("table-screen").hidden = false;
  $("mode-live-select").value = payload.learning_mode;
  render();
}

// ── Rendering ──────────────────────────────────────────────

function render() {
  if (!state || !state.hand) return;
  const hand = state.hand;
  const hero = hand.players.find((p) => p.is_hero);
  const villain = hand.players.find((p) => !p.is_hero);
  const heroIndex = hand.players.indexOf(hero);
  const villainIndex = hand.players.indexOf(villain);
  const complete = hand.is_complete;

  // Header
  $("stat-hands").textContent = state.hands_played;
  const netEl = $("stat-net");
  netEl.textContent = signed(state.hero_net_total);
  netEl.className = "hstat-value " + (state.hero_net_total >= 0 ? "pos" : "neg");

  // Seats
  renderSeat("hero", hero, heroIndex, hand, complete);
  renderSeat("villain", villain, villainIndex, hand, complete);

  // Board
  const boardEl = $("board");
  clear(boardEl);
  const winningCards = collectWinningCards(hand);
  for (let i = 0; i < 5; i += 1) {
    const code = hand.board[i];
    boardEl.appendChild(
      code
        ? cardEl(code, { highlight: complete && winningCards.has(code) })
        : cardEl(null)
    );
  }

  $("pot-value").textContent = hand.pot_total;
  $("street-label").textContent = prettyStreet(hand.street);
  $("made-hand").textContent =
    state.decision && state.decision.made_hand ? state.decision.made_hand : "";

  renderBanner(hand, complete);
  renderActions(hand, complete);
  renderMathPanel(complete);
  renderLog(hand);
  renderSession();
}

function renderSeat(which, player, index, hand, complete) {
  const seat = $(`${which}-seat`);
  seat.classList.toggle("active", !complete && hand.to_act_index === index);
  seat.classList.toggle("folded", player.has_folded);

  $(`${which}-name`).textContent = player.name;
  $(`${which}-stack`).textContent = player.stack;
  $(`${which}-pos`).textContent = player.position;
  $(`${which}-dealer`).hidden = hand.button_index !== index;

  const betEl = $(`${which}-bet`);
  betEl.hidden = player.committed_street <= 0;
  betEl.textContent = player.committed_street;

  const cardsEl = $(`${which}-cards`);
  clear(cardsEl);
  const winningCards = collectWinningCards(hand);
  if (player.hole_cards) {
    player.hole_cards.forEach((code) =>
      cardsEl.appendChild(
        cardEl(code, { highlight: complete && winningCards.has(code) })
      )
    );
  } else if (!player.has_folded) {
    cardsEl.appendChild(cardEl(null, { faceDown: true }));
    cardsEl.appendChild(cardEl(null, { faceDown: true }));
  }
}

/* Cards that make the winning hand, so a showdown highlights what won. */
function collectWinningCards(hand) {
  const winners = new Set();
  const result = hand.result;
  if (!result || !result.went_to_showdown) return winners;
  result.winners.forEach((pid) => {
    const entry = result.showdown[pid];
    if (entry) entry.cards.forEach((code) => winners.add(code));
  });
  return winners;
}

function prettyStreet(street) {
  return street.charAt(0) + street.slice(1).toLowerCase();
}

function renderBanner(hand, complete) {
  const banner = $("hand-banner");
  if (!complete || !hand.result) { banner.hidden = true; return; }

  clear(banner);
  const summary = document.createElement("div");
  summary.textContent = hand.result.summary;

  const net = hand.result.net.hero || 0;
  const netEl = document.createElement("span");
  netEl.className = "net " + (net >= 0 ? "pos" : "neg");
  netEl.textContent = `${signed(net)} chips`;

  banner.append(summary, netEl);
  banner.hidden = false;
}

// ── Action bar ─────────────────────────────────────────────

function renderActions(hand, complete) {
  const container = $("action-buttons");
  const raisePanel = $("raise-panel");
  const nextBtn = $("next-hand-btn");
  clear(container);
  raisePanel.hidden = true;
  $("action-error").textContent = "";

  if (complete) {
    container.hidden = true;
    nextBtn.hidden = false;
    return;
  }
  container.hidden = false;
  nextBtn.hidden = true;

  if (!state.awaiting_hero) {
    const waiting = document.createElement("p");
    waiting.className = "locked-note";
    waiting.textContent = "Opponent is acting…";
    container.appendChild(waiting);
    return;
  }

  const legal = hand.legal_actions;
  if (!legal) return;

  if (legal.can_check) {
    container.appendChild(actionButton("Check", "btn-check", () => submit("check")));
  }
  if (legal.can_call) {
    container.appendChild(
      actionButton(`Call ${legal.call_amount}`, "btn-call", () => submit("call"))
    );
  }
  if (legal.can_bet || legal.can_raise) {
    const label = legal.can_bet ? "Bet" : "Raise";
    container.appendChild(
      actionButton(label, "btn-raise", () => openRaisePanel(legal, label))
    );
  }
  if (legal.can_fold) {
    // Folding when checking is free is legal but never correct, so it is
    // de-emphasised rather than presented as an equal option.
    const cls = legal.can_check ? "btn-secondary-action" : "btn-fold";
    container.appendChild(actionButton("Fold", cls, () => submit("fold")));
  }
}

function actionButton(label, className, handler) {
  const button = document.createElement("button");
  button.className = `btn ${className}`;
  button.textContent = label;
  button.addEventListener("click", handler);
  return button;
}

function openRaisePanel(legal, label) {
  const panel = $("raise-panel");
  const slider = $("raise-slider");
  const amount = $("raise-amount");

  slider.min = legal.min_raise_to;
  slider.max = legal.max_raise_to;
  slider.value = legal.min_raise_to;
  amount.min = legal.min_raise_to;
  amount.max = legal.max_raise_to;
  amount.value = legal.min_raise_to;

  slider.oninput = () => { amount.value = slider.value; };
  amount.oninput = () => { slider.value = amount.value; };

  const presets = $("sizing-presets");
  clear(presets);
  const options = (state.decision && state.decision.bet_options) || [];
  options.forEach((option) => {
    const button = document.createElement("button");
    button.className = "btn";
    button.textContent = `${option.label} (${option.amount})`;
    button.addEventListener("click", () => {
      slider.value = option.amount;
      amount.value = option.amount;
    });
    presets.appendChild(button);
  });

  $("raise-confirm-btn").textContent = label;
  $("raise-confirm-btn").onclick = () =>
    submit(legal.can_bet ? "bet" : "raise", Number(amount.value));
  $("raise-cancel").onclick = () => { panel.hidden = true; };

  panel.hidden = false;
}

async function submit(action, amount = 0) {
  $("action-error").textContent = "";
  try {
    state = await api(`/sessions/${state.session_id}/actions`, {
      method: "POST",
      body: JSON.stringify({ action, amount }),
    });
    mathRevealed = false;
    render();
  } catch (error) {
    $("action-error").textContent = error.message;
  }
}

async function nextHand() {
  try {
    state = await api(`/sessions/${state.session_id}/hands`, { method: "POST" });
    mathRevealed = false;
    render();
  } catch (error) {
    $("action-error").textContent = error.message;
  }
}

// ── Maths panel ────────────────────────────────────────────

/* Learning mode decides whether the analysis is visible *before* the
   decision. The server always computes it; gating is presentational. */
function mathVisible(complete) {
  switch (state.learning_mode) {
    case "beginner":     return true;
    case "intermediate": return mathRevealed || complete;
    case "challenge":    return complete;
    case "analysis":     return complete;
    default:             return true;
  }
}

function renderMathPanel(complete) {
  const body = $("math-body");
  const revealBtn = $("reveal-btn");
  clear(body);

  const decision = state.decision;
  const visible = mathVisible(complete);

  revealBtn.hidden = !(state.learning_mode === "intermediate" && decision && !visible);
  revealBtn.onclick = () => { mathRevealed = true; render(); };

  if (!decision) {
    const note = document.createElement("p");
    note.className = "locked-note";
    note.textContent = complete
      ? "Hand complete. Deal the next hand to see a new decision."
      : "Waiting for your turn…";
    body.appendChild(note);
    return;
  }

  if (!visible) {
    const note = document.createElement("p");
    note.className = "locked-note";
    note.textContent =
      state.learning_mode === "challenge"
        ? "Challenge mode — decide on your own. The maths is revealed after the hand."
        : "Hidden while you decide. Reveal it, or let the hand finish.";
    body.appendChild(note);
    return;
  }

  metric(body, "Street", prettyStreet(decision.street));
  metric(body, "Position", decision.position);
  metric(body, "Pot", decision.pot);
  if (decision.to_call > 0) {
    metric(body, "To call", decision.to_call);
    metric(body, "Pot odds — equity needed", `${decision.pot_odds_pct}%`, true);
  }
  if (decision.made_hand) metric(body, "Your hand", decision.made_hand);
  metric(body, "Your stack", decision.stack);
  if (decision.stack_to_pot_ratio !== null) {
    metric(body, "Stack-to-pot", `${decision.stack_to_pot_ratio}×`);
  }

  decision.explanations.forEach((note) => {
    const box = document.createElement("div");
    box.className = "explain";

    const title = document.createElement("div");
    title.className = "explain-title";
    title.textContent = note.title;
    box.appendChild(title);

    if (note.formula) {
      const formula = document.createElement("div");
      formula.className = "explain-math";
      renderMath(formula, note.formula);
      box.appendChild(formula);
    }
    if (note.substitution) {
      const sub = document.createElement("div");
      sub.className = "explain-math";
      renderMath(sub, note.substitution);
      box.appendChild(sub);
    }

    const text = document.createElement("div");
    text.className = "explain-body";
    text.textContent = note.body;
    box.appendChild(text);

    body.appendChild(box);
  });

  if (decision.pending && decision.pending.length) {
    const list = document.createElement("div");
    list.className = "pending-list";
    const heading = document.createElement("p");
    heading.className = "section-title";
    heading.textContent = "Not yet available";
    list.appendChild(heading);

    decision.pending.forEach((item) => {
      const row = document.createElement("div");
      row.className = "pending-item";
      const label = document.createElement("span");
      label.textContent = item.label;
      const ms = document.createElement("span");
      ms.className = "ms";
      ms.textContent = item.milestone;
      row.append(label, ms);
      list.appendChild(row);
    });
    body.appendChild(list);
  }
}

function metric(parent, label, value, highlight = false) {
  const row = document.createElement("div");
  row.className = "metric-row";
  const labelEl = document.createElement("span");
  labelEl.className = "metric-label";
  labelEl.textContent = label;
  const valueEl = document.createElement("span");
  valueEl.className = "metric-value" + (highlight ? " highlight" : "");
  valueEl.textContent = value;
  row.append(labelEl, valueEl);
  parent.appendChild(row);
}

// ── Log & session ──────────────────────────────────────────

function renderLog(hand) {
  const log = $("hand-log");
  clear(log);
  const names = {};
  hand.players.forEach((p) => { names[p.player_id] = p.name; });

  hand.action_log.forEach((record) => {
    const entry = document.createElement("div");
    entry.className = "log-entry";

    const street = document.createElement("span");
    street.className = "log-street";
    street.textContent = prettyStreet(record.street);

    const text = document.createElement("span");
    text.className = "log-text";
    const who = document.createElement("span");
    who.className = "who";
    who.textContent = names[record.player_id] || record.player_id;
    text.appendChild(who);

    let verb = ` ${record.action.replace("_", " ")}`;
    if (record.action === "bet" || record.action === "raise") {
      verb += " to ";
    } else if (record.chips_committed > 0) {
      verb += " ";
    }
    text.appendChild(document.createTextNode(verb));

    if (record.action === "bet" || record.action === "raise") {
      const amount = document.createElement("span");
      amount.className = "amt";
      amount.textContent = record.amount;
      text.appendChild(amount);
    } else if (record.chips_committed > 0) {
      const amount = document.createElement("span");
      amount.className = "amt";
      amount.textContent = record.chips_committed;
      text.appendChild(amount);
    }
    if (record.is_all_in) text.appendChild(document.createTextNode(" (all-in)"));

    entry.append(street, text);
    log.appendChild(entry);
  });
  log.scrollTop = log.scrollHeight;
}

function renderSession() {
  const grid = $("session-grid");
  clear(grid);
  const cells = [
    ["Hands", state.hands_played, null],
    ["Net", signed(state.hero_net_total), state.hero_net_total >= 0 ? "pos" : "neg"],
    ["Blinds", `${state.config.small_blind}/${state.config.big_blind}`, null],
    ["Rebuys", state.rebuys, null],
  ];
  cells.forEach(([key, value, cls]) => {
    const cell = document.createElement("div");
    cell.className = "session-cell";
    const k = document.createElement("div");
    k.className = "k";
    k.textContent = key;
    const v = document.createElement("div");
    v.className = "v" + (cls ? ` ${cls}` : "");
    v.textContent = value;
    cell.append(k, v);
    grid.appendChild(cell);
  });
}

// ── Wiring ─────────────────────────────────────────────────

async function resumeSession() {
  const sessionId = localStorage.getItem(STORAGE_KEY);
  if (!sessionId) return false;
  try {
    const payload = await api(`/sessions/${sessionId}`);
    showTable(payload);
    return true;
  } catch (_) {
    localStorage.removeItem(STORAGE_KEY);
    return false;
  }
}

async function init() {
  await loadSetupOptions();

  $("sit-btn").addEventListener("click", sitDown);
  $("next-hand-btn").addEventListener("click", nextHand);
  $("leave-btn").addEventListener("click", () => {
    localStorage.removeItem(STORAGE_KEY);
    state = null;
    $("table-screen").hidden = true;
    $("setup-screen").hidden = false;
  });
  $("mode-live-select").addEventListener("change", async (event) => {
    state = await api(`/sessions/${state.session_id}/learning-mode`, {
      method: "POST",
      body: JSON.stringify({ learning_mode: event.target.value }),
    });
    mathRevealed = false;
    updateLiveModeHint();
    render();
  });

  await resumeSession();
  updateLiveModeHint();
}

function updateLiveModeHint() {
  const mode = modes.find((m) => m.key === $("mode-live-select").value);
  $("mode-live-hint").textContent = mode ? mode.description : "";
}

document.addEventListener("DOMContentLoaded", init);
