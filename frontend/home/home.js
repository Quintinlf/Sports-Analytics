/* =========================================================
   Home — section links, opponent quick-picks, You vs the AI.
   Dependency-free, like the other surfaces.
   ========================================================= */

const $ = (id) => document.getElementById(id);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} → ${response.status}`);
  return response.json();
}

// ── Poker quick-picks ──────────────────────────────────────

async function loadOpponents() {
  const box = $("opponent-chips");
  try {
    const { opponents } = await getJson("/api/poker/opponents");
    opponents.forEach((profile) => {
      const chip = el("a", "chip", profile.name);
      chip.href = `/poker?play=1&opponent=${encodeURIComponent(profile.key)}`;
      chip.title = profile.description || "";
      box.appendChild(chip);
    });
  } catch (_) {
    box.appendChild(el("span", "muted", "Opponent list unavailable."));
  }
}

// ── You vs the AI ──────────────────────────────────────────

function stat(label, value) {
  const box = el("div", "stat");
  box.appendChild(el("span", "stat-label", label));
  box.appendChild(el("span", "stat-value", value));
  return box;
}

function gameRow(game, tag) {
  const li = el("li");
  const left = el("span", "matchup", game.matchup);
  const right = el("span", "pick", game.your_pick);
  right.appendChild(tag);
  li.appendChild(left);
  li.appendChild(right);
  li.title = `${game.game_date} · model: ${game.model_pick}`;
  return li;
}

function fillList(list, games, tagFor, emptyText) {
  list.replaceChildren();
  if (!games.length) {
    list.appendChild(el("li", "empty", emptyText));
    return;
  }
  games.forEach((game) => list.appendChild(gameRow(game, tagFor(game))));
}

function pValueText(p) {
  if (p === null || p === undefined) return "–";
  return p < 0.001 ? "< 0.001" : p.toFixed(3);
}

async function loadMirror() {
  let board;
  try {
    board = await getJson("/api/mirror/scoreboard");
  } catch (error) {
    $("mirror-verdict").textContent = "Scoreboard unavailable right now.";
    $("mirror-explain").textContent = String(error.message || error);
    return;
  }

  const owner = board.owner || {};
  if (owner.found) {
    $("mirror-you-label").textContent = owner.name || "You";
    const pickUrl = `/feedback?reviewer_id=${encodeURIComponent(owner.reviewer_id)}`;
    $("mirror-pick-link").href = pickUrl;
    $("predictions-link").href = pickUrl;
  }

  $("mirror-verdict").textContent = board.settled_games
    ? board.verdict
    : "No settled games yet. The first result starts the count.";
  $("mirror-you").textContent = board.you_correct;
  $("mirror-ai").textContent = board.ai_correct;

  const o = board.overrides;
  const stats = $("mirror-stats");
  stats.replaceChildren(
    stat("Settled games", String(board.settled_games)),
    stat("Rode the model", String(board.picks.auto + board.picks.agreed)),
    stat("Overrides", String(board.picks.overrode)),
    stat("Overrides W-L", `${o.won}-${o.lost}`),
    stat("Could be luck? (p)", pValueText(o.p_value)),
  );
  if (board.picks.late_not_scored) {
    stats.appendChild(stat("Late picks (not scored)", String(board.picks.late_not_scored)));
  }
  if (owner.same_name_accounts) {
    const others = owner.same_name_accounts;
    $("mirror-explain").textContent +=
      ` ${others} other account${others === 1 ? " is" : "s are"} also named ${owner.name};` +
      " picks made from those don't count here.";
  }

  fillList($("mirror-upcoming"), board.upcoming, (game) =>
    game.source === "auto" ? el("span", "tag auto", "= model") : el("span", "tag you", "yours"),
    "No upcoming games.");
  fillList($("mirror-recent"), board.recent_overrides, (game) => {
    if (game.you_correct && !game.ai_correct) return el("span", "tag won", "beat AI");
    if (game.ai_correct && !game.you_correct) return el("span", "tag lost", "AI right");
    return el("span", "tag neither", "both wrong");
  }, "No overrides yet. So far it's the AI against itself.");
}

// ── Everyone vs the AI ─────────────────────────────────────

function rowOf(text, span) {
  const tr = el("tr");
  const td = el("td", "muted", text);
  td.colSpan = span;
  tr.appendChild(td);
  return tr;
}

async function loadLeaderboard() {
  const body = $("leaderboard");
  let data;
  try { data = await getJson("/api/mirror/leaderboard"); }
  catch (error) { body.replaceChildren(rowOf(`Leaderboard unavailable: ${error.message}`, 6)); return; }
  body.replaceChildren();
  if (!data.leaderboard.length) { body.appendChild(rowOf("No graded picks yet.", 6)); return; }
  const pct = (x) => (x === null ? "–" : `${Math.round(x * 100)}%`);
  data.leaderboard.forEach((r) => {
    const tr = el("tr", r.is_owner ? "owner" : "");
    [r.is_owner ? `${r.name} (you)` : r.name, String(r.picks), `${r.correct} · ${pct(r.accuracy)}`,
      `${r.model_correct} · ${pct(r.model_accuracy)}`, `${r.won}-${r.lost}`]
      .forEach((t) => tr.appendChild(el("td", "", t)));
    tr.appendChild(el("td", r.lead > 0 ? "pos" : r.lead < 0 ? "neg" : "", r.lead > 0 ? `+${r.lead}` : String(r.lead)));
    body.appendChild(tr);
  });
}

// ── Powerball ──────────────────────────────────────────────

function ballsInto(box, ticket) {
  box.replaceChildren();
  if (!ticket) { box.appendChild(el("span", "muted", "–")); return; }
  const drawn = new Set((ticket.drawn_whites || "").split(" ").filter(Boolean).map(Number));
  ticket.whites.split(" ").forEach((n) => box.appendChild(el("span", `ball${drawn.has(Number(n)) ? " hit" : ""}`, n)));
  const hit = ticket.drawn_special !== null && ticket.drawn_special === ticket.special;
  box.appendChild(el("span", `ball red${hit ? " hit" : ""}`, String(ticket.special).padStart(2, "0")));
}

let pbOwner = null;
const money = (x) => `$${Number(x).toLocaleString()}`;

async function loadPowerball() {
  let data;
  try { data = await getJson("/api/lottery/powerball"); }
  catch (error) { $("pb-next").textContent = "Powerball unavailable right now."; return; }
  pbOwner = data.owner.reviewer_id;
  const when = new Date(data.draw_time);
  $("pb-next").textContent = "Next draw " + when.toLocaleString(undefined,
    { weekday: "long", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
  $("pb-note").textContent = data.note +
    (data.status.fetch_error ? " The draw results could not be reached just now; grading will catch up." : "");
  ballsInto($("pb-model"), data.model_next);
  ballsInto($("pb-you"), data.your_next);
  $("pb-you-label").textContent = data.your_next && data.your_next.source === "manual"
    ? "Your ticket (your numbers)" : "Your ticket (= the model's)";

  const L = data.ledger;
  $("pb-ledger").replaceChildren(
    stat("Draws graded", String(L.you.tickets)),
    stat("You: spent / won", `${money(L.you.spent)} / ${money(L.you.won)}`),
    stat("Model: spent / won", `${money(L.model.spent)} / ${money(L.model.won)}`),
  );
  const hist = $("pb-history");
  hist.replaceChildren();
  data.history.slice(0, 8).forEach((r) => {
    const li = el("li");
    const ticket = el("span", "matchup", `${r.draw_date}  ${r.whites} [${String(r.special).padStart(2, "0")}]`);
    const result = r.graded
      ? `${r.matched_white}${r.matched_special ? " + PB" : ""} · ${r.jackpot ? "JACKPOT" : money(r.prize || 0)}`
      : "awaiting draw";
    const right = el("span", "pick", result);
    right.appendChild(el("span", `tag ${r.source === "manual" ? "you" : "auto"}`, r.source === "manual" ? "yours" : "= model"));
    li.append(ticket, right);
    hist.appendChild(li);
  });
  if (!data.history.length) hist.appendChild(el("li", "empty", "No past draws yet. The first graded draw appears here."));
}

function wirePowerballForm() {
  $("pb-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const whites = [1, 2, 3, 4, 5].map((i) => Number($(`pb-w${i}`).value));
    const special = Number($("pb-pb").value);
    const msg = $("pb-msg");
    try {
      const response = await fetch("/api/lottery/powerball/picks", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ whites, special, reviewer_id: pbOwner }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || response.status);
      msg.textContent = `Saved for ${payload.draw_date}. It locks when the balls are drawn.`;
      loadPowerball();
    } catch (error) { msg.textContent = `Not saved: ${error.message}`; }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  loadOpponents();
  loadMirror();
  loadLeaderboard();
  loadPowerball();
  wirePowerballForm();
});
