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

document.addEventListener("DOMContentLoaded", () => {
  loadOpponents();
  loadMirror();
});
