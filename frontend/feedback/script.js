/* AI Sports Analyst Feedback Platform — frontend logic */
"use strict";

const API = "";  // same origin

const SPORT_EMOJI = { MLB: "⚾", NBA: "🏀", FIFA: "⚽", SOCCER: "⚽" };
const SPORT_COLORS = {
  MLB:  "#3b82f6",
  NBA:  "#f59e0b",
  FIFA: "#22c55e",
};

// State
let currentReviewer = null;   // { reviewer_id, name, display_name, ... }
let onboardingQuestions = [];
let currentResearchQuestion = null;
let currentPrediction = null; // full prediction object
let currentReviewId = null;   // after pregame submit
let selectedSport = "ALL";
let lastPregamePick = null;
let lastPregameAgreeWithModel = null;

// ---------------------------------------------------------------------------
// Toast
// ---------------------------------------------------------------------------
function toast(msg, type = "info") {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.className = `show ${type}`;
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.className = ""; }, 3500);
}

// ---------------------------------------------------------------------------
// Stars
// ---------------------------------------------------------------------------
function initStars() {
  const stars = document.querySelectorAll(".star-btn");
  stars.forEach(btn => {
    btn.addEventListener("click", () => {
      const val = +btn.dataset.val;
      document.getElementById("reviewer-confidence").value = val;
      stars.forEach(s => s.className = "star-btn " + (+s.dataset.val <= val ? "lit" : "dim"));
    });
    btn.addEventListener("mouseenter", () => {
      const val = +btn.dataset.val;
      stars.forEach(s => s.className = "star-btn " + (+s.dataset.val <= val ? "lit" : "dim"));
    });
  });
  document.querySelector(".stars-group").addEventListener("mouseleave", () => {
    const val = +document.getElementById("reviewer-confidence").value || 0;
    stars.forEach(s => s.className = "star-btn " + (+s.dataset.val <= val ? "lit" : "dim"));
  });
}

// ---------------------------------------------------------------------------
// Analyst profile & onboarding
// ---------------------------------------------------------------------------
function welcomeLabel(data) {
  return data.display_name || data.name || "Analyst";
}

function applyReviewerProfile(data) {
  document.getElementById("reviewer-display-name").textContent = `Welcome, ${welcomeLabel(data)}`;

  const roleBadge = document.getElementById("analyst-role-badge");
  if (data.analyst_role && data.analyst_role !== "analyst") {
    roleBadge.textContent = data.analyst_role.replace(/_/g, " ");
    roleBadge.style.display = "inline-block";
  } else {
    roleBadge.style.display = "none";
  }
}

async function refreshOnboardingAlert() {
  if (!currentReviewer) return;
  const alertEl = document.getElementById("onboarding-alert");
  const res = await fetch(
    `${API}/api/feedback/onboarding/status?reviewer_id=${encodeURIComponent(currentReviewer.reviewer_id)}`
  );
  if (!res.ok) {
    alertEl.style.display = "none";
    return;
  }
  const status = await res.json();
  if (status.completed) {
    alertEl.style.display = "none";
    return;
  }
  alertEl.style.display = "block";
  alertEl.textContent = `Onboarding: ${status.unanswered_count} question(s) remaining — your reasoning helps train the model.`;
}

async function loadOnboardingQuestions() {
  const res = await fetch(`${API}/api/feedback/onboarding/questions`);
  if (!res.ok) return [];
  return res.json();
}

function renderOnboardingModal(questions) {
  const root = document.getElementById("onboarding-questions");
  root.innerHTML = "";
  for (const q of questions) {
    const block = document.createElement("div");
    block.className = "onboarding-question";
    const prompts = (q.prompts || []).map(p => `<p style="margin-bottom:8px">${p}</p>`).join("");
    block.innerHTML = `
      <h3>${q.title}</h3>
      <div class="body">${q.body_markdown.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")}</div>
      ${prompts}
      <label for="onboard-${q.question_id}">Your answer</label>
      <textarea id="onboard-${q.question_id}" data-question-id="${q.question_id}"></textarea>
    `;
    root.appendChild(block);
  }
}

async function maybeShowOnboarding() {
  if (!currentReviewer) return;
  const statusRes = await fetch(
    `${API}/api/feedback/onboarding/status?reviewer_id=${encodeURIComponent(currentReviewer.reviewer_id)}`
  );
  if (!statusRes.ok) return;
  const status = await statusRes.json();
  if (status.completed) return;

  onboardingQuestions = await loadOnboardingQuestions();
  if (!onboardingQuestions.length) return;

  renderOnboardingModal(onboardingQuestions);
  document.getElementById("onboarding-modal").style.display = "flex";
}

async function submitOnboarding() {
  if (!currentReviewer || !onboardingQuestions.length) return;

  const answers = [];
  for (const q of onboardingQuestions) {
    const el = document.getElementById(`onboard-${q.question_id}`);
    const answer = (el?.value || "").trim();
    if (!answer) {
      toast("Please answer all onboarding questions", "error");
      return;
    }
    answers.push({ question_id: q.question_id, answer });
  }

  const res = await fetch(`${API}/api/feedback/onboarding/answers`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reviewer_id: currentReviewer.reviewer_id, answers }),
  });
  if (!res.ok) {
    toast("Could not save onboarding answers", "error");
    return;
  }

  document.getElementById("onboarding-modal").style.display = "none";
  toast("Onboarding saved — thank you for training the model", "success");
  refreshOnboardingAlert();
}

function applyReviewerSession(data, history, customSections) {
  currentReviewer = {
    reviewer_id: data.reviewer_id,
    name: data.name,
    display_name: data.display_name || data.name,
    analyst_role: data.analyst_role,
    onboarding_completed_at: data.onboarding_completed_at,
  };

  document.getElementById("reviewer-login").style.display = "none";
  document.getElementById("reviewer-panel").style.display = "block";
  document.getElementById("reviewer-history-wrap").style.display = "block";
  document.getElementById("prediction-section").style.display = "block";

  applyReviewerProfile(data);
  renderReviewerStats(data.stats || data);
  renderHistory(history || data.history || []);
  renderCustomSections(customSections || data.custom_sections || []);

  loadPredictions();
  refreshOnboardingAlert();
  maybeShowOnboarding();
  loadResearchQuestion();
  refreshPendingCaseStudies();
}

function renderMarkdownMath(el, text) {
  if (!el || !text) return;
  if (typeof marked !== "undefined") {
    el.innerHTML = marked.parse(text);
  } else {
    el.textContent = text;
  }
  if (typeof renderMathInElement !== "undefined") {
    renderMathInElement(el, { delimiters: [{ left: "$$", right: "$$", display: true }, { left: "$", right: "$", display: false }] });
  }
}

async function loadResearchQuestion() {
  if (!currentReviewer) return;
  const card = document.getElementById("research-question-card");
  try {
    const res = await fetch(
      `${API}/api/feedback/research/current?reviewer_id=${encodeURIComponent(currentReviewer.reviewer_id)}`
    );
    if (!res.ok) { card.style.display = "none"; return; }
    currentResearchQuestion = await res.json();
    card.style.display = "block";
    document.getElementById("research-title").textContent = currentResearchQuestion.title;
    const areaEl = document.getElementById("research-knowledge-area");
    areaEl.textContent = currentResearchQuestion.knowledge_area || "Research";
    renderMarkdownMath(document.getElementById("research-body"), currentResearchQuestion.body_markdown);
    const promptsRoot = document.getElementById("research-prompts");
    promptsRoot.innerHTML = "";
    const existing = currentResearchQuestion.existing_answer?.answer || "";
    (currentResearchQuestion.prompts || []).forEach((prompt, i) => {
      const block = document.createElement("div");
      block.className = "onboarding-question";
      block.innerHTML = `<label>${prompt}</label><textarea id="research-answer-${i}" data-prompt="${prompt}">${existing}</textarea>`;
      promptsRoot.appendChild(block);
    });
    loadComments("research_question", currentResearchQuestion.question_id, "research-comments");
    const compose = document.getElementById("research-comment-compose");
    if (compose) compose.style.display = "block";
  } catch {
    card.style.display = "none";
  }
}

async function submitResearchAnswer() {
  if (!currentReviewer || !currentResearchQuestion) return;
  const prompts = currentResearchQuestion.prompts || [];
  const parts = prompts.map((p, i) => {
    const el = document.getElementById(`research-answer-${i}`);
    return `${p}\n${(el?.value || "").trim()}`;
  });
  const answer = parts.join("\n\n").trim();
  if (!answer) { toast("Please answer the research question", "error"); return; }
  const res = await fetch(`${API}/api/feedback/research/answers`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      reviewer_id: currentReviewer.reviewer_id,
      answers: [{
        question_id: currentResearchQuestion.question_id,
        answer,
        knowledge_area: currentResearchQuestion.knowledge_area,
      }],
    }),
  });
  if (!res.ok) { toast("Could not save research answer", "error"); return; }
  toast("Research answer saved — thank you", "success");
  loadResearchQuestion();
}

async function refreshPendingCaseStudies() {
  if (!currentReviewer) return;
  const alertEl = document.getElementById("case-study-alert");
  const res = await fetch(
    `${API}/api/feedback/case-studies/pending?reviewer_id=${encodeURIComponent(currentReviewer.reviewer_id)}`
  );
  if (!res.ok) { alertEl.style.display = "none"; return; }
  const pending = await res.json();
  if (!pending.length) {
    alertEl.style.display = "none";
    document.getElementById("case-study-section").style.display = "none";
    return;
  }
  alertEl.style.display = "block";
  alertEl.textContent = `${pending.length} case study(ies) needed — you beat the AI. Help teach the model why.`;
  const first = pending[0];
  document.getElementById("case-study-section").style.display = "block";
  document.getElementById("case-study-review-id").value = first.review_id;
  loadComments("case_study", first.review_id, "case-study-comments");
}

async function submitCaseStudy() {
  const reviewId = document.getElementById("case-study-review-id").value;
  if (!reviewId || !currentReviewer) return;
  const payload = {
    review_id: reviewId,
    reviewer_id: currentReviewer.reviewer_id,
    ai_missed: document.getElementById("cs-ai-missed").value.trim(),
    decision_factors: document.getElementById("cs-decision-factors").value.trim(),
    missing_variables: document.getElementById("cs-missing-variables").value.trim(),
    data_sources: document.getElementById("cs-data-sources").value.trim(),
    confidence_rating: +document.getElementById("cs-confidence").value || 4,
  };
  if (!payload.ai_missed || !payload.decision_factors) {
    toast("Please complete the case study fields", "error");
    return;
  }
  const res = await fetch(`${API}/api/feedback/case-studies`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) { toast("Could not save case study", "error"); return; }
  toast("Case study saved — published to training data", "success");
  refreshPendingCaseStudies();
}

async function loadComments(targetType, targetId, containerId) {
  const root = document.getElementById(containerId);
  if (!root) return;
  const res = await fetch(
    `${API}/api/feedback/comments?target_type=${encodeURIComponent(targetType)}&target_id=${encodeURIComponent(targetId)}`
  );
  if (!res.ok) { root.innerHTML = ""; return; }
  const comments = await res.json();
  root.innerHTML = "<p class='conf-label'>Discussion</p>";
  for (const c of comments) {
    const el = document.createElement("div");
    el.className = "comment-item";
    const name = c.first_name || c.name || "Analyst";
    el.innerHTML = `<strong>${name}</strong><p>${c.body}</p>`;
    root.appendChild(el);
  }
}

async function submitComment(targetType, targetId, inputId, containerId) {
  if (!currentReviewer) { toast("Please log in first", "error"); return; }
  const input = document.getElementById(inputId);
  const body = (input?.value || "").trim();
  if (!body) { toast("Please enter a comment", "error"); return; }
  const res = await fetch(`${API}/api/feedback/comments`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      reviewer_id: currentReviewer.reviewer_id,
      target_type: targetType,
      target_id: targetId,
      body,
    }),
  });
  if (!res.ok) { toast("Could not post comment", "error"); return; }
  input.value = "";
  toast("Comment posted", "success");
  loadComments(targetType, targetId, containerId);
}

function updateDisagreeFields() {
  const pickEl = document.querySelector('input[name="who-wins"]:checked');
  const group = document.getElementById("primary-variable-group");
  if (!pickEl || !currentPrediction) { group.style.display = "none"; return; }
  const pick = pickEl.value === "home" ? currentPrediction.home_team : currentPrediction.away_team;
  const disagree = pick.toLowerCase() !== (currentPrediction.predicted_winner || "").toLowerCase();
  group.style.display = disagree ? "block" : "none";
}

// ---------------------------------------------------------------------------
// Reviewer login
// ---------------------------------------------------------------------------
async function handleReviewerLogin() {
  const name = document.getElementById("reviewer-name-input").value.trim();
  if (!name) { toast("Please enter your name", "error"); return; }

  const res = await fetch(`${API}/api/feedback/reviewers`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!res.ok) { toast("Could not load reviewer", "error"); return; }

  const data = await res.json();
  applyReviewerSession(data, data.history, data.custom_sections);
}

function renderCustomSections(sections) {
  const root = document.getElementById("reviewer-custom-sections");
  root.innerHTML = "";
  if (!sections || !sections.length) return;
  for (const section of sections) {
    const el = document.createElement("div");
    el.className = "custom-section";
    el.innerHTML = `<h4>${section.title}</h4><p>${section.content}</p>`;
    root.appendChild(el);
  }
}

function renderReviewerStats(stats) {
  document.getElementById("stat-total").textContent    = stats.total_reviews;
  document.getElementById("stat-agree").textContent    = stats.agree_pct + "%";
  document.getElementById("stat-beat").textContent     = stats.beat_ai;
  document.getElementById("stat-acc").textContent      = stats.reviewer_accuracy + "%";

  const sportChips = document.getElementById("sport-chips");
  sportChips.innerHTML = "";
  const emojis = { MLB: "⚾", NBA: "🏀", FIFA: "⚽" };
  for (const [sport, s] of Object.entries(stats.by_sport || {})) {
    const chip = document.createElement("span");
    chip.className = "sport-chip";
    chip.innerHTML = `<span class="sport-name">${emojis[sport] || ""} ${sport}</span>`
      + `<span class="sport-stat">${s.reviews} reviews · ${s.beat_ai} beat AI</span>`;
    sportChips.appendChild(chip);
  }
  if (!Object.keys(stats.by_sport || {}).length) {
    sportChips.innerHTML = `<span class="sport-chip"><span class="sport-stat">No reviews yet — pick a prediction below</span></span>`;
  }
}

function renderHistory(history) {
  const tbody = document.getElementById("history-tbody");
  tbody.innerHTML = "";
  if (!history.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty-state" style="padding:20px;text-align:center;color:var(--text-dim)">No past predictions yet.</td></tr>`;
    return;
  }
  const badgeMap = {
    beat_ai:      { cls: "badge-beat-ai",   txt: "✅ You beat the AI" },
    ai_right:     { cls: "badge-ai-right",  txt: "❌ AI was right" },
    both_correct: { cls: "badge-both",      txt: "🤝 Both correct" },
    reviewer_right:{ cls: "badge-rev-right",txt: "⭐ You were right" },
    both_wrong:   { cls: "badge-pending",   txt: "Both wrong" },
    pending:      { cls: "badge-pending",   txt: "⏳ Pending" },
  };
  for (const row of history) {
    const b = badgeMap[row.badge] || badgeMap.pending;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.game_date}</td>
      <td>${SPORT_EMOJI[row.sport] || ""} ${row.matchup}</td>
      <td>${row.ai_pick}</td>
      <td>${row.reviewer_pick}</td>
      <td>${row.actual_winner || "—"}</td>
      <td><span class="badge ${b.cls}">${b.txt}</span></td>
    `;
    tbody.appendChild(tr);
  }
}

// ---------------------------------------------------------------------------
// Predictions list
// ---------------------------------------------------------------------------
function setActiveSportTab(sport) {
  selectedSport = sport;
  console.log("[Feedback] Active sport tab:", sport);
  document.querySelectorAll(".sport-tab").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.sport === sport);
  });
}

function setupSportTabs() {
  document.querySelectorAll(".sport-tab").forEach(btn => {
    btn.addEventListener("click", () => {
      const sport = btn.dataset.sport || "ALL";
      setActiveSportTab(sport);
      currentPrediction = null;
      document.getElementById("ai-card").style.display = "none";
      document.getElementById("pregame-section").style.display = "none";
      document.getElementById("postgame-section").style.display = "none";
      loadPredictions();
    });
  });
}

async function loadPredictions() {
  const grid = document.getElementById("pred-grid");
  grid.innerHTML = `<div class="spinner"></div>`;

  const query = selectedSport !== "ALL" ? `?sport=${encodeURIComponent(selectedSport)}` : "";
  const res = await fetch(`${API}/api/feedback/predictions${query}`, { cache: "no-store" });
  if (!res.ok) { grid.innerHTML = `<p style="color:var(--red)">Failed to load predictions.</p>`; return; }
  const preds = await res.json();
  console.log(`[Feedback] Loaded ${preds.length} predictions for filter=${selectedSport}`);

  grid.innerHTML = "";
  if (!preds.length) {
    grid.innerHTML = `<div class="empty-state"><div class="icon">🔍</div><p>No predictions available.</p></div>`;
    return;
  }

  for (const p of preds) {
    const sportUi = p.sport_ui || p.sport;
    const emoji = SPORT_EMOJI[sportUi] || "🏟";
    const leagueLabel = p.league || "";
    const isWorldCup = /world cup/i.test(leagueLabel);
    const tile = document.createElement("div");
    tile.className = "pred-tile" + (isWorldCup ? " pred-tile--world-cup" : "");
    tile.dataset.id = p.prediction_id;
    const dotCls = p.settled ? "settled" : "unsettled";
    tile.innerHTML = `
      <div class="sport-emoji">${emoji}</div>
      ${leagueLabel ? `<div class="game-league">${leagueLabel}</div>` : ""}
      <div class="game-teams">${p.away_team} <span style="color:var(--muted)">@</span> ${p.home_team}</div>
      <div class="game-date">
        <span class="settled-dot ${dotCls}"></span>
        ${p.game_date} · <span class="conf-badge ${(p.confidence_level||'low').toLowerCase()}">${p.confidence_level}</span>
      </div>
    `;
    tile.addEventListener("click", () => selectPrediction(p.prediction_id, tile));
    grid.appendChild(tile);
  }
}

async function selectPrediction(id, tileEl) {
  document.querySelectorAll(".pred-tile").forEach(t => t.classList.remove("active"));
  tileEl.classList.add("active");

  const res = await fetch(`${API}/api/feedback/predictions/${id}`);
  if (!res.ok) { toast("Failed to load prediction", "error"); return; }
  currentPrediction = await res.json();

  renderAICard(currentPrediction);
  await loadMissingFactors(currentPrediction.sport_ui || currentPrediction.sport);
  showPregameSection();

  // Reset postgame
  document.getElementById("postgame-section").style.display = "none";
  currentReviewId = null;
}

// ---------------------------------------------------------------------------
// AI card
// ---------------------------------------------------------------------------
function shortLabel(label) {
  const map = {
    "ELO Difference": "ELO",
    "Average Point Differential": "POINT DIFF",
    "Recent Form (10)": "FORM",
    "Recent Form": "FORM",
    "Last 10 Games": "FORM",
    "Injury Impact": "INJURIES",
    "Injury Status": "INJURIES",
    "Home Court Advantage": "HOME ADV",
    "Home Field Advantage": "HOME ADV",
    "Home Advantage": "HOME ADV",
    "Starting Pitcher ERA": "PITCHER",
    "Starting Pitcher Strength": "PITCHER",
    "Bullpen Ranking": "BULLPEN",
    "Bullpen Strength": "BULLPEN",
    "Run Differential": "RUN DIFF",
    "Goals For / Against": "GOALS",
    "Possession %": "POSSESSION",
    "Offensive Rating": "OFF RATING",
    "Defensive Rating": "DEF RATING",
  };
  return map[label] || label.split(" ").slice(0, 2).join(" ").toUpperCase();
}

function renderAICard(pred) {
  const sportUi = pred.sport_ui || pred.sport;
  const emoji = SPORT_EMOJI[sportUi] || "🏟";

  // Matchup
  document.getElementById("ai-sport-emoji").textContent = emoji;
  document.getElementById("ai-sport-label").textContent = sportUi;
  document.getElementById("ai-home-name").textContent = pred.home_team;
  document.getElementById("ai-away-name").textContent = pred.away_team;
  document.getElementById("ai-home-initials").textContent = initials(pred.home_team);
  document.getElementById("ai-away-initials").textContent = initials(pred.away_team);
  document.getElementById("ai-game-meta").textContent =
    `${sportUi} · ${pred.league || ""} · ${pred.game_date}`;
  const offseason = document.getElementById("offseason-banner");
  if (pred.is_fallback && pred.offseason_notice) {
    offseason.style.display = "block";
    offseason.textContent = pred.offseason_notice;
  } else {
    offseason.style.display = "none";
    offseason.textContent = "";
  }

  // Winner + confidence
  document.getElementById("ai-winner").textContent = pred.predicted_winner;
  const confPct = Math.round((pred.confidence_pct || 0.5) * 100);
  document.getElementById("conf-pct-label").textContent = confPct + "%";
  // Animate bar after render
  setTimeout(() => {
    document.getElementById("conf-bar-fill").style.width = confPct + "%";
  }, 50);

  // Confidence badge
  const cl = (pred.confidence_level || "low").toLowerCase();
  document.getElementById("conf-badge-label").textContent = pred.confidence_level;
  document.getElementById("conf-badge-label").className = `conf-badge ${cl}`;

  const exps = pred.explanations || [];
  const cardsRoot = document.getElementById("explanation-cards");
  cardsRoot.innerHTML = "";
  if (exps.length) {
    for (const e of exps) {
      const card = document.createElement("div");
      card.className = "metric-card";
      card.innerHTML = `
        <div class="k">${shortLabel(e.label)}</div>
        <div class="v">${e.value ?? "—"}</div>
      `;
      cardsRoot.appendChild(card);
    }
  } else {
    cardsRoot.innerHTML = `<p style="color:var(--text-dim);font-size:.82rem">No explanation metrics available.</p>`;
  }

  // Weighted reasoning bars (normalized, no raw JSON)
  const container = document.getElementById("feature-bars");
  container.innerHTML = "";
  if (exps.length) {
    for (const e of exps) {
      const pct = Math.round((e.weight || 0) * 100);
      const row = document.createElement("div");
      row.className = "feature-row";
      row.innerHTML = `
        <span class="feature-label">${e.label}</span>
        <div class="feature-bar-track">
          <div class="feature-bar-fill" style="width:0%" data-pct="${pct}"></div>
        </div>
        <span class="feature-weight">${pct}%</span>
      `;
      container.appendChild(row);
    }
    setTimeout(() => {
      container.querySelectorAll(".feature-bar-fill").forEach(el => {
        el.style.width = el.dataset.pct + "%";
      });
    }, 80);
  } else {
    container.innerHTML = "";
  }

  document.getElementById("exp-betting").innerHTML = `
    <div class="metric-cards">
      <div class="metric-card"><div class="k">Predicted Winner</div><div class="v">${pred.predicted_winner || "—"}</div></div>
      <div class="metric-card"><div class="k">Win Probability</div><div class="v">${Math.round((pred.win_probability || pred.confidence_pct || 0) * 100)}%</div></div>
      <div class="metric-card"><div class="k">Confidence</div><div class="v">${pred.confidence_level || "—"}</div></div>
    </div>
  `;

  const mlbBlock = document.getElementById("mlb-context-block");
  const warnBlock = document.getElementById("missing-data-warnings");
  if ((pred.sport_ui || pred.sport) === "MLB" && pred.starting_pitchers) {
    const hp = pred.starting_pitchers.home || {};
    const ap = pred.starting_pitchers.away || {};
    mlbBlock.style.display = "block";
    mlbBlock.innerHTML = `
      <div class="conf-label" style="margin-bottom:8px">Starting Pitchers</div>
      <div class="metric-cards">
        <div class="metric-card"><div class="k">Away: ${ap.name || "TBD"}</div><div class="v">ERA ${ap.era ?? "—"} · WHIP ${ap.whip ?? "—"}</div></div>
        <div class="metric-card"><div class="k">Home: ${hp.name || "TBD"}</div><div class="v">ERA ${hp.era ?? "—"} · WHIP ${hp.whip ?? "—"}</div></div>
      </div>`;
  } else {
    mlbBlock.style.display = "none";
    mlbBlock.innerHTML = "";
  }
  const warnings = pred.missing_data_warnings || [];
  if (warnings.length) {
    warnBlock.style.display = "block";
    warnBlock.textContent = "Missing data: " + warnings.map(w => w.replace(/_/g, " ")).join(", ");
  } else {
    warnBlock.style.display = "none";
    warnBlock.textContent = "";
  }

  document.getElementById("ai-card").style.display = "block";
}

function initials(name) {
  return name.split(" ").map(w => w[0]).join("").slice(0, 3).toUpperCase();
}

// ---------------------------------------------------------------------------
// Missing factors checkboxes
// ---------------------------------------------------------------------------
async function loadMissingFactors(sport) {
  const res = await fetch(`${API}/api/feedback/missing-factors/${sport}`);
  if (!res.ok) return;
  const data = await res.json();

  const container = document.getElementById("missing-factors-group");
  container.innerHTML = "";
  for (const factor of data.factors) {
    const id = "mf-" + factor.replace(/\s+/g, "-").toLowerCase();
    const label = document.createElement("label");
    label.className = "check-item";
    label.innerHTML = `<input type="checkbox" name="missing_factor" value="${factor}"> ${factor}`;
    container.appendChild(label);
  }
}

// ---------------------------------------------------------------------------
// Pregame section
// ---------------------------------------------------------------------------
function showPregameSection() {
  const sec = document.getElementById("pregame-section");
  sec.style.display = "block";

  // Reset form state
  document.querySelectorAll('input[name="who-wins"]').forEach(r => r.checked = false);
  document.getElementById("reviewer-confidence").value = "";
  document.querySelectorAll(".star-btn").forEach(s => s.className = "star-btn dim");
  document.querySelectorAll('input[name="bet-size"]').forEach(r => r.checked = false);
  document.querySelectorAll('input[name="missing_factor"]').forEach(r => r.checked = false);
  document.getElementById("pregame-notes").value = "";
  document.getElementById("primary-decision-variable").value = "";
  document.getElementById("disagree-explain").value = "";
  document.getElementById("primary-variable-group").style.display = "none";
  document.querySelectorAll('input[name="who-wins"]').forEach(r => {
    r.onchange = updateDisagreeFields;
  });

  // Update team labels in who-wins
  if (currentPrediction) {
    document.getElementById("who-home-label").textContent = `🏠 ${currentPrediction.home_team}`;
    document.getElementById("who-away-label").textContent = `✈️ ${currentPrediction.away_team}`;
  }
}

async function submitPregame() {
  if (!currentReviewer) { toast("Please enter your name first", "error"); return; }
  if (!currentPrediction) { toast("Select a prediction first", "error"); return; }

  const pickEl = document.querySelector('input[name="who-wins"]:checked');
  if (!pickEl) { toast("Pick a winner", "error"); return; }
  const confVal = +document.getElementById("reviewer-confidence").value;
  if (!confVal) { toast("Select your confidence (stars)", "error"); return; }
  const betEl = document.querySelector('input[name="bet-size"]:checked');
  if (!betEl) { toast("Select bet size", "error"); return; }

  const pick = pickEl.value === "home" ? currentPrediction.home_team : currentPrediction.away_team;
  const agreeWithModel = pick.toLowerCase() === currentPrediction.predicted_winner.toLowerCase();
  lastPregamePick = pick;
  lastPregameAgreeWithModel = agreeWithModel;
  const missingFactors = [...document.querySelectorAll('input[name="missing_factor"]:checked')]
    .map(el => el.value);

  let notes = document.getElementById("pregame-notes").value.trim() || null;
  const disagreeExplain = document.getElementById("disagree-explain").value.trim();
  if (!agreeWithModel && disagreeExplain) {
    notes = [notes, disagreeExplain].filter(Boolean).join("\n\n");
  }

  const payload = {
    prediction_id: currentPrediction.prediction_id,
    reviewer_id: currentReviewer.reviewer_id,
    reviewer_pick: pick,
    reviewer_confidence: confVal,
    would_bet: betEl.value,
    agree_with_model: agreeWithModel,
    missing_factors: missingFactors,
    pregame_notes: notes,
    primary_decision_variable: agreeWithModel ? null : (document.getElementById("primary-decision-variable").value || null),
  };

  const res = await fetch(`${API}/api/feedback/prediction-reviews`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (res.status === 409) {
    toast("You already reviewed this prediction", "error");
    // Still try to show postgame if settled
    if (currentPrediction.settled) showPostgameSection(null);
    return;
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    toast(err.detail || "Failed to submit review", "error");
    return;
  }

  const data = await res.json();
  currentReviewId = data.review_id;
  toast("Pregame review saved ✓", "success");

  document.getElementById("pregame-section").style.display = "none";

  if (currentPrediction.settled) {
    showPostgameSection(currentReviewId);
  } else {
    toast("Game hasn't settled yet — check back after the final score is in.", "info");
  }

  // Refresh stats
  refreshReviewerStats();
}

// ---------------------------------------------------------------------------
// Postgame section
// ---------------------------------------------------------------------------
function showPostgameSection(reviewId) {
  if (!currentPrediction || !currentPrediction.settled) return;

  const pred = currentPrediction;
  document.getElementById("pg-home-score").textContent = pred.actual_home_score ?? "—";
  document.getElementById("pg-away-score").textContent = pred.actual_away_score ?? "—";
  document.getElementById("pg-home-team").textContent  = pred.home_team;
  document.getElementById("pg-away-team").textContent  = pred.away_team;

  const modelCorrect = pred.correct === 1 || pred.correct === true;
  document.getElementById("ai-result-icon").textContent = modelCorrect ? "✅" : "❌";
  document.getElementById("ai-result-desc").textContent = modelCorrect
    ? `${pred.predicted_winner} won — AI correct`
    : `${pred.predicted_winner} predicted, ${pred.actual_winner} won — AI incorrect`;

  // Reviewer result unknown until submitted
  document.getElementById("reviewer-result-icon").textContent = "⏳";
  document.getElementById("reviewer-result-desc").textContent = "Submit postgame to see your result";

  document.getElementById("postgame-review-id").value = reviewId || "";
  document.getElementById("deep-analysis-fields").style.display = "none";
  document.getElementById("structured-explanation").value = "";
  document.getElementById("factor-tags").value = "";
  document.getElementById("factor-importance").value = "";
  document.querySelectorAll('input[name="should-feature"]').forEach(r => r.checked = false);

  const unlockDeepAnalysis = Boolean(
    pred.actual_winner &&
    !lastPregameAgreeWithModel &&
    lastPregamePick &&
    String(lastPregamePick).toLowerCase() === String(pred.actual_winner).toLowerCase() &&
    !(pred.correct === 1 || pred.correct === true)
  );
  if (unlockDeepAnalysis) {
    document.getElementById("deep-analysis-fields").style.display = "block";
  }

  // Load postgame missing factors checkboxes
  fetch(`${API}/api/feedback/missing-factors/${pred.sport_ui || pred.sport}`)
    .then(r => r.json())
    .then(data => {
      const cg = document.getElementById("postgame-factors-group");
      cg.innerHTML = "";
      for (const f of data.postgame_factors) {
        const label = document.createElement("label");
        label.className = "check-item";
        label.innerHTML = `<input type="checkbox" name="pg_factor" value="${f}"> ${f}`;
        cg.appendChild(label);
      }
    });

  document.getElementById("postgame-section").style.display = "block";
}

async function submitPostgame() {
  const reviewId = document.getElementById("postgame-review-id").value;
  if (!reviewId) {
    toast("No pregame review found. Submit pregame first.", "error");
    return;
  }

  const factors = [...document.querySelectorAll('input[name="pg_factor"]:checked')]
    .map(el => el.value);
  const reason = document.getElementById("postgame-reason").value.trim() || null;

  const payload = {
    review_id: reviewId,
    followup_missing_factors: factors,
    followup_reason: reason,
    structured_explanation: document.getElementById("structured-explanation").value.trim() || null,
    factor_tags: (document.getElementById("factor-tags").value || "")
      .split(",")
      .map(s => s.trim())
      .filter(Boolean),
    should_be_feature: (() => {
      const selected = document.querySelector('input[name="should-feature"]:checked');
      if (!selected) return null;
      return selected.value === "yes";
    })(),
    importance: (() => {
      const raw = document.getElementById("factor-importance").value;
      if (!raw) return null;
      const n = Number(raw);
      return Number.isFinite(n) ? n : null;
    })(),
  };

  const res = await fetch(`${API}/api/feedback/review-outcomes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (res.status === 409) { toast("Outcome already submitted", "error"); return; }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    toast(err.detail || "Failed to submit outcome", "error");
    return;
  }

  const data = await res.json();
  toast("Postgame reflection saved ✓", "success");

  // Update reviewer result
  document.getElementById("reviewer-result-icon").textContent = data.reviewer_correct ? "✅" : "❌";
  document.getElementById("reviewer-result-desc").textContent = data.reviewer_correct
    ? "You called it right"
    : "Wrong pick this time";

  // Beat AI banner
  if (data.reviewer_beat_model) {
    document.getElementById("beat-ai-banner").style.display = "block";
    if (document.getElementById("deep-analysis-fields").style.display === "none") {
      document.getElementById("deep-analysis-fields").style.display = "block";
    }
  }

  // Disable submit button
  document.getElementById("postgame-submit-btn").disabled = true;
  document.getElementById("postgame-submit-btn").textContent = "Submitted ✓";

  refreshReviewerStats();
  if (data.reviewer_beat_model) {
    refreshPendingCaseStudies();
  }
}

async function refreshReviewerStats() {
  if (!currentReviewer) return;
  const res = await fetch(`${API}/api/feedback/reviewers/${currentReviewer.reviewer_id}/stats`);
  if (!res.ok) return;
  const data = await res.json();
  renderReviewerStats(data);

  const histRes = await fetch(`${API}/api/feedback/reviewers/${currentReviewer.reviewer_id}/history`);
  if (histRes.ok) renderHistory(await histRes.json());
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  initStars();
  setupSportTabs();

  // Auto-login when reviewer_id is present in the URL (e.g. from an email link)
  const params = new URLSearchParams(window.location.search);
  const urlReviewerId = params.get("reviewer_id");
  const urlSport = (params.get("sport") || "").toUpperCase();
  if (["MLB", "NBA", "FIFA", "ALL"].includes(urlSport)) {
    setActiveSportTab(urlSport);
  } else {
    setActiveSportTab("ALL");
  }
  if (urlReviewerId) {
    fetch(`${API}/api/feedback/reviewers/${encodeURIComponent(urlReviewerId)}/stats`)
      .then(r => r.ok ? r.json() : null)
      .then(async data => {
        if (!data) return;
        const histRes = await fetch(`${API}/api/feedback/reviewers/${encodeURIComponent(urlReviewerId)}/history`);
        const history = histRes.ok ? await histRes.json() : [];
        const sectRes = await fetch(`${API}/api/feedback/reviewers/${encodeURIComponent(urlReviewerId)}/custom-sections`);
        const sections = sectRes.ok ? await sectRes.json() : [];
        applyReviewerSession(data, history, sections);
      })
      .catch(() => {});
  }

  document.getElementById("login-btn").addEventListener("click", handleReviewerLogin);
  document.getElementById("reviewer-name-input").addEventListener("keydown", e => {
    if (e.key === "Enter") handleReviewerLogin();
  });

  document.getElementById("onboarding-submit-btn").addEventListener("click", submitOnboarding);
  document.getElementById("onboarding-alert").addEventListener("click", maybeShowOnboarding);
  document.getElementById("research-submit-btn").addEventListener("click", submitResearchAnswer);
  document.getElementById("research-comment-btn").addEventListener("click", () => {
    if (!currentResearchQuestion) return;
    submitComment(
      "research_question",
      currentResearchQuestion.question_id,
      "research-comment-input",
      "research-comments"
    );
  });
  document.getElementById("case-study-submit-btn").addEventListener("click", submitCaseStudy);
  document.getElementById("case-study-comment-btn").addEventListener("click", () => {
    const reviewId = document.getElementById("case-study-review-id").value;
    if (!reviewId) return;
    submitComment("case_study", reviewId, "case-study-comment-input", "case-study-comments");
  });
  document.getElementById("case-study-alert").addEventListener("click", refreshPendingCaseStudies);

  document.getElementById("pregame-submit-btn").addEventListener("click", submitPregame);
  document.getElementById("postgame-submit-btn").addEventListener("click", submitPostgame);
});
