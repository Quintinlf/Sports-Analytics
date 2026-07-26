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

// Inert math placeholders — marked must not rewrite these (unlike __MATH_*__).
const MATH_BLOCK_TOKEN = (i) => `%%MATH_BLOCK_${i}%%`;
const MATH_INLINE_TOKEN = (i) => `%%MATH_INLINE_${i}%%`;

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/** True when row is a discovered fixture without an AI model score. */
function isScheduleOnlyPrediction(pred) {
  if (!pred) return false;
  if (pred.metrics && pred.metrics.schedule_only) return true;
  if (String(pred.predicted_winner || "").toLowerCase() === "scheduled") return true;
  if (String(pred.confidence_level || "").toUpperCase() === "N/A" && pred.is_fallback) {
    return true;
  }
  return false;
}

function sanitizeHtml(html) {
  if (typeof DOMPurify !== "undefined") {
    return DOMPurify.sanitize(html, {
      USE_PROFILES: { html: true, mathMl: true, svg: true },
      ADD_ATTR: ["class", "style"],
      ADD_TAGS: ["semantics", "annotation", "annotation-xml"],
    });
  }
  return escapeHtml(html);
}

function setSanitizedHtml(el, html) {
  if (!el) return;
  el.innerHTML = sanitizeHtml(html);
}

function clearChildren(el) {
  if (!el) return;
  while (el.firstChild) el.removeChild(el.firstChild);
}

function appendText(el, tag, text, className) {
  const child = document.createElement(tag);
  if (className) child.className = className;
  child.textContent = text;
  el.appendChild(child);
  return child;
}

function simpleMarkdownBold(text) {
  const parts = String(text ?? "").split(/(\*\*[^*]+\*\*)/g);
  const frag = document.createDocumentFragment();
  for (const part of parts) {
    if (/^\*\*[^*]+\*\*$/.test(part)) {
      const strong = document.createElement("strong");
      strong.textContent = part.slice(2, -2);
      frag.appendChild(strong);
    } else {
      frag.appendChild(document.createTextNode(part));
    }
  }
  return frag;
}

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
  const first = (data.first_name || "").trim();
  if (first) return first;
  const display = (data.display_name || data.name || "").trim();
  if (display) return display.split(/\s+/)[0];
  return "Analyst";
}

function promptText(prompt) {
  if (!prompt) return "";
  if (typeof prompt === "string") return prompt;
  return prompt.prompt || "";
}

function applyReviewerProfile(data) {
  document.getElementById("reviewer-display-name").textContent = `Welcome, ${welcomeLabel(data)}`;

  const roleBadge = document.getElementById("analyst-role-badge");
  if (data.analyst_role && data.analyst_role !== "analyst") {
    roleBadge.textContent = data.analyst_role
      .replace(/_/g, " ")
      .replace(/\b\w/g, c => c.toUpperCase());
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
  clearChildren(root);
  for (const q of questions) {
    const block = document.createElement("div");
    block.className = "onboarding-question";
    const prompt = (q.prompts || [])[0] || {};
    const promptLine = promptText(prompt);

    appendText(block, "h3", q.title || "");
    const body = document.createElement("div");
    body.className = "body";
    body.appendChild(simpleMarkdownBold(q.body_markdown || ""));
    block.appendChild(body);

    if (promptLine) appendText(block, "p", promptLine, "onboarding-prompt");
    if (prompt.example) appendText(block, "p", prompt.example, "onboarding-example");

    const label = document.createElement("label");
    label.htmlFor = `onboard-${q.question_id}`;
    label.textContent = "Your answer";
    block.appendChild(label);

    const textarea = document.createElement("textarea");
    textarea.id = `onboard-${q.question_id}`;
    textarea.dataset.questionId = q.question_id;
    textarea.placeholder = prompt.placeholder || "";
    block.appendChild(textarea);

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

  const titleEl = document.getElementById("onboarding-welcome-title");
  if (titleEl) {
    titleEl.textContent = `Welcome, ${welcomeLabel(currentReviewer)}`;
  }

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

  try {
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
  } catch {
    toast("Network error — please retry", "error");
  }
}

function skipOnboarding() {
  document.getElementById("onboarding-modal").style.display = "none";
  toast("Onboarding skipped — you can complete it later from the alert banner", "info");
}

function applyReviewerSession(data, history, customSections) {
  currentReviewer = {
    reviewer_id: data.reviewer_id,
    name: data.name,
    first_name: data.first_name,
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
  const placeholders = [];
  let work = text;

  work = work.replace(/\$\$([\s\S]+?)\$\$/g, (_, math) => {
    const id = placeholders.length;
    placeholders.push({ type: "display", math: math.trim() });
    return MATH_BLOCK_TOKEN(id);
  });
  work = work.replace(/(?<!\$)\$([^\$\n]+?)\$(?!\$)/g, (_, math) => {
    const id = placeholders.length;
    placeholders.push({ type: "inline", math: math.trim() });
    return MATH_INLINE_TOKEN(id);
  });

  let html = typeof marked !== "undefined" ? marked.parse(work) : work.replace(/\n/g, "<br>");

  for (let i = 0; i < placeholders.length; i++) {
    const p = placeholders[i];
    let rendered = escapeHtml(p.math);
    try {
      if (typeof katex !== "undefined") {
        rendered = katex.renderToString(p.math, {
          throwOnError: false,
          displayMode: p.type === "display",
        });
      }
    } catch {
      rendered = escapeHtml(p.math);
    }
    const token = p.type === "display" ? MATH_BLOCK_TOKEN(i) : MATH_INLINE_TOKEN(i);
    html = html.split(token).join(rendered);
  }

  setSanitizedHtml(el, html);
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
    areaEl.dataset.area = currentResearchQuestion.knowledge_area || "Research";
    renderMarkdownMath(document.getElementById("research-body"), currentResearchQuestion.body_markdown);
    const promptsRoot = document.getElementById("research-prompts");
    clearChildren(promptsRoot);
    const existing = currentResearchQuestion.existing_answer?.answer || "";
    (currentResearchQuestion.prompts || []).forEach((prompt, i) => {
      const labelText = promptText(prompt);
      const block = document.createElement("div");
      block.className = "onboarding-question";
      const label = document.createElement("label");
      label.textContent = labelText;
      const textarea = document.createElement("textarea");
      textarea.id = `research-answer-${i}`;
      textarea.dataset.prompt = labelText;
      textarea.value = existing;
      block.appendChild(label);
      block.appendChild(textarea);
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
    const label = promptText(p);
    return `${label}\n${(el?.value || "").trim()}`;
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
  if (!res.ok) { clearChildren(root); return; }
  const comments = await res.json();
  clearChildren(root);
  appendText(root, "p", "Discussion", "conf-label");
  for (const c of comments) {
    const el = document.createElement("div");
    el.className = "comment-item";
    const name = c.first_name || c.name || "Analyst";
    appendText(el, "strong", name);
    appendText(el, "p", c.body || "");
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

function resolveWhoWinsPick(pickEl) {
  if (!pickEl || !currentPrediction) return null;
  if (pickEl.value === "home") return currentPrediction.home_team;
  if (pickEl.value === "away") return currentPrediction.away_team;
  if (pickEl.value === "draw") return "Draw";
  return null;
}

function updateDisagreeFields() {
  const pickEl = document.querySelector('input[name="who-wins"]:checked');
  const group = document.getElementById("primary-variable-group");
  if (!pickEl || !currentPrediction) { group.style.display = "none"; return; }
  const pick = resolveWhoWinsPick(pickEl);
  const disagree = (pick || "").toLowerCase() !== (currentPrediction.predicted_winner || "").toLowerCase();
  group.style.display = disagree ? "block" : "none";
}

// ---------------------------------------------------------------------------
// Reviewer login
// ---------------------------------------------------------------------------
async function handleReviewerLogin() {
  const name = document.getElementById("reviewer-name-input").value.trim();
  if (!name) { toast("Please enter your name", "error"); return; }

  try {
    const res = await fetch(`${API}/api/feedback/reviewers`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (!res.ok) { toast("Could not load reviewer", "error"); return; }

    const data = await res.json();
    applyReviewerSession(data, data.history, data.custom_sections);
  } catch {
    toast("Network error — please retry", "error");
  }
}

function renderCustomSections(sections) {
  const root = document.getElementById("reviewer-custom-sections");
  clearChildren(root);
  if (!sections || !sections.length) return;
  for (const section of sections) {
    const el = document.createElement("div");
    el.className = "custom-section";
    appendText(el, "h4", section.title || "");
    appendText(el, "p", section.content || "");
    root.appendChild(el);
  }
}

function renderReviewerStats(stats) {
  document.getElementById("stat-total").textContent    = stats.total_reviews;
  document.getElementById("stat-agree").textContent    = stats.agree_pct + "%";
  document.getElementById("stat-beat").textContent     = stats.beat_ai;
  document.getElementById("stat-acc").textContent      = stats.reviewer_accuracy + "%";

  const sportChips = document.getElementById("sport-chips");
  clearChildren(sportChips);
  const emojis = { MLB: "⚾", NBA: "🏀", FIFA: "⚽" };
  for (const [sport, s] of Object.entries(stats.by_sport || {})) {
    const chip = document.createElement("span");
    chip.className = "sport-chip";
    appendText(chip, "span", `${emojis[sport] || ""} ${sport}`, "sport-name");
    appendText(chip, "span", `${s.reviews} reviews · ${s.beat_ai} beat AI`, "sport-stat");
    sportChips.appendChild(chip);
  }
  if (!Object.keys(stats.by_sport || {}).length) {
    const chip = document.createElement("span");
    chip.className = "sport-chip";
    appendText(chip, "span", "No reviews yet — pick a prediction below", "sport-stat");
    sportChips.appendChild(chip);
  }
}

function renderHistory(history) {
  const tbody = document.getElementById("history-tbody");
  clearChildren(tbody);
  if (!history.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 7;
    td.className = "empty-state";
    td.style.padding = "20px";
    td.style.textAlign = "center";
    td.style.color = "var(--text-dim)";
    td.textContent = "No past predictions yet.";
    tr.appendChild(td);
    tbody.appendChild(tr);
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
    const cells = [
      row.game_date,
      `${SPORT_EMOJI[row.sport] || ""} ${row.matchup}`,
      row.ai_pick,
      row.reviewer_pick,
      row.actual_winner || "—",
    ];
    for (const text of cells) {
      appendText(tr, "td", text);
    }
    const badgeTd = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = `badge ${b.cls}`;
    badge.textContent = b.txt;
    badgeTd.appendChild(badge);
    tr.appendChild(badgeTd);
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
  clearChildren(grid);
  const spinner = document.createElement("div");
  spinner.className = "spinner";
  grid.appendChild(spinner);

  const query = selectedSport !== "ALL" ? `?sport=${encodeURIComponent(selectedSport)}` : "";
  let res;
  try {
    res = await fetch(`${API}/api/feedback/predictions${query}`, { cache: "no-store" });
  } catch {
    clearChildren(grid);
    const err = document.createElement("p");
    err.style.color = "var(--red)";
    err.textContent = "Network error — please retry.";
    grid.appendChild(err);
    toast("Network error — please retry", "error");
    return;
  }
  if (!res.ok) {
    clearChildren(grid);
    const err = document.createElement("p");
    err.style.color = "var(--red)";
    err.textContent = "Failed to load predictions.";
    grid.appendChild(err);
    return;
  }
  const preds = await res.json();
  console.log(`[Feedback] Loaded ${preds.length} predictions for filter=${selectedSport}`);

  clearChildren(grid);
  if (!preds.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    appendText(empty, "div", "🔍", "icon");
    appendText(empty, "p", "No predictions available.");
    grid.appendChild(empty);
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

    appendText(tile, "div", emoji, "sport-emoji");
    if (leagueLabel) appendText(tile, "div", leagueLabel, "game-league");

    const teams = document.createElement("div");
    teams.className = "game-teams";
    teams.appendChild(document.createTextNode(`${p.away_team} `));
    const at = document.createElement("span");
    at.style.color = "var(--muted)";
    at.textContent = "@";
    teams.appendChild(at);
    teams.appendChild(document.createTextNode(` ${p.home_team}`));
    tile.appendChild(teams);

    const scheduleOnly = isScheduleOnlyPrediction(p);
    const dateRow = document.createElement("div");
    dateRow.className = "game-date";
    const dot = document.createElement("span");
    dot.className = `settled-dot ${dotCls}`;
    dateRow.appendChild(dot);
    dateRow.appendChild(document.createTextNode(` ${p.game_date} · `));
    const conf = document.createElement("span");
    if (scheduleOnly) {
      conf.className = "conf-badge schedule-only";
      conf.textContent = "Scheduled";
    } else {
      conf.className = `conf-badge ${(p.confidence_level || "low").toLowerCase()}`;
      conf.textContent = p.confidence_level || "";
    }
    dateRow.appendChild(conf);
    tile.appendChild(dateRow);

    tile.addEventListener("click", () => selectPrediction(p.prediction_id, tile));
    grid.appendChild(tile);
  }
}

async function selectPrediction(id, tileEl) {
  document.querySelectorAll(".pred-tile").forEach(t => t.classList.remove("active"));
  tileEl.classList.add("active");

  try {
    const res = await fetch(`${API}/api/feedback/predictions/${id}`);
    if (!res.ok) { toast("Failed to load prediction", "error"); return; }
    currentPrediction = await res.json();
  } catch {
    toast("Network error — please retry", "error");
    return;
  }

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
function isMeaningfulExplanation(e) {
  // Accept legacy rows that only have label/weight (no value/detail).
  const v = e?.value ?? e?.detail ?? e?.label;
  if (v === null || v === undefined) return false;
  const s = String(v).trim();
  if (!s || s === "—" || s === "-" || s === "–") return false;
  if (/^data\s*pending$/i.test(s)) return false;
  return true;
}

const MISSING_DATA_LABELS = {
  probable_starter_unconfirmed: "Starting lineup unavailable",
  lineup_unavailable: "Starting lineup unavailable",
  injury_report_unavailable: "Injury report unavailable",
  advanced_metrics_unavailable: "Advanced metrics unavailable",
  pitcher_stats_unavailable: "Starter stats unavailable",
  bullpen_workload_unavailable: "Bullpen workload unavailable",
};

function formatMissingWarning(code) {
  return MISSING_DATA_LABELS[code] || code.replace(/_/g, " ");
}

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
    "Home Starter ERA": "PITCHER",
    "Away Starter ERA": "PITCHER",
    "Starting Pitcher Strength": "PITCHER",
    "Bullpen Ranking": "BULLPEN",
    "Bullpen Strength": "BULLPEN",
    "Run Differential": "RUN DIFF",
    "Recent scoring (runs / game)": "SCORING",
    "Recent run prevention": "PREVENTION",
    "Last-10 win rate": "FORM",
    "Recent scoring": "SCORING",
    "Elo rating edge": "ELO",
    "Home court advantage": "HOME ADV",
    "Expected goals (xG)": "xG",
    "Goals scored": "GOALS",
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
  const provenanceEl = document.getElementById("ai-provenance-meta");
  if (provenanceEl) {
    const parts = [];
    if (pred.model_name) parts.push(pred.model_name);
    if (pred.data_source) parts.push(`source: ${pred.data_source}`);
    if (pred.created_at) parts.push(`predicted ${pred.created_at}`);
    provenanceEl.textContent = parts.join(" · ");
  }
  const scheduleOnly = isScheduleOnlyPrediction(pred);
  const kindLabel = document.getElementById("prediction-kind-label");
  if (kindLabel) {
    kindLabel.textContent = scheduleOnly ? "Scheduled fixture" : "AI Predicts";
  }

  const offseason = document.getElementById("offseason-banner");
  if (scheduleOnly || pred.is_fallback) {
    offseason.style.display = "block";
    offseason.textContent = pred.offseason_notice
      || (scheduleOnly
        ? "Fixture discovered, but model unavailable for this competition."
        : "Not a live model prediction — demo/fallback data.");
  } else {
    offseason.style.display = "none";
    offseason.textContent = "";
  }

  // Winner + confidence
  document.getElementById("ai-winner").textContent = scheduleOnly
    ? "No AI prediction"
    : pred.predicted_winner;
  const confPct = scheduleOnly ? 0 : Math.round((pred.confidence_pct || 0.5) * 100);
  document.getElementById("conf-pct-label").textContent = scheduleOnly ? "—" : confPct + "%";
  // Animate bar after render
  setTimeout(() => {
    document.getElementById("conf-bar-fill").style.width = confPct + "%";
  }, 50);

  // Confidence badge
  const confBadge = document.getElementById("conf-badge-label");
  if (scheduleOnly) {
    confBadge.textContent = "Scheduled";
    confBadge.className = "conf-badge schedule-only";
  } else {
    const cl = (pred.confidence_level || "low").toLowerCase();
    confBadge.textContent = pred.confidence_level;
    confBadge.className = `conf-badge ${cl}`;
  }

  // Prefer why_factors (model-input grounded); fall back to legacy explanations.
  const whyFactors = Array.isArray(pred.why_factors) && pred.why_factors.length
    ? pred.why_factors
    : (pred.explanations || []).map(e => ({
        label: e.label,
        detail: e.value || e.label,
        strength: e.weight,
      }));
  const whyHeading = document.getElementById("why-ai-heading");
  const whyBlock = document.getElementById("why-ai-block");
  const whyEmpty = document.getElementById("why-ai-empty");
  if (whyHeading) {
    if (scheduleOnly) {
      whyHeading.textContent = "Schedule only — no AI explanation";
    } else {
      whyHeading.textContent = pred.predicted_winner
        && String(pred.predicted_winner).toLowerCase() !== "scheduled"
        ? `Why the AI picked ${pred.predicted_winner}`
        : "Why the AI thinks this";
    }
  }
  const cardsRoot = document.getElementById("explanation-cards");
  const featureSection = document.getElementById("feature-bars");
  clearChildren(cardsRoot);
  const meaningfulWhy = whyFactors.filter(isMeaningfulExplanation);
  // Never hide the why block on mobile — empty state is clearer than a missing section.
  if (whyBlock) whyBlock.style.display = "block";
  if (meaningfulWhy.length) {
    for (const f of meaningfulWhy) {
      const card = document.createElement("div");
      card.className = "metric-card";
      appendText(card, "div", shortLabel(f.label), "k");
      appendText(card, "div", String(f.detail || f.label), "v");
      cardsRoot.appendChild(card);
    }
    if (whyEmpty) whyEmpty.style.display = "none";
  } else if (whyEmpty) {
    whyEmpty.style.display = scheduleOnly ? "none" : "block";
  }

  // Relative strength bars from why_factors
  clearChildren(featureSection);
  if (meaningfulWhy.length) {
    for (const f of meaningfulWhy) {
      const pct = Math.round((Number(f.strength) || 0) * 100);
      const row = document.createElement("div");
      row.className = "feature-row";
      appendText(row, "span", f.label || "", "feature-label");
      const track = document.createElement("div");
      track.className = "feature-bar-track";
      const fill = document.createElement("div");
      fill.className = "feature-bar-fill";
      fill.style.width = "0%";
      fill.dataset.pct = String(pct);
      track.appendChild(fill);
      row.appendChild(track);
      appendText(row, "span", `${pct}%`, "feature-weight");
      featureSection.appendChild(row);
    }
    setTimeout(() => {
      featureSection.querySelectorAll(".feature-bar-fill").forEach(el => {
        el.style.width = el.dataset.pct + "%";
      });
    }, 80);
  }

  const riskBlock = document.getElementById("risk-factors-block");
  const riskList = document.getElementById("risk-factors-list");
  const risks = Array.isArray(pred.risk_factors) ? pred.risk_factors : [];
  if (riskBlock && riskList) {
    clearChildren(riskList);
    if (risks.length) {
      riskBlock.style.display = "block";
      for (const r of risks) {
        const detail = r.detail || r.label || r.code || "";
        appendText(riskList, "li", `⚠ ${detail}`);
      }
    } else {
      riskBlock.style.display = "none";
    }
  }

  const bettingRoot = document.getElementById("exp-betting");
  clearChildren(bettingRoot);
  const bettingCards = document.createElement("div");
  bettingCards.className = "metric-cards";
  const bettingItems = [
    ["Predicted Winner", pred.predicted_winner || "—"],
    ["Win Probability", `${Math.round((pred.win_probability || pred.confidence_pct || 0) * 100)}%`],
    ["Confidence", pred.confidence_level || "—"],
  ];
  for (const [k, v] of bettingItems) {
    const card = document.createElement("div");
    card.className = "metric-card";
    appendText(card, "div", k, "k");
    appendText(card, "div", v, "v");
    bettingCards.appendChild(card);
  }
  bettingRoot.appendChild(bettingCards);

  const mlbBlock = document.getElementById("mlb-context-block");
  const warnBlock = document.getElementById("missing-data-warnings");
  if ((pred.sport_ui || pred.sport) === "MLB" && pred.starting_pitchers) {
    const hp = pred.starting_pitchers.home || {};
    const ap = pred.starting_pitchers.away || {};
    mlbBlock.style.display = "block";
    clearChildren(mlbBlock);
    appendText(mlbBlock, "div", "Starting Pitchers", "conf-label").style.marginBottom = "8px";
    const pitcherCards = document.createElement("div");
    pitcherCards.className = "metric-cards";
    for (const [side, pitcher] of [["Away", ap], ["Home", hp]]) {
      const card = document.createElement("div");
      card.className = "metric-card";
      appendText(card, "div", `${side}: ${pitcher.name || "TBD"}`, "k");
      appendText(card, "div", `ERA ${pitcher.era ?? "—"} · WHIP ${pitcher.whip ?? "—"}`, "v");
      pitcherCards.appendChild(card);
    }
    mlbBlock.appendChild(pitcherCards);
  } else {
    mlbBlock.style.display = "none";
    clearChildren(mlbBlock);
  }
  const warnings = pred.missing_data_warnings || [];
  if (warnings.length) {
    warnBlock.style.display = "block";
    clearChildren(warnBlock);
    appendText(warnBlock, "div", "Missing information", "conf-label").style.marginBottom = "6px";
    const ul = document.createElement("ul");
    ul.className = "missing-info-list";
    for (const w of warnings) {
      appendText(ul, "li", `⚠ ${formatMissingWarning(w)}`);
    }
    warnBlock.appendChild(ul);
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
  clearChildren(container);
  for (const factor of data.factors) {
    const label = document.createElement("label");
    label.className = "check-item";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.name = "missing_factor";
    input.value = factor;
    label.appendChild(input);
    label.appendChild(document.createTextNode(` ${factor}`));
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

  // Update team labels in who-wins; show Draw only for soccer.
  if (currentPrediction) {
    document.getElementById("who-home-label").textContent = `🏠 ${currentPrediction.home_team}`;
    document.getElementById("who-away-label").textContent = `✈️ ${currentPrediction.away_team}`;
    const sportUi = (currentPrediction.sport_ui || currentPrediction.sport || "").toUpperCase();
    const drawWrap = document.getElementById("pick-draw-wrap");
    if (drawWrap) {
      drawWrap.style.display = (sportUi === "FIFA" || sportUi === "SOCCER") ? "" : "none";
    }
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

  const pick = resolveWhoWinsPick(pickEl);
  if (!pick) { toast("Pick a winner", "error"); return; }
  const agreeWithModel = pick.toLowerCase() === (currentPrediction.predicted_winner || "").toLowerCase();
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

  try {
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
  } catch {
    toast("Network error — please retry", "error");
  }
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
  document.getElementById("beat-ai-banner").style.display = unlockDeepAnalysis ? "block" : "none";
  if (unlockDeepAnalysis) {
    document.getElementById("deep-analysis-fields").style.display = "block";
  }

  // Load postgame missing factors checkboxes
  fetch(`${API}/api/feedback/missing-factors/${pred.sport_ui || pred.sport}`)
    .then(r => r.json())
    .then(data => {
      const cg = document.getElementById("postgame-factors-group");
      clearChildren(cg);
      for (const f of data.postgame_factors) {
        const label = document.createElement("label");
        label.className = "check-item";
        const input = document.createElement("input");
        input.type = "checkbox";
        input.name = "pg_factor";
        input.value = f;
        label.appendChild(input);
        label.appendChild(document.createTextNode(` ${f}`));
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
  const deepVisible = document.getElementById("deep-analysis-fields").style.display !== "none";
  if (deepVisible && !reason) {
    toast(
      "If your prediction was correct and the AI was wrong, explain what information the model missed.",
      "error"
    );
    return;
  }

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

  try {
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

    // Successful analyst override — collect reasoning (not ranking)
    if (data.successful_analyst_override || data.reviewer_beat_model) {
      document.getElementById("beat-ai-banner").style.display = "block";
      const promptEl = document.getElementById("override-followup-prompt");
      if (promptEl && data.override_followup_prompt) {
        promptEl.textContent = data.override_followup_prompt;
      }
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
  } catch {
    toast("Network error — please retry", "error");
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
      .then(r => {
        if (!r.ok) throw new Error(`status ${r.status}`);
        return r.json();
      })
      .then(async data => {
        if (!data) throw new Error("empty reviewer");
        const histRes = await fetch(`${API}/api/feedback/reviewers/${encodeURIComponent(urlReviewerId)}/history`);
        const history = histRes.ok ? await histRes.json() : [];
        const sectRes = await fetch(`${API}/api/feedback/reviewers/${encodeURIComponent(urlReviewerId)}/custom-sections`);
        const sections = sectRes.ok ? await sectRes.json() : [];
        applyReviewerSession(data, history, sections);
      })
      .catch(() => {
        toast("Could not auto-login from link — please enter your name", "error");
        document.getElementById("reviewer-login").style.display = "block";
        document.getElementById("reviewer-panel").style.display = "none";
      });
  }

  document.getElementById("login-btn").addEventListener("click", handleReviewerLogin);
  document.getElementById("reviewer-name-input").addEventListener("keydown", e => {
    if (e.key === "Enter") handleReviewerLogin();
  });

  document.getElementById("onboarding-submit-btn").addEventListener("click", submitOnboarding);
  const skipBtn = document.getElementById("onboarding-skip-btn");
  if (skipBtn) skipBtn.addEventListener("click", skipOnboarding);
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
