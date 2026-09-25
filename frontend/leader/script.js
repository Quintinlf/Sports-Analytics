"use strict";

const API = "";
let reviewerId = null;

function qs(id) { return document.getElementById(id); }

function statusClass(status) {
  const s = String(status || "").toLowerCase();
  if (["ok", "success", "true"].includes(s)) return "ok";
  if (["lagging", "warn", "warning", "unknown", "partial"].includes(s)) return "warn";
  if (["fail", "failed", "error", "false", "bad"].includes(s)) return "bad";
  return "";
}

function metricEl(label, value, tone) {
  const div = document.createElement("div");
  div.className = "metric";
  const l = document.createElement("div");
  l.className = "label";
  l.textContent = label;
  const v = document.createElement("div");
  v.className = "value " + (tone || "");
  v.textContent = value == null ? "—" : String(value);
  div.appendChild(l);
  div.appendChild(v);
  return div;
}

async function leaderFetch(path) {
  const sep = path.includes("?") ? "&" : "?";
  const url = reviewerId
    ? `${API}${path}${sep}reviewer_id=${encodeURIComponent(reviewerId)}`
    : `${API}${path}`;
  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

function renderHealth(h) {
  const grid = qs("health-grid");
  grid.replaceChildren();
  grid.appendChild(metricEl("DB", h.db_ok ? "ok" : "fail", h.db_ok ? "ok" : "bad"));
  grid.appendChild(metricEl("Predictions today", h.predictions_today));
  grid.appendChild(metricEl("Stale", h.stale_predictions, h.stale_predictions > 0 ? "warn" : "ok"));
  grid.appendChild(metricEl("Settlement", h.settlement_status, statusClass(h.settlement_status)));
  grid.appendChild(metricEl("Last successful run", h.last_successful_run || "—"));
  grid.appendChild(metricEl("Last email", h.last_email_run || "—"));
  qs("health-stamp").textContent = h.db_fingerprint || "";

  const row = qs("pipeline-row");
  row.replaceChildren();
  const pipes = h.pipelines || {};
  for (const sport of ["NBA", "MLB", "FIFA"]) {
    const p = pipes[sport] || {};
    const el = document.createElement("div");
    el.className = "pipe";
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = sport + " pipeline";
    const st = document.createElement("div");
    st.className = "status " + statusClass(p.status);
    st.textContent = String(p.status || "unknown");
    const detail = document.createElement("div");
    detail.className = "detail";
    const bits = [];
    if (p.last_run_at) bits.push(p.last_run_at);
    if (p.predictions_count != null) bits.push(`${p.predictions_count} preds`);
    if (p.detail) bits.push(p.detail);
    detail.textContent = bits.join(" · ") || "No recent run logged";
    el.appendChild(name);
    el.appendChild(st);
    el.appendChild(detail);
    row.appendChild(el);
  }
}

function renderPerformance(p) {
  const o = p.overall || {};
  const strip = qs("perf-overall");
  strip.replaceChildren();
  for (const [label, val] of [
    ["Settled", o.settled],
    ["Correct", o.correct],
    ["Accuracy", o.accuracy_pct != null ? `${o.accuracy_pct}%` : "—"],
    ["Brier (approx)", o.brier_approx != null ? o.brier_approx : "—"],
  ]) {
    const item = document.createElement("div");
    item.className = "item";
    const strong = document.createElement("strong");
    strong.textContent = val == null ? "—" : String(val);
    const span = document.createElement("span");
    span.textContent = label;
    item.appendChild(strong);
    item.appendChild(span);
    strip.appendChild(item);
  }

  qs("perf-sport").innerHTML = tableHtml(
    ["Sport", "Settled", "Correct", "Accuracy"],
    (p.by_sport || []).map((r) => [
      r.sport,
      r.settled,
      r.correct,
      r.accuracy_pct != null ? `${r.accuracy_pct}%` : "—",
    ])
  );
  qs("perf-conf").innerHTML = tableHtml(
    ["Confidence", "Settled", "Correct", "Accuracy"],
    (p.by_confidence || []).map((r) => [
      r.confidence_level,
      r.settled,
      r.correct,
      r.accuracy_pct != null ? `${r.accuracy_pct}%` : "—",
    ])
  );
}

function tableHtml(headers, rows) {
  if (!rows.length) return "<p class='muted small'>No data yet.</p>";
  let html = "<table><thead><tr>";
  for (const h of headers) html += `<th>${escapeHtml(h)}</th>`;
  html += "</tr></thead><tbody>";
  for (const row of rows) {
    html += "<tr>";
    for (const cell of row) html += `<td>${escapeHtml(cell)}</td>`;
    html += "</tr>";
  }
  html += "</tbody></table>";
  return html;
}

function escapeHtml(v) {
  return String(v ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderReviewers(payload) {
  const reviewers = payload.reviewers || payload || [];
  const list = Array.isArray(reviewers) ? reviewers : [];
  qs("reviewers-table").innerHTML = tableHtml(
    ["Name", "Role", "Reviews", "Agree %", "Beat AI", "Accuracy", "Last review"],
    list.map((r) => {
      const s = r.stats || {};
      return [
        r.name || r.reviewer_id,
        r.analyst_role || "analyst",
        s.total_reviews ?? r.total_reviews ?? 0,
        s.agree_pct != null ? `${s.agree_pct}%` : "—",
        s.beat_ai ?? s.beat_count ?? 0,
        s.reviewer_accuracy != null ? `${s.reviewer_accuracy}%` : "—",
        r.last_review_at || "—",
      ];
    })
  );
}

function renderFailures(f) {
  const root = qs("failures");
  root.replaceChildren();
  const items = f.pipeline_failures || [];
  if (!items.length) {
    const empty = document.createElement("p");
    empty.className = "muted small";
    empty.textContent = "No recent pipeline failures.";
    root.appendChild(empty);
  } else {
    for (const item of items.slice(0, 15)) {
      const el = document.createElement("div");
      el.className = "fail-item";
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = `${item.sport || "?"} · ${item.status || "fail"} · ${item.run_at || ""}`;
      const msg = document.createElement("div");
      msg.textContent = item.error_message || item.detail || "Failed run";
      el.appendChild(meta);
      el.appendChild(msg);
      root.appendChild(el);
    }
  }
}

async function loadDashboard() {
  const [health, perf, reviewers, failures] = await Promise.all([
    leaderFetch("/api/leader/health"),
    leaderFetch("/api/leader/performance"),
    leaderFetch("/api/leader/reviewers"),
    leaderFetch("/api/leader/failures"),
  ]);
  renderHealth(health);
  renderPerformance(perf);
  renderReviewers(reviewers);
  renderFailures(failures);
  qs("gate").style.display = "none";
  qs("dashboard").style.display = "block";
}

async function loginByName() {
  const name = qs("leader-name").value.trim();
  const err = qs("gate-error");
  err.style.display = "none";
  if (!name) {
    err.textContent = "Enter your name.";
    err.style.display = "block";
    return;
  }
  try {
    const res = await fetch(`${API}/api/feedback/reviewers`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (!res.ok) throw new Error("Could not resolve reviewer");
    const data = await res.json();
    if (String(data.analyst_role || "").toLowerCase() !== "leader") {
      throw new Error("This account is not a leader. Use the analyst view at /feedback.");
    }
    reviewerId = data.reviewer_id;
    const url = new URL(window.location.href);
    url.searchParams.set("reviewer_id", reviewerId);
    window.history.replaceState({}, "", url.toString());
    await loadDashboard();
  } catch (e) {
    err.textContent = e.message || String(e);
    err.style.display = "block";
  }
}

async function lookupProvenance() {
  const id = qs("prov-id").value.trim();
  const out = qs("prov-out");
  if (!id) {
    out.textContent = "Enter a prediction_id.";
    return;
  }
  try {
    const data = await leaderFetch(`/api/leader/predictions/${encodeURIComponent(id)}/provenance`);
    out.textContent = JSON.stringify(data, null, 2);
    out.style.color = "var(--text)";
  } catch (e) {
    out.textContent = e.message || String(e);
    out.style.color = "var(--bad)";
  }
}

async function boot() {
  qs("leader-login-btn").addEventListener("click", loginByName);
  qs("refresh-btn").addEventListener("click", () => {
    if (reviewerId) loadDashboard().catch((e) => alert(e.message));
  });
  qs("prov-btn").addEventListener("click", lookupProvenance);

  const params = new URLSearchParams(window.location.search);
  const rid = params.get("reviewer_id");
  if (rid) {
    reviewerId = rid;
    try {
      await loadDashboard();
    } catch (e) {
      qs("gate-error").textContent = e.message || String(e);
      qs("gate-error").style.display = "block";
      reviewerId = null;
    }
  }
}

boot();
