/* =========================================================
   Shuffle & Fourier Lab — client

   The server does the exact arithmetic (riffle table, cut distribution,
   Fourier coefficients) and stores the long simulations. This file draws
   them, and computes P̂(m)^k and its inverse transform for the k slider:
   that is 52 complex powers and one 52-point inverse DFT, which the
   server's own walk table checks at k = 1, 2, 3, 5, 10, 25.
   ========================================================= */

const LAB = "/api/poker/lab";
const N = 52;
const SVG_NS = "http://www.w3.org/2000/svg";

const $ = (id) => document.getElementById(id);

async function getJson(path) {
  const response = await fetch(LAB + path);
  const payload = await response.json().catch(() => null);
  if (!response.ok) throw new Error((payload && payload.detail) || `Request failed (${response.status})`);
  return payload;
}

function svg(tag, attrs = {}, parent = null) {
  const node = document.createElementNS(SVG_NS, tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  if (parent) parent.appendChild(node);
  return node;
}

function metric(parent, label, value, note = "") {
  const box = document.createElement("div");
  box.className = "metric";
  const title = document.createElement("span");
  title.textContent = label;
  const number = document.createElement("b");
  number.textContent = value;
  box.append(title, number);
  if (note) {
    const small = document.createElement("small");
    small.textContent = note;
    box.appendChild(small);
  }
  parent.appendChild(box);
}

const cardHue = (label) => `hsl(${(label / N) * 300}, 72%, 56%)`;

// ── 1. Seven shuffles ──────────────────────────────────────

async function drawRiffles() {
  const data = await getJson("/riffles?max_riffles=12");
  $("riffle-formula").textContent = data.formula;
  const chart = $("riffle-chart");
  chart.replaceChildren();
  const left = 44, bottom = 220, width = 580, height = 190;
  const barWidth = width / data.table.length;

  svg("line", { x1: left, y1: bottom, x2: left + width, y2: bottom, class: "axis" }, chart);
  [0, 0.5, 1].forEach((v) => {
    const y = bottom - v * height;
    svg("line", { x1: left, y1: y, x2: left + width, y2: y, class: "axis",
                  "stroke-dasharray": v === 0.5 ? "4 4" : "" }, chart);
    svg("text", { x: left - 8, y: y + 4, "text-anchor": "end", class: "axis-text" }, chart).textContent = v.toFixed(1);
  });

  data.table.forEach(({ riffles, tv_distance }) => {
    const x = left + riffles * barWidth + 6;
    const h = tv_distance * height;
    const key = riffles === data.riffles_needed_half;
    svg("rect", { x, y: bottom - h, width: barWidth - 12, height: Math.max(h, 1), rx: 3,
                  fill: key ? "#f59e0b" : "#3b82f6",
                  opacity: key ? 1 : 0.75 }, chart);
    svg("text", { x: x + (barWidth - 12) / 2, y: bottom + 16, "text-anchor": "middle", class: "axis-text" }, chart)
      .textContent = riffles;
    svg("text", { x: x + (barWidth - 12) / 2, y: bottom - h - 6, "text-anchor": "middle", class: "axis-text" }, chart)
      .textContent = tv_distance > 0.995 ? "1" : tv_distance.toFixed(2);
  });
  svg("text", { x: left + width / 2, y: bottom + 34, "text-anchor": "middle", class: "axis-text" }, chart)
    .textContent = "riffles";

  const seven = data.table[data.riffles_needed_half].tv_distance;
  const four = data.table[4].tv_distance;
  $("riffle-caption").textContent =
    `Nothing happens for four riffles (distance ${four.toFixed(3)}), then it falls off a cliff. ` +
    `${data.riffles_needed_half} is the first count below 0.5 (${seven.toFixed(3)}). ` +
    "A poker room's standard hand shuffle does three.";
}

// ── 2. Watch a shuffle ─────────────────────────────────────

let procedures = [];

function drawFrames(result) {
  const box = $("frames");
  box.replaceChildren();
  result.frames.forEach((frame) => {
    const label = document.createElement("div");
    label.className = "frame-label";
    const name = document.createElement("span");
    name.textContent = frame.step;
    const rising = document.createElement("span");
    rising.innerHTML = `rising sequences <b>${frame.rising_sequences}</b>`;
    label.append(name, rising);

    const strip = document.createElement("div");
    strip.className = "strip";
    frame.arrangement.forEach((cardLabel) => {
      const cell = document.createElement("span");
      cell.style.background = cardHue(cardLabel);
      strip.appendChild(cell);
    });
    box.append(label, strip);
  });
}

function drawReport(procedure, settings) {
  const box = $("procedure-metrics");
  box.replaceChildren();
  const report = procedure.report;
  if (!report) {
    metric(box, "Stored report", "missing", "run scripts/build_shuffle_lab.py");
    return;
  }
  metric(box, "Rising sequences", report.mean_rising_sequences.toFixed(1), "random deck ≈ 26.5");
  metric(box, "Pairs still in order", `${(report.order_preservation * 100).toFixed(1)}%`,
         "random = 50%; read both sides");
  metric(box, "Flagged non-random", `${Math.round(report.significant_fraction * 100)}% of runs`,
         "a fair shuffle ≈ 5%");
  if (report.exact_tv !== null && report.exact_tv !== undefined) {
    metric(box, "Exact distance", report.exact_tv.toFixed(3), "Bayer–Diaconis, riffles only");
  }
  $("report-settings").textContent = settings
    ? `${settings.trials.toLocaleString()} simulated shuffles per measure, seed ${settings.seed}.`
    : "";
}

async function shuffle() {
  const key = $("procedure-select").value;
  const procedure = procedures.find((p) => p.key === key);
  $("procedure-description").textContent = procedure ? procedure.description : "";
  drawFrames(await getJson(`/shuffle?procedure=${encodeURIComponent(key)}`));
}

async function setupProcedures() {
  const data = await getJson("/procedures");
  procedures = data.procedures;
  const select = $("procedure-select");
  procedures.forEach((p) => {
    const option = document.createElement("option");
    option.value = p.key;
    option.textContent = `${p.key} (${p.steps.join(" → ")})`;
    select.appendChild(option);
  });
  select.value = "casino standard";
  const onChange = () => {
    drawReport(procedures.find((p) => p.key === select.value), data.report_settings);
    shuffle();
  };
  select.addEventListener("change", onChange);
  $("shuffle-btn").addEventListener("click", shuffle);
  onChange();
}

// ── 3. Cuts and characters ─────────────────────────────────

let cuts = null;

const cmul = (a, b) => ({ re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re });

function cpow(z, k) {
  const r = Math.hypot(z.re, z.im) ** k;
  const theta = Math.atan2(z.im, z.re) * k;
  return { re: r * Math.cos(theta), im: r * Math.sin(theta) };
}

/** Distribution after k cuts, by inverting P̂(m)^k: P_k(j) = (1/n) Σ_m P̂(m)^k ω^(−jm). */
function distributionAfter(k) {
  const powered = cuts.coefficients.map((c) => cpow(c, k));
  const out = [];
  for (let j = 0; j < N; j++) {
    let re = 0;
    for (let m = 0; m < N; m++) {
      const angle = (-2 * Math.PI * j * m) / N;
      re += cmul(powered[m], { re: Math.cos(angle), im: Math.sin(angle) }).re;
    }
    out.push(re / N);
  }
  return out;
}

function unitCircle(root) {
  root.replaceChildren();
  svg("circle", { cx: 0, cy: 0, r: 100, fill: "none", stroke: "#1e2d42" }, root);
  svg("line", { x1: -118, y1: 0, x2: 118, y2: 0, class: "axis" }, root);
  svg("line", { x1: 0, y1: -118, x2: 0, y2: 118, class: "axis" }, root);
  svg("text", { x: 104, y: -4, class: "axis-text" }, root).textContent = "1";
  svg("text", { x: 4, y: -104, class: "axis-text" }, root).textContent = "i";
}

function arrow(root, x, y, colour) {
  svg("line", { x1: 0, y1: 0, x2: x, y2: y, stroke: colour, "stroke-width": 2.5 }, root);
  svg("circle", { cx: x, cy: y, r: 4, fill: colour }, root);
}

function drawWinding() {
  const m = Number($("m-slider").value);
  $("m-value").textContent = m;
  const root = $("winding");
  unitCircle(root);
  const maxP = Math.max(...cuts.distribution);
  cuts.distribution.forEach((p, j) => {
    if (p < 1e-6) return;
    const angle = (2 * Math.PI * j * m) / N;
    svg("circle", { cx: 100 * Math.cos(angle), cy: -100 * Math.sin(angle),
                    r: 1.5 + 7 * Math.sqrt(p / maxP), fill: cardHue(j), opacity: 0.8 }, root);
  });
  const c = cuts.coefficients[m];
  arrow(root, 100 * c.re, -100 * c.im, "#f8fafc");
  svg("text", { x: -122, y: 124, class: "axis-text" }, root).textContent =
    `|P̂(${m})| = ${c.modulus.toFixed(4)}`;
}

function drawSpiral() {
  const k = Number($("k-slider").value);
  $("k-value").textContent = k;
  const root = $("spiral");
  unitCircle(root);
  cuts.coefficients.slice(1).forEach((c) => {
    const z = cpow(c, k);
    svg("circle", { cx: 100 * z.re, cy: -100 * z.im, r: 2.6, fill: cardHue(c.m), opacity: 0.9 }, root);
  });

  const sumSq = cuts.coefficients.slice(1).reduce((s, c) => s + c.modulus ** (2 * k), 0);
  const bound = Math.sqrt(sumSq) / 2;
  const dist = distributionAfter(k);
  const tv = dist.reduce((s, p) => s + Math.abs(p - 1 / N), 0) / 2;

  const box = $("cut-metrics");
  box.replaceChildren();
  metric(box, "Distance from a random rotation", tv.toFixed(4), `after ${k} cut${k === 1 ? "" : "s"}`);
  metric(box, "Diaconis–Shahshahani bound", bound > 1 ? "> 1 (says nothing)" : bound.toFixed(4), "always ≥ the exact value");
  metric(box, "Distance from a shuffled deck", cuts.never_mix === 1 ? "1.000…" : cuts.never_mix.toFixed(6),
         "1 − 52/52!, for every k");
  const check = cuts.walk.find((w) => w.cuts === k);
  if (check) {
    metric(box, "Server check (direct convolution)", check.tv_on_rotations.toFixed(4), "computed without Fourier");
  }
}

function describeCuts() {
  $("sd-value").textContent = `${cuts.sd.toFixed(1)} cards`;
  const worst = Math.max(...cuts.coefficients.slice(1).map((c) => c.modulus));
  $("punchline").textContent =
    `The largest |P̂(m)| is ${worst.toFixed(3)}, so the walk does settle, but only to a uniformly random ` +
    "rotation. A rotation keeps every card next to the same neighbours, and cuts can only ever reach " +
    "52 of the 52! orders. However many times you cut, you are exactly as far from a shuffled deck as " +
    "when you started. A cut protects against a dealer who knows the bottom card; it does not shuffle.";
}

async function loadCuts() {
  const sd = Number($("sd-slider").value);
  cuts = await getJson(`/cuts?sd=${sd}&max_cuts=25`);
  describeCuts();
  drawWinding();
  drawSpiral();
}

function setupCuts() {
  let timer = null;
  $("sd-slider").addEventListener("input", () => {
    $("sd-value").textContent = `${Number($("sd-slider").value).toFixed(1)} cards`;
    clearTimeout(timer);
    timer = setTimeout(loadCuts, 150);
  });
  $("m-slider").addEventListener("input", () => cuts && drawWinding());
  $("k-slider").addEventListener("input", () => cuts && drawSpiral());
  return loadCuts();
}

// ── Boot ───────────────────────────────────────────────────

function showError(sectionId, error) {
  const target = $(sectionId);
  const p = document.createElement("p");
  p.className = "error-text";
  p.textContent = `Could not load: ${error.message}`;
  target.after(p);
}

document.addEventListener("DOMContentLoaded", () => {
  drawRiffles().catch((e) => showError("riffle-chart", e));
  setupProcedures().catch((e) => showError("frames", e));
  setupCuts().catch((e) => showError("cut-metrics", e));
});
