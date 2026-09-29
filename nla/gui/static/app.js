// TinyNLA Thought Reader — frontend (no build step; D3 from CDN).
"use strict";

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const state = {
  text: "",
  tokens: [],
  selected: null,
  fveByPos: new Map(),   // position -> best self-check FVE seen
  samples: [],           // explanations for the selected token
  controls: null,
  reading: false,
};

const STOP = new Set(("the a an and or of to in on at is are was were be it this that with for as by " +
  "from about text its their his her they he she likely next will would could can may which what " +
  "into about has have had not but so than then there these those more most some any").split(" "));

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path}: ${res.status} ${await res.text()}`);
  return res.json();
}

// ---------------------------------------------------------------- run header
async function loadRun() {
  try {
    const r = await api("/api/run");
    const ev = r.eval
      ? `held-out FVE <b>${r.eval.av_fve.toFixed(3)}</b> · summary ${r.eval.summary_fve.toFixed(3)} · shuffled ${r.eval.av_fve_shuffled.toFixed(3)}`
      : "no eval report yet (run training.eval_nla)";
    $("run").innerHTML =
      `<span>model <b>${r.target}</b></span><span>layer <b>${r.layer}</b></span>` +
      `<span>stage <b>${r.stage.toUpperCase()}</b></span><span>${ev}</span>`;
  } catch (e) {
    $("run").innerHTML = `<span class="error">${e.message}</span>`;
  }
}

// ---------------------------------------------------------------- tokens
async function tokenize() {
  state.text = $("text").value;
  if (!state.text.trim()) return;
  $("analyze").disabled = true;
  try {
    const { tokens } = await api("/api/tokenize", { text: state.text });
    state.tokens = tokens;
    state.fveByPos.clear();
    const last = [...tokens].reverse().find((t) => t.eligible);
    renderTokens();
    if (last) select(last.position);
  } catch (e) {
    $("tokens").innerHTML = `<p class="error">${e.message}</p>`;
  } finally {
    $("analyze").disabled = false;
  }
}

function tokenColor() {
  const by = $("colorBy").value;
  const eligible = state.tokens.filter((t) => t.eligible);
  if (by === "fve") {
    const scale = d3.scaleSequential(d3.interpolateRdYlGn).domain([-0.2, 0.6]).clamp(true);
    return {
      fill: (t) => (state.fveByPos.has(t.position) ? scale(state.fveByPos.get(t.position)) : "transparent"),
      legend: { interp: d3.interpolateRdYlGn, lo: "−0.2", hi: "0.6", label: "self-check FVE (read tokens only)" },
    };
  }
  const ext = d3.extent(eligible, (t) => t.norm);
  const scale = d3.scaleSequential(d3.interpolateBlues).domain([ext[0] ?? 0, (ext[1] ?? 1) * 1.15]);
  return {
    fill: (t) => (t.eligible ? scale(t.norm) : "transparent"),
    legend: { interp: d3.interpolateBlues, lo: (ext[0] ?? 0).toFixed(0), hi: (ext[1] ?? 0).toFixed(0), label: "activation norm" },
  };
}

function renderTokens() {
  const { fill, legend } = tokenColor();
  const box = $("tokens");
  box.innerHTML = "";
  for (const t of state.tokens) {
    const span = document.createElement("span");
    span.className = "tok" + (t.eligible ? "" : " ineligible") + (t.position === state.selected ? " selected" : "");
    span.textContent = t.token;
    span.style.background = fill(t);
    // pick text colour against the fill, not the theme (pale fills + dark theme = invisible text)
    const bg = d3.color(fill(t));
    if (bg && bg.opacity > 0) span.style.color = d3.hsl(bg).l < 0.55 ? "#fff" : "#111";
    span.title = `#${t.position} · norm ${t.norm.toFixed(1)} · model predicts ${JSON.stringify(t.top_next)}` +
      (t.eligible ? "" : " · too early to read reliably");
    if (t.eligible) span.onclick = () => select(t.position);
    box.appendChild(span);
  }
  const stops = d3.range(0, 1.01, 0.1).map((x) => legend.interp(x)).join(",");
  $("legend").innerHTML =
    `<span>${legend.label}</span><span>${legend.lo}</span>` +
    `<span class="bar" style="background:linear-gradient(90deg,${stops})"></span><span>${legend.hi}</span>`;
}

function select(position) {
  const t = state.tokens[position];
  if (!t || !t.eligible || state.reading) return;
  state.selected = position;
  state.samples = [];
  state.controls = null;
  renderTokens();
  $("selected").innerHTML =
    `token <code>${escapeHtml(t.token)}</code> · #${t.position} · norm ${t.norm.toFixed(1)} · ` +
    `model's next-token guess <code>${escapeHtml(t.top_next)}</code>`;
  $("explanations").innerHTML = `<p class="empty">Press <b>Read token</b> to verbalize this activation.</p>`;
  $("read").disabled = false;
  $("controls").innerHTML = "";
  $("compare").innerHTML = "";
}

function move(delta) {
  if (state.selected === null) return;
  let p = state.selected + delta;
  while (p >= 0 && p < state.tokens.length && !state.tokens[p].eligible) p += delta;
  if (p >= 0 && p < state.tokens.length) select(p);
}

// ---------------------------------------------------------------- reading (SSE)
async function readToken() {
  if (state.selected === null || state.reading) return;
  state.reading = true;
  $("read").disabled = true;
  const n = +$("samples").value;
  state.samples = [];
  renderExplanations(n);

  const body = { text: state.text, position: state.selected, n_samples: n };
  const controlsP = api("/api/controls", { text: state.text, position: state.selected })
    .then((c) => { state.controls = c; renderControls(); })
    .catch((e) => { $("controls").innerHTML = `<p class="error">${e.message}</p>`; });

  try {
    const res = await fetch("/api/read", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`/api/read: ${res.status} ${await res.text()}`);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buf.indexOf("\n\n")) >= 0) {
        handleEvent(buf.slice(0, cut), n);
        buf = buf.slice(cut + 2);
      }
    }
  } catch (e) {
    $("explanations").insertAdjacentHTML("beforeend", `<p class="error">${e.message}</p>`);
  } finally {
    await controlsP;
    state.reading = false;
    $("read").disabled = false;
  }
}

function handleEvent(raw, n) {
  let event = "message", data = "";
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (event === "error") {
    $("explanations").insertAdjacentHTML("beforeend", `<p class="error">${JSON.parse(data).message}</p>`);
  } else if (event === "message") {
    const item = JSON.parse(data);
    state.samples.push(item);
    const best = Math.max(item.fve, state.fveByPos.get(item.position) ?? -Infinity);
    state.fveByPos.set(item.position, best);
    renderExplanations(n);
    renderControls();
    if ($("colorBy").value === "fve") renderTokens();
  }
}

// ---------------------------------------------------------------- explanations
function recurringWords() {
  const counts = new Map();
  for (const s of state.samples) {
    for (const w of new Set(words(s.text))) counts.set(w, (counts.get(w) ?? 0) + 1);
  }
  return new Set([...counts].filter(([, c]) => c >= 2).map(([w]) => w));
}

function words(text) {
  return (text.toLowerCase().match(/[a-z][a-z'-]{2,}/g) ?? []).filter((w) => !STOP.has(w));
}

function highlight(text, recurring) {
  return escapeHtml(text).replace(/[A-Za-z][A-Za-z'-]{2,}/g,
    (w) => (recurring.has(w.toLowerCase()) ? `<mark>${w}</mark>` : w));
}

function fveColor(v) {
  return v >= 0.3 ? css("--good") : v > 0.05 ? css("--warn") : css("--bad");
}

function renderExplanations(n) {
  const box = $("explanations");
  const recurring = state.samples.length > 1 ? recurringWords() : new Set();
  box.innerHTML = "";
  state.samples.forEach((s, i) => {
    const div = document.createElement("div");
    div.className = "expl";
    div.innerHTML =
      `<div class="meta"><span class="badge" style="background:${fveColor(s.fve)}">FVE ${s.fve.toFixed(3)}</span>` +
      `<span>cos ${s.cosine.toFixed(3)}</span><span>sample ${i + 1}</span>` +
      (s.ok ? "" : `<span class="error">no closing tag</span>`) + `</div>` +
      `<div>${highlight(s.text, recurring) || "<i>(empty)</i>"}</div>`;
    div.onclick = () => {
      box.querySelectorAll(".expl").forEach((e) => e.classList.remove("active"));
      div.classList.add("active");
      compare(s.text);
    };
    box.appendChild(div);
  });
  for (let i = state.samples.length; i < n && state.reading; i++) {
    box.insertAdjacentHTML("beforeend", `<div class="expl pending">sampling explanation ${i + 1}…</div>`);
  }
  if (recurring.size) {
    box.insertAdjacentHTML("beforeend",
      `<p class="note">Highlighted words recur across samples; per the NLA paper, recurring claims are more trustworthy.</p>`);
  }
}

// ---------------------------------------------------------------- charts
function renderControls() {
  const rows = [];
  if (state.samples.length) {
    const f = state.samples.map((s) => s.fve);
    rows.push({ label: "best sample", value: d3.max(f), color: css("--accent") });
    rows.push({ label: "mean of samples", value: d3.mean(f), color: css("--accent") });
  }
  if (state.controls) {
    rows.push({ label: "shuffled explanations", value: state.controls.shuffled_fve, color: css("--muted") });
    rows.push({ label: "average activation", value: state.controls.mean_fve, color: css("--muted") });
  }
  hbars($("controls"), rows);
}

function hbars(el, rows) {
  el.innerHTML = "";
  if (!rows.length) return;
  const W = 460, rowH = 26, left = 150, right = 50, H = rows.length * rowH + 24;
  const lo = Math.min(-0.2, d3.min(rows, (r) => r.value)), hi = Math.max(0.6, d3.max(rows, (r) => r.value));
  const x = d3.scaleLinear().domain([lo, hi]).range([left, W - right]);
  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  svg.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", 0).attr("y2", H - 20)
    .attr("stroke", css("--line")).attr("stroke-width", 2);
  const g = svg.selectAll("g.r").data(rows).join("g").attr("transform", (_, i) => `translate(0,${i * rowH})`);
  g.append("text").attr("x", left - 8).attr("y", 15).attr("text-anchor", "end").text((r) => r.label);
  g.append("rect").attr("x", (r) => x(Math.min(0, r.value))).attr("y", 4)
    .attr("width", (r) => Math.abs(x(r.value) - x(0))).attr("height", rowH - 10)
    .attr("rx", 3).attr("fill", (r) => r.color);
  g.append("text").attr("x", (r) => x(Math.max(0, r.value)) + 5).attr("y", 15).text((r) => r.value.toFixed(3));
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - 20})`)
    .call(d3.axisBottom(x).ticks(5));
}

async function compare(explanation) {
  $("compare").innerHTML = `<p class="note">reconstructing…</p>`;
  try {
    const c = await api("/api/compare", { text: state.text, position: state.selected, explanation });
    dimBars($("compare"), c);
  } catch (e) {
    $("compare").innerHTML = `<p class="error">${e.message}</p>`;
  }
}

function dimBars(el, c) {
  el.innerHTML = "";
  const data = c.dims.map((d, i) => ({ dim: d, gold: c.gold[i], recon: c.recon[i] }));
  const W = 460, H = 210, m = { t: 22, r: 8, b: 34, l: 38 };
  const x0 = d3.scaleBand().domain(data.map((d) => d.dim)).range([m.l, W - m.r]).padding(0.2);
  const x1 = d3.scaleBand().domain(["gold", "recon"]).range([0, x0.bandwidth()]).padding(0.08);
  const ext = d3.extent(data.flatMap((d) => [d.gold, d.recon, 0]));
  const y = d3.scaleLinear().domain(ext).nice().range([H - m.b, m.t]);
  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  svg.append("text").attr("x", m.l).attr("y", 13)
    .text(`${data.length} dims where true and reconstructed differ most · FVE ${c.fve.toFixed(3)} · cos ${c.cosine.toFixed(3)}`);
  const g = svg.selectAll("g.d").data(data).join("g").attr("transform", (d) => `translate(${x0(d.dim)},0)`);
  for (const key of ["gold", "recon"]) {
    g.append("rect").attr("x", x1(key)).attr("width", x1.bandwidth())
      .attr("y", (d) => y(Math.max(0, d[key]))).attr("height", (d) => Math.abs(y(d[key]) - y(0)))
      .attr("fill", css(key === "gold" ? "--gold" : "--recon"));
  }
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${y(0)})`)
    .call(d3.axisBottom(x0).tickSize(0)).selectAll("text").attr("transform", "rotate(-45)")
    .attr("text-anchor", "end").attr("dy", "0.9em").attr("dx", "-0.3em");
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4));
  const lg = svg.append("g").attr("transform", `translate(${W - 190},${H - 4})`);
  [["true", "--gold"], ["reconstruction", "--recon"]].forEach(([t, v], i) => {
    lg.append("rect").attr("x", i * 62).attr("y", -9).attr("width", 10).attr("height", 10).attr("fill", css(v));
    lg.append("text").attr("x", i * 62 + 14).attr("y", 0).text(t);
  });
}

// ---------------------------------------------------------------- utils & wiring
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

$("analyze").onclick = tokenize;
$("read").onclick = readToken;
$("colorBy").onchange = renderTokens;
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;
  if (e.key === "ArrowLeft") move(-1);
  else if (e.key === "ArrowRight") move(1);
  else if (e.key === "Enter") readToken();
});
loadRun();
tokenize();
