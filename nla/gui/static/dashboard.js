// TinyNLA training dashboard — polls the log-derived API; never touches the GPU.
"use strict";

const $ = (id) => document.getElementById(id);
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const STAGE_NAMES = { datagen: "Data", ar_sft: "Reconstructor SFT", av_sft: "Verbalizer SFT", rl: "RL (GRPO)", eval: "Eval" };

async function get(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

const fmtDur = (s) => {
  if (s == null) return "?";
  s = Math.round(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h}h ${String(m).padStart(2, "0")}m` : `${m}m ${String(s % 60).padStart(2, "0")}s`;
};

// ------------------------------------------------------------------ summary
async function refreshSummary() {
  let s;
  try { s = await get("/api/dashboard/summary"); } catch (e) { $("run").textContent = e.message; return; }

  const stale = s.run.log_age_s != null && s.run.log_age_s > 180 && !s.finished;
  $("run").innerHTML =
    `<span>run <b>${s.run.name}</b></span><span>model <b>${s.run.target}</b> · layer <b>${s.run.layer}</b></span>` +
    `<span>samples <b>${s.run.num_samples ?? "?"}</b></span>` +
    `<span>injection scale <b>${s.run.injection_scale ? (+s.run.injection_scale).toFixed(1) : "—"}</b></span>` +
    `<span class="${stale ? "stale" : ""}">log updated ${fmtDur(s.run.log_age_s)} ago${stale ? " — stalled?" : ""}</span>`;

  $("stages").innerHTML = Object.entries(s.stages).map(([k, v], i) =>
    (i ? `<span class="arrow">→</span>` : "") + `<span class="stage ${v}" title="${v}">${STAGE_NAMES[k]}</span>`).join("");

  const cur = s.current && s.progress[s.current];
  if (s.failed) {
    $("progress").innerHTML = `<span class="fail">Stage ${STAGE_NAMES[s.failed]} failed — see log messages.</span>`;
  } else if (s.finished) {
    $("progress").innerHTML = `<span class="pass">Pipeline complete.</span>`;
  } else if (cur) {
    const post = Object.entries(cur.postfix).map(([k, v]) => `${k} <b>${v}</b>`).join(" · ");
    $("progress").innerHTML =
      `<b>${STAGE_NAMES[s.current]}</b> · ${cur.desc} · ${cur.n} / ${cur.total} (${cur.pct}%)` +
      `<div class="bar-outer"><div class="bar-inner" style="width:${cur.pct}%"></div></div>` +
      `elapsed <b>${fmtDur(cur.elapsed_s)}</b> · remaining in this bar <b>${fmtDur(cur.eta_s)}</b> · ` +
      `${cur.sec_per_it ? cur.sec_per_it.toFixed(2) + " s/it" : ""}${post ? " · " + post : ""}`;
  } else {
    $("progress").textContent = s.current ? `${STAGE_NAMES[s.current]} starting…` : "waiting for the pipeline log…";
  }

  renderGpu(s.gpu);
  renderFinal(s.final_eval);
  const box = $("messages");
  const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 5;
  box.textContent = s.messages.join("\n");
  if (atBottom) box.scrollTop = box.scrollHeight;
}

function gauge(label, value, max, unit, warn, hot) {
  const pct = Math.min(100, (100 * value) / max);
  const cls = value >= hot ? "hot" : value >= warn ? "warm" : "";
  return `<div class="gauge">${label} <b>${value.toFixed(0)}${unit}</b>${max !== 100 ? ` / ${max.toFixed(0)}${unit}` : ""}` +
    `<div class="bar-outer"><div class="bar-inner ${cls}" style="width:${pct}%"></div></div></div>`;
}

function renderGpu(g) {
  if (!g) { $("gpu").innerHTML = `<p class="empty">nvidia-smi not available</p>`; return; }
  $("gpu").innerHTML = `<div class="muted" style="margin-bottom:8px">${g.name}</div>` +
    gauge("utilization", g.util, 100, "%", 101, 101) +
    gauge("memory", g.mem_used, g.mem_total, " MB", 0.9 * g.mem_total, 0.97 * g.mem_total) +
    gauge("temperature", g.temp, 100, "°C", 80, 88);
}

function renderFinal(ev) {
  if (!ev) return;
  const rows = [["verbalizer FVE", ev.av_fve], ["shuffled", ev.av_fve_shuffled], ["input summary", ev.summary_fve],
    ["input summary (SFT reconstructor)", ev.summary_fve_sft_ar], ["mean direction (fair baseline)", ev.mean_direction_fve],
    ["raw mean", ev.mean_fve], ["parse ok", ev.parse_ok], ["unique", ev.unique_frac]].filter(([, v]) => v != null);
  $("final").innerHTML =
    `<div class="kv">${rows.map(([k, v]) => `<span>${k}</span><b>${(+v).toFixed(4)}</b>`).join("")}</div>` +
    `<ul class="checks">${Object.entries(ev.checks).map(([k, v]) =>
      `<li><span class="${v ? "pass" : "fail"}">${v ? "PASS" : "FAIL"}</span> ${k.replace(/_/g, " ")}</li>`).join("")}</ul>` +
    `<p class="note">${ev.n_test} held-out activations · ${ev.n_samples} sample(s) each</p>`;
}

// ------------------------------------------------------------------ charts
function ema(values, alpha) {
  let m = null;
  return values.map((v) => (m = m === null ? v : alpha * v + (1 - alpha) * m));
}

function thin(rows, max = 700) {
  if (rows.length <= max) return rows;
  const stride = Math.ceil(rows.length / max);
  return rows.filter((_, i) => i % stride === 0 || i === rows.length - 1);
}

// series: [{name, color, points:[{x,y}], dashed, raw, width}]
function lineChart(el, series, opt = {}) {
  el.innerHTML = "";
  series = series.filter((s) => s.points.length);
  if (!series.length) { el.innerHTML = `<p class="empty">${opt.empty ?? "no data yet"}</p>`; return; }
  const W = opt.width ?? 520, H = opt.height ?? 190, m = { t: 8, r: 12, b: 26, l: 46 };
  const all = series.flatMap((s) => s.points);
  const xs = d3.extent(all, (p) => p.x);
  if (xs[0] === xs[1]) xs[1] = xs[0] + 1;
  const x = (opt.logx ? d3.scaleSymlog().constant(10) : d3.scaleLinear()).domain(xs).range([m.l, W - m.r]);
  let ys = d3.extent(all.concat((opt.refs ?? []).map((r) => ({ y: r.y }))), (p) => p.y);
  if (opt.yDomain) ys = opt.yDomain;
  const pad = (ys[1] - ys[0]) * 0.08 || 0.05;
  const y = d3.scaleLinear().domain([ys[0] - pad, ys[1] + pad]).nice().range([H - m.b, m.t]);

  const legend = document.createElement("div");
  legend.className = "legend-row";
  legend.innerHTML = series.filter((s) => !s.raw).map((s) =>
    `<span><i style="background:${s.color};${s.dashed ? "opacity:.6" : ""}"></i>${s.name}</span>`).join("") +
    (opt.refs ?? []).map((r) => `<span><i style="background:${r.color};opacity:.5"></i>${r.name}</span>`).join("");
  el.appendChild(legend);

  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(6, "~s"));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5));
  for (const r of opt.refs ?? []) {
    svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(r.y)).attr("y2", y(r.y))
      .attr("stroke", r.color).attr("stroke-dasharray", "4 4").attr("opacity", 0.6);
  }
  const line = d3.line().x((p) => x(p.x)).y((p) => y(p.y));
  for (const s of series) {
    svg.append("path").datum(s.points).attr("class", s.raw ? "raw" : "line").attr("d", line)
      .attr("stroke", s.color).attr("stroke-dasharray", s.dashed ? "6 4" : null);
    if (s.dots) {
      svg.selectAll(null).data(s.points).join("circle").attr("cx", (p) => x(p.x)).attr("cy", (p) => y(p.y))
        .attr("r", 3.5).attr("fill", s.color).append("title").text((p) => `${s.name}: ${p.y.toFixed(4)} @ ${p.x}`);
    }
  }
  if (opt.xlabel) svg.append("text").attr("x", W - m.r).attr("y", H - m.b - 5).attr("text-anchor", "end")
    .attr("class", "muted").style("font-size", "11px").text(opt.xlabel);
}

// SFT bars are "epoch k" with n/total -> one global step axis
const globalStep = (r) => {
  const k = +(/epoch (\d+)/.exec(r.bar)?.[1] ?? 0);
  return k * r.total + r.n;
};

async function refreshMetrics() {
  let m;
  try { m = await get("/api/dashboard/metrics"); } catch { return; }
  const A = css("--accent"), G = css("--good"), W = css("--warn"), B = css("--bad"), M = css("--muted");

  // --- RL validation: the key chart
  const rlVal = m.jsonl.rl.filter((r) => r.split === "val");
  const pts = (rows, k) => rows.map((r) => ({ x: r.step, y: r[k] })).filter((p) => p.y != null);
  lineChart($("rlval"), [
    { name: "verbalizer explanations", color: A, points: pts(rlVal, "av_fve"), dots: true },
    { name: "input summaries", color: G, points: pts(rlVal, "summary_fve"), dashed: true, dots: true },
    { name: "shuffled", color: M, points: pts(rlVal, "av_fve_shuffled"), dashed: true, dots: true },
  ], { xlabel: "RL step", width: 760, height: 220, empty: "first RL evaluation pending" });

  // --- RL live reward / KL (from log postfix; jsonl once flushed)
  const live = thin(m.live.rl ?? []);
  const rx = live.map((r) => r.n);
  const rew = live.map((r) => r.r);
  const rewEma = ema(rew, 0.08);
  lineChart($("reward"), [
    { name: "raw", color: A, raw: true, points: rx.map((x, i) => ({ x, y: rew[i] })) },
    { name: "reward (EMA)", color: A, points: rx.map((x, i) => ({ x, y: rewEma[i] })) },
  ], { xlabel: "RL step", empty: "RL not started" });
  lineChart($("kl"), [
    { name: "KL to SFT", color: W, points: live.map((r) => ({ x: r.n, y: r.kl })) },
    { name: "parse ok", color: G, points: live.map((r) => ({ x: r.n, y: r.ok })) },
  ], { xlabel: "RL step", empty: "RL not started" });

  // --- SFT stages
  const sftLoss = (rows) => {
    const t = thin(rows.filter((r) => r.loss != null)).map((r) => ({ x: globalStep(r), y: r.loss }));
    const e = ema(t.map((p) => p.y), 0.05);
    return [{ name: "raw", color: A, raw: true, points: t },
            { name: "loss (EMA)", color: A, points: t.map((p, i) => ({ x: p.x, y: e[i] })) }];
  };
  // the log may only cover a later run: fall back to the checkpoint's saved train losses
  const saved = (rows) => rows.filter((r) => r.loss != null && r.split !== "val")
                              .map((r) => ({ bar: "", total: 0, n: r.step, loss: r.loss }));
  const lossRows = (s) => (m.live[s]?.length ? m.live[s] : saved(m.jsonl[s] ?? []));
  lineChart($("arloss"), sftLoss(lossRows("ar_sft")), { xlabel: "batch", height: 150 });
  lineChart($("avloss"), sftLoss(lossRows("av_sft")), { xlabel: "batch", height: 150 });

  const arVal = m.jsonl.ar_sft.filter((r) => r.split === "val");
  lineChart($("arval"), [
    { name: "val FVE", color: A, points: pts(arVal, "fve"), dots: true },
    { name: "shuffled", color: M, points: pts(arVal, "fve_shuffled"), dashed: true, dots: true },
  ], { xlabel: "optimizer step", height: 150, refs: [{ name: "mean = 0", y: 0, color: M }] });

  const avVal = m.jsonl.av_sft.filter((r) => r.split === "val");
  lineChart($("avval"), [
    { name: "verbalizer", color: A, points: pts(avVal, "av_fve"), dots: true },
    { name: "input summaries", color: G, points: pts(avVal, "summary_fve"), dashed: true, dots: true },
    { name: "shuffled", color: M, points: pts(avVal, "av_fve_shuffled"), dashed: true, dots: true },
  ], { xlabel: "optimizer step", height: 150 });
}

refreshSummary();
refreshMetrics();
setInterval(refreshSummary, 10000);
setInterval(refreshMetrics, 30000);
