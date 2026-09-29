// TinyNLA contrast page: what is different in the model's state between two prompts?
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const CLS = { changed_text: "in", shared_text: "shared", beyond_text: "beyond" };
const pct = (x) => `${Math.round(100 * x)}%`;

async function compare() {
  const prompt = $("prompt").value, baseline = $("baseline").value;
  if (!prompt.trim() || !baseline.trim()) return;
  const answer = $("answer").value.trim();
  $("go").disabled = true;
  $("status").textContent = "reading the answer under both prompts… (can take a minute)";
  try {
    const res = await fetch("/api/contrast", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, baseline, answer: answer ? " " + answer : null, n_samples: 4 }),
    });
    if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
    render(await res.json());
    $("status").textContent = "";
  } catch (e) {
    $("status").innerHTML = `<span class="error">${esc(e.message)}</span>`;
  } finally {
    $("go").disabled = false;
  }
}

function chips(rows) {
  if (!rows.length) return `<span class="muted">nothing clear</span>`;
  return rows.map((r) => `<span class="chip ${CLS[r.source]}" title="${esc(r.source_text)}">${esc(r.theme)}` +
    `<small>${pct(r.share_target)} → ${pct(r.share_baseline)}</small></span>`).join("");
}

function render(r) {
  $("result").classList.remove("hidden");
  if (!$("answer").value.trim()) $("answer").value = r.answer.trim();
  $("summary").textContent = r.summary;
  const ms = r.meaning_shift;
  const shift = ms ? (ms.significant
    ? `meaning changed: yes (chance of a fluke ${ms.p_value < 0.001 ? "under 0.1%" : Math.round(100 * ms.p_value) + "%"})`
    : `meaning changed: not clearly (${Math.round(100 * ms.p_value)}% chance this much change is a fluke)`) : "";
  $("coverage").textContent = [shift, `${r.tokens_read} words of the answer read under each prompt`,
    `${r.n_target_reads} vs ${r.n_baseline_reads} reliable reads`, `${r.model}, layer ${r.layer}`]
    .filter(Boolean).join(" · ");
  $("more").innerHTML = chips(r.more);
  $("less").innerHTML = chips(r.less);
  $("lensmore").innerHTML = r.lens_more.length
    ? r.lens_more.map((w) => `<span class="chip">${esc(w.word)}</span>`).join("")
    : `<span class="muted">no clear words</span>`;
  $("limits").innerHTML = r.limits.map((l) => `<li>${esc(l)}</li>`).join("");
}

$("go").addEventListener("click", compare);
