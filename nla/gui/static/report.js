// TinyNLA plain-language report: verdict card first, details on demand.
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const VERDICT = {
  none: "No reliable read", weak: "Weak read", moderate: "Moderate read",
  strong: "Strong read", uncalibrated: "Not calibrated",
};
const LABEL = { echoes_input: "in text", beyond_input: "beyond text", invented_specifics: "invented name/number" };

async function explain() {
  const prompt = $("prompt").value.trim();
  if (!prompt) return;
  const answer = $("answer").value.trim();
  $("go").disabled = true;
  $("status").textContent = "reading the model's internal state at every word of the answer… (can take a minute)";
  try {
    const res = await fetch("/api/report", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, answer: answer ? " " + answer.replace(/^\s+/, "") : null, n_samples: 3 }),
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

function render(r) {
  $("result").classList.remove("hidden");
  if (!$("answer").value.trim()) $("answer").value = r.answer.trim();

  const v = $("verdict");
  v.className = `verdict ${r.overall_strength}`;
  const vr = r.verdict_range;
  const title = vr && vr.is_range
    ? `${VERDICT[vr.low].replace(" read", "")} to ${VERDICT[vr.high].toLowerCase()}`
    : VERDICT[r.overall_strength] ?? r.overall_strength;
  v.textContent = `${title}: ${r.overall_strength_text.split(": ").slice(1).join(": ") || r.overall_strength_text}`;
  $("summary").textContent = r.summary;
  $("coverage").textContent = `${r.coverage_text} ${r.consistency ? r.consistency.text : ""} Model: ${r.model}, layer ${r.layer} (${r.stage}).`;

  $("themes").innerHTML = r.themes.length
    ? r.themes.map((t) => {
        const cls = t.in_input ? "in" : t.named_entity ? "named" : "beyond";
        return `<span class="chip ${cls}" title="appeared in ${Math.round(100 * t.share_of_explanations)}% of reliable reads, at ${t.tokens.length} word(s)">` +
          `${esc(t.named_entity ? t.theme.replace(/\b\w/g, (c) => c.toUpperCase()) : t.theme)}<small>${Math.round(100 * t.share_of_explanations)}%</small>` +
          (t.corroborated ? `<small class="ok" title="also found by the independent logit lens">✓ confirmed</small>` : "") +
          (t.anticipated ? `<small class="early" title="reflected in the model's state ${t.words_ahead} words before the text states it">⏩ ${t.words_ahead} words early</small>` : "") +
          `</span>`;
      }).join("")
    : `<span class="muted">No theme recurred across reliable reads.</span>`;

  const u = $("unusual");
  u.classList.toggle("hidden", !r.unusual);
  if (r.unusual) u.textContent = "⚠ " + r.unusual.text;
  const uv = r.unverified_themes || [];
  $("unverified-card").classList.toggle("hidden", !uv.length);
  $("unverified").innerHTML = uv.map((t) =>
    `<span class="chip unverified ${t.in_input ? "in" : "beyond"}" title="in ${Math.round(100 * t.share_of_rejected_reads)}% of rejected reads, at ${t.tokens.length} word(s)">` +
    `${esc(t.theme)}<small>${Math.round(100 * t.share_of_rejected_reads)}% · unverified</small></span>`).join("");

  $("lenskind").textContent = r.lens_kind || "lens";
  $("leaning").innerHTML = (r.lens_leaning || []).length
    ? r.lens_leaning.map((w) => `<span class="chip" title="in the lens's top 10 at ${Math.round(100 * w.share_of_tokens)}% of words">${esc(w.word)}<small>${Math.round(100 * w.share_of_tokens)}%</small></span>`).join("")
    : `<span class="muted">No second-opinion data.</span>`;

  $("limits").innerHTML = r.limits.map((l) => `<li>${esc(l)}</li>`).join("");

  // specialist detail: answer words shaded by read strength, then every claim
  $("answerline").innerHTML = r.tokens.map((t) =>
    `<span class="w ${t.strength}" title="strength: ${t.strength}, best FVE ${t.best_fve.toFixed(3)}">${esc(t.token)}</span>`).join("");
  $("tokens").innerHTML = r.tokens.map((t) =>
    `<div class="tok-row"><b>${esc(t.token)}</b> · #${t.position} · ${t.strength} · ${t.reliable_reads}/${t.explanations.length} reliable reads` +
    (t.unusual ? ` · state more unusual than ${Math.round(100 * t.unusual.percentile)}% of ordinary states${t.unusual.flag ? " ⚠" : ""}` : "") +
    t.claims.map((c) => `<div class="claim"><span class="lab ${c.label}">${LABEL[c.label]}</span>${esc(c.text)}</div>`).join("") +
    `</div>`).join("");
}

$("go").onclick = explain;
