"""
nla/report.py

Answer-level "what was the model thinking?" report (stakeholder report, idea 2).

Pure function over per-token readings, so it is testable without a model:

    readings = [{"position", "token", "explanations": [{"text", "fve"}, ...]}, ...]
    build_report(readings, input_text, calibration) -> dict

Aggregation follows the NLA paper's reading heuristics: trust THEMES that
recur across samples and adjacent tokens, not individual claims; and
explanations should be selective (Miller 2019), so only recurring themes
are surfaced, each with its evidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional

from nla.calibrate import BUCKET_TEXT, Calibration, answer_verdict, verdict_range
from nla.claims import LABEL_TEXT, _norm, content_terms, specifics, surface_forms, tag_explanation
from nla.unusual import unusual_summary

LIMITS = [
    "These are reconstructions of the model's internal state from a small, weak decoder, not the model's own words.",
    "Specific names, places and numbers in explanations are usually invented; trust recurring themes, not details.",
    "A theme 'not in the text' may be the model's own expectation, or noise; treat it as a lead to check, not a fact.",
    "Signal strength compares each read with reads of the wrong text; 'none' means no evidence either way.",
    "'Seen before it was written' can mean the topic simply made it predictable: in tests this reader names the "
    "upcoming word 7x more often than chance, but about as often as a summary of the text so far does.",
]


def build_report(
    readings: List[Dict],
    input_text: str,
    calibration: Optional[Calibration],
    min_share: float = 0.25,
    max_themes: int = 8,
    token_texts: Optional[List[str]] = None,
    flag_quantile: float = 0.99,
) -> Dict:
    """
    token_texts: every token of `input_text`, in order. When given, themes are
    checked for ANTICIPATION: reflected in reads before the text first states
    them (planning-ahead evidence; NLA paper rhymes, Future Lens - Pal et al. 2023).
    """
    def reliable(e):   # passes the shuffled control (or no calibration to judge by)
        return calibration is None or calibration.bucket(e["fve"]) != "none"

    # themes come only from reads that beat descriptions of other text
    all_expl = [(r, e) for r in readings for e in r["explanations"] if reliable(e)]
    n_expl = len(all_expl)

    # ---- per-token strength and claims
    tokens = []
    for r in readings:
        fves = [e["fve"] for e in r["explanations"]]
        best = max(fves) if fves else float("nan")
        bucket = calibration.bucket(best) if calibration else "uncalibrated"
        claims = [c.to_dict() for e in r["explanations"] for c in tag_explanation(e["text"], input_text)]
        tokens.append({
            "position": r["position"],
            "token": r["token"],
            "best_fve": best,
            "strength": bucket,
            "reliable_reads": sum(reliable(e) for e in r["explanations"]),
            "unusual": r.get("unusual"),
            "lens": r.get("lens", []),
            "explanations": r["explanations"],
            "claims": claims,
        })

    # ---- themes: content terms recurring across explanations
    support = defaultdict(set)      # term -> explanation indices
    where = defaultdict(set)        # term -> token positions
    shown = defaultdict(Counter)    # term -> surface words, for display
    named = defaultdict(int)        # term -> mentions written as a proper name
    for i, (r, e) in enumerate(all_expl):
        names = {w.lower() for sp in specifics(e["text"]) for w in sp.split()}
        for t, surface in surface_forms(e["text"]).items():
            support[t].add(i)
            where[t].add(r["position"])
            shown[t][surface] += 1
            named[t] += surface in names
    _merge_variants(support, where, shown, named)

    # stems of words the lens ranked highly anywhere in the answer
    # corroboration uses the logit lens when present (its agreement is the stronger
    # predictor of a correct read: 3.0x vs 1.8x for the tuned lens), else any lens
    check_key = "lens_check" if any(r.get("lens_check") for r in readings) else "lens"
    lens_stems = {_norm(w["word"]) for r in readings for w in r.get(check_key, [])}
    has_lens = any(r.get(check_key) for r in readings)

    input_terms = content_terms(input_text)
    input_lower = input_text.lower()
    themes = []
    for term, idx in support.items():
        share = len(idx) / max(1, n_expl)
        if share < min_share or len(idx) < 2:
            continue
        themes.append({
            "theme": shown[term].most_common(1)[0][0],
            "stem": term,
            "share_of_explanations": round(share, 3),
            "tokens": sorted(where[term]),
            "in_input": term in input_terms or term in input_lower,
            "named_entity": named[term] > len(idx) / 2,
            "mean_fve": round(sum(all_expl[i][1]["fve"] for i in idx) / len(idx), 4),
            **_anticipation(term, sorted(where[term]), token_texts),
            "corroborated": has_lens and term in lens_stems,
        })
    themes.sort(key=lambda t: (-t["share_of_explanations"], -t["mean_fve"]))
    themes = themes[:max_themes]

    consistency = _split_half(readings, reliable, min_share, max_themes)

    reliable_tokens = sum(t["reliable_reads"] > 0 for t in tokens)
    typical = calibration.reliable_rate if calibration else None
    overall = (answer_verdict(reliable_tokens, len(tokens), typical)
               if calibration else "uncalibrated")
    vr = verdict_range(reliable_tokens, len(tokens), typical) if calibration else None

    unverified = _unverified_themes(readings, reliable, input_text, {t["stem"] for t in themes})
    unusual = unusual_summary([bool((r.get("unusual") or {}).get("flag")) for r in readings], flag_quantile)

    return {
        "unverified_themes": unverified,
        "unusual": unusual,
        "overall_strength": overall,
        "overall_strength_text": _strength_text(overall, vr),
        "verdict_range": vr,
        "summary": _narrative(themes, overall, unverified),
        "reliable_tokens": reliable_tokens,
        "typical_reliable_rate": typical,
        "coverage_text": (f"Reliable reads on {reliable_tokens} of {len(tokens)} answer tokens"
                          + (f" (this tool typically gets {typical:.0%})." if typical else ".")),
        "themes": themes,
        "tokens": tokens,
        "label_text": LABEL_TEXT,
        "limits": LIMITS,
        "n_tokens_read": len(readings),
        "n_explanations": n_expl,
        "lens_leaning": _leaning(readings),
        "consistency": consistency,
        "n_explanations_total": sum(len(r["explanations"]) for r in readings),
    }


def _narrative(themes: List[Dict], overall: str, unverified: Optional[List[Dict]] = None) -> str:
    if overall == "none" or not themes:
        if unverified:
            return ("None of the reads could be verified, but they kept returning to "
                    + _join([t["theme"] for t in unverified]) + ". Treat this as unverified: it can mean one "
                    "idea dominated the model's state (our check penalises explanations that leave the rest "
                    "out), or it can be noise.")
        return ("The reader found no reliable signal for this answer: its explanations were no "
                "better than descriptions of unrelated text. Do not draw conclusions from them.")
    inside = [t["theme"] for t in themes if t["in_input"]]
    beyond = [t["theme"] for t in themes if not t["in_input"] and not t["named_entity"]]
    names = [t["theme"].title() for t in themes
             if not t["in_input"] and t["named_entity"] and not t.get("corroborated")]
    real_names = [t["theme"].title() for t in themes
                  if not t["in_input"] and t["named_entity"] and t.get("corroborated")]
    parts = []   # the verdict is shown separately; the summary says what, not how strong
    if inside:
        parts.append("While producing this answer, the model's internal state repeatedly reflected "
                     + _join(inside) + " (all also present in the text).")
    if beyond:
        parts.append("It also repeatedly reflected " + _join(beyond)
                     + ", which the text does not mention; this may be the model's own expectation or inference, "
                       "or noise. Check before relying on it.")
    confirmed = [t["theme"] for t in themes if t.get("corroborated") and not t["named_entity"]]
    if confirmed:
        parts.append("A second, independent method (a lens that reads the same internal state "
                     "without our decoder) also pointed to " + _join(confirmed) + ", so these are the "
                     "most trustworthy themes.")
    ahead = [t for t in themes if t.get("anticipated")]
    if ahead:
        parts.append("Some ideas showed up in the model's internal state before it wrote them: "
                     + _join([f"{t['theme']} ({t['words_ahead']} words early)" for t in ahead])
                     + ". This suggests it was already heading there.")
    if real_names:
        parts.append("The model genuinely associated the text with " + _join(real_names) + ": both our decoder and "
                     "the independent lens found it, though the text never mentions it. That shows what the model "
                     "linked the text to, not that it is true.")
    if names:
        parts.append("Explanations also kept naming " + _join(names) + " (found by our decoder only), which the text never mentions. "
                     "Treat this as an association the model made (the kind of setting it linked the text to), "
                     "not as a fact it knows.")
    return " ".join(parts)


def _strength_text(overall: str, vr: Optional[Dict]) -> str:
    if not vr or not vr["is_range"]:
        return BUCKET_TEXT[overall]
    lo, hi = vr["interval"]
    return (f"{vr['low']} to {vr['high']} signal: with this few words we can't pin it down more precisely "
            f"(reliable share {lo:.0%}-{hi:.0%}, 95% range)")


def _recurring_stems(expls: List[str], min_share: float, max_themes: int) -> set:
    support = Counter()
    for text in expls:
        support.update(set(content_terms(text)))
    n = max(1, len(expls))
    ranked = [t for t, c in support.most_common() if c >= 2 and c / n >= min_share]
    return set(ranked[:max_themes])


def _split_half(readings: List[Dict], reliable, min_share: float, max_themes: int) -> Dict:
    """
    Split-half reliability of the theme list (psychometrics): themes from even-
    vs odd-numbered samples of every token, overlap = Jaccard, stepped up to full
    length with the Spearman-Brown formula 2r / (1 + r). Estimates how much of the
    theme list would come back if the model were read again - without re-running it.
    """
    halves = ([], [])
    for r in readings:
        for k, e in enumerate(r["explanations"]):
            if reliable(e):
                halves[k % 2].append(e["text"])
    if min(len(h) for h in halves) < 2:
        return {"score": None, "label": "unknown",
                "text": "Not enough reads to estimate how repeatable this report is (use 2+ samples)."}
    a, b = (_recurring_stems(h, min_share, max_themes) for h in halves)
    if not a and not b:
        return {"score": None, "label": "unknown", "text": "No recurring themes to compare."}
    j = len(a & b) / len(a | b)
    sb = 2 * j / (1 + j)
    label = "high" if sb >= 0.6 else "medium" if sb >= 0.35 else "low"
    return {"score": round(sb, 3), "split_half_jaccard": round(j, 3), "label": label,
            "text": f"Repeatability: {label}. If the model were read again, roughly {sb:.0%} of these themes "
                    f"would be expected to recur (split-half estimate)."}


def _leaning(readings: List[Dict], k: int = 8) -> List[Dict]:
    """Words the lens most often ranked top-10 across the answer."""
    c = Counter(w["word"].lower() for r in readings for w in r.get("lens", []))
    n = max(1, len(readings))
    return [{"word": w, "share_of_tokens": round(m / n, 3)} for w, m in c.most_common(k)]


def _anticipation(term: str, read_positions: List[int], token_texts: Optional[List[str]]) -> Dict:
    """First position the text states `term` vs first position reads reflect it."""
    if not token_texts or not read_positions:
        return {"anticipated": False, "words_ahead": None}
    first_read = read_positions[0]
    first_text = None
    for pos in range(len(token_texts)):
        if term in content_terms("".join(token_texts[: pos + 1])):
            first_text = pos
            break
    if first_text is not None and first_text > first_read:
        return {"anticipated": True, "words_ahead": first_text - first_read}
    return {"anticipated": False, "words_ahead": None}


def _merge_variants(support, where, shown, named, min_prefix: int = 5) -> None:
    """Merge terms where one is a prefix of the other (syria/syrian, conflict/conflicts)."""
    terms = sorted(support, key=len)
    for i, short in enumerate(terms):
        if short not in support or len(short) < min_prefix:
            continue
        for long in terms[i + 1:]:
            if long in support and long.startswith(short) and len(long) - len(short) <= 3:
                support[short] |= support.pop(long)
                where[short] |= where.pop(long)
                shown[short] += shown.pop(long)
                named[short] += named.pop(long, 0)


def _join(words: List[str]) -> str:
    quoted = [f"'{w}'" for w in words]
    return quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " and " + quoted[-1]


# words the verbalizer uses everywhere; they recur in rejected reads of ordinary text too
GENERIC = {"different", "used", "number", "specific", "specifically", "various", "overall", "certain",
           "general", "type", "typ", "variou", "includ", "involv", "describ", "mention", "discuss"}


def _unverified_themes(readings: List[Dict], reliable, input_text: str, exclude: set,
                       max_themes: int = 5, strong_share: float = 0.6) -> List[Dict]:
    """
    Themes that recur across reads the reliability check REJECTED.

    The check asks whether the reconstructor can rebuild the whole state from the
    explanation, so an explanation that is right but incomplete fails it: when one
    idea dominates the state (planted-concept steering at medium strength), every
    read named the concept and every read was rejected (/loop iterations 9-10).
    Recurrence across samples and positions is independent evidence (the NLA
    paper's reading heuristic), so strongly recurring themes among rejected reads
    are shown - labelled unverified - instead of being silently dropped.
    """
    rejected = [(r, e) for r in readings for e in r["explanations"] if e.get("ok", True) and not reliable(e)]
    support, where, shown = defaultdict(int), defaultdict(set), defaultdict(Counter)
    for r, e in rejected:
        for t, surface in surface_forms(e["text"]).items():
            support[t] += 1
            where[t].add(r["position"])
            shown[t][surface] += 1
    n = len(rejected)
    input_terms, input_lower = content_terms(input_text), input_text.lower()
    out = []
    for t, k in support.items():
        if t in exclude or t in GENERIC or not _consistent_enough(k / max(1, n), len(where[t]), n):
            continue
        out.append({"theme": shown[t].most_common(1)[0][0], "stem": t,
                    "share_of_rejected_reads": round(k / n, 3), "tokens": sorted(where[t]),
                    "in_input": t in input_terms or t in input_lower})
    out.sort(key=lambda x: -x["share_of_rejected_reads"])
    # A real takeover shows up as a CLUSTER of related themes ("goal, points, scoring,
    # match"); false shows on ordinary text were mostly one lone vague word ("someone",
    # "idea"). Show the card only for 2+ themes or one very strong theme (iteration 11:
    # ordinary false shows 20% -> 7%, see stakeholder_report_iterations).
    if len(out) < 2 and not (out and out[0]["share_of_rejected_reads"] >= strong_share):
        return []
    return out[:max_themes]


def _consistent_enough(share: float, n_positions: int, n_rejected: int) -> bool:
    """
    Should a theme found only in REJECTED reads be shown to a stakeholder?

    share         fraction of rejected reads that mention the theme (0-1)
    n_positions   number of distinct answer words whose reads mention it
    n_rejected    total rejected reads (e.g. 16 words x 3 samples = up to 48)
    """
    # Calibrated on 6 steered vs 6 ordinary prompts (16 words x 3 samples):
    #   takeovers: 26-48 rejected reads, concept at 7-15 words, share 0.37-0.81
    #   ordinary:   6-23 rejected reads, top theme at 1-7 words, share up to 0.67
    # Share alone does not separate them; the amount of rejected evidence and the
    # breadth across answer words do.
    return n_rejected >= 24 and n_positions >= 6 and share >= 0.35

