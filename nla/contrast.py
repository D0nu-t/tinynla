"""
nla/contrast.py

"What is different from normal?" - the contrastive stakeholder view.

People ask "why P rather than Q?" and find contrastive explanations more useful
than complete ones (Miller 2019). Differencing matched hidden states cancels the
shared content and surfaces what separates them (Contrastive Projection, arXiv
2609.09902). In TinyNLA's planted-concept benchmark the change-vs-unsteered
reading was the best detector of faint hidden influences
(training/hidden_state_nla.py).

So: read the SAME answer tokens under two prompts (e.g. with and without a
suspected hint) and report

    more / less    themes whose share of reliable NLA reads changed significantly
                   (two-proportion z-test), each labelled by where it comes from:
                     changed_text   the words that differ between the prompts mention it
                     shared_text    both prompts mention it; the model weights it differently
                     beyond_text    neither prompt mentions it (possibly internal)
    lens_more/less the lens read of the averaged state DIFFERENCE (independent view)

Pure functions over readings (as produced by NLAReader.read_many), so testable
without a model.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional

from nla.calibrate import Calibration
from nla.claims import _norm, content_terms, surface_forms

LIMITS = [
    "A difference means the model's internal state changed, not that the change caused the answer.",
    "Themes marked 'the changed words mention it' may simply echo the text you added.",
    "Only differences that are statistically clear with this many reads are shown; "
    "no difference listed does not prove there is none.",
    "The same answer is read under both prompts, so anything the answer itself says shows up in both "
    "and is not listed as a difference.",
]


def theme_counts(readings: List[Dict], calibration: Optional[Calibration]):
    """stem -> number of reliable explanations mentioning it; plus total and display words."""
    counts, shown, n = Counter(), defaultdict(Counter), 0
    for r in readings:
        for e in r["explanations"]:
            if calibration is not None and calibration.bucket(e["fve"]) == "none":
                continue
            n += 1
            for stem, surface in surface_forms(e["text"]).items():
                counts[stem] += 1
                shown[stem][surface] += 1
    return counts, n, shown


def two_prop_z(k1: int, n1: int, k2: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return 0.0
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return 0.0 if se == 0 else (k1 / n1 - k2 / n2) / se


def benjamini_hochberg(pvals: List[float], q: float) -> List[bool]:
    """Which hypotheses survive a false-discovery-rate of q (Benjamini & Hochberg 1995)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    cutoff = 0
    for rank, i in enumerate(order, 1):
        if pvals[i] <= rank / m * q:
            cutoff = rank
    keep = [False] * m
    for i in order[:cutoff]:
        keep[i] = True
    return keep


def _mentions(stem: str, terms: set, text_lower: str) -> bool:
    # the stemmer is crude ("wedding" -> "wedd", "weddings" -> "wedding"), so also
    # accept the stem as a substring of the text, as nla.report does for in_input
    return stem in terms or stem in text_lower


def _source(stem: str, target: tuple, base: tuple) -> str:
    in_t, in_b = _mentions(stem, *target), _mentions(stem, *base)
    if in_t and in_b:
        return "shared_text"
    if in_t or in_b:
        return "changed_text"
    return "beyond_text"


SOURCE_TEXT = {
    "changed_text": "the words that differ between the two prompts mention this",
    "shared_text": "both prompts mention this; the model weighs it differently",
    "beyond_text": "neither prompt mentions this (possibly the model's own association)",
}


def meaning_shift(target: List[Dict], baseline: List[Dict], calibration: Optional[Calibration],
                  embed: Callable[[List[str]], "torch.Tensor"], n_perm: int = 2000, seed: int = 0) -> Dict:
    """
    ONE test for "did the meaning of the reads change?": at each aligned answer
    position, the difference between the mean sentence embedding of the target's
    and the baseline's reliable explanations; statistic = norm of the average
    difference; null = random sign flips per position (paired permutation test).
    Pairing by position respects the correlation between reads of neighbouring
    words, and a single test has no multiple-comparison problem.
    """
    import torch

    def ok(e):
        return e.get("ok", True) and (calibration is None or calibration.bucket(e["fve"]) != "none")
    diffs = []
    for rt, rb in zip(target, baseline):
        tt = [e["text"] for e in rt["explanations"] if ok(e)]
        tb = [e["text"] for e in rb["explanations"] if ok(e)]
        if tt and tb:
            diffs.append(embed(tt).mean(0) - embed(tb).mean(0))
    if len(diffs) < 3:
        return {"p_value": 1.0, "effect": 0.0, "n_positions": len(diffs), "direction": None}
    d = torch.stack(diffs)
    stat = d.mean(0).norm().item()
    g = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, (n_perm, len(d)), generator=g).float() * 2 - 1
    null = (signs @ d / len(d)).norm(dim=-1)
    p = (1 + (null >= stat).sum().item()) / (1 + n_perm)
    return {"p_value": p, "effect": stat, "n_positions": len(d), "direction": d.mean(0)}


def contrast_themes(target: List[Dict], baseline: List[Dict], calibration: Optional[Calibration],
                    target_text: str, baseline_text: str, min_diff: float = 0.15,
                    fdr: float = 0.05, max_themes: int = 8,
                    embed: Optional[Callable] = None, alpha: float = 0.05, min_share: float = 0.1) -> Dict:
    """
    Two modes.

    With `embed` (a sentence embedder; the default in NLAReader.contrast): a single
    meaning-level permutation test decides WHETHER anything changed (meaning_shift);
    only then are words described, ranked by alignment with the shift direction.
    Pooling synonyms this way ("wedding", "bride", "ceremony") is what gave the
    planted-concept benchmark its power.

    Without it: word-level tests. Every theme word is its own test and a comparison
    involves hundreds of words, so a per-word z >= 2 lets several differences through
    by chance (a neutral placebo produced as many as a real hint did); words are kept
    at a false-discovery rate of `fdr` (Benjamini-Hochberg). Low power: synonyms
    split the evidence.
    """
    if embed is not None:
        # The reliability filter measures "could this state be reconstructed?", which
        # fails for UNUSUAL states: under medium steering every read named the planted
        # concept yet all 48 were marked unreliable. The permutation test controls
        # false alarms by itself (null: ~5%), so meaning mode uses every read.
        calibration = None
    kt, nt, shown_t = theme_counts(target, calibration)
    kb, nb, shown_b = theme_counts(baseline, calibration)
    tt = (content_terms(target_text), target_text.lower())
    bt = (content_terms(baseline_text), baseline_text.lower())
    stems = sorted(set(kt) | set(kb))
    zs = [two_prop_z(kt[st], nt, kb[st], nb) for st in stems]
    keep = benjamini_hochberg([math.erfc(abs(z) / math.sqrt(2)) for z in zs], fdr)
    shift = meaning_shift(target, baseline, calibration, embed) if embed is not None else None

    rows = []
    for stem, z, ok in zip(stems, zs, keep):
        a, b = kt[stem], kb[stem]
        diff = (a / nt if nt else 0) - (b / nb if nb else 0)
        if shift is None:
            if not ok or abs(diff) < min_diff:
                continue
        elif shift["p_value"] >= alpha or max(a / max(1, nt), b / max(1, nb)) < min_share or diff == 0:
            continue
        word = (shown_t[stem] + shown_b[stem]).most_common(1)[0][0]
        src = _source(stem, tt, bt)
        rows.append({"theme": word, "stem": stem, "share_target": round(a / max(1, nt), 3),
                     "share_baseline": round(b / max(1, nb), 3), "difference": round(diff, 3),
                     "z": round(z, 2), "source": src, "source_text": SOURCE_TEXT[src]})
    if shift is not None and rows:
        # rank by alignment with the meaning shift; keep only words that point its way
        dirn = shift["direction"]
        dirn = dirn / dirn.norm()
        align = (embed([r["theme"] for r in rows]) @ dirn).tolist()
        for r, a in zip(rows, align):
            r["alignment"] = round(a, 3)
        rows = [r for r in rows if (r["alignment"] > 0) == (r["difference"] > 0)]
        rows.sort(key=lambda r: -abs(r["alignment"]))
    else:
        rows.sort(key=lambda r: -abs(r["difference"]))
    more = [r for r in rows if r["difference"] > 0][:max_themes]
    less = [r for r in rows if r["difference"] < 0][:max_themes]
    # non-contrastive view for comparison: the target's most frequent themes
    top = [(shown_t[st].most_common(1)[0][0], kt[st] / max(1, nt)) for st, _ in kt.most_common(max_themes)]
    out = {"more": more, "less": less, "n_target_reads": nt, "n_baseline_reads": nb,
           "target_top": [{"theme": w, "share": round(sh, 3)} for w, sh in top]}
    if shift is not None:
        out["meaning_shift"] = {"p_value": round(shift["p_value"], 4), "effect": round(shift["effect"], 4),
                                "n_positions": shift["n_positions"], "significant": shift["p_value"] < alpha}
    return out


def contrast_summary(res: Dict) -> str:
    def words(rows):
        return ", ".join(f"“{r['theme']}”" for r in rows)
    if not res["more"] and not res["less"]:
        return ("No clear difference: with this many reads, the model's state while writing this answer "
                "looks about the same under both prompts.")
    parts = []
    beyond = [r for r in res["more"] if r["source"] == "beyond_text"]
    echoed = [r for r in res["more"] if r["source"] != "beyond_text"]
    if echoed:
        parts.append(f"With your prompt the model is thinking more about {words(echoed)}, "
                     f"which the prompts themselves mention.")
    if beyond:
        parts.append(f"It is also thinking more about {words(beyond)}, which neither prompt mentions.")
    if res["less"]:
        parts.append(f"Compared with the baseline it is thinking less about {words(res['less'])}.")
    if res.get("lens_more"):
        parts.append("The independent lens view of the difference leans towards "
                     + ", ".join(f"“{w['word']}”" for w in res["lens_more"][:4]) + ".")
    return " ".join(parts)
