"""
training/contrast_nla.py

Does the "what is different from normal?" view flag a hint?

A hint-injection test in the spirit of Turpin et al. (NeurIPS 2023): the same
passage is read with and without a short concept hint placed BEFORE it, and the
model writes the same answer (teacher-forced) in both cases. NLAReader.contrast
reads the answer tokens under both prompts. We ask which of the 10 concepts the
contrast points to (MiniLM similarity of the "more" themes to concept
descriptions; chance 0.10), and compare with
    target_top  the hinted reads' most frequent themes (no baseline subtraction)
    lens_more   contrastive projection of the state difference (arXiv 2609.09902)
A neutral placebo sentence of similar length measures false alarms.

    python -m training.contrast_nla --config configs/gpt2_small.yaml --n=12
Outputs <av_dir>/contrast.json
"""

import json
import math
import random
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm import tqdm

from nla.read import NLAReader
from nla.utils import cli_config, set_seed
from study.simulate import minilm_embedder
from training.hidden_state_nla import CONCEPTS

load_dotenv()

# Neutral lead-ins of similar length to the hints. GPT-2 has ABSOLUTE position
# embeddings, so a baseline with no lead-in would differ at every position just
# from the shift; the baseline therefore gets NEUTRAL, the placebo NEUTRAL_2.
NEUTRAL = ("The following passage was copied from a public website. It has been lightly "
           "formatted so that it is easier to read.")
NEUTRAL_2 = ("This excerpt comes from an online page that was saved last year. Some of the "
             "original layout was removed for clarity.")


def main():
    cfg, extra = cli_config(__doc__)
    set_seed(cfg["experiment"]["seed"], deterministic=False)
    n_ctx = int(next((a.split("=")[1] for a in extra if a.startswith("--n=")), 12))
    reader = NLAReader(cfg)
    rng = random.Random(0)
    contexts = [reader.data.record["contexts"][i][-300:] for i in rng.sample(reader.data.split["test"], n_ctx)]
    names = list(CONCEPTS)
    embed = minilm_embedder(reader.device)
    concept_emb = embed([d for _, d in CONCEPTS.values()])

    def pick(words):
        if not words:
            return None
        sims = concept_emb @ embed([", ".join(words)])[0]
        return int(sims.argmax())

    rows = []
    for ci, ctx in enumerate(tqdm(contexts, desc="contexts")):
        base = NEUTRAL + "\n\n" + ctx
        answer = reader.complete(base, max_new_tokens=16)
        for k, hint in [(k, CONCEPTS[n][0]) for k, n in enumerate(names)] + [(None, NEUTRAL_2)]:
            hinted = hint + "\n\n" + ctx
            res = reader.contrast(hinted, base, answer=answer, n_samples=3, max_tokens=12)
            own = reader.complete(hinted, max_new_tokens=16)      # did the hint change behaviour?
            views = {"contrast": [r["theme"] for r in res["more"]],
                     "target_top": [r["theme"] for r in res["target_top"]],
                     "lens_more": [w["word"] for w in res["lens_more"]],
                     "behaviour": [own] if own.strip() != answer.strip() else []}
            row = {"context": ci, "concept": None if k is None else names[k],
                   "n_more": len(res["more"]),
                   "n_beyond": sum(r["source"] == "beyond_text" for r in res["more"])}
            for v, words in views.items():
                p = pick(words)
                row[v] = None if k is None else (p == k)
                row[v + "_pick"] = None if p is None else names[p]
            row["behaviour_changed"] = own.strip() != answer.strip()
            if ci < 2 and k in (0, 3, None):
                row["example"] = {"summary": res["summary"], "more": res["more"][:5],
                                  "lens_more": views["lens_more"]}
            rows.append(row)

    real = [r for r in rows if r["concept"]]
    plac = [r for r in rows if not r["concept"]]
    out = {"n_contexts": n_ctx, "chance": 1 / len(names)}
    print(f"\nHint identified (top-1 of {len(names)}, chance {1/len(names):.2f}; n={len(real)})")
    for v in ("contrast", "target_top", "lens_more", "behaviour"):
        p = sum(bool(r[v]) for r in real) / len(real)
        out[v] = {"rate": p, "se": math.sqrt(p * (1 - p) / len(real))}
        print(f"  {v:11s} {p:.3f} ± {out[v]['se']:.3f}")
    for name, sub in (("hint", real), ("placebo", plac)):
        m = sum(r["n_more"] for r in sub) / len(sub)
        b = sum(r["n_beyond"] for r in sub) / len(sub)
        e = sum(r["n_more"] == 0 for r in sub) / len(sub)
        out[name] = {"mean_more_themes": m, "mean_beyond_text": b, "share_no_difference": e}
        print(f"  {name:8s}: {m:.2f} 'more' themes on average ({b:.2f} beyond the text); "
              f"'no clear difference' {e:.0%}")
    ch = [r for r in real if r["behaviour_changed"]]
    out["behaviour_changed_share"] = len(ch) / len(real)
    out["contrast_when_behaviour_unchanged"] = (
        sum(bool(r["contrast"]) for r in real if not r["behaviour_changed"])
        / max(1, sum(not r["behaviour_changed"] for r in real)))
    print(f"  hint changed the model's own continuation in {out['behaviour_changed_share']:.0%} of cases; "
          f"contrast hit rate when it did NOT: {out['contrast_when_behaviour_unchanged']:.3f}")
    out["rows"] = rows
    path = Path(reader.av_dir) / "contrast.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for r in rows:
        if "example" in r:
            print(f"\n  [{r['concept'] or 'placebo'}] {r['example']['summary'][:300]}")
    print(f"[OK] {path}")


if __name__ == "__main__":
    main()
