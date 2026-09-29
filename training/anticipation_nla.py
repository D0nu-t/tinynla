"""
training/anticipation_nla.py

Does the NLA reveal what the model is ABOUT to say?

For held-out activations whose true next token is a content word, measure
how often the word appears in:
    av        AV explanations of the activation (model state only)
    summary   the warm-start summary of the text so far (cannot see the future)
    shuffled  AV explanations of a DIFFERENT activation (controls for words
              that are simply common in explanations)
    prompt    the text so far itself (the word was already mentioned)

av > shuffled and av > summary is evidence that the layer-K state encodes
upcoming content and the verbalizer puts it into words - a small-scale
analogue of the NLA paper's planning results; related to Future Lens
(Pal et al., CoNLL 2023), which decodes subsequent tokens from one hidden state.

    python -m training.anticipation_nla --config configs/gpt2_small.yaml
"""

import json
import math
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer

from nla.av import ActivationVerbalizer
from nla.claims import STOP, _norm, content_terms
from nla.evaluate import derange
from nla.lens import Lens, top_words
from nla.runs import load_data, load_target, model_dirs
from nla.utils import cli_config, resolve_device, set_seed

load_dotenv()


def rate(hits):
    n = len(hits)
    p = sum(hits) / n if n else float("nan")
    se = math.sqrt(p * (1 - p) / n) if n else float("nan")
    return {"rate": p, "se": se, "n": n}


def main():
    cfg, _ = cli_config(__doc__)
    set_seed(cfg["experiment"]["seed"], deterministic=False)
    device = resolve_device(cfg)
    n_samples = cfg.get("anticipation", {}).get("n_samples", 3)
    max_items = cfg.get("anticipation", {}).get("max_items", 1000)

    data = load_data(cfg)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["target_name"])
    rec = data.record

    # held-out items whose next token is a content word
    items = []
    for i in data.split["test"] + data.split["val"]:
        word = tok.decode([rec["next_tokens"][i]]).strip().lower()
        if len(word) >= 4 and word.isalpha() and word not in STOP:
            items.append((i, word))
    items = items[:max_items]
    print(f"[anticipation] {len(items)} held-out items with a content-word next token")

    av_dir, _ = model_dirs(cfg, cfg.get("eval", {}).get("stage", "auto"))
    av = ActivationVerbalizer.load(av_dir).to(device).eval()

    explanations, singles = [], []
    idx = [i for i, _ in items]
    for s in tqdm(range(0, len(idx), 16), desc="decode"):
        outs = av.generate(data.acts[idx[s:s + 16]], n_samples=n_samples)
        for j in range(len(outs) // n_samples):
            group = outs[j * n_samples:(j + 1) * n_samples]
            explanations.append(" ".join(o["text"] for o in group))
            singles.append(group[0]["text"])   # one sample: same budget as one summary

    def mentions(text, word):
        return _norm(word) in content_terms(text)

    # independent second method: logit lens on the SAME stored layer-K states
    lm = load_target(cfg, device)
    lens = Lens(lm, cfg["model"]["target_name"], cfg["model"]["layer"])
    print(f"[anticipation] second method: {lens.kind}")
    lens_words = []
    for s in range(0, len(idx), 64):
        lp = lens(data.acts[idx[s:s + 64]].to(device))
        lens_words += [{w["word"].lower() for w in top_words(row, tok, k=10)} for row in lp]
    del lm

    perm = derange(len(items), seed=1).tolist()
    av_hits, sum_hits, shuf_hits, prompt_hits, av_new = [], [], [], [], []
    av1_hits, av1_new, sum_new, shuf1_new = [], [], [], []
    lens_new, av1_given_lens, av1_given_nolens = [], [], []
    examples = []
    for k, (i, word) in enumerate(items):
        in_prompt = mentions(rec["contexts"][i], word)
        a = mentions(explanations[k], word)
        av_hits.append(a)
        sum_hits.append(mentions(rec["explanations"][i], word))
        shuf_hits.append(mentions(explanations[perm[k]], word))
        prompt_hits.append(in_prompt)
        a1 = mentions(singles[k], word)
        av1_hits.append(a1)
        if not in_prompt:
            av_new.append(a)          # anticipated a word NOT yet in the text
            av1_new.append(a1)
            sum_new.append(mentions(rec["explanations"][i], word))
            shuf1_new.append(mentions(singles[perm[k]], word))
            lh = word in lens_words[k]
            lens_new.append(lh)
            (av1_given_lens if lh else av1_given_nolens).append(a1)
            if a and len(examples) < 8:
                examples.append({"context_tail": rec["contexts"][i][-160:], "next_word": word,
                                 "explanations": explanations[k][:400]})

    report = {
        "n_items": len(items),
        "n_samples_per_item": n_samples,
        "av": rate(av_hits),
        "summary": rate(sum_hits),
        "shuffled_av": rate(shuf_hits),
        "already_in_text": rate(prompt_hits),
        "av_when_word_not_yet_in_text": rate(av_new),
        "av_1_sample": rate(av1_hits),
        "not_yet_in_text__av_1_sample": rate(av1_new),
        "not_yet_in_text__summary": rate(sum_new),
        "not_yet_in_text__shuffled_1_sample": rate(shuf1_new),
        "not_yet_in_text__lens_top10": rate(lens_new),
        "not_yet_in_text__av_1_sample_when_lens_hits": rate(av1_given_lens),
        "not_yet_in_text__av_1_sample_when_lens_misses": rate(av1_given_nolens),
        "examples_new_word_anticipated": examples,
    }
    out = Path(av_dir) / "anticipation.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nHow often the TRUE NEXT WORD is named (± 1 s.e.):")
    for k in ("av", "av_1_sample", "summary", "shuffled_av", "already_in_text",
              "av_when_word_not_yet_in_text", "not_yet_in_text__av_1_sample",
              "not_yet_in_text__summary", "not_yet_in_text__shuffled_1_sample",
              "not_yet_in_text__lens_top10", "not_yet_in_text__av_1_sample_when_lens_hits",
              "not_yet_in_text__av_1_sample_when_lens_misses"):
        r = report[k]
        print(f"  {k:<30} {r['rate']:.3f} ± {r['se']:.3f}  (n={r['n']})")
    for e in examples[:4]:
        print(f"\n  …{e['context_tail'][-90:]!r}  -> next '{e['next_word']}'\n    AV: {e['explanations'][:200]}")
    print(f"\n[OK] {out}")


if __name__ == "__main__":
    main()
