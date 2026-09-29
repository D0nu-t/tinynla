"""
study/make_pilot.py

Build a forward-simulation pilot study (Hase & Bansal, ACL 2020; human-grounded
evaluation per Doshi-Velez & Kim 2017) for the stakeholder report.

Each item: a prompt, the TinyNLA report of the target model's state over the
prompt's last words (read BEFORE it answers - the answer never enters the
report), and two continuations:
    real        the target model's own greedy continuation
    distractor  a plausible continuation from a different model (distilgpt2)
Participants pick which one the target model wrote and rate confidence 1-5.
Half the items show the report (within-subject; counterbalanced by form).

Before recruiting anyone, two simulated participants test whether the report
carries usable signal at all:
    report_matcher   picks the option sharing more content words with the report themes
    prompt_matcher   picks the option sharing more content words with the prompt
plus the meaning-based check in study/simulate.py (embeddings + shuffled-report
control), which is the one to trust: word overlap ties on most items.

    python -m study.make_pilot --n 16 --out study/pilot
"""

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.claims import content_terms
from nla.read import NLAReader
from nla.utils import load_config, parse_overrides, utf8_stdio

HTML_TEMPLATE = Path(__file__).with_name("pilot_template.html")


@torch.no_grad()
def continue_text(model, tok, prompt, device, n=16, temperature=0.0, seed=0):
    ids = tok(prompt, return_tensors="pt").to(device)
    torch.manual_seed(seed)
    out = model.generate(**ids, max_new_tokens=n, do_sample=temperature > 0,
                         temperature=temperature if temperature > 0 else None, top_k=0,
                         top_p=0.95 if temperature > 0 else None,
                         pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return " ".join(text.split())          # join lines; the model often starts with a newline


def plausible(text: str) -> bool:
    return len([w for w in text.split() if any(c.isalpha() for c in w)]) >= 5


def diverges(a: str, b: str, k: int = 3) -> bool:
    return a.lower().split()[:k] != b.lower().split()[:k]


def overlap(a: str, b_terms: set) -> int:
    return len(content_terms(a) & b_terms)


def main():
    utf8_stdio()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gpt2_small.yaml")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--read_last", type=int, default=8, help="prompt words whose state is reported")
    ap.add_argument("--out", default="study/pilot")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="config override, e.g. --set rl.save_dir=checkpoints/gpt2_small_L8/rl2")
    ap.add_argument("--distractor", choices=["other_model", "unlikely_sample"], default="unlikely_sample",
                    help="other_model: same-topic text from distilgpt2 (tests fine style - too hard for a "
                         "gist-level report); unlikely_sample: a coherent nucleus sample (T=1, p=0.95) from the SAME "
                         "model that diverges from its greedy continuation within 3 words")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=parse_overrides(args.set))
    reader = NLAReader(cfg)
    dev = reader.device
    other_tok = AutoTokenizer.from_pretrained("distilgpt2")
    other = AutoModelForCausalLM.from_pretrained("distilgpt2").to(dev).eval()

    rng = random.Random(args.seed)
    contexts = [reader.data.record["contexts"][i] for i in reader.data.split["test"]]
    rng.shuffle(contexts)

    items = []
    for ctx in contexts:
        if len(items) >= args.n:
            break
        ids = reader.tokenizer(ctx).input_ids
        if len(ids) < reader.min_position + args.read_last + 4:
            continue
        prompt = reader.tokenizer.decode(ids)
        real = continue_text(reader.target, reader.tokenizer, prompt, dev)
        if not plausible(real):
            continue
        fake = None
        if args.distractor == "other_model":
            fake = continue_text(other, other_tok, prompt, dev)
        else:
            # a coherent alternative this model COULD write (nucleus sample), diverging early
            for attempt in range(6):
                cand = continue_text(reader.target, reader.tokenizer, prompt, dev,
                                     temperature=1.0, seed=1000 * len(items) + attempt)
                if plausible(cand) and diverges(cand, real):
                    fake = cand
                    break
        if not fake or not plausible(fake) or fake == real:
            continue

        # report on the prompt's final words: the model's state BEFORE it answers
        head = reader.tokenizer.decode(ids[: -args.read_last])
        tail = reader.tokenizer.decode(ids[-args.read_last:])
        rep = reader.report(head, tail, n_samples=3)

        order = rng.random() < 0.5
        options = [real, fake] if order else [fake, real]
        items.append({
            "id": len(items),
            "prompt": prompt,
            "options": options,
            "answer": 0 if order else 1,
            "report": {k: rep[k] for k in ("overall_strength_text", "summary", "themes", "limits",
                                           "coverage_text", "consistency", "lens_leaning")},
        })
        print(f"[pilot] item {len(items)}/{args.n}: {rep['overall_strength']}")

    # ---- simulated participants (information check, not a substitute for people)
    def sim(chooser):
        return sum(chooser(it) == it["answer"] for it in items) / len(items)

    def report_choice(it):
        terms = {t["stem"] for t in it["report"]["themes"]} | \
                {w["word"].lower() for w in it["report"]["lens_leaning"]}
        s = [overlap(o, terms) for o in it["options"]]
        return s.index(max(s)) if s[0] != s[1] else random.Random(it["id"]).randint(0, 1)

    def prompt_choice(it):
        terms = content_terms(it["prompt"])
        s = [overlap(o, terms) for o in it["options"]]
        return s.index(max(s)) if s[0] != s[1] else random.Random(it["id"]).randint(0, 1)

    sims = {"report_matcher": sim(report_choice), "prompt_matcher": sim(prompt_choice), "chance": 0.5}
    try:   # meaning-based check with a shuffled-report control (study/simulate.py)
        from study.simulate import describe, embedding_check, minilm_embedder
        sims["embedding"] = embedding_check(items, minilm_embedder())
    except OSError as e:          # embedding model not downloadable/cached
        print(f"[pilot] embedding check skipped: {e}")

    # ---- counterbalanced forms: form A shows the report on even items, form B on odd
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"distractor": args.distractor, "n": len(items), "read_last": args.read_last}
    (out / "items.json").write_text(json.dumps({"meta": meta, "items": items, "simulated": sims}, indent=2),
                                    encoding="utf-8")
    template = HTML_TEMPLATE.read_text(encoding="utf-8")
    for form, parity in (("A", 0), ("B", 1)):
        public = [{k: v for k, v in it.items() if k != "answer"} | {"show_report": it["id"] % 2 == parity}
                  for it in items]
        html = template.replace("/*__ITEMS__*/[]", json.dumps(public)).replace("__FORM__", form)
        (out / f"pilot_form_{form}.html").write_text(html, encoding="utf-8")

    print("\nWord-overlap simulated participants (accuracy):",
          json.dumps({k: v for k, v in sims.items() if k != "embedding"}))
    if "embedding" in sims:
        print("Meaning-based check (study/simulate.py):\n" + describe(sims["embedding"]))
    print(f"[OK] {out}/items.json, pilot_form_A.html, pilot_form_B.html")


if __name__ == "__main__":
    main()
