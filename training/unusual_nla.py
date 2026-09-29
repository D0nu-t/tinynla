"""
training/unusual_nla.py

Validate the "unusual state" detector (nla/unusual.py) before it is shown to anyone:
    held-out test states   flagged ~1% by construction (checks the fit generalises)
    steered states         concept / random directions at several strengths
                           (planted-concept design of training/hidden_state_nla.py)
    odd text               code, another language, repetition, gibberish
and whether flagged states are the ones the reliability check rejects (the blind
spot found in /loop iteration 9).

    python -m training.unusual_nla --config configs/gpt2_small.yaml
Writes <data.output_dir>/unusual.pt (the fitted detector) and <av_dir>/unusual.json
"""

import json
import random
from pathlib import Path

import torch
from dotenv import load_dotenv

import nla.model_adapter as ma
from nla.read import NLAReader
from nla.unusual import UnusualnessModel
from nla.utils import cli_config, set_seed
from training.hidden_state_nla import CONCEPTS, concept_vectors, steer, typical_norm

load_dotenv()

ODD_TEXT = {
    "python code": "def main():\n    for i in range(10):\n        if i % 2 == 0:\n            print(i, sum(x for x in range(i)))\n    return None",
    "french": "Le gouvernement a annoncé hier une nouvelle réforme des retraites, qui suscite déjà de vives critiques dans les syndicats.",
    "repetition": "the the the the the the the the the the the the the the the the the the the the the the",
    "gibberish": "qzv plorth xandu vekk mibble frenzo tarquil oompf zyxt blorvenshire quatch invorsk",
    "ordinary": "The city council met on Tuesday to discuss the new budget for local schools and road repairs.",
}


@torch.no_grad()
def states(reader, text, min_pos, vec=None, steer_layer=None):
    ids = reader.tokenizer(text, return_tensors="pt").input_ids.to(reader.device)
    ctx = steer(reader.target, steer_layer, vec) if vec is not None else _null()
    with ctx, ma.capture_block_output(reader.target, reader.layer) as cap:
        reader.target(ids)
    return cap["hidden"][0, min_pos:].float().cpu()


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main():
    cfg, _ = cli_config(__doc__)
    set_seed(cfg["experiment"]["seed"], deterministic=False)
    reader = NLAReader(cfg)
    data = reader.data
    cache = Path(cfg["data"]["output_dir"]) / "unusual.pt"
    if cache.exists():
        cache.unlink()                     # always refit here
    det = UnusualnessModel.for_data(data, cache)
    mp = reader.min_position
    out = {"threshold": det.threshold, "flag_quantile": det.flag_quantile}

    test = data.acts[data.split["test"]]
    out["test_flag_rate"] = sum(a["flag"] for a in det.assess(test)) / len(test)
    print(f"[unusual] held-out test states flagged: {out['test_flag_rate']:.3f} (target ~{1 - det.flag_quantile:.2f})")

    rng = random.Random(0)
    ctxs = [data.record["contexts"][i][-300:] for i in rng.sample(data.split["test"], 8)]
    L = max(1, reader.layer // 2)
    vecs = concept_vectors(reader, L)
    base = typical_norm(reader, L, ctxs)
    g = torch.Generator().manual_seed(1)
    rand = torch.nn.functional.normalize(torch.randn(len(CONCEPTS), vecs.shape[1], generator=g), dim=-1)
    out["steered"] = {}
    for kind, vs in (("concept", vecs), ("random", rand.to(vecs.device))):
        for s in (0.125, 0.25, 0.5, 1.0):
            flags = []
            for c in ctxs:
                for k in range(len(CONCEPTS)):
                    h = states(reader, c, mp, (s * base * vs[k]).to(reader.device, reader.target.dtype), L)
                    flags += [a["flag"] for a in det.assess(h)]
            rate = sum(flags) / len(flags)
            out["steered"][f"{kind}_{s}"] = rate
            print(f"[unusual] {kind:7s} steering {s:<5}: {rate:.3f} of states flagged")

    out["odd_text"] = {}
    for name, text in ODD_TEXT.items():
        a = det.assess(states(reader, text, min(mp, 3)))
        out["odd_text"][name] = {"flag_rate": sum(x["flag"] for x in a) / len(a),
                                 "median_percentile": sorted(x["percentile"] for x in a)[len(a) // 2]}
        print(f"[unusual] {name:12s}: {out['odd_text'][name]['flag_rate']:.2f} flagged, "
              f"median percentile {out['odd_text'][name]['median_percentile']:.3f}")

    # does "unusual" explain "unreliable"? (steered reads at 0.5, one context)
    if reader.calibration is not None:
        h = states(reader, ctxs[0], mp, (0.5 * base * vecs[0]).to(reader.device, reader.target.dtype), L)[-8:]
        with steer(reader.target, L, (0.5 * base * vecs[0]).to(reader.device, reader.target.dtype)):
            n = len(reader.tokenizer(ctxs[0]).input_ids)
            rd = reader.read_many(ctxs[0], list(range(n - 8, n)), n_samples=3)
        fl = det.assess(h)
        rel = [any(reader.calibration.bucket(e["fve"]) != "none" for e in r["explanations"]) for r in rd]
        out["steered_0.5_flagged_and_unreliable"] = sum(f["flag"] and not r for f, r in zip(fl, rel)) / len(rel)
        print(f"[unusual] steered 0.5: {out['steered_0.5_flagged_and_unreliable']:.2f} of tokens are flagged "
              f"AND have no reliable read (the blind spot this flag covers)")

    p = Path(reader.av_dir) / "unusual.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] {cache} and {p}")


if __name__ == "__main__":
    main()
