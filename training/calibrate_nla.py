"""
training/calibrate_nla.py

Calibrate the plain-language signal strength used in stakeholder reports.

    python -m training.calibrate_nla --config configs/gpt2_small.yaml

For `calibrate.n` validation activations, draw two independent AV samples:
  - sample 1 (own) and sample 1 re-paired with the wrong activation (shuffled)
    set the thresholds (nla.calibrate.Calibration);
  - sample 2 is a test-retest check: the bucket of sample 1 must predict the
    FVE of sample 2 (monotonic means), or the buckets carry no information.

Writes calibration.json next to the evaluated AV.
"""

import json

import torch
from dotenv import load_dotenv
from tqdm import tqdm

from nla.ar import ActivationReconstructor
from nla.av import ActivationVerbalizer
from nla.calibrate import Calibration, is_monotonic, retest_by_bucket
from nla.evaluate import derange, reconstruct
from nla.metrics import per_sample_fve
from nla.runs import load_data, model_dirs
from nla.utils import cli_config, resolve_device, set_seed

load_dotenv()


def main():
    cfg, _ = cli_config(__doc__)
    set_seed(cfg["experiment"]["seed"], deterministic=False)
    device = resolve_device(cfg)
    n = cfg.get("calibrate", {}).get("n", 300)

    data = load_data(cfg)
    idx = data.split["val"][:n]
    acts = data.acts[idx]

    av_dir, ar_dir = model_dirs(cfg, cfg.get("eval", {}).get("stage", "auto"))
    av = ActivationVerbalizer.load(av_dir).to(device).eval()
    ar = ActivationReconstructor.load(ar_dir).to(device).eval()
    print(f"[calibrate] AV {av_dir} | AR {ar_dir} | {len(idx)} validation activations")

    first, second = [], []
    for i in tqdm(range(0, len(idx), 16), desc="decode x2"):
        outs = av.generate(acts[i:i + 16], n_samples=2)
        first += [o["text"] for o in outs[0::2]]
        second += [o["text"] for o in outs[1::2]]

    def score(texts, gold):
        return per_sample_fve(reconstruct(ar, texts), gold, data.variance, data.scale).tolist()

    own1 = score(first, acts)
    own2 = score(second, acts)
    shuffled = score([first[j] for j in derange(len(first)).tolist()], acts)

    cal = Calibration.from_samples(own1, shuffled)
    cal.retest = retest_by_bucket(cal, own1, own2)
    path = cal.save(av_dir)

    ok = is_monotonic(cal.retest)
    print("\n" + json.dumps({k: v for k, v in cal.__dict__.items() if k != "retest"}, indent=2))
    print("\nTest-retest (bucket of sample 1 -> mean FVE of independent sample 2):")
    for b, r in cal.retest.items():
        m = r["second_sample_fve_mean"]
        print(f"  {b:<9} n={r['n']:<4} {'' if m is None else f'{m:+.4f}'}")
    print(f"\n{'PASS' if ok else 'FAIL'}  buckets are monotonic in an independent sample")
    print(f"[OK] {path}")


if __name__ == "__main__":
    main()
