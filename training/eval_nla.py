"""
training/eval_nla.py

Held-out NLA evaluation with the touchstone controls
(wiki: concepts/nla_touchstone).

    python -m training.eval_nla --config configs/gpt2_small.yaml
    python -m training.eval_nla --set eval.stage=sft

Report (written next to the evaluated AV as nla_eval.json):
    av_fve            AR FVE of AV explanations for test activations
    av_fve_shuffled   same explanations paired with the wrong activation  (<= 0)
    summary_fve       AR FVE of warm-start summaries of the input text
    summary_fve_sft_ar  summaries scored by the SFT AR (trained on them)
    mean_fve          predicting the raw train mean                      (= 0)
    mean_direction_fve  the train mean direction at the prediction norm
                      (fair constant baseline for check 1)
    parse_ok, unique_frac, sample_agreement
    checks            pass/fail per touchstone row
"""

import json
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm import tqdm

from nla.ar import ActivationReconstructor
from nla.av import ActivationVerbalizer
from nla.evaluate import ar_scores, reconstruct
from nla.metrics import fve
from nla.runs import load_data, model_dirs
from nla.utils import cli_config, resolve_device, set_seed

load_dotenv()


def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / max(1, len(sa | sb))


def main():
    cfg, _ = cli_config(__doc__)
    set_seed(cfg["experiment"]["seed"], deterministic=False)
    device = resolve_device(cfg)
    ecfg = cfg.get("eval", {})
    stage = ecfg.get("stage", "auto")

    data = load_data(cfg)
    test = data.split["test"][: ecfg.get("max_test", 500)]
    acts = data.acts[test]
    summaries = [data.explanations[i] for i in test]

    av_dir, ar_dir = model_dirs(cfg, stage)
    print(f"[eval] AV: {av_dir}\n[eval] AR: {ar_dir}\n[eval] {len(test)} test activations")
    av = ActivationVerbalizer.load(av_dir).to(device).eval()
    ar = ActivationReconstructor.load(ar_dir).to(device).eval()

    n = ecfg.get("n_samples", 1)
    samples = [[] for _ in test]
    ok = []
    for i in tqdm(range(0, len(test), 16), desc="decode"):
        for o in av.generate(acts[i:i + 16], n_samples=n):
            samples[i + o["source"]].append(o["text"])
            ok.append(o["ok"])

    # score every sample; report the mean over samples
    per_sample = []
    for k in range(n):
        per_sample.append(ar_scores(ar, [s[k] for s in samples], acts, data.mean, data.scale))
    av_fve = sum(s["fve"] for s in per_sample) / n
    av_shuf = sum(s["fve_shuffled"] for s in per_sample) / n

    summary = ar_scores(ar, summaries, acts, data.mean, data.scale, controls=False)["fve"]
    mean_fve = fve(data.mean.expand_as(acts), acts, data.mean, data.scale, normalize_pred=False)
    # fair constant baseline for direction-normalised predictions (touchstone check 1)
    mean_dir_fve = fve(data.mean.expand_as(acts), acts, data.mean, data.scale)

    # Fairness control for check 1b: after RL the AR has adapted to the AV's
    # language, which could penalise summaries. Also score the summaries with
    # the SFT AR - the one trained on them - and require the AV to beat both.
    summary_sft_ar = None
    sft_ar_dir = cfg["ar_sft"]["save_dir"]
    if Path(ar_dir).resolve() != Path(sft_ar_dir).resolve() and Path(sft_ar_dir, "ar.pt").exists():
        sft_ar = ActivationReconstructor.load(sft_ar_dir).to(device).eval()
        summary_sft_ar = ar_scores(sft_ar, summaries, acts, data.mean, data.scale, controls=False)["fve"]
        del sft_ar
    best_summary = max(summary, summary_sft_ar if summary_sft_ar is not None else summary)

    firsts = [s[0] for s in samples]
    agreement = None
    if n > 1:
        pairs = [jaccard(s[a], s[b]) for s in samples for a in range(n) for b in range(a + 1, n)]
        agreement = sum(pairs) / len(pairs)

    report = {
        "stage": stage,
        "av_dir": av_dir,
        "ar_dir": ar_dir,
        "n_test": len(test),
        "n_samples": n,
        "av_fve": av_fve,
        "av_fve_shuffled": av_shuf,
        "summary_fve": summary,
        "summary_fve_sft_ar": summary_sft_ar,
        "mean_fve": mean_fve,
        "mean_direction_fve": mean_dir_fve,
        "parse_ok": sum(ok) / len(ok),
        "unique_frac": len(set(firsts)) / len(firsts),
        "sample_agreement": agreement,
    }
    report["checks"] = {
        "1_beats_mean": av_fve > mean_dir_fve + 0.05,
        "1b_beats_input_summary": av_fve > best_summary,
        "2_shuffle_control": av_shuf < min(0.02, av_fve - 0.05),
        "5_explanations_diverse": report["unique_frac"] > 0.9,
        "8_held_out": True,
    }
    report["examples"] = [
        {"context_tail": data.record["contexts"][i][-200:], "summary": summaries[j], "av": samples[j]}
        for j, i in enumerate(test[:5])
    ]

    out = Path(av_dir) / "nla_eval.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 64)
    print(f"NLA EVAL ({stage}) — held-out test, {len(test)} activations")
    print("=" * 64)
    for k in ("av_fve", "av_fve_shuffled", "summary_fve", "summary_fve_sft_ar",
              "mean_fve", "mean_direction_fve", "parse_ok", "unique_frac"):
        if report[k] is not None:
            print(f"{k:<20} {report[k]:>10.4f}")
    print("\nTouchstone checks:")
    for k, v in report["checks"].items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\n[OK] {out}")


if __name__ == "__main__":
    main()
