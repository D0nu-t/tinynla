"""
training/eval_functional.py

Stage 2b: Functional activation patching evaluation.

Evaluates whether reconstructed activations preserve downstream computation.
Three conditions tested against original (unpatched) logits:
  - reconstructed  — text-conditioned AR output
  - random         — Gaussian noise (same shape)
  - zero           — zero vector

Additionally runs an interpolation sweep:
  h_patch = alpha * h_reconstructed + (1-alpha) * h_original
  across alphas in cfg["evaluation"]["interpolation"]["alphas"]

This sweep diagnoses whether KL divergence is caused by:
  (a) reconstruction error — KL rises steeply even at small alpha
  (b) patching shock — KL only blows up at high alpha
  These have different fixes.

Outputs:
    <save_dir>/metrics.json        — full results dict
    <save_dir>/interpolation.json  — per-alpha KL/topk/cosine
"""

import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.dataset import ActivationDataset
from nla.dataset import SequenceActivationDataset
from nla.evaluation import (
    evaluate_all_conditions_sequence,
    kl_divergence,
    logit_cosine_similarity,
    perplexity_shift,
    run_interpolation_sweep,
    topk_overlap,
    evaluate_condition_sequence,
)
from nla.reconstructor import ActivationReconstructor
from nla.reconstructor import TokenLevelReconstructor
from nla.tracking import WandbTracker
from nla.utils import load_config, resolve_device, set_seed

load_dotenv()


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    import numpy as np
    return float(np.std(values)) if values else 0.0


def main():
    cfg = load_config()
    device = resolve_device(cfg)
    set_seed(cfg["experiment"]["seed"])

    print(f"\n[INFO] Device: {device}")
    if device == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    save_dir = Path(cfg["training"]["save_dir"])

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------
    tracker = None
    if cfg["tracking"]["use_wandb"]:
        tracker = WandbTracker(
            project=cfg["tracking"]["project"],
            run_name=cfg["tracking"]["run_name"] + "_functional",
            config=cfg,
        )

    # ------------------------------------------------------------------
    # Load target LM
    # ------------------------------------------------------------------
    model_name = cfg["model"]["target_name"]
    print(f"[INFO] Loading LM: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    lm.eval()

    layer_idx = cfg["activation"]["layer_idx"]
    max_length = cfg["activation"]["max_length"]
    topk = cfg["evaluation"]["topk"]

    # ------------------------------------------------------------------
    # Load AR
    # ------------------------------------------------------------------
    buffer_path = Path(cfg["dataset"]["output_dir"]) / "buffer.pt"
    dataset = SequenceActivationDataset(str(buffer_path))
    sample_dim = dataset[0]["activation"].shape[-1]

    checkpoint = save_dir / "best_model.pt"
    print(f"[INFO] Loading AR from: {checkpoint}")
    ar = TokenLevelReconstructor(
        encoder_name=cfg["training"].get("encoder_name", "distilbert-base-uncased"),
        output_dim=sample_dim,
        hidden_dim=cfg["training"]["hidden_dim"],
    ).to(device)
    ar.load_state_dict(torch.load(checkpoint, map_location=device))
    ar.eval()

    # ------------------------------------------------------------------
    # Evaluation loop — 4-condition table
    # ------------------------------------------------------------------
    num_eval = min(cfg["evaluation"]["num_eval_samples"], len(dataset))
    samples = dataset.samples[:num_eval]

    cond_metrics: Dict[str, Dict[str, List[float]]] = {
        "reconstructed": {"kl_divergence": [], "topk_overlap": [], "logit_cosine": []},
        "random":        {"kl_divergence": [], "topk_overlap": [], "logit_cosine": []},
        "zero":          {"kl_divergence": [], "topk_overlap": [], "logit_cosine": []},
    }
    ppl_shifts = []
    interpolation_samples = []

    print(f"\n[INFO] Evaluating {num_eval} samples (4-condition table)...")

    for item in tqdm(samples):
        text = item["text"]
        description = item["description"]

        with torch.no_grad():
            reconstructed = ar([description], device)

        result = evaluate_all_conditions_sequence(
            model=lm,
            tokenizer=tokenizer,
            text=text,
            reconstructed=reconstructed,
            layer_idx=layer_idx,
            device=device,
            topk=topk,
            max_length=max_length,
        )

        for cond in ("reconstructed", "random", "zero"):
            for metric, val in result[cond].items():
                cond_metrics[cond][metric].append(val)

        if cfg["evaluation"].get("perplexity_shift", False):
            ppl_ratio = perplexity_shift(
                model=lm,
                tokenizer=tokenizer,
                text=text,
                patch_vector=reconstructed,
                layer_idx=layer_idx,
                device=device,
                max_length=max_length,
            )
            ppl_shifts.append(ppl_ratio)

        interp_cfg = cfg["evaluation"].get("interpolation", {})
        if interp_cfg.get("enabled") and len(interpolation_samples) < interp_cfg.get("num_pairs", 50):
            interpolation_samples.append({"text": text, "reconstructed": reconstructed})

    # ------------------------------------------------------------------
    # Aggregate 4-condition metrics
    # ------------------------------------------------------------------
    results: Dict = {}
    for cond, metrics in cond_metrics.items():
        for metric, values in metrics.items():
            results[f"{cond}/{metric}_mean"] = _mean(values)
            results[f"{cond}/{metric}_std"] = _std(values)

    if ppl_shifts:
        results["reconstructed/perplexity_shift_mean"] = _mean(ppl_shifts)
        results["reconstructed/perplexity_shift_std"] = _std(ppl_shifts)

    # ------------------------------------------------------------------
    # Interpolation sweep
    # ------------------------------------------------------------------
    interp_results = {}
    interp_cfg = cfg["evaluation"].get("interpolation", {})
    if interp_cfg.get("enabled") and interpolation_samples:
        alphas = interp_cfg.get("alphas", [0.0, 0.25, 0.5, 0.75, 1.0])
        print(f"\n[INFO] Running interpolation sweep over {len(interpolation_samples)} samples...")
        interp_results = run_interpolation_sweep(
            model=lm,
            tokenizer=tokenizer,
            samples=interpolation_samples,
            alphas=alphas,
            layer_idx=layer_idx,
            device=device,
            topk=topk,
            max_length=max_length,
        )

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("FUNCTIONAL EVALUATION")
    print("=" * 60)

    header = f"{'Condition':<16} {'KL':>10} {'Top-k':>10} {'Cos':>10}"
    print(header)
    print("-" * len(header))
    for cond in ("reconstructed", "random", "zero"):
        kl  = results.get(f"{cond}/kl_divergence_mean", 0.0)
        ovr = results.get(f"{cond}/topk_overlap_mean", 0.0)
        cos = results.get(f"{cond}/logit_cosine_mean", 0.0)
        print(f"{cond:<16} {kl:>10.4f} {ovr:>10.4f} {cos:>10.4f}")

    if ppl_shifts:
        ppl_m = results["reconstructed/perplexity_shift_mean"]
        print(f"\nPerplexity shift (reconstructed / original): {ppl_m:.4f}")
        print("  -> 1.0 = perfect preservation, >1.0 = degradation")

    if interp_results:
        print("\nInterpolation sweep (KL divergence by alpha):")
        print(f"  {'alpha':>6}  {'KL':>10}  {'top-k':>10}")
        for alpha in sorted(interp_results):
            r = interp_results[alpha]
            print(f"  {alpha:>6.2f}  {r['kl_divergence_mean']:>10.4f}  {r['topk_overlap_mean']:>10.4f}")

    print()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    if interp_results:
        serializable = {str(k): v for k, v in interp_results.items()}
        with open(save_dir / "interpolation.json", "w") as f:
            json.dump(serializable, f, indent=2)

    # ------------------------------------------------------------------
    # WandB
    # ------------------------------------------------------------------
    if tracker:
        wandb_metrics = {f"eval/{k}": v for k, v in results.items()}
        if interp_results:
            for alpha, r in interp_results.items():
                for k, v in r.items():
                    wandb_metrics[f"interp/alpha_{alpha:.2f}/{k}"] = v
        tracker.log(wandb_metrics)
        tracker.finish()

    print("[OK] Functional evaluation complete.")


if __name__ == "__main__":
    main()