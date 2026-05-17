"""
training/eval_functional.py

Stage 2b: Functional activation patching evaluation (sequence mode).

For each sample:
  1. Tokenize the original text and run the frozen LM to get original logits.
  2. Reconstruct the full [seq_len, hidden_dim] activation trajectory from
     the sample's description using TokenLevelReconstructor.
  3. Patch the trajectory back using SequenceInterpolationPatcher.
  4. Compare original vs patched logits under three conditions:
       reconstructed — AR output
       random        — Gaussian noise, same shape
       zero          — zeros, same shape

Additionally runs an interpolation sweep:
  h_patch[t] = alpha * h_recon[t] + (1 - alpha) * h_orig[t]  per position
  across alphas in cfg["evaluation"]["interpolation"]["alphas"]

Outputs:
    <save_dir>/metrics.json        — 4-condition + PPL shift results
    <save_dir>/interpolation.json  — per-alpha KL/topk/cosine
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.dataset import SequenceActivationDataset
from nla.evaluation import (
    evaluate_all_conditions_sequence,
    perplexity_shift,
    run_interpolation_sweep_sequence,
)
from nla.reconstructor import TokenLevelReconstructor
from nla.tracking import WandbTracker
from nla.utils import load_config, resolve_device, set_seed

load_dotenv()


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
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
            mode=cfg["tracking"].get("mode", "offline"),
        )

    # ------------------------------------------------------------------
    # Load target LM (frozen)
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
    # Load dataset and AR
    # ------------------------------------------------------------------
    buffer_path = Path(cfg["dataset"]["output_dir"]) / "buffer.pt"
    dataset = SequenceActivationDataset(str(buffer_path))

    # Derive hidden_dim from buffer, not config, to be safe
    hidden_dim = dataset[0]["activation_sequence"].shape[-1]

    checkpoint = save_dir / "best_model.pt"
    print(f"[INFO] Loading AR from: {checkpoint}")
    ar = TokenLevelReconstructor(
        hidden_dim=hidden_dim,
        n_layers=cfg["training"]["decoder_layers"],
        n_heads=cfg["training"]["decoder_heads"],
        max_len=cfg["activation"]["max_length"],
        encoder_name=cfg["training"].get("encoder_name", "distilgpt2"),
    ).to(device)
    ar.load_state_dict(torch.load(checkpoint, map_location=device))
    ar.eval()

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    num_eval = min(cfg["evaluation"]["num_eval_samples"], len(dataset))
    samples = dataset.samples[:num_eval]

    cond_metrics: Dict[str, Dict[str, List[float]]] = {
        "reconstructed": {"kl_divergence": [], "topk_overlap": [], "logit_cosine": []},
        "random":        {"kl_divergence": [], "topk_overlap": [], "logit_cosine": []},
        "zero":          {"kl_divergence": [], "topk_overlap": [], "logit_cosine": []},
    }
    ppl_shifts: List[float] = []
    interpolation_samples: List[Dict] = []

    print(f"\n[INFO] Evaluating {num_eval} samples (4-condition table)...")

    for item in tqdm(samples):
        text = item["text"]
        description = item["description"]

        # Tokenize the original text to get its actual seq_len for reconstruction
        toks = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        original_seq_len = toks["input_ids"].shape[1]

        # Reconstruct [1, original_seq_len, hidden_dim]
        with torch.no_grad():
            recon_seq = ar(
                [description],
                seq_len=original_seq_len,
                device=device,
            )   # [1, seq_len, hidden_dim]

        # Squeeze batch dim for patcher: [seq_len, hidden_dim]
        recon_seq_2d = recon_seq.squeeze(0)

        # 4-condition table — uses SequenceInterpolationPatcher internally
        result = evaluate_all_conditions_sequence(
            model=lm,
            tokenizer=tokenizer,
            text=text,
            reconstructed_sequence=recon_seq_2d,   # correct kwarg name
            layer_idx=layer_idx,
            device=device,
            topk=topk,
            max_length=max_length,
        )

        for cond in ("reconstructed", "random", "zero"):
            for metric, val in result[cond].items():
                cond_metrics[cond][metric].append(val)

        # Perplexity shift
        if cfg["evaluation"].get("perplexity_shift", False):
            ppl_ratio = perplexity_shift(
                model=lm,
                tokenizer=tokenizer,
                text=text,
                patch_tensor=recon_seq_2d,   # correct kwarg name
                layer_idx=layer_idx,
                device=device,
                max_length=max_length,
                sequence_mode=True,
            )
            ppl_shifts.append(ppl_ratio)

        # Collect for interpolation sweep
        interp_cfg = cfg["evaluation"].get("interpolation", {})
        if (
            interp_cfg.get("enabled")
            and len(interpolation_samples) < interp_cfg.get("num_pairs", 50)
        ):
            # run_interpolation_sweep_sequence expects key "reconstructed_sequence"
            interpolation_samples.append({
                "text": text,
                "reconstructed_sequence": recon_seq_2d,
            })

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
    # Interpolation sweep (sequence mode)
    # ------------------------------------------------------------------
    interp_results = {}
    interp_cfg = cfg["evaluation"].get("interpolation", {})
    if interp_cfg.get("enabled") and interpolation_samples:
        alphas = interp_cfg.get("alphas", [0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
        print(
            f"\n[INFO] Running interpolation sweep over "
            f"{len(interpolation_samples)} samples..."
        )
        interp_results = run_interpolation_sweep_sequence(
            model=lm,
            tokenizer=tokenizer,
            samples=interpolation_samples,   # each has "reconstructed_sequence"
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
            print(
                f"  {alpha:>6.2f}  "
                f"{r['kl_divergence_mean']:>10.4f}  "
                f"{r['topk_overlap_mean']:>10.4f}"
            )

    print()

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    if interp_results:
        with open(save_dir / "interpolation.json", "w") as f:
            json.dump({str(k): v for k, v in interp_results.items()}, f, indent=2)

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