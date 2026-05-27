"""
training/eval_functional.py

Stage 3 functional evaluation for TinyNLA.

Trajectory-level activation patching evaluation.

This script evaluates whether reconstructed activation trajectories preserve
model behavior when patched back into the frozen transformer.

Pipeline
--------
For each sample:

1. tokenize original text
2. run frozen LM to obtain original logits
3. reconstruct full activation trajectory:
       [seq_len, hidden_dim]
4. patch trajectory back into residual stream
5. compare patched vs original behavior

Evaluated conditions:
    reconstructed  — AR output
    random         — Gaussian noise trajectory
    zero           — null trajectory

Additionally:
    - interpolation alpha sweep
    - perplexity shift
    - manifold metrics
    - trajectory geometry metrics

Outputs
-------
metrics.json
interpolation.json
manifold.json

v3 upgrades
-----------
- trajectory-level evaluation
- manifold diagnostics
- sequence-aware patching
- robust aggregation
- automatic hidden_dim inference
- trajectory cosine tracking
- patching stability diagnostics
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.dataset import SequenceActivationDataset
from nla.evaluation import (
    evaluate_all_conditions_sequence,
    perplexity_shift,
    run_interpolation_sweep_sequence,
)
from nla.metrics import (
    cosine_similarity_metric,
    manifold_offmanifold_ratio,
)
from nla.reconstructor import TokenLevelReconstructor
from nla.tracking import WandbTracker
from nla.utils import load_config, resolve_device, set_seed

load_dotenv()


# ============================================================================
# Utilities
# ============================================================================

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _flatten_sequence(x: torch.Tensor) -> torch.Tensor:
    """
    [seq, hidden] -> [seq * hidden]
    """
    return x.reshape(-1)


# ============================================================================
# Main
# ============================================================================

def main():
    cfg = load_config()

    device = resolve_device(cfg)
    set_seed(cfg["experiment"]["seed"])

    print("\n" + "=" * 70)
    print("TinyNLA — Functional Evaluation")
    print("=" * 70)

    print(f"[INFO] Device: {device}")

    if device == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    # ----------------------------------------------------------------------
    # Paths
    # ----------------------------------------------------------------------

    save_dir = Path(cfg["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    # Tracking
    # ----------------------------------------------------------------------

    tracker = None

    if cfg["tracking"]["use_wandb"]:
        tracker = WandbTracker(
            project=cfg["tracking"]["project"],
            run_name=cfg["tracking"]["run_name"] + "_functional",
            config=cfg,
            mode=cfg["tracking"].get("mode", "offline"),
        )

    # ----------------------------------------------------------------------
    # Frozen target LM
    # ----------------------------------------------------------------------

    model_name = cfg["model"]["target_name"]

    print(f"\n[INFO] Loading target LM: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    lm = AutoModelForCausalLM.from_pretrained(
        model_name
    ).to(device)

    lm.eval()

    layer_idx = cfg["activation"]["layer_idx"]
    max_length = cfg["activation"]["max_length"]
    topk = cfg["evaluation"]["topk"]

    # ----------------------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------------------

    buffer_path = (
        Path(cfg["dataset"]["output_dir"])
        / "buffer.pt"
    )

    print(f"[INFO] Loading dataset: {buffer_path}")

    dataset = SequenceActivationDataset(str(buffer_path))

    hidden_dim = dataset[0]["activation_sequence"].shape[-1]

    print(f"[INFO] Hidden dim inferred from dataset: {hidden_dim}")

    # ----------------------------------------------------------------------
    # Load AR
    # ----------------------------------------------------------------------

    checkpoint = save_dir / "best_model.pt"

    print(f"[INFO] Loading reconstructor checkpoint:")
    print(f"       {checkpoint}")

    ar = TokenLevelReconstructor(
        hidden_dim=hidden_dim,
        n_layers=cfg["training"]["decoder_layers"],
        n_heads=cfg["training"]["decoder_heads"],
        max_len=max_length,
        encoder_name=cfg["training"].get(
            "encoder_name",
            "distilgpt2",
        ),
    ).to(device)

    ar.load_state_dict(
        torch.load(checkpoint, map_location=device)
    )

    ar.eval()

    # ----------------------------------------------------------------------
    # Evaluation state
    # ----------------------------------------------------------------------

    num_eval = min(
        cfg["evaluation"]["num_eval_samples"],
        len(dataset),
    )

    samples = dataset.samples[:num_eval]

    cond_metrics: Dict[str, Dict[str, List[float]]] = {
        "reconstructed": {
            "kl_divergence": [],
            "topk_overlap": [],
            "logit_cosine": [],
        },
        "random": {
            "kl_divergence": [],
            "topk_overlap": [],
            "logit_cosine": [],
        },
        "zero": {
            "kl_divergence": [],
            "topk_overlap": [],
            "logit_cosine": [],
        },
    }

    ppl_shifts: List[float] = []

    # trajectory geometry
    sequence_cosines: List[float] = []

    # manifold
    manifold_original: List[torch.Tensor] = []
    manifold_reconstructed: List[torch.Tensor] = []

    interpolation_samples: List[Dict] = []

    # ----------------------------------------------------------------------
    # Main evaluation loop
    # ----------------------------------------------------------------------

    print(f"\n[INFO] Evaluating {num_eval} samples...")

    for item in tqdm(samples):

        text = item["text"]
        description = item["description"]

        original_sequence = item["activation_sequence"]

        toks = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

        seq_len = toks["input_ids"].shape[1]

        # --------------------------------------------------------------
        # Reconstruct trajectory
        # --------------------------------------------------------------

        with torch.no_grad():

            recon_sequence = ar(
                [description],
                seq_len=seq_len,
                device=device,
            )

        recon_sequence = recon_sequence.squeeze(0).cpu()

        # --------------------------------------------------------------
        # Trajectory cosine similarity
        # --------------------------------------------------------------

        original_flat = _flatten_sequence(
            original_sequence[:seq_len]
        )

        recon_flat = _flatten_sequence(
            recon_sequence[:seq_len]
        )

        seq_cos = F.cosine_similarity(
            original_flat.unsqueeze(0),
            recon_flat.unsqueeze(0),
            dim=-1,
        ).item()

        sequence_cosines.append(seq_cos)

        # --------------------------------------------------------------
        # Functional metrics
        # --------------------------------------------------------------

        result = evaluate_all_conditions_sequence(
            model=lm,
            tokenizer=tokenizer,
            text=text,
            reconstructed_sequence=recon_sequence,
            layer_idx=layer_idx,
            device=device,
            topk=topk,
            max_length=max_length,
        )

        for cond in ("reconstructed", "random", "zero"):
            for metric, val in result[cond].items():
                cond_metrics[cond][metric].append(val)

        # --------------------------------------------------------------
        # Perplexity shift
        # --------------------------------------------------------------

        if cfg["evaluation"].get("perplexity_shift", False):

            ppl_ratio = perplexity_shift(
                model=lm,
                tokenizer=tokenizer,
                text=text,
                patch_tensor=recon_sequence,
                layer_idx=layer_idx,
                device=device,
                max_length=max_length,
                sequence_mode=True,
            )

            ppl_shifts.append(ppl_ratio)

        # --------------------------------------------------------------
        # Manifold metrics
        # --------------------------------------------------------------

        # pooled representation for manifold diagnostics
        manifold_original.append(
            original_sequence[:seq_len]
            .mean(dim=0)
            .cpu()
        )

        manifold_reconstructed.append(
            recon_sequence[:seq_len]
            .mean(dim=0)
            .cpu()
        )

        # --------------------------------------------------------------
        # Interpolation sweep collection
        # --------------------------------------------------------------

        interp_cfg = cfg["evaluation"].get(
            "interpolation",
            {},
        )

        if (
            interp_cfg.get("enabled")
            and len(interpolation_samples)
            < interp_cfg.get("num_pairs", 50)
        ):
            interpolation_samples.append({
                "text": text,
                "reconstructed_sequence": recon_sequence,
            })

    # ==========================================================================
    # Aggregate functional metrics
    # ==========================================================================

    results: Dict = {}

    for cond, metrics in cond_metrics.items():

        for metric, values in metrics.items():

            results[f"{cond}/{metric}_mean"] = _mean(values)
            results[f"{cond}/{metric}_std"] = _std(values)

    # ----------------------------------------------------------------------
    # Geometry metrics
    # ----------------------------------------------------------------------

    results["geometry/trajectory_cosine_mean"] = _mean(
        sequence_cosines
    )

    results["geometry/trajectory_cosine_std"] = _std(
        sequence_cosines
    )

    # ----------------------------------------------------------------------
    # Perplexity shift
    # ----------------------------------------------------------------------

    if ppl_shifts:

        results["reconstructed/perplexity_shift_mean"] = _mean(
            ppl_shifts
        )

        results["reconstructed/perplexity_shift_std"] = _std(
            ppl_shifts
        )

    # ==========================================================================
    # Manifold diagnostics
    # ==========================================================================

    manifold_original_tensor = torch.stack(
        manifold_original
    )

    manifold_reconstructed_tensor = torch.stack(
        manifold_reconstructed
    )

    manifold_metrics = {
        "mean_cosine_similarity": cosine_similarity_metric(
            manifold_original_tensor,
            manifold_reconstructed_tensor,
        ),
        "offmanifold_ratio": manifold_offmanifold_ratio(
            manifold_original_tensor,
            manifold_reconstructed_tensor,
        ),
    }

    # ==========================================================================
    # Interpolation sweep
    # ==========================================================================

    interp_results = {}

    interp_cfg = cfg["evaluation"].get(
        "interpolation",
        {},
    )

    if (
        interp_cfg.get("enabled")
        and interpolation_samples
    ):

        alphas = interp_cfg.get(
            "alphas",
            [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        )

        print(
            f"\n[INFO] Running interpolation sweep "
            f"over {len(interpolation_samples)} samples..."
        )

        interp_results = run_interpolation_sweep_sequence(
            model=lm,
            tokenizer=tokenizer,
            samples=interpolation_samples,
            alphas=alphas,
            layer_idx=layer_idx,
            device=device,
            topk=topk,
            max_length=max_length,
        )

    # ==========================================================================
    # Console output
    # ==========================================================================

    print()
    print("=" * 70)
    print("FUNCTIONAL EVALUATION")
    print("=" * 70)

    header = (
        f"{'Condition':<16}"
        f"{'KL':>12}"
        f"{'Top-k':>12}"
        f"{'Cosine':>12}"
    )

    print(header)
    print("-" * len(header))

    for cond in ("reconstructed", "random", "zero"):

        kl = results.get(
            f"{cond}/kl_divergence_mean",
            0.0,
        )

        overlap = results.get(
            f"{cond}/topk_overlap_mean",
            0.0,
        )

        cos = results.get(
            f"{cond}/logit_cosine_mean",
            0.0,
        )

        print(
            f"{cond:<16}"
            f"{kl:>12.4f}"
            f"{overlap:>12.4f}"
            f"{cos:>12.4f}"
        )

    print()

    print("=" * 70)
    print("GEOMETRY")
    print("=" * 70)

    print(
        f"Trajectory cosine similarity: "
        f"{results['geometry/trajectory_cosine_mean']:.4f}"
    )

    print()

    print("=" * 70)
    print("MANIFOLD")
    print("=" * 70)

    for k, v in manifold_metrics.items():
        print(f"{k:<32} {v:.6f}")

    if ppl_shifts:

        print()
        print("=" * 70)
        print("PERPLEXITY SHIFT")
        print("=" * 70)

        print(
            f"PPL ratio (patched/original): "
            f"{results['reconstructed/perplexity_shift_mean']:.4f}"
        )

    if interp_results:

        print()
        print("=" * 70)
        print("INTERPOLATION SWEEP")
        print("=" * 70)

        print(
            f"{'alpha':>8}"
            f"{'KL':>12}"
            f"{'Top-k':>12}"
            f"{'Cosine':>12}"
        )

        for alpha in sorted(interp_results):

            r = interp_results[alpha]

            print(
                f"{alpha:>8.2f}"
                f"{r['kl_divergence_mean']:>12.4f}"
                f"{r['topk_overlap_mean']:>12.4f}"
                f"{r['logit_cosine_mean']:>12.4f}"
            )

    print()

    # ==========================================================================
    # Save artifacts
    # ==========================================================================

    with open(save_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(save_dir / "manifold.json", "w") as f:
        json.dump(manifold_metrics, f, indent=2)

    if interp_results:

        with open(save_dir / "interpolation.json", "w") as f:

            json.dump(
                {
                    str(k): v
                    for k, v in interp_results.items()
                },
                f,
                indent=2,
            )

    # ==========================================================================
    # WandB
    # ==========================================================================

    if tracker:

        wandb_metrics = {
            f"eval/{k}": v
            for k, v in results.items()
        }

        for k, v in manifold_metrics.items():
            wandb_metrics[f"manifold/{k}"] = v

        if interp_results:

            for alpha, r in interp_results.items():

                for k, v in r.items():

                    wandb_metrics[
                        f"interp/alpha_{alpha:.2f}/{k}"
                    ] = v

        tracker.log(wandb_metrics)
        tracker.finish()

    print("[OK] Functional evaluation complete.")


if __name__ == "__main__":
    main()