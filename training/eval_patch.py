"""
training/eval_patch.py

Stage 2a: Geometric reconstruction evaluation (trajectory mode).

Evaluates:
    1. Token-level cosine similarity
    2. Sequence trajectory drift
    3. Temporal smoothness preservation
    4. Manifold consistency

This replaces pooled-era geometric evaluation.

Key principle:
    Hidden-state trajectories are structured dynamical objects,
    not unordered sets of vectors.

Metrics:
    token_cosine_mean
        Mean cosine over valid positions.

    trajectory_cosine_mean
        Cosine between flattened trajectories.

    delta_cosine_mean
        Cosine between first-order temporal differences.

    manifold_l2_mean
        Mean L2 distance between normalized trajectories.

Outputs:
    checkpoints/.../geometry.json
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from tqdm import tqdm

from nla.dataset import SequenceActivationDataset
from nla.reconstructor import TokenLevelReconstructor
from nla.utils import load_config, resolve_device, set_seed

load_dotenv()


# ============================================================================
# Metrics
# ============================================================================

def masked_token_cosine(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """
    Mean token-wise cosine similarity.

    pred,target:
        [batch, seq_len, hidden_dim]

    mask:
        [batch, seq_len]
    """
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)

    cosine = (pred * target).sum(dim=-1)

    masked = cosine * mask.float()

    denom = mask.float().sum().clamp(min=1.0)

    return (masked.sum() / denom).item()


def trajectory_cosine(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    Cosine between flattened trajectories.
    """
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)

    pred_flat = F.normalize(pred_flat, dim=-1)
    target_flat = F.normalize(target_flat, dim=-1)

    return (pred_flat * target_flat).sum(dim=-1).mean().item()


def temporal_delta_cosine(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    Compare trajectory dynamics:
        Δh[t] = h[t+1] - h[t]
    """
    if pred.shape[1] < 2:
        return 1.0

    pred_delta = pred[:, 1:, :] - pred[:, :-1, :]
    target_delta = target[:, 1:, :] - target[:, :-1, :]

    pred_delta = F.normalize(pred_delta, dim=-1)
    target_delta = F.normalize(target_delta, dim=-1)

    cosine = (pred_delta * target_delta).sum(dim=-1)

    return cosine.mean().item()


def manifold_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    Mean L2 distance between normalized trajectories.
    """
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)

    l2 = torch.norm(pred - target, dim=-1)

    return l2.mean().item()


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_sample(
    model: TokenLevelReconstructor,
    item: Dict,
    device: str,
) -> Dict[str, float]:
    """
    Evaluate one trajectory reconstruction sample.
    """
    description = item["description"]

    target_seq = item["activation_sequence"]     # [seq_len, hidden_dim]

    seq_len = target_seq.shape[0]

    with torch.no_grad():
        pred_seq = model(
            [description],
            seq_len=seq_len,
            device=device,
        )   # [1, seq_len, hidden_dim]

    target_seq = target_seq.unsqueeze(0).to(device)

    mask = torch.ones(
        1,
        seq_len,
        dtype=torch.bool,
        device=device,
    )

    return {
        "token_cosine": masked_token_cosine(
            pred_seq,
            target_seq,
            mask,
        ),
        "trajectory_cosine": trajectory_cosine(
            pred_seq,
            target_seq,
        ),
        "delta_cosine": temporal_delta_cosine(
            pred_seq,
            target_seq,
        ),
        "manifold_l2": manifold_l2(
            pred_seq,
            target_seq,
        ),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    cfg = load_config()

    device = resolve_device(cfg)

    set_seed(cfg["experiment"]["seed"])

    print(f"\n[INFO] Device: {device}")

    if device == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    # ----------------------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------------------

    buffer_path = Path(cfg["dataset"]["output_dir"]) / "buffer.pt"

    dataset = SequenceActivationDataset(str(buffer_path))

    print(f"[INFO] Loaded dataset: {len(dataset)} samples")

    hidden_dim = dataset[0]["activation_sequence"].shape[-1]

    # ----------------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------------

    model = TokenLevelReconstructor(
        hidden_dim=hidden_dim,
        n_layers=cfg["training"]["decoder_layers"],
        n_heads=cfg["training"]["decoder_heads"],
        max_len=cfg["activation"]["max_length"],
        encoder_name=cfg["training"].get(
            "encoder_name",
            "distilgpt2",
        ),
    ).to(device)

    checkpoint = (
        Path(cfg["training"]["save_dir"])
        / "best_model.pt"
    )

    print(f"[INFO] Loading checkpoint: {checkpoint}")

    model.load_state_dict(
        torch.load(checkpoint, map_location=device)
    )

    model.eval()

    # ----------------------------------------------------------------------
    # Evaluation Loop
    # ----------------------------------------------------------------------

    metric_lists = {
        "token_cosine": [],
        "trajectory_cosine": [],
        "delta_cosine": [],
        "manifold_l2": [],
    }

    print("\n[INFO] Running geometric trajectory evaluation...")

    with torch.no_grad():
        for item in tqdm(dataset.samples, desc="eval_patch"):
            metrics = evaluate_sample(
                model=model,
                item=item,
                device=device,
            )

            for k, v in metrics.items():
                metric_lists[k].append(v)

    # ----------------------------------------------------------------------
    # Aggregate
    # ----------------------------------------------------------------------

    results = {}

    for metric, values in metric_lists.items():
        results[f"{metric}_mean"] = float(np.mean(values))
        results[f"{metric}_std"] = float(np.std(values))

    # ----------------------------------------------------------------------
    # Console Output
    # ----------------------------------------------------------------------

    print()
    print("=" * 72)
    print("GEOMETRIC TRAJECTORY EVALUATION")
    print("=" * 72)

    print(
        f"{'Metric':<28} {'Mean':>12} {'Std':>12}"
    )

    print("-" * 72)

    for metric in (
        "token_cosine",
        "trajectory_cosine",
        "delta_cosine",
        "manifold_l2",
    ):
        mean = results[f"{metric}_mean"]
        std = results[f"{metric}_std"]

        print(
            f"{metric:<28} "
            f"{mean:>12.4f} "
            f"{std:>12.4f}"
        )

    print()

    print("Interpretation:")
    print("  token_cosine       -> local reconstruction quality")
    print("  trajectory_cosine  -> global sequence geometry")
    print("  delta_cosine       -> temporal dynamics preservation")
    print("  manifold_l2        -> manifold deviation")

    # ----------------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------------

    save_dir = Path(cfg["training"]["save_dir"])

    with open(save_dir / "geometry.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Saved geometry metrics to:")
    print(f"     {save_dir / 'geometry.json'}")

    print("\n[OK] Geometric evaluation complete.")


if __name__ == "__main__":
    main()