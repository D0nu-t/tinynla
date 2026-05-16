"""
training/eval_patch.py

Stage 2a: Geometric reconstruction evaluation (sequence mode).

For each sample, reconstructs the full [seq_len, hidden_dim] activation
trajectory from its description, then measures per-position cosine similarity
against the ground-truth trajectory stored in the buffer.

Cosine is computed only over valid (non-padded) positions using the mask.

This is a necessary but insufficient success criterion — high cosine confirms
geometry is recovered but does not guarantee functional fidelity under patching.
Run eval_functional.py for the causal test.
"""

import numpy as np
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from tqdm import tqdm

from nla.dataset import SequenceActivationDataset, sequence_collate
from nla.reconstructor import TokenLevelReconstructor
from nla.utils import load_config, resolve_device

load_dotenv()


def masked_cosine_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """
    Mean cosine similarity over valid (non-padded) positions.

    pred, target: [batch, seq_len, hidden_dim]
    mask:         [batch, seq_len]  — True at valid positions
    """
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    cosine = (pred * target).sum(dim=-1)        # [batch, seq_len]
    masked = cosine * mask.float()
    n_valid = mask.float().sum().clamp(min=1.0)
    return (masked.sum() / n_valid).item()


def main():
    cfg = load_config()
    device = resolve_device(cfg)

    buffer_path = Path(cfg["dataset"]["output_dir"]) / "buffer.pt"
    dataset = SequenceActivationDataset(str(buffer_path))

    # Determine hidden_dim from first sample
    hidden_dim = dataset[0]["activation_sequence"].shape[-1]

    # Build model with config-driven params
    model = TokenLevelReconstructor(
        hidden_dim=hidden_dim,
        n_layers=cfg["training"]["decoder_layers"],
        n_heads=cfg["training"]["decoder_heads"],
        max_len=cfg["activation"]["max_length"],
        encoder_name=cfg["training"].get("encoder_name", "distilgpt2"),
    ).to(device)

    checkpoint = Path(cfg["training"]["save_dir"]) / "best_model.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    # Evaluate sample-by-sample to handle variable seq_len correctly.
    # Batching requires padding; per-sample avoids the complexity for eval.
    sims: List[float] = []

    with torch.no_grad():
        for item in tqdm(dataset.samples, desc="eval_patch"):
            description = item["description"]
            target_seq = item["activation_sequence"]   # [seq_len, hidden_dim]
            seq_len = target_seq.shape[0]

            # Reconstruct trajectory at the original seq_len
            pred_seq = model(
                [description],
                seq_len=seq_len,
                device=device,
            )   # [1, seq_len, hidden_dim]

            target_on_device = target_seq.unsqueeze(0).to(device)   # [1, seq_len, hidden_dim]
            mask = torch.ones(1, seq_len, dtype=torch.bool, device=device)

            sim = masked_cosine_metric(pred_seq, target_on_device, mask)
            sims.append(sim)

    mean_sim = float(np.mean(sims))
    std_sim = float(np.std(sims))
    print(f"\nmean cosine similarity: {mean_sim:.4f}  (std={std_sim:.4f})")
    print("[OK] Geometric evaluation complete.")


if __name__ == "__main__":
    main()