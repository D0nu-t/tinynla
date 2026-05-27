"""
training/eval_manifold.py

Stage 4: Manifold fidelity evaluation.

Measures whether reconstructed activation trajectories lie on the natural
manifold of the target model's residual stream.

Metrics
-------
cosine_similarity       — mean cosine between original and reconstructed
euclidean_distance      — mean L2 distance
knn_overlap             — fraction of shared k-nearest neighbours
manifold_consistency    — how well reconstructed vectors fit local neighbourhoods
local_density_ratio     — density preservation (1.0 = natural manifold)
centroid_distance       — global centroid shift
pca_projection_error    — deviation from principal activation subspace

Outputs
-------
checkpoints/ar/manifold_metrics.json

Usage
-----
python -m training.eval_manifold
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

import dotenv
dotenv.load_dotenv()
print("Environment variables loaded from .env")

from nla.dataset import SequenceActivationDataset, sequence_collate
from nla.reconstructor import TokenLevelReconstructor
from nla.utils import load_config, resolve_device, set_seed


# ============================================================================
# Manifold metrics
# ============================================================================

def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> float:
    return F.cosine_similarity(x, y, dim=-1).mean().item()


def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    return torch.norm(x - y, dim=-1).mean().item()


def compute_knn_overlap(
    original: np.ndarray,
    reconstructed: np.ndarray,
    k: int = 10,
) -> float:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(original)

    orig_idx = nn.kneighbors(original, return_distance=False)[:, 1:]
    recon_idx = nn.kneighbors(reconstructed, return_distance=False)[:, 1:]

    overlap = [
        len(set(o) & set(r)) / k
        for o, r in zip(orig_idx, recon_idx)
    ]
    return float(np.mean(overlap))


def manifold_consistency(
    original: np.ndarray,
    reconstructed: np.ndarray,
    k: int = 10,
) -> float:
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(original)
    _, indices = nn.kneighbors(reconstructed)

    scores = []
    for i, nbrs in enumerate(indices):
        local_centroid = original[nbrs].mean(axis=0)
        denom = np.linalg.norm(reconstructed[i]) * np.linalg.norm(local_centroid) + 1e-8
        sim = (reconstructed[i] @ local_centroid) / denom
        scores.append(sim)

    return float(np.mean(scores))


def local_density_ratio(
    original: np.ndarray,
    reconstructed: np.ndarray,
    k: int = 10,
) -> float:
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(original)

    d_orig, _ = nn.kneighbors(original)
    d_recon, _ = nn.kneighbors(reconstructed)

    density_orig = np.mean(d_orig[:, 1:])
    density_recon = np.mean(d_recon[:, 1:])

    return float(density_orig / (density_recon + 1e-8))


def centroid_distance(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.linalg.norm(original.mean(axis=0) - reconstructed.mean(axis=0)))


def pca_manifold_score(
    original: np.ndarray,
    reconstructed: np.ndarray,
    n_components: int = 32,
) -> float:
    n_components = min(n_components, original.shape[1], len(original) - 1)
    pca = PCA(n_components=n_components)
    pca.fit(original)

    projected = pca.inverse_transform(pca.transform(reconstructed))
    error = np.mean(np.linalg.norm(reconstructed - projected, axis=-1))
    return float(error)


# ============================================================================
# Main
# ============================================================================

@torch.no_grad()
def evaluate_manifold() -> Dict:
    cfg = load_config()
    set_seed(cfg["experiment"]["seed"])
    device = resolve_device(cfg)

    dataset_path = Path(cfg["dataset"]["output_dir"]) / "buffer.pt"
    checkpoint_dir = Path(cfg["training"]["save_dir"])

    print(f"[INFO] Loading dataset from {dataset_path}")

    dataset = SequenceActivationDataset(str(dataset_path))

    # Derive hidden_dim from data — NOT from cfg["training"]["hidden_dim"].
    # cfg["training"]["hidden_dim"] is the MLP intermediate dim (2048, legacy).
    # The actual activation dimension is always dataset[0]["activation_sequence"].shape[-1].
    hidden_dim = dataset[0]["activation_sequence"].shape[-1]

    print(f"[INFO] Hidden dim inferred from dataset: {hidden_dim}")

    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=sequence_collate,  # required: returns "texts" and "activation_sequences"
    )

    # Build model with same hidden_dim used at training time
    model = TokenLevelReconstructor(
        hidden_dim=hidden_dim,
        n_layers=cfg["training"]["decoder_layers"],
        n_heads=cfg["training"]["decoder_heads"],
        max_len=cfg["activation"]["max_length"],
        encoder_name=cfg["training"].get("encoder_name", "distilgpt2"),
    )

    checkpoint_path = checkpoint_dir / "best_model.pt"
    print(f"[INFO] Loading checkpoint {checkpoint_path}")

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )
    model.to(device)
    model.eval()

    original_vecs: List[torch.Tensor] = []
    reconstructed_vecs: List[torch.Tensor] = []

    print("[INFO] Running manifold evaluation...")

    for batch in loader:
        # sequence_collate returns:
        #   "texts"                 — List[str]
        #   "activation_sequences"  — [batch, max_seq_len, hidden_dim]
        #   "seq_lens"              — [batch]
        #   "mask"                  — [batch, max_seq_len]

        texts = batch["texts"]
        target = batch["activation_sequences"].to(device)   # [B, max_len, hidden]
        seq_lens = batch["seq_lens"]                        # [B]

        # Reconstruct using the padded batch max seq_len
        seq_len = target.shape[1]

        pred = model(texts, seq_len=seq_len, device=device)   # [B, seq_len, hidden]

        # Mean-pool over sequence (valid positions only via mask)
        mask = batch["mask"].to(device).unsqueeze(-1).float()  # [B, max_len, 1]

        target_pooled = (target * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pred_pooled = (pred * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        target_pooled = F.normalize(target_pooled, dim=-1)
        pred_pooled = F.normalize(pred_pooled, dim=-1)

        original_vecs.append(target_pooled.cpu())
        reconstructed_vecs.append(pred_pooled.cpu())

    original = torch.cat(original_vecs).numpy()
    reconstructed = torch.cat(reconstructed_vecs).numpy()

    print("[INFO] Computing metrics...")

    metrics = {
        "cosine_similarity": cosine_similarity(
            torch.tensor(original), torch.tensor(reconstructed)
        ),
        "euclidean_distance": euclidean_distance(
            torch.tensor(original), torch.tensor(reconstructed)
        ),
        "knn_overlap": compute_knn_overlap(original, reconstructed),
        "manifold_consistency": manifold_consistency(original, reconstructed),
        "local_density_ratio": local_density_ratio(original, reconstructed),
        "centroid_distance": centroid_distance(original, reconstructed),
        "pca_projection_error": pca_manifold_score(original, reconstructed),
    }

    save_path = checkpoint_dir / "manifold_metrics.json"

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== MANIFOLD RESULTS ===")
    for k, v in metrics.items():
        print(f"{k:28s}: {v:.6f}")

    print(f"\n[OK] Saved metrics → {save_path}")

    return metrics


if __name__ == "__main__":
    evaluate_manifold()