"""
training/eval_manifold.py

Evaluate whether reconstructed activations lie on the natural
activation manifold of the target model.

This is the primary diagnostic for:

1. Reconstruction manifold fidelity
2. Patching stability prediction
3. Detecting off-manifold collapse
4. Comparing pooled vs sequence reconstruction

Metrics
-------
- cosine similarity
- euclidean distance
- nearest-neighbor overlap
- kNN manifold consistency
- local density ratio
- centroid distance
- PCA explained reconstruction quality

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
from nla.dataset import SequenceActivationDataset
from nla.dataset import ActivationDataset
from nla.reconstructor import (
    ActivationReconstructor,
    TokenLevelReconstructor,
)
from nla.utils import (
    load_config,
    resolve_device,
    set_seed,
)


# ============================================================
# Helpers
# ============================================================


def cosine_similarity(
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    return F.cosine_similarity(x, y, dim=-1).mean().item()


def euclidean_distance(
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    return torch.norm(x - y, dim=-1).mean().item()


def compute_knn_overlap(
    original: np.ndarray,
    reconstructed: np.ndarray,
    k: int = 10,
) -> float:
    """
    Fraction of shared nearest neighbors.

    Higher = reconstructed vector lies in
    same neighborhood of latent space.
    """

    nn = NearestNeighbors(
        n_neighbors=k + 1,
        metric="cosine",
    )

    nn.fit(original)

    orig_idx = nn.kneighbors(
        original,
        return_distance=False,
    )[:, 1:]

    recon_idx = nn.kneighbors(
        reconstructed,
        return_distance=False,
    )[:, 1:]

    overlap = []

    for o, r in zip(orig_idx, recon_idx):
        overlap.append(
            len(set(o) & set(r)) / k
        )

    return float(np.mean(overlap))


def manifold_consistency(
    original: np.ndarray,
    reconstructed: np.ndarray,
    k: int = 10,
) -> float:
    """
    Measures whether reconstructed activations
    remain inside local manifold neighborhoods.

    For each reconstructed vector:
        find kNN in original manifold

    Then compare to source neighborhood.
    """

    nn = NearestNeighbors(
        n_neighbors=k,
        metric="cosine",
    )

    nn.fit(original)

    _, indices = nn.kneighbors(
        reconstructed
    )

    consistency_scores = []

    for i, nbrs in enumerate(indices):
        local_centroid = original[nbrs].mean(axis=0)

        sim = (
            reconstructed[i]
            @ local_centroid
        ) / (
            np.linalg.norm(reconstructed[i])
            * np.linalg.norm(local_centroid)
            + 1e-8
        )

        consistency_scores.append(sim)

    return float(np.mean(consistency_scores))


def local_density_ratio(
    original: np.ndarray,
    reconstructed: np.ndarray,
    k: int = 10,
) -> float:
    """
    Compare local density around points.

    ratio ≈ 1
        natural manifold density

    ratio << 1
        sparse / collapsed

    ratio >> 1
        overcompressed cluster
    """

    nn = NearestNeighbors(
        n_neighbors=k,
        metric="euclidean",
    )

    nn.fit(original)

    d_orig, _ = nn.kneighbors(original)
    d_recon, _ = nn.kneighbors(reconstructed)

    density_orig = np.mean(d_orig[:, 1:])
    density_recon = np.mean(d_recon[:, 1:])

    return float(
        density_orig
        / (density_recon + 1e-8)
    )


def centroid_distance(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> float:
    """
    Distance between global centroids.
    """

    orig_centroid = original.mean(axis=0)
    recon_centroid = reconstructed.mean(axis=0)

    return float(
        np.linalg.norm(
            orig_centroid
            - recon_centroid
        )
    )


def pca_manifold_score(
    original: np.ndarray,
    reconstructed: np.ndarray,
    n_components: int = 32,
) -> float:
    """
    Measures whether reconstructed activations
    remain inside principal manifold subspace.

    Higher explained variance retained
    = better geometric faithfulness.
    """

    n_components = min(
        n_components,
        original.shape[1],
        len(original) - 1,
    )

    pca = PCA(
        n_components=n_components
    )

    pca.fit(original)

    projected = pca.inverse_transform(
        pca.transform(reconstructed)
    )

    error = np.mean(
        np.linalg.norm(
            reconstructed - projected,
            axis=-1,
        )
    )

    return float(error)


# ============================================================
# Main Evaluation
# ============================================================


@torch.no_grad()
def evaluate_manifold() -> Dict:

    cfg = load_config()

    set_seed(cfg["experiment"]["seed"])

    device = resolve_device(
        cfg["device"]
    )

    dataset_path = Path(
        cfg["dataset"]["output_dir"]
    ) / "buffer.pt"

    checkpoint_dir = Path(
        cfg["training"]["save_dir"]
    )

    print(
        f"[INFO] Loading dataset from {dataset_path}"
    )

    dataset = SequenceActivationDataset(
        dataset_path,
        
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["training"][
            "batch_size"
        ],
        shuffle=False
    )

    reconstructor_type = cfg[
        "training"
    ].get(
        "reconstructor_type"  )

    hidden_dim = cfg["training"][
        "hidden_dim"
    ]

    max_seq_len = cfg["activation"].get(
        "max_length",
        128,
    )

    if reconstructor_type == "token_decoder":

        model = TokenLevelReconstructor(
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            n_layers=cfg["training"]["decoder_layers"],
            n_heads=cfg["training"]["decoder_heads"],
            encoder_name=cfg["training"].get(
                "encoder_name",
                "distilgpt2",
            )
        )

    else:

        model = ActivationReconstructor(
            hidden_dim=hidden_dim
        )

    checkpoint_path = (
        checkpoint_dir
        / "best_model.pt"
    )

    print(
        f"[INFO] Loading checkpoint {checkpoint_path}"
    )

    state = torch.load(
        checkpoint_path,
        map_location=device,

    )

    model.load_state_dict(
        state
    )

    model.to(device)
    model.eval()

    original = []
    reconstructed = []

    print(
        "[INFO] Running manifold evaluation..."
    )

    for batch in loader:

        descriptions = batch[
            "description"
        ]

        target = batch[
            "activation_sequence"
        ].to(device)

        pred = model(
            descriptions
        )

        if pred.ndim == 3:
            pred = pred.mean(dim=1)

        pred = F.normalize(
            pred,
            dim=-1,
        )

        target = F.normalize(
            target,
            dim=-1,
        )

        original.append(
            target.cpu()
        )

        reconstructed.append(
            pred.cpu()
        )

    original = torch.cat(
        original
    ).numpy()

    reconstructed = torch.cat(
        reconstructed
    ).numpy()

    print(
        "[INFO] Computing metrics..."
    )

    metrics = {
        "cosine_similarity":
            cosine_similarity(
                torch.tensor(
                    original
                ),
                torch.tensor(
                    reconstructed
                ),
            ),

        "euclidean_distance":
            euclidean_distance(
                torch.tensor(
                    original
                ),
                torch.tensor(
                    reconstructed
                ),
            ),

        "knn_overlap":
            compute_knn_overlap(
                original,
                reconstructed,
            ),

        "manifold_consistency":
            manifold_consistency(
                original,
                reconstructed,
            ),

        "local_density_ratio":
            local_density_ratio(
                original,
                reconstructed,
            ),

        "centroid_distance":
            centroid_distance(
                original,
                reconstructed,
            ),

        "pca_projection_error":
            pca_manifold_score(
                original,
                reconstructed,
            ),
    }

    save_path = (
        checkpoint_dir
        / "manifold_metrics.json"
    )

    with open(
        save_path,
        "w",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    print("\n=== MANIFOLD RESULTS ===")

    for k, v in metrics.items():
        print(
            f"{k:28s}: {v:.6f}"
        )

    print(
        f"\n[OK] Saved metrics → {save_path}"
    )

    return metrics


if __name__ == "__main__":
    evaluate_manifold()