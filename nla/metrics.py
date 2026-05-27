"""
nla/metrics.py

Geometric reconstruction metrics.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List


def cosine_similarity_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return F.cosine_similarity(pred, target).mean().item()


def aggregate_metrics(values: List[float]) -> Dict[str, float]:
    """Return mean and std for a list of scalar metric values."""
    if not values:
        return {"mean": 0.0, "std": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std())}

def manifold_offmanifold_ratio(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    threshold_std: float = 2.0,
) -> float:
    """
    Fraction of reconstructed points lying unusually
    far from the original activation manifold.

    Uses distance-to-centroid z-score thresholding.
    """

    centroid = original.mean(dim=0)

    orig_dist = torch.norm(
        original - centroid,
        dim=-1,
    )

    recon_dist = torch.norm(
        reconstructed - centroid,
        dim=-1,
    )

    mean_dist = orig_dist.mean()
    std_dist = orig_dist.std()

    threshold = mean_dist + (
        threshold_std * std_dist
    )

    offmanifold = (
        recon_dist > threshold
    ).float()

    return offmanifold.mean().item()