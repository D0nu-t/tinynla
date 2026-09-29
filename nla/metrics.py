"""
nla/metrics.py

Reconstruction metrics.

Primary NLA metric (Fraser-Taliente et al. 2026; kitft docs/inference.md):

    FVE = 1 - E||g - p||^2 / E||g - mu||^2

where g, p are the gold / predicted activations scaled to a common norm
(`mse_scale`, default sqrt(d) -> direction-only) and mu is the mean of the
*scaled* golds, deliberately NOT re-normalised (re-normalising the mean
inflates FVE). FVE = 0 means "no better than guessing the mean".
"""

import math
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

ScaleSpec = Optional[Union[str, float]]


def resolve_scale(scale: ScaleSpec, d_model: int) -> Optional[float]:
    """'sqrt_d_model' -> sqrt(d); float -> float; None -> None (raw)."""
    if scale is None:
        return None
    if scale == "sqrt_d_model":
        return math.sqrt(d_model)
    return float(scale)


def rescale(x: torch.Tensor, scale: Optional[float]) -> torch.Tensor:
    """Scale each row to L2 norm `scale` (None leaves x unchanged)."""
    x = x.float()
    if scale is None:
        return x
    return F.normalize(x, dim=-1) * scale


def normalized_mse(
    pred: torch.Tensor,
    gold: torch.Tensor,
    scale: Optional[float],
) -> torch.Tensor:
    """Per-row MSE after rescaling both sides. Returns [N]."""
    return ((rescale(pred, scale) - rescale(gold, scale)) ** 2).mean(dim=-1)


def per_sample_fve(
    pred: torch.Tensor,
    gold: torch.Tensor,
    variance: float,
    scale: Optional[float],
) -> torch.Tensor:
    """
    Per-row FVE on the dataset scale: 1 - ||g - p||^2 / Var_train, where
    Var_train = E||g - mean||^2 over training activations. Averaging it over
    a dataset gives `fve` (same numerator, same denominator). Returns [N].
    """
    err = ((rescale(pred, scale) - rescale(gold, scale).to(pred.device)) ** 2).sum(dim=-1)
    return 1.0 - err / variance


def fve(
    pred: torch.Tensor,
    gold: torch.Tensor,
    mean: torch.Tensor,
    scale: Optional[float],
    normalize_pred: bool = True,
) -> float:
    """
    Fraction of variance explained over a set of activations.

    Args:
        pred:   [N, d] reconstructions
        gold:   [N, d] true activations (raw)
        mean:   [d] mean of rescale(gold_train, scale) - from TRAIN data
        scale:  common norm (see resolve_scale)
        normalize_pred: False only to score the mean itself as a baseline
    """
    g = rescale(gold, scale)
    p = rescale(pred, scale) if normalize_pred else pred.float()
    mu = mean.float().to(g.device)

    err = ((g - p.to(g.device)) ** 2).sum(dim=-1).mean()
    var = ((g - mu) ** 2).sum(dim=-1).mean()

    return float(1.0 - err / var)


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