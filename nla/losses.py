"""
nla/losses.py

Reconstruction loss functions.

cosine_loss          — 1 - cosine_similarity; primary training objective
sequence_cosine_loss — same, applied token-by-token for sequence reconstructors
combined_loss        — weighted sum of cosine and MSE; more stable early in training
"""

import torch
import torch.nn.functional as F


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    1 - mean cosine similarity between pred and target.

    Both inputs are normalized internally; this loss is scale-invariant
    and compatible with L2-normalized activation vectors.

    pred, target: [batch, dim]
    """
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return 1 - (pred * target).sum(dim=-1).mean()


def sequence_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Cosine loss applied position-by-position for sequence reconstructors.

    pred, target: [batch, seq_len, dim]
    """
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return 1 - (pred * target).sum(dim=-1).mean()


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.8,
) -> torch.Tensor:
    """
    alpha * cosine_loss + (1 - alpha) * MSE.

    MSE over normalized vectors penalizes both directional and magnitude error.
    Useful early in training when cosine loss alone has a flat gradient landscape.

    pred, target: [batch, dim]
    """
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    c_loss = 1 - (pred_n * target_n).sum(dim=-1).mean()
    mse = F.mse_loss(pred_n, target_n)
    return alpha * c_loss + (1 - alpha) * mse