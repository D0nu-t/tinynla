"""
nla/losses.py

Reconstruction loss functions.

cosine_loss                  — 1 - cosine_similarity; pooled baseline
sequence_cosine_loss         — position-by-position cosine; no masking
masked_sequence_cosine_loss  — position-by-position cosine with padding mask (v3)
combined_loss                — α·cosine + (1-α)·MSE; useful early in training
"""

import torch
import torch.nn.functional as F


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    1 - mean cosine similarity. Scale-invariant; works on L2-normalized vectors.
    pred, target: [batch, hidden_dim]
    """
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return 1 - (pred * target).sum(dim=-1).mean()


def sequence_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Position-by-position cosine loss without masking.
    Assumes all positions are valid (uniform sequence length in batch).
    pred, target: [batch, seq_len, hidden_dim]
    """
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return 1 - (pred * target).sum(dim=-1).mean()


def masked_sequence_cosine_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Position-by-position cosine loss with a boolean padding mask.

    Only valid (non-padded) positions contribute to the loss.
    This is the correct loss for variable-length sequence batches.

    Args:
        pred:   [batch, seq_len, hidden_dim]
        target: [batch, seq_len, hidden_dim]
        mask:   [batch, seq_len]  — True at valid positions, False at padding

    Returns:
        Scalar loss: 1 - (mean cosine over valid positions)
    """
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)

    cosine = (pred * target).sum(dim=-1)    # [batch, seq_len]

    # Zero out padded positions, sum over valid, normalize by count
    masked_cosine = cosine * mask.float()
    n_valid = mask.float().sum().clamp(min=1.0)

    return 1 - masked_cosine.sum() / n_valid


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.8,
) -> torch.Tensor:
    """
    α·cosine_loss + (1-α)·MSE over normalized vectors.
    Useful early in training when cosine loss gradient landscape is flat.
    pred, target: [batch, hidden_dim]
    """
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    c_loss = 1 - (pred_n * target_n).sum(dim=-1).mean()
    mse = F.mse_loss(pred_n, target_n)
    return alpha * c_loss + (1 - alpha) * mse