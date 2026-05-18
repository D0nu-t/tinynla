"""
nla/losses.py

Reconstruction loss functions.

cosine_loss                      — pooled cosine loss
sequence_cosine_loss             — tokenwise cosine loss
masked_sequence_cosine_loss      — tokenwise cosine with padding mask
combined_loss                    — pooled cosine + MSE
combined_sequence_loss           — sequence cosine + MSE
masked_combined_sequence_loss    — masked sequence cosine + MSE (v3 default)
"""

import torch
import torch.nn.functional as F


# ==========================================================
# Pooled losses
# ==========================================================

def cosine_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    1 - cosine similarity.

    Args:
        pred:   [batch, hidden_dim]
        target: [batch, hidden_dim]
    """
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)

    cosine = (pred * target).sum(dim=-1)

    return 1 - cosine.mean()


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.8,
) -> torch.Tensor:
    """
    α·cosine + (1-α)·MSE

    Useful early in training.

    Args:
        pred:   [batch, hidden_dim]
        target: [batch, hidden_dim]
    """
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)

    cosine = (
        pred_n * target_n
    ).sum(dim=-1).mean()

    c_loss = 1 - cosine

    mse = F.mse_loss(
        pred_n,
        target_n,
    )

    return alpha * c_loss + (1 - alpha) * mse


# ==========================================================
# Sequence losses
# ==========================================================

def sequence_cosine_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Tokenwise cosine loss.

    Args:
        pred:   [batch, seq_len, hidden_dim]
        target: [batch, seq_len, hidden_dim]
    """
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)

    cosine = (
        pred * target
    ).sum(dim=-1)

    return 1 - cosine.mean()


def masked_sequence_cosine_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Tokenwise cosine loss with padding mask.

    Args:
        pred:   [batch, seq_len, hidden_dim]
        target: [batch, seq_len, hidden_dim]
        mask:   [batch, seq_len]
    """
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)

    cosine = (
        pred * target
    ).sum(dim=-1)

    mask = mask.float()

    masked_cosine = cosine * mask

    n_valid = (
        mask.sum()
        .clamp(min=1.0)
    )

    return 1 - (
        masked_cosine.sum()
        / n_valid
    )


def combined_sequence_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.8,
) -> torch.Tensor:
    """
    Sequence cosine + MSE.

    Args:
        pred:   [batch, seq_len, hidden_dim]
        target: [batch, seq_len, hidden_dim]
    """
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)

    cosine = (
        pred_n * target_n
    ).sum(dim=-1)

    c_loss = 1 - cosine.mean()

    mse = F.mse_loss(
        pred_n,
        target_n,
    )

    return alpha * c_loss + (
        1 - alpha
    ) * mse


def masked_combined_sequence_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.8,
) -> torch.Tensor:
    """
    v3 default loss.

    Masked tokenwise cosine + masked MSE.

    Args:
        pred:   [batch, seq_len, hidden_dim]
        target: [batch, seq_len, hidden_dim]
        mask:   [batch, seq_len]
    """
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)

    cosine = (
        pred_n * target_n
    ).sum(dim=-1)

    mask_f = mask.float()

    n_valid = (
        mask_f.sum()
        .clamp(min=1.0)
    )

    cosine_loss_val = 1 - (
        (cosine * mask_f).sum()
        / n_valid
    )

    hidden_dim = pred.shape[-1]

    expanded_mask = (
        mask_f.unsqueeze(-1)
        .expand_as(pred_n)
    )

    mse = (
        ((pred_n - target_n) ** 2)
        * expanded_mask
    ).sum()

    mse = mse / (
        n_valid * hidden_dim
    )

    return (
        alpha * cosine_loss_val
        + (1 - alpha) * mse
    )