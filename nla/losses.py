"""
nla/losses.py

Trajectory-aware reconstruction losses for sequence activation modeling.

Implemented losses
------------------
cosine_loss
    Legacy pooled cosine loss.

sequence_cosine_loss
    Per-token cosine loss without masking.

masked_sequence_cosine_loss
    Per-token cosine loss with padding mask.

trajectory_velocity_loss
    Penalizes mismatch in first-order trajectory dynamics:
        Δh_t = h_t - h_{t-1}

trajectory_acceleration_loss
    Penalizes mismatch in second-order trajectory curvature:
        Δ²h_t = Δh_t - Δh_{t-1}

trajectory_magnitude_loss
    Penalizes norm mismatch across positions.

combined_sequence_loss
    Main v3 objective:
        cosine
      + velocity consistency
      + acceleration consistency
      + optional magnitude regularization

This upgrades reconstruction from static token geometry
to trajectory-level manifold dynamics.
"""

from typing import Dict

import torch
import torch.nn.functional as F


# ============================================================================
# Legacy pooled losses
# ============================================================================

def cosine_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    1 - mean cosine similarity.

    pred, target:
        [batch, hidden_dim]
    """
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)

    return 1.0 - (pred * target).sum(dim=-1).mean()


# ============================================================================
# Sequence cosine losses
# ============================================================================

def sequence_cosine_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Unmasked token-wise cosine loss.

    pred, target:
        [batch, seq_len, hidden_dim]
    """
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)

    cosine = (pred * target).sum(dim=-1)

    return 1.0 - cosine.mean()


def masked_sequence_cosine_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Token-wise cosine loss with padding mask.

    Args:
        pred:
            [batch, seq_len, hidden_dim]

        target:
            [batch, seq_len, hidden_dim]

        mask:
            [batch, seq_len]
            True at valid positions.
    """
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)

    cosine = (pred * target).sum(dim=-1)

    masked_cosine = cosine * mask.float()

    n_valid = mask.float().sum().clamp(min=1.0)

    return 1.0 - (masked_cosine.sum() / n_valid)


# ============================================================================
# Trajectory dynamics losses
# ============================================================================

def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Utility masked mean.

    values:
        [batch, seq_len]

    mask:
        [batch, seq_len]
    """
    values = values * mask.float()

    denom = mask.float().sum().clamp(min=1.0)

    return values.sum() / denom


def trajectory_velocity_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Match first-order trajectory dynamics.

    Δh_t = h_t - h_{t-1}

    Encourages smooth manifold evolution consistency.
    """

    pred_vel = pred[:, 1:, :] - pred[:, :-1, :]
    target_vel = target[:, 1:, :] - target[:, :-1, :]

    pred_vel = F.normalize(pred_vel, dim=-1)
    target_vel = F.normalize(target_vel, dim=-1)

    cosine = (pred_vel * target_vel).sum(dim=-1)

    vel_mask = mask[:, 1:] & mask[:, :-1]

    return 1.0 - _masked_mean(cosine, vel_mask)


def trajectory_acceleration_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Match second-order trajectory curvature.

    Δ²h_t = Δh_t - Δh_{t-1}

    Captures turning structure of the activation manifold.
    """

    pred_vel = pred[:, 1:, :] - pred[:, :-1, :]
    target_vel = target[:, 1:, :] - target[:, :-1, :]

    pred_acc = pred_vel[:, 1:, :] - pred_vel[:, :-1, :]
    target_acc = target_vel[:, 1:, :] - target_vel[:, :-1, :]

    pred_acc = F.normalize(pred_acc, dim=-1)
    target_acc = F.normalize(target_acc, dim=-1)

    cosine = (pred_acc * target_acc).sum(dim=-1)

    acc_mask = (
        mask[:, 2:]
        & mask[:, 1:-1]
        & mask[:, :-2]
    )

    return 1.0 - _masked_mean(cosine, acc_mask)


def trajectory_magnitude_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Match hidden-state norms across positions.

    Helps preserve residual stream energy profile.
    """

    pred_norm = pred.norm(dim=-1)
    target_norm = target.norm(dim=-1)

    mse = (pred_norm - target_norm).pow(2)

    return _masked_mean(mse, mask)


# ============================================================================
# Main v3 trajectory objective
# ============================================================================

def combined_sequence_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    cosine_weight: float = 1.0,
    velocity_weight: float = 0.25,
    acceleration_weight: float = 0.10,
    magnitude_weight: float = 0.05,
    return_components: bool = False,
):
    """
    Main trajectory-aware reconstruction objective.

    Total loss:

        L =
            cosine
          + λ_v * velocity
          + λ_a * acceleration
          + λ_m * magnitude

    This upgrades the model from:
        static hidden matching

    to:
        trajectory manifold reconstruction.
    """

    cosine = masked_sequence_cosine_loss(
        pred,
        target,
        mask,
    )

    velocity = trajectory_velocity_loss(
        pred,
        target,
        mask,
    )

    acceleration = trajectory_acceleration_loss(
        pred,
        target,
        mask,
    )

    magnitude = trajectory_magnitude_loss(
        pred,
        target,
        mask,
    )

    total = (
        cosine_weight * cosine
        + velocity_weight * velocity
        + acceleration_weight * acceleration
        + magnitude_weight * magnitude
    )

    if return_components:
        return total, {
            "cosine": cosine.detach(),
            "velocity": velocity.detach(),
            "acceleration": acceleration.detach(),
            "magnitude": magnitude.detach(),
        }

    return total


# ============================================================================
# Legacy combined pooled loss
# ============================================================================

def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.8,
) -> torch.Tensor:
    """
    Legacy pooled cosine + MSE loss.

    pred, target:
        [batch, hidden_dim]
    """

    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)

    cosine = 1.0 - (pred_n * target_n).sum(dim=-1).mean()

    mse = F.mse_loss(pred_n, target_n)

    return alpha * cosine + (1.0 - alpha) * mse