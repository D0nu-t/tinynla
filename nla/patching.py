"""
nla/patching.py

Activation patching mechanisms for injecting reconstructed activations
back into a frozen transformer's forward pass.

Phase 1–3 upgrade:
------------------
Canonical patching now supports trajectory-level functional fidelity.

Primary:
    SequenceInterpolationPatcher
        - per-position trajectory patching
        - supports masking
        - supports sequence alignment
        - optional prefix-only patching
        - geometry-preserving interpolation

Legacy (retained for ablation):
    LastTokenPatcher
    BroadcastPatcher
    InterpolationPatcher

Canonical alias:
    ActivationPatcher = SequenceInterpolationPatcher

Design principles:
------------------
1. Sequence-aware patching:
    pooled-vector broadcast destroys causal structure.

2. Functional fidelity:
    preserve token-position computation.

3. Safe alignment:
    mismatched sequence lengths handled gracefully.

4. Research flexibility:
    enable ablations without rewriting infrastructure.
"""

from __future__ import annotations

from typing import Optional

import torch


# ============================================================
# Legacy patchers (retained for ablation)
# ============================================================


class LastTokenPatcher:
    """
    Legacy patcher.

    Replaces only hidden[:, -1, :].

    Useful only for:
        - historical comparison
        - patching shock replication
        - ablation studies

    Geometrically mismatched with:
        - mean pooling
        - sequence extraction
    """

    def __init__(self, replacement_vector: torch.Tensor):
        self.replacement = replacement_vector

    def hook_fn(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            hidden = outputs[0].clone()
            hidden[:, -1, :] = self.replacement.to(hidden.device)
            return (hidden,) + outputs[1:]

        hidden = outputs.clone()
        hidden[:, -1, :] = self.replacement.to(hidden.device)
        return hidden


class BroadcastPatcher:
    """
    Legacy pooled patcher.

    Broadcasts one vector across all positions.

    Useful for:
        - pooled baselines
        - geometry ablations

    Limitation:
        destroys token-level causal structure.
    """

    def __init__(self, replacement_vector: torch.Tensor):
        if replacement_vector.dim() == 1:
            replacement_vector = replacement_vector.unsqueeze(0)

        self.replacement = replacement_vector

    def hook_fn(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            hidden = outputs[0].clone()

            replacement = (
                self.replacement
                .to(hidden.device)
                .unsqueeze(1)
                .expand_as(hidden)
            )

            hidden[:] = replacement
            return (hidden,) + outputs[1:]

        hidden = outputs.clone()

        replacement = (
            self.replacement
            .to(hidden.device)
            .unsqueeze(1)
            .expand_as(hidden)
        )

        hidden[:] = replacement
        return hidden


class InterpolationPatcher:
    """
    Legacy pooled interpolation patcher.

    h_patch =
        α * h_reconstructed
        + (1 - α) * h_original

    broadcast over all positions.

    Useful for:
        - pooled-vector baselines
        - comparison with sequence patching

    alpha:
        0.0 → identity
        1.0 → full replacement
    """

    def __init__(
        self,
        replacement_vector: torch.Tensor,
        alpha: float = 1.0,
    ):
        if replacement_vector.dim() == 1:
            replacement_vector = replacement_vector.unsqueeze(0)

        self.replacement = replacement_vector
        self.alpha = alpha

    def hook_fn(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            hidden = outputs[0].clone()

            replacement = (
                self.replacement
                .to(hidden.device)
                .unsqueeze(1)
                .expand_as(hidden)
            )

            patched = (
                self.alpha * replacement
                + (1.0 - self.alpha) * hidden
            )

            return (patched,) + outputs[1:]

        hidden = outputs.clone()

        replacement = (
            self.replacement
            .to(hidden.device)
            .unsqueeze(1)
            .expand_as(hidden)
        )

        return (
            self.alpha * replacement
            + (1.0 - self.alpha) * hidden
        )

    def with_alpha(self, alpha: float):
        return InterpolationPatcher(
            replacement_vector=self.replacement,
            alpha=alpha,
        )


# ============================================================
# Canonical v4 trajectory patcher
# ============================================================


class SequenceInterpolationPatcher:
    """
    Canonical trajectory-level patcher.

    Performs per-position interpolation:

        h_patch[t] =
            α * h_recon[t]
            + (1 - α) * h_orig[t]

    for token positions t.

    This preserves causal structure and matches:

        extract_sequence()
            ↔
        TokenLevelReconstructor
            ↔
        sequence functional evaluation

    Supports:
    ---------
    - sequence-length mismatch
    - prefix-only patching
    - optional attention-mask patching
    - interpolation sweeps
    - trajectory-level functional fidelity

    Args:
        reconstructed_sequence:
            [seq_len, hidden_dim]
            OR
            [1, seq_len, hidden_dim]

        alpha:
            interpolation strength

        patch_mode:
            "prefix"
                patch first N positions

            "suffix"
                patch final N positions

            "full"
                patch aligned positions

        attention_mask:
            optional [seq_len]
            only patch valid tokens

    Example:
    --------
    patcher = SequenceInterpolationPatcher(
        reconstructed_sequence=recon_seq,
        alpha=0.5,
    )

    handle = layer.register_forward_hook(
        patcher.hook_fn
    )
    """

    VALID_PATCH_MODES = {
        "prefix",
        "suffix",
        "full",
    }

    def __init__(
        self,
        reconstructed_sequence: torch.Tensor,
        alpha: float = 1.0,
        patch_mode: str = "full",
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if reconstructed_sequence.dim() == 3:
            reconstructed_sequence = (
                reconstructed_sequence.squeeze(0)
            )

        if reconstructed_sequence.dim() != 2:
            raise ValueError(
                "Expected reconstructed_sequence shape "
                "[seq_len, hidden_dim]"
            )

        if patch_mode not in self.VALID_PATCH_MODES:
            raise ValueError(
                f"Invalid patch_mode={patch_mode}. "
                f"Expected one of "
                f"{self.VALID_PATCH_MODES}"
            )

        self.recon = reconstructed_sequence
        self.alpha = float(alpha)
        self.patch_mode = patch_mode
        self.attention_mask = attention_mask

    # ========================================================
    # Hook entry point
    # ========================================================

    def hook_fn(self, module, inputs, outputs):
        """
        Forward hook entry point.
        """

        if isinstance(outputs, tuple):
            hidden = outputs[0].clone()

            patched = self._patch_hidden(hidden)

            return (patched,) + outputs[1:]

        hidden = outputs.clone()

        return self._patch_hidden(hidden)

    # ========================================================
    # Core patching
    # ========================================================

    def _patch_hidden(
        self,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        hidden:
            [batch, seq_len, hidden_dim]
        """

        batch_size, input_seq_len, hidden_dim = hidden.shape

        recon = self.recon.to(hidden.device)

        recon_seq_len = recon.shape[0]

        patch_len = min(
            input_seq_len,
            recon_seq_len,
        )

        if patch_len == 0:
            return hidden

        # -----------------------------------
        # sequence alignment
        # -----------------------------------

        if self.patch_mode == "prefix":
            hidden_slice = slice(0, patch_len)
            recon_slice = slice(0, patch_len)

        elif self.patch_mode == "suffix":
            hidden_slice = slice(
                input_seq_len - patch_len,
                input_seq_len,
            )

            recon_slice = slice(
                recon_seq_len - patch_len,
                recon_seq_len,
            )

        else:
            # full aligned patching
            hidden_slice = slice(0, patch_len)
            recon_slice = slice(0, patch_len)

        recon_chunk = recon[recon_slice]

        recon_chunk = recon_chunk.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )

        original_chunk = hidden[:, hidden_slice, :]

        patched_chunk = (
            self.alpha * recon_chunk
            + (1.0 - self.alpha) * original_chunk
        )

        # -----------------------------------
        # optional masking
        # -----------------------------------

        if self.attention_mask is not None:
            mask = (
                self.attention_mask[:patch_len]
                .to(hidden.device)
                .bool()
                .unsqueeze(0)
                .unsqueeze(-1)
            )

            original_chunk = hidden[:, hidden_slice, :]

            patched_chunk = torch.where(
                mask,
                patched_chunk,
                original_chunk,
            )

        hidden[:, hidden_slice, :] = patched_chunk

        return hidden

    # ========================================================
    # Convenience API
    # ========================================================

    def with_alpha(
        self,
        alpha: float,
    ) -> "SequenceInterpolationPatcher":
        """
        Create copy with new interpolation strength.
        """

        return SequenceInterpolationPatcher(
            reconstructed_sequence=self.recon,
            alpha=alpha,
            patch_mode=self.patch_mode,
            attention_mask=self.attention_mask,
        )


# ============================================================
# Canonical alias
# ============================================================

ActivationPatcher = SequenceInterpolationPatcher