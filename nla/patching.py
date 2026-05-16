"""
nla/patching.py

Activation patching mechanisms for injecting reconstructed vectors
back into a frozen transformer's forward pass.

v3 primary:
  SequenceInterpolationPatcher  — takes [seq_len, hidden_dim]; patches each
                                   position independently with interpolation.
                                   Correct match for sequence-mode extraction.

Legacy (retained for ablation):
  LastTokenPatcher    — replaces hidden[:, -1, :] only
  BroadcastPatcher    — broadcasts one vector across all positions
  InterpolationPatcher — blends one vector across all positions (pooled mode)

ActivationPatcher = SequenceInterpolationPatcher  (v3 canonical alias)

Usage (sequence mode):
    patcher = SequenceInterpolationPatcher(recon_seq, alpha=0.5)
    handle = model.transformer.h[layer_idx].register_forward_hook(patcher.hook_fn)
    outputs = model(**inputs)
    handle.remove()
"""

import torch


class LastTokenPatcher:
    """
    Legacy. Replaces only hidden[:, -1, :].
    Mismatched with mean-pool extraction. Retained for ablation.
    """

    def __init__(self, replacement_vector: torch.Tensor):
        self.replacement = replacement_vector

    def hook_fn(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            hidden = outputs[0].clone()
            hidden[:, -1, :] = self.replacement
            return (hidden,) + outputs[1:]
        outputs = outputs.clone()
        outputs[:, -1, :] = self.replacement
        return outputs


class BroadcastPatcher:
    """
    Replaces all token positions with a single reconstructed vector.
    Geometrically coherent with mean-pool extraction but forces uniform state.
    Retained for ablation.
    """

    def __init__(self, replacement_vector: torch.Tensor):
        if replacement_vector.dim() == 1:
            replacement_vector = replacement_vector.unsqueeze(0)
        self.replacement = replacement_vector

    def hook_fn(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            hidden = outputs[0].clone()
            hidden[:] = self.replacement.unsqueeze(1).expand_as(hidden)
            return (hidden,) + outputs[1:]
        outputs = outputs.clone()
        outputs[:] = self.replacement.unsqueeze(1).expand_as(outputs)
        return outputs


class InterpolationPatcher:
    """
    Pooled soft patching: h_patch = α·h_recon + (1-α)·h_orig broadcast over all positions.

    Legacy pooled-mode patcher. Use SequenceInterpolationPatcher for sequence mode.

    alpha=0.0 → identity; alpha=1.0 → full replacement.
    """

    def __init__(self, replacement_vector: torch.Tensor, alpha: float = 1.0):
        if replacement_vector.dim() == 1:
            replacement_vector = replacement_vector.unsqueeze(0)
        self.replacement = replacement_vector
        self.alpha = alpha

    def hook_fn(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            hidden = outputs[0].clone()
            replacement = self.replacement.unsqueeze(1).expand_as(hidden)
            patched = self.alpha * replacement + (1.0 - self.alpha) * hidden
            return (patched,) + outputs[1:]
        outputs = outputs.clone()
        replacement = self.replacement.unsqueeze(1).expand_as(outputs)
        return self.alpha * replacement + (1.0 - self.alpha) * outputs

    def with_alpha(self, alpha: float) -> "InterpolationPatcher":
        return InterpolationPatcher(self.replacement, alpha=alpha)


class SequenceInterpolationPatcher:
    """
    Per-position soft patching for sequence-level reconstructions.

    h_patch[t] = α·h_recon[t] + (1-α)·h_orig[t]   for t in [0, patch_len)

    patch_len = min(input_seq_len, recon_seq_len).
    Positions beyond patch_len are left untouched.

    This is the geometrically correct patcher for sequence-mode extraction:
    each position in the residual stream is replaced by the corresponding
    position in the reconstructed trajectory, not a broadcast summary.

    Args:
        reconstructed_sequence: Tensor[seq_len, hidden_dim] or [1, seq_len, hidden_dim]
        alpha:                  0.0 = identity, 1.0 = full per-position replacement.
    """

    def __init__(self, reconstructed_sequence: torch.Tensor, alpha: float = 1.0):
        # Normalize to [seq_len, hidden_dim]
        if reconstructed_sequence.dim() == 3:
            reconstructed_sequence = reconstructed_sequence.squeeze(0)
        self.recon = reconstructed_sequence    # [recon_seq_len, hidden_dim]
        self.alpha = alpha

    def hook_fn(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            hidden = outputs[0].clone()         # [batch, input_seq_len, hidden_dim]
            self._patch_inplace(hidden)
            return (hidden,) + outputs[1:]
        outputs = outputs.clone()
        self._patch_inplace(outputs)
        return outputs

    def _patch_inplace(self, hidden: torch.Tensor) -> None:
        """hidden: [batch, input_seq_len, hidden_dim] — modified in place."""
        input_seq_len = hidden.shape[1]
        recon_seq_len = self.recon.shape[0]
        patch_len = min(input_seq_len, recon_seq_len)

        # Move recon to same device as hidden
        recon = self.recon[:patch_len].to(hidden.device)   # [patch_len, hidden_dim]
        recon = recon.unsqueeze(0)                          # [1, patch_len, hidden_dim]

        hidden[:, :patch_len, :] = (
            self.alpha * recon
            + (1.0 - self.alpha) * hidden[:, :patch_len, :]
        )

    def with_alpha(self, alpha: float) -> "SequenceInterpolationPatcher":
        return SequenceInterpolationPatcher(self.recon, alpha=alpha)


# v3 canonical alias
ActivationPatcher = SequenceInterpolationPatcher