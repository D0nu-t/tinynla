"""
nla/patching.py

Activation patching mechanisms for injecting reconstructed vectors
back into a frozen transformer's forward pass.

Three classes:

  LastTokenPatcher      — legacy; replaces hidden[:, -1, :]
                          Mismatched with mean-pooled extraction; causes
                          distributional shock. Retained for ablation.

  BroadcastPatcher      — replaces all token positions with the reconstructed
                          vector. Geometrically coherent with mean-pool extraction
                          but still causes full replacement shock at alpha=1.

  InterpolationPatcher  — h_patch = alpha * h_reconstructed + (1-alpha) * h_original
                          Correct tool for diagnosing patching shock magnitude.
                          alpha=0.0 is identity; alpha=1.0 is full replacement.
                          Sweep alpha to get a KL-vs-injection-strength curve.

Usage:
    patcher = InterpolationPatcher(reconstructed_vec, alpha=0.5)
    handle = model.transformer.h[layer_idx].register_forward_hook(patcher.hook_fn)
    outputs = model(**inputs)
    handle.remove()
"""

import torch


class LastTokenPatcher:
    """
    Legacy patcher. Replaces only the last token position.

    Mismatched with mean-pooled extraction: the reconstructed vector
    is a sequence summary injected into a single causal position.
    Retained for ablation comparison with BroadcastPatcher.
    """

    def __init__(self, replacement_vector: torch.Tensor):
        # replacement_vector: [1, hidden_dim] or [hidden_dim]
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
    Replaces all token positions with the reconstructed vector.

    Geometrically coherent with mean-pool extraction: the injected
    vector represents the full sequence, and is broadcast uniformly
    across all positions. Still causes full replacement shock.
    Use InterpolationPatcher with alpha=1.0 for equivalent effect
    with alpha-sweep capability.
    """

    def __init__(self, replacement_vector: torch.Tensor):
        # replacement_vector: [1, hidden_dim] or [hidden_dim]
        if replacement_vector.dim() == 1:
            replacement_vector = replacement_vector.unsqueeze(0)  # [1, hidden]
        self.replacement = replacement_vector

    def hook_fn(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            hidden = outputs[0].clone()         # [batch, seq, hidden]
            # Broadcast replacement across all positions
            hidden[:] = self.replacement.unsqueeze(1).expand_as(hidden)
            return (hidden,) + outputs[1:]
        outputs = outputs.clone()
        outputs[:] = self.replacement.unsqueeze(1).expand_as(outputs)
        return outputs


class InterpolationPatcher:
    """
    Soft patching: h_patch = alpha * h_reconstructed + (1 - alpha) * h_original

    alpha=0.0 → identity (no change; KL should be ~0)
    alpha=1.0 → full replacement (equivalent to BroadcastPatcher)

    Sweeping alpha gives a KL-vs-injection-strength curve that isolates
    how much of the downstream distributional shift is caused by the
    reconstruction error vs. the replacement shock itself.

    Args:
        replacement_vector: [1, hidden_dim] or [hidden_dim] — reconstructed activation.
        alpha:              Interpolation weight in [0, 1].
    """

    def __init__(self, replacement_vector: torch.Tensor, alpha: float = 1.0):
        if replacement_vector.dim() == 1:
            replacement_vector = replacement_vector.unsqueeze(0)
        self.replacement = replacement_vector
        self.alpha = alpha

    def hook_fn(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            hidden = outputs[0].clone()         # [batch, seq, hidden]
            replacement = self.replacement.unsqueeze(1).expand_as(hidden)
            patched = self.alpha * replacement + (1.0 - self.alpha) * hidden
            return (patched,) + outputs[1:]
        outputs = outputs.clone()
        replacement = self.replacement.unsqueeze(1).expand_as(outputs)
        return self.alpha * replacement + (1.0 - self.alpha) * outputs

    # Convenience: return a new patcher with a different alpha
    def with_alpha(self, alpha: float) -> "InterpolationPatcher":
        return InterpolationPatcher(self.replacement, alpha=alpha)


# Canonical alias — default patcher used by training scripts
ActivationPatcher = InterpolationPatcher