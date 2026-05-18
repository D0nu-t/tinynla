"""
nla/hooks.py

Forward hook infrastructure.

ActivationHook  — single-layer capture
MultiLayerHook  — captures multiple transformer layers in one forward pass

v3 upgrades:
  - Safe hook lifecycle management
  - Deterministic activation capture
  - Explicit cleanup support
  - Robust tuple-output handling
  - Optional CPU offloading
"""

from typing import Dict, List, Optional

import torch


# ============================================================================
# Single-layer hook
# ============================================================================

class ActivationHook:
    """
    Capture activations from a single transformer block.

    Stores activations sequentially in case multiple forwards occur.

    Args:
        detach:
            If True, detach from graph.

        move_to_cpu:
            If True, immediately move activations to CPU.
            Useful for reducing GPU memory pressure.
    """

    def __init__(
        self,
        detach: bool = True,
        move_to_cpu: bool = False,
    ):
        self.detach = detach
        self.move_to_cpu = move_to_cpu
        self.activations: List[torch.Tensor] = []

    def hook_fn(
        self,
        module,
        inputs,
        outputs,
    ):
        """
        Forward hook callback.
        """
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs

        if self.detach:
            hidden = hidden.detach()

        if self.move_to_cpu:
            hidden = hidden.cpu()

        self.activations.append(hidden)

    def clear(self):
        """Clear cached activations."""
        self.activations.clear()

    def latest(self) -> Optional[torch.Tensor]:
        """
        Return most recent activation.

        Returns:
            Tensor or None
        """
        if not self.activations:
            return None
        return self.activations[-1]


# ============================================================================
# Multi-layer hook
# ============================================================================

class MultiLayerHook:
    """
    Capture activations from multiple transformer blocks
    in a single forward pass.

    Usage:
        hook = MultiLayerHook()
        hook.register(model, [1, 3, 5])

        model(**inputs)

        acts = hook.activations
        hook.remove()
    """

    def __init__(
        self,
        detach: bool = True,
        move_to_cpu: bool = False,
    ):
        self.detach = detach
        self.move_to_cpu = move_to_cpu

        self.activations: Dict[int, torch.Tensor] = {}
        self._handles: List = []

    # ------------------------------------------------------------------
    # Hook creation
    # ------------------------------------------------------------------

    def _make_hook(
        self,
        layer_idx: int,
    ):
        """
        Create closure-bound hook for one layer.
        """

        def fn(module, inputs, outputs):

            hidden = (
                outputs[0]
                if isinstance(outputs, tuple)
                else outputs
            )

            if self.detach:
                hidden = hidden.detach()

            if self.move_to_cpu:
                hidden = hidden.cpu()

            self.activations[layer_idx] = hidden

        return fn

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        model,
        layer_indices: List[int],
    ):
        """
        Register hooks on transformer layers.

        Args:
            model:
                HuggingFace GPT-style model.

            layer_indices:
                Transformer block indices.
        """
        self.remove()

        for idx in layer_indices:

            handle = (
                model.transformer.h[idx]
                .register_forward_hook(
                    self._make_hook(idx)
                )
            )

            self._handles.append(handle)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def clear(self):
        """Clear cached activations."""
        self.activations.clear()

    def remove(self):
        """
        Remove all active hooks safely.
        """
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass

        self._handles.clear()
        self.activations.clear()

    def __del__(self):
        try:
            self.remove()
        except Exception:
            pass