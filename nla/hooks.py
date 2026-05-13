"""
nla/hooks.py

Forward hook infrastructure.

ActivationHook  — single-layer, backward-compatible
MultiLayerHook  — captures N layers in one forward pass; call remove() after use
"""

import torch
from typing import Dict, List


class ActivationHook:
    def __init__(self):
        self.activations: List[torch.Tensor] = []

    def hook_fn(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            hidden = outputs[0]
        else:
            hidden = outputs
        self.activations.append(hidden.detach())

    def clear(self):
        self.activations = []


class MultiLayerHook:
    """
    Registers hooks on multiple transformer blocks simultaneously.
    One forward pass captures all target layers.

    Usage:
        hook = MultiLayerHook()
        hook.register(model, [3, 5, 7])
        model(**inputs)
        activations = hook.activations   # {layer_idx: Tensor}
        hook.remove()
    """

    def __init__(self):
        self.activations: Dict[int, torch.Tensor] = {}
        self._handles: List = []

    def _make_hook(self, idx: int):
        def fn(module, inputs, outputs):
            if isinstance(outputs, tuple):
                hidden = outputs[0]
            else:
                hidden = outputs
            self.activations[idx] = hidden.detach()
        return fn

    def register(self, model, layer_indices: List[int]):
        for idx in layer_indices:
            handle = model.transformer.h[idx].register_forward_hook(
                self._make_hook(idx)
            )
            self._handles.append(handle)

    def clear(self):
        self.activations = {}

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        self.activations = {}