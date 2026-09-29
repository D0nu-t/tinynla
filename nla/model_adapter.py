"""
nla/model_adapter.py

Uniform access to the parts of a HuggingFace causal LM that an NLA needs,
so the rest of the package never hard-codes `model.transformer.h`.

Supported layouts:
    GPT-2 / GPT-Neo          model.transformer.h        final norm: transformer.ln_f
    Llama / Qwen / Mistral   model.model.layers         final norm: model.norm
    Gemma                    as Llama, plus embed_scale = sqrt(d)
    GPT-NeoX / Pythia        model.gpt_neox.layers      final norm: gpt_neox.final_layer_norm

"Block K output" everywhere in this package means the residual stream
leaving transformer block K (0-indexed), i.e. hidden_states[K + 1].
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterator, List, Tuple

import torch
import torch.nn as nn


# (attribute path to the block list, attribute path to the final norm)
_LAYOUTS: List[Tuple[str, str]] = [
    ("transformer.h", "transformer.ln_f"),
    ("model.layers", "model.norm"),
    ("gpt_neox.layers", "gpt_neox.final_layer_norm"),
    # base (headless) models loaded with AutoModel
    ("h", "ln_f"),
    ("layers", "norm"),
]


def _get_attr(obj, path: str):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _set_attr(obj, path: str, value) -> None:
    *parents, last = path.split(".")
    for part in parents:
        obj = getattr(obj, part)
    setattr(obj, last, value)


def _unwrap(model: nn.Module) -> nn.Module:
    """See through a PEFT (LoRA) wrapper to the transformers model inside."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def _layout(model: nn.Module) -> Tuple[str, str]:
    for blocks_path, norm_path in _LAYOUTS:
        try:
            blocks = _get_attr(model, blocks_path)
        except AttributeError:
            continue
        if isinstance(blocks, nn.ModuleList):
            return blocks_path, norm_path
    raise ValueError(
        f"Unsupported architecture {type(model).__name__}: "
        f"add its block/norm paths to nla.model_adapter._LAYOUTS"
    )


# ============================================================================
# Accessors
# ============================================================================

def blocks(model: nn.Module) -> nn.ModuleList:
    model = _unwrap(model)
    return _get_attr(model, _layout(model)[0])


def num_layers(model: nn.Module) -> int:
    return len(blocks(model))


def hidden_size(model: nn.Module) -> int:
    cfg = model.config
    return getattr(cfg, "hidden_size", None) or cfg.n_embd


def embed_tokens(model: nn.Module) -> nn.Module:
    return model.get_input_embeddings()


def final_norm(model: nn.Module) -> nn.Module:
    """The norm applied after the last block (ln_f / norm / final_layer_norm)."""
    model = _unwrap(model)
    return _get_attr(model, _layout(model)[1])


def embed_scale(model: nn.Module) -> float:
    """
    Multiplier the architecture applies after the embedding lookup.

    Gemma scales embeddings by sqrt(d) inside its forward; when we build
    inputs_embeds ourselves we must apply it manually (the "Gemma gotcha").
    """
    if "gemma" in getattr(model.config, "model_type", ""):
        return math.sqrt(hidden_size(model))
    return 1.0


def default_layer(model: nn.Module) -> int:
    """About two-thirds depth, matching the released NLA checkpoints."""
    # Gemma-3-12B: 32/48, Llama-3.3-70B: 53/80  ->  index ≈ n * 2/3
    return min(num_layers(model) - 1, round(num_layers(model) * 2 / 3))


# ============================================================================
# Capture
# ============================================================================

@contextmanager
def capture_block_output(
    model: nn.Module,
    layer: int,
) -> Iterator[dict]:
    """
    Record the residual stream leaving block `layer` during forward passes.

    Usage:
        with capture_block_output(model, 8) as cap:
            model(**toks)
        h = cap["hidden"]   # [batch, seq, d]
    """
    cap: dict = {}

    def hook(module, inputs, outputs):
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        cap["hidden"] = hidden.detach()

    handle = blocks(model)[layer].register_forward_hook(hook)
    try:
        yield cap
    finally:
        handle.remove()


class _StopForward(Exception):
    pass


@torch.no_grad()
def block_output(model: nn.Module, layer: int, **inputs) -> torch.Tensor:
    """
    The residual stream leaving block `layer`, WITHOUT running the rest of the model.

    A full forward also computes the output layer: a score for every vocabulary
    word at every position (Qwen: 152k words; a 224 MiB tensor per extraction
    batch, which ran the 4 GB GPU out of memory). Raising from the block's hook
    stops the pass as soon as the needed state exists.
    """
    cap: dict = {}

    def hook(module, args, outputs):
        cap["hidden"] = (outputs[0] if isinstance(outputs, tuple) else outputs).detach()
        raise _StopForward

    handle = blocks(model)[layer].register_forward_hook(hook)
    try:
        model(**inputs)
    except _StopForward:
        pass
    finally:
        handle.remove()
    return cap["hidden"]


# ============================================================================
# Truncation (for the Activation Reconstructor)
# ============================================================================

def truncate(model: nn.Module, layer: int) -> nn.Module:
    """
    Keep blocks 0..layer (inclusive) and drop the final norm, in place.

    The truncated model's last hidden state then equals the full model's
    block-`layer` output, which is what the AR builds its readout on.
    """
    blocks_path, norm_path = _layout(model)

    kept = nn.ModuleList(list(blocks(model))[: layer + 1])
    _set_attr(model, blocks_path, kept)
    _set_attr(model, norm_path, nn.Identity())

    for attr in ("n_layer", "num_hidden_layers"):
        if hasattr(model.config, attr):
            setattr(model.config, attr, layer + 1)

    return model


@torch.no_grad()
def block_norms(
    model: nn.Module,
    input_ids: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    """Per-token L2 norm of block-`layer` output. Returns [seq]."""
    with capture_block_output(model, layer) as cap:
        model(input_ids=input_ids)
    return cap["hidden"][0].norm(dim=-1)
