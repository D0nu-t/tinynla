"""
nla/activations.py

Activation extraction from frozen transformer target models.

ActivationExtractor      — single layer, pooled or sequence output
MultiLayerExtractor      — N layers captured in one forward pass

Both models are loaded eval/no_grad; the target model is never updated.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Literal

from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.hooks import ActivationHook, MultiLayerHook


PoolingMode = Literal["mean", "last"]


def _pool(hidden: torch.Tensor, mode: PoolingMode) -> torch.Tensor:
    """
    hidden: [batch, seq, hidden_dim]
    returns: [batch, hidden_dim]
    """
    if mode == "mean":
        return hidden.mean(dim=1)
    if mode == "last":
        return hidden[:, -1, :]
    raise ValueError(f"Unknown pooling mode: {mode!r}")


class ActivationExtractor:
    """
    Extracts residual-stream activations from a single frozen layer.

    Args:
        model_name:  HuggingFace identifier.
        layer_idx:   Transformer block index.
        device:      Torch device string.
        max_length:  Tokenizer truncation length.
        normalize:   Apply L2 normalization to output vectors.
    """

    def __init__(
        self,
        model_name: str,
        layer_idx: int,
        device: str = "cpu",
        max_length: int = 64,
        normalize: bool = True,
    ):
        self.device = device
        self.layer_idx = layer_idx
        self.max_length = max_length
        self.normalize = normalize

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()

        self._hook = ActivationHook()
        self.model.transformer.h[layer_idx].register_forward_hook(self._hook.hook_fn)

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(
        self,
        text: str,
        mode: str = "pooled",
        pooling: PoolingMode = "mean",
    ) -> torch.Tensor:
        """
        Dispatch to extract_pooled or extract_sequence.

        Returns Tensor on CPU:
          pooled   → [hidden_dim]
          sequence → [seq_len, hidden_dim]
        """
        if mode == "sequence":
            return self.extract_sequence(text)
        return self.extract_pooled(text, pooling=pooling)

    @torch.no_grad()
    def extract_sequence(self, text: str) -> torch.Tensor:
        """Returns [seq_len, hidden_dim] on CPU, optionally L2-normalized per token."""
        seq = self._run(text)               # [1, seq, hidden]
        seq = seq.squeeze(0)               # [seq, hidden]
        if self.normalize:
            seq = F.normalize(seq, dim=-1)
        return seq.cpu()

    @torch.no_grad()
    def extract_pooled(
        self,
        text: str,
        pooling: PoolingMode = "mean",
    ) -> torch.Tensor:
        """Returns [hidden_dim] on CPU, L2-normalized."""
        hidden = self._run(text)            # [1, seq, hidden]
        vec = _pool(hidden, pooling)        # [1, hidden]
        vec = F.normalize(vec, dim=-1)
        return vec.squeeze(0).cpu()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self, text: str) -> torch.Tensor:
        self._hook.clear()
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        self.model(**inputs)
        hidden = self._hook.activations[0]      # [1, seq, hidden] or [seq, hidden]
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        return hidden


class MultiLayerExtractor:
    """
    Extracts pooled activations from multiple layers in one forward pass.

    Returns:
        Dict[layer_idx, Tensor[hidden_dim]]  — each vector L2-normalized on CPU.
    """

    def __init__(
        self,
        model_name: str,
        layer_indices: List[int],
        pooling: PoolingMode = "mean",
        device: str = "cpu",
        max_length: int = 64,
        normalize: bool = True,
    ):
        self.device = device
        self.layer_indices = layer_indices
        self.pooling = pooling
        self.max_length = max_length
        self.normalize = normalize

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()

        self._hook = MultiLayerHook()
        self._hook.register(self.model, layer_indices)

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @torch.no_grad()
    def extract(self, text: str) -> Dict[int, torch.Tensor]:
        self._hook.clear()
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        self.model(**inputs)

        result = {}
        for idx, hidden in self._hook.activations.items():
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)
            vec = _pool(hidden, self.pooling)
            if self.normalize:
                vec = F.normalize(vec, dim=-1)
            result[idx] = vec.squeeze(0).cpu()
        return result

        