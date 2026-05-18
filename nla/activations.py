"""
nla/activations.py

Activation extraction from frozen transformer target models.

ActivationExtractor      — single-layer extraction (sequence + pooled)
MultiLayerExtractor      — multi-layer extraction in one forward pass

v3 upgrades:
  - Native sequence extraction (trajectory-level)
  - Attention-mask-aware mean pooling
  - Optional token-level normalization
  - Robust GPT-family tokenizer handling (pad=eos)
  - Safe hook cleanup
  - Consistent device + dtype behavior
"""

from typing import Dict, List, Literal, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.hooks import ActivationHook, MultiLayerHook


PoolingMode = Literal["mean", "last"]


# ============================================================================
# Helpers
# ============================================================================

def _masked_mean_pool(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Attention-mask-aware mean pooling.

    hidden:          [batch, seq_len, hidden_dim]
    attention_mask:  [batch, seq_len]

    Returns:
        [batch, hidden_dim]
    """
    mask = attention_mask.unsqueeze(-1).float()

    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)

    return summed / counts


def _pool(
    hidden: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    mode: PoolingMode,
) -> torch.Tensor:
    """
    Pool hidden states into a single vector.

    Args:
        hidden:           [batch, seq_len, hidden_dim]
        attention_mask:   [batch, seq_len] or None
        mode:
            "mean" -> masked mean
            "last" -> final valid token

    Returns:
        [batch, hidden_dim]
    """
    if mode == "mean":
        if attention_mask is None:
            return hidden.mean(dim=1)
        return _masked_mean_pool(hidden, attention_mask)

    if mode == "last":
        if attention_mask is None:
            return hidden[:, -1, :]

        last_positions = (
            attention_mask.sum(dim=1) - 1
        ).clamp(min=0)

        batch_idx = torch.arange(
            hidden.shape[0],
            device=hidden.device,
        )

        return hidden[batch_idx, last_positions]

    raise ValueError(f"Unknown pooling mode: {mode!r}")


# ============================================================================
# Single-layer extractor
# ============================================================================

class ActivationExtractor:
    """
    Extract residual-stream activations from one frozen transformer layer.

    Supports:
      - Sequence extraction:
            [seq_len, hidden_dim]

      - Pooled extraction:
            [hidden_dim]

    Args:
        model_name:
            HuggingFace causal LM identifier.

        layer_idx:
            Transformer block index.

        device:
            Torch device.

        max_length:
            Token truncation length.

        normalize:
            If True:
                sequence -> tokenwise L2 normalization
                pooled   -> vector L2 normalization
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

        self.model = (
            AutoModelForCausalLM
            .from_pretrained(model_name)
            .to(device)
        )
        self.model.eval()

        # GPT-family models often lack pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = (
                self.tokenizer.eos_token_id
            )

        self._hook = ActivationHook()

        self._hook_handle = (
            self.model.transformer.h[layer_idx]
            .register_forward_hook(self._hook.hook_fn)
        )

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        text: str,
        mode: str = "pooled",
        pooling: PoolingMode = "mean",
    ) -> torch.Tensor:
        """
        Dispatch method.

        Returns:
            pooled:
                [hidden_dim]

            sequence:
                [seq_len, hidden_dim]
        """
        if mode == "sequence":
            return self.extract_sequence(text)

        return self.extract_pooled(
            text,
            pooling=pooling,
        )

    @torch.no_grad()
    def extract_sequence(
        self,
        text: str,
    ) -> torch.Tensor:
        """
        Extract token-level trajectory.

        Returns:
            [seq_len, hidden_dim]

        Padding tokens are removed using the attention mask.
        """
        hidden, attention_mask = self._run(text)

        hidden = hidden.squeeze(0)
        attention_mask = attention_mask.squeeze(0)

        valid_len = int(attention_mask.sum().item())
        hidden = hidden[:valid_len]

        if self.normalize:
            hidden = F.normalize(hidden, dim=-1)

        return hidden.cpu()

    @torch.no_grad()
    def extract_pooled(
        self,
        text: str,
        pooling: PoolingMode = "mean",
    ) -> torch.Tensor:
        """
        Extract pooled activation vector.

        Returns:
            [hidden_dim]
        """
        hidden, attention_mask = self._run(text)

        pooled = _pool(
            hidden=hidden,
            attention_mask=attention_mask,
            mode=pooling,
        )

        if self.normalize:
            pooled = F.normalize(
                pooled,
                dim=-1,
            )

        return pooled.squeeze(0).cpu()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(
        self,
        text: str,
    ):
        """
        Execute forward pass and capture hidden activations.

        Returns:
            hidden:          [1, seq_len, hidden_dim]
            attention_mask:  [1, seq_len]
        """
        self._hook.clear()

        toks = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        ).to(self.device)

        self.model(**toks)

        hidden = self._hook.activations[0]

        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)

        return hidden, toks["attention_mask"]

    def close(self):
        """Remove hook explicitly."""
        if hasattr(self, "_hook_handle"):
            self._hook_handle.remove()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ============================================================================
# Multi-layer extractor
# ============================================================================

class MultiLayerExtractor:
    """
    Extract pooled activations from multiple layers in one forward pass.

    Returns:
        Dict[layer_idx, Tensor[hidden_dim]]

    All outputs are returned on CPU.
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

        self.model = (
            AutoModelForCausalLM
            .from_pretrained(model_name)
            .to(device)
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )
            self.model.config.pad_token_id = (
                self.tokenizer.eos_token_id
            )

        self._hook = MultiLayerHook()
        self._hook.register(
            self.model,
            layer_indices,
        )

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @torch.no_grad()
    def extract(
        self,
        text: str,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract pooled activations from multiple layers.

        Returns:
            {
                layer_idx: Tensor[hidden_dim]
            }
        """
        self._hook.clear()

        toks = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        ).to(self.device)

        self.model(**toks)

        result = {}

        for idx, hidden in self._hook.activations.items():

            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)

            pooled = _pool(
                hidden=hidden,
                attention_mask=toks["attention_mask"],
                mode=self.pooling,
            )

            if self.normalize:
                pooled = F.normalize(
                    pooled,
                    dim=-1,
                )

            result[idx] = (
                pooled.squeeze(0)
                .detach()
                .cpu()
            )

        return result