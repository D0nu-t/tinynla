"""
nla/reconstructor.py

Activation reconstruction models.

Two architectures:

  ActivationReconstructor   — DistilBERT encoder + MLP projection
                              Stable baseline; used for all standard runs.

  TokenLevelReconstructor   — BGE encoder + TransformerDecoder
                              Experimental; reconstructs token-level sequence.

Both expose:
    forward(texts: List[str], device: str) -> Tensor[batch, hidden_dim]

This signature is what all training and evaluation scripts expect.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from transformers import AutoModel, AutoTokenizer


# ===========================================================================
# Stable Baseline: DistilBERT + MLP
# ===========================================================================

class ActivationReconstructor(nn.Module):
    """
    Text -> pooled activation vector.

    Architecture:
        text
          -> DistilBERT (frozen or fine-tuned)
          -> mean pool over token dimension
          -> 2-layer MLP
          -> L2-normalized activation vector

    Args:
        encoder_name:  HuggingFace encoder identifier.
        output_dim:    Target hidden dimension (must match target LM).
        hidden_dim:    MLP intermediate dimension.
    """

    def __init__(
        self,
        encoder_name: str = "distilbert-base-uncased",
        output_dim: int = 768,
        hidden_dim: int = 2048,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)

        encoder_dim = self.encoder.config.hidden_size

        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode_text(self, texts: List[str], device: str) -> torch.Tensor:
        toks = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        ).to(device)
        outputs = self.encoder(**toks)
        # Mean pool over token dimension
        return outputs.last_hidden_state.mean(dim=1)    # [batch, encoder_dim]

    def forward(self, texts: List[str], device: str) -> torch.Tensor:
        """
        Returns L2-normalized vectors: [batch, output_dim]
        """
        pooled = self.encode_text(texts, device)
        vec = self.projector(pooled)
        return F.normalize(vec, dim=-1)


# ===========================================================================
# Experimental: BGE + TransformerDecoder (token-level)
# ===========================================================================

class TokenLevelReconstructor(nn.Module):
    """
    Text -> token-level activation sequence.

    Architecture:
        text
          -> BGE encoder (cross-attention memory)
          -> learned positional queries [seq_len, embed_dim]
          -> TransformerDecoder
          -> linear projection to target activation dim

    The decoder uses the encoder's output as memory and learns
    position-specific queries — effectively a conditional sequence generator
    that produces a latent trajectory rather than a single vector.

    For pooled-vector experiments, call .forward_pooled() which mean-pools
    the decoder output.

    Args:
        hidden_dim:    Target activation dimension.
        n_layers:      Decoder layer count.
        n_heads:       Decoder attention heads.
        max_len:       Maximum sequence length.
        encoder_name:  BGE or compatible encoder.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int = 4,
        n_heads: int = 8,
        max_len: int = 64,
        encoder_name: str = "BAAI/bge-base-en-v1.5",
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)

        embed_dim = self.encoder.config.hidden_size

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            batch_first=True,
            dim_feedforward=embed_dim * 4,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Learned positional queries; one per target sequence position
        self.positional_queries = nn.Parameter(
            torch.randn(1, max_len, embed_dim) * 0.02
        )

        self.output_proj = nn.Linear(embed_dim, hidden_dim)
        self.max_len = max_len

    def encode_text(self, texts: List[str], device: str) -> torch.Tensor:
        toks = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        ).to(device)
        return self.encoder(**toks).last_hidden_state   # [batch, seq, embed_dim]

    def forward(
        self,
        texts: List[str],
        seq_len: int,
        device: str,
    ) -> torch.Tensor:
        """
        Returns token-level activations: [batch, seq_len, hidden_dim]
        """
        memory = self.encode_text(texts, device)    # [batch, seq, embed_dim]
        B = memory.shape[0]
        queries = self.positional_queries[:, :seq_len].expand(B, -1, -1)
        decoded = self.decoder(tgt=queries, memory=memory)
        return self.output_proj(decoded)            # [batch, seq_len, hidden_dim]

    def forward_pooled(self, texts: List[str], device: str) -> torch.Tensor:
        """
        Returns mean-pooled, L2-normalized vector: [batch, hidden_dim]
        Allows TokenLevelReconstructor to be used in pooled-vector pipelines.
        """
        seq = self.forward(texts, self.max_len, device)
        vec = seq.mean(dim=1)
        return F.normalize(vec, dim=-1)