"""
nla/reconstructor.py

Activation reconstruction models.

TokenLevelReconstructor  — distilgpt2 (causal) encoder + TransformerDecoder
                           Produces [batch, seq_len, hidden_dim] per input text.
                           v3 default; matches GPT-2's causal inductive bias.

ActivationReconstructor  — DistilBERT (bidirectional) encoder + MLP
                           Produces [batch, hidden_dim] per input text.
                           Legacy pooled baseline; retained for ablation.

Both are usable in isolation. TokenLevelReconstructor is the primary model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from transformers import AutoModel, AutoTokenizer


# ===========================================================================
# v3 Primary: distilgpt2 (causal) + TransformerDecoder
# ===========================================================================

class TokenLevelReconstructor(nn.Module):
    """
    Text -> token-level activation sequence.

    Architecture:
        description text
          -> distilgpt2 (causal; left-to-right context)
          -> hidden states [desc_seq, 768]  (used as cross-attention memory)
          -> learned positional queries [target_seq_len, 768]
          -> TransformerDecoder (queries attend to encoder memory)
          -> linear projection
          -> [target_seq_len, hidden_dim]

    The causal encoder matches GPT-2's inductive bias. DistilBERT (bidirectional)
    is geometrically incompatible because GPT-2 residual streams are causally
    asymmetric — each position encodes only the prefix, not the full sequence.

    distilgpt2 does not define a padding token by default.
    We set pad_token = eos_token, which is standard practice for decoder-only
    models used in encoder mode.

    Args:
        hidden_dim:    Target activation dimension (768 for GPT-2).
        n_layers:      Number of TransformerDecoder layers.
        n_heads:       Number of attention heads (embed_dim must be divisible by n_heads).
        max_len:       Maximum target sequence length (must be >= activation.max_length).
        encoder_name:  Causal encoder identifier. Default: "distilgpt2" (768-dim, causal).
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        n_layers: int = 4,
        n_heads: int = 8,
        max_len: int = 64,
        encoder_name: str = "distilgpt2",
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)

        # distilgpt2 has no pad token — use eos as pad (standard for GPT-family)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.encoder.config.pad_token_id = self.tokenizer.eos_token_id

        embed_dim = self.encoder.config.hidden_size   # 768 for distilgpt2

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            batch_first=True,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Learned positional queries — one per target position
        # Initialized near zero to start close to the encoder representation
        self.positional_queries = nn.Parameter(
            torch.randn(1, max_len, embed_dim) * 0.02
        )

        self.output_proj = nn.Linear(embed_dim, hidden_dim)
        self.max_len = max_len

    def encode_text(self, texts: List[str], device: str) -> torch.Tensor:
        """
        Run texts through the causal encoder.

        Returns:
            [batch, desc_seq_len, embed_dim]  — all hidden states as decoder memory.
        """
        toks = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            # We do NOT fine-tune the encoder in this stage;
            # gradients only flow through the decoder and output_proj.
            out = self.encoder(**toks)

        return out.last_hidden_state    # [batch, desc_seq, embed_dim]

    def forward(
        self,
        texts: List[str],
        seq_len: int,
        device: str,
    ) -> torch.Tensor:
        """
        Reconstruct a token-level activation sequence.

        Args:
            texts:   List of description strings (batch).
            seq_len: Number of target positions to generate.
                     Should match the original text's tokenized length.
            device:  Torch device string.

        Returns:
            [batch, seq_len, hidden_dim]  — reconstructed activation trajectory.
        """
        memory = self.encode_text(texts, device)    # [batch, desc_seq, embed_dim]
        B = memory.shape[0]

        seq_len = min(seq_len, self.max_len)
        queries = self.positional_queries[:, :seq_len].expand(B, -1, -1)

        decoded = self.decoder(tgt=queries, memory=memory)
        return self.output_proj(decoded)            # [batch, seq_len, hidden_dim]

    def forward_pooled(self, texts: List[str], device: str) -> torch.Tensor:
        """
        Convenience: generate max_len positions then mean-pool.
        Allows this model to be used in pooled-vector evaluation pipelines.

        Returns:
            [batch, hidden_dim]  — L2-normalized mean-pooled reconstruction.
        """
        seq = self.forward(texts, self.max_len, device)
        vec = seq.mean(dim=1)
        return F.normalize(vec, dim=-1)


# ===========================================================================
# Legacy: DistilBERT (bidirectional) + MLP
# ===========================================================================

class ActivationReconstructor(nn.Module):
    """
    Text -> pooled activation vector.

    Architecture:
        text -> DistilBERT -> mean pool -> MLP -> L2-normalize -> [hidden_dim]

    Legacy baseline. DistilBERT is bidirectional, which is geometrically
    incompatible with GPT-2's causal residual streams. Retained for ablation
    comparison against TokenLevelReconstructor.

    Args:
        encoder_name:  HuggingFace encoder identifier.
        output_dim:    Target hidden dimension (768 for GPT-2).
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
        return outputs.last_hidden_state.mean(dim=1)    # [batch, encoder_dim]

    def forward(self, texts: List[str], device: str) -> torch.Tensor:
        """Returns L2-normalized vectors: [batch, output_dim]"""
        pooled = self.encode_text(texts, device)
        vec = self.projector(pooled)
        return F.normalize(vec, dim=-1)