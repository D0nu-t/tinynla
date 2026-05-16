"""
nla/reconstructor.py

Activation reconstruction models.

TokenLevelReconstructor  — distilgpt2 (causal) encoder + TransformerDecoder
                           Produces [batch, seq_len, hidden_dim].
                           v3 default; causal encoder matches GPT-2's inductive bias.

ActivationReconstructor  — DistilBERT (bidirectional) encoder + MLP
                           Produces [batch, hidden_dim].
                           Legacy pooled baseline; retained for ablation.

Key design decisions:
  - distilgpt2 has no pad token; we set pad=eos (standard GPT-family practice).
  - The causal encoder is frozen during training; gradients flow only through
    the TransformerDecoder and output_proj.
  - TokenLevelReconstructor.forward() requires seq_len so it can generate
    exactly as many positions as the original text has tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from transformers import AutoModel, AutoTokenizer


# ===========================================================================
# v3 Primary: distilgpt2 (causal) + TransformerDecoder
# ===========================================================================

class TokenLevelReconstructor(nn.Module):
    """
    Text description -> token-level activation sequence.

    Architecture:
        description
          -> distilgpt2 encoder (frozen, causal, left-to-right)
          -> hidden states [desc_seq_len, 768]  as cross-attention memory
          -> learned positional queries [target_seq_len, 768]
          -> TransformerDecoder
          -> linear projection
          -> [target_seq_len, hidden_dim]

    Why distilgpt2 over DistilBERT:
        GPT-2 residual streams encode only the causal prefix at each position.
        A bidirectional encoder (DistilBERT) integrates full-sequence context
        symmetrically, which is geometrically incompatible with that structure.
        distilgpt2 shares GPT-2's left-to-right inductive bias and embedding space.

    Args:
        hidden_dim:    Target activation dimension (768 for GPT-2).
        n_layers:      TransformerDecoder layer count.
        n_heads:       Attention heads (embed_dim must be divisible by n_heads).
        max_len:       Maximum target seq_len (>= activation.max_length in config).
        encoder_name:  Causal encoder. Default "distilgpt2" (768-dim, 6 layers).
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

        # distilgpt2 has no pad token by default — use eos as pad.
        # This is standard practice for GPT-family models used as encoders.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.encoder.config.pad_token_id = self.tokenizer.eos_token_id

        embed_dim = self.encoder.config.hidden_size    # 768 for distilgpt2

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            batch_first=True,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Learned positional queries — one per target position.
        # Small init keeps early outputs close to zero, avoiding gradient spikes.
        self.positional_queries = nn.Parameter(
            torch.randn(1, max_len, embed_dim) * 0.02
        )

        self.output_proj = nn.Linear(embed_dim, hidden_dim)
        self.max_len = max_len

    def encode_text(self, texts: List[str], device: str) -> torch.Tensor:
        """
        Encode descriptions through the frozen causal encoder.

        Returns:
            [batch, desc_seq_len, embed_dim]  used as cross-attention memory.
        """
        toks = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = self.encoder(**toks)

        return out.last_hidden_state    # [batch, desc_seq, embed_dim]

    def forward(
        self,
        texts: List[str],
        seq_len: int,
        device: str,
    ) -> torch.Tensor:
        """
        Reconstruct a token-level activation trajectory.

        Args:
            texts:   Description strings, one per sample in batch.
            seq_len: Number of target positions to generate.
                     Pass the actual tokenized length of the original text
                     so the reconstructed trajectory has the right shape for patching.
            device:  Torch device string.

        Returns:
            [batch, seq_len, hidden_dim]
        """
        memory = self.encode_text(texts, device)    # [batch, desc_seq, embed_dim]
        B = memory.shape[0]

        seq_len = min(seq_len, self.max_len)
        queries = self.positional_queries[:, :seq_len].expand(B, -1, -1)

        decoded = self.decoder(tgt=queries, memory=memory)
        return self.output_proj(decoded)            # [batch, seq_len, hidden_dim]

    def forward_pooled(self, texts: List[str], device: str) -> torch.Tensor:
        """
        Generate max_len positions then mean-pool to a single vector.
        Allows TokenLevelReconstructor to be dropped into pooled-mode eval pipelines.

        Returns:
            [batch, hidden_dim]  L2-normalized.
        """
        seq = self.forward(texts, self.max_len, device)
        return F.normalize(seq.mean(dim=1), dim=-1)


# ===========================================================================
# Legacy: DistilBERT (bidirectional) + MLP
# ===========================================================================

class ActivationReconstructor(nn.Module):
    """
    Text -> pooled activation vector.

    Architecture:
        text -> DistilBERT -> mean pool -> MLP -> L2-normalize -> [hidden_dim]

    Legacy baseline. DistilBERT's bidirectional attention is geometrically
    incompatible with GPT-2's causal residual streams. Retained for ablation.

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
        return self.encoder(**toks).last_hidden_state.mean(dim=1)

    def forward(self, texts: List[str], device: str) -> torch.Tensor:
        """Returns L2-normalized vectors: [batch, output_dim]"""
        return F.normalize(self.projector(self.encode_text(texts, device)), dim=-1)