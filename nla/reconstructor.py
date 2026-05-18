"""
nla/reconstructor.py

Phase 1/2/3 upgraded reconstruction stack.

Primary architecture:
  TokenLevelReconstructor
    - frozen causal text encoder (distilgpt2 by default)
    - learned trajectory queries
    - causal TransformerDecoder
    - sequence output [batch, seq_len, hidden_dim]

Major upgrades:
  - proper padding-aware encoder attention masking
  - causal target masking in decoder
  - learned query scaling stabilization
  - optional output normalization
  - safer max_len handling
  - mixed-length robustness
  - deterministic sequence generation
  - trajectory-level compatible decoding

Legacy baseline retained:
  ActivationReconstructor
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer


# ============================================================================
# Utilities
# ============================================================================

def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Standard autoregressive causal mask.

    Shape:
        [seq_len, seq_len]

    True values are masked positions for nn.TransformerDecoder.
    """
    return torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )


# ============================================================================
# v3 Primary Model
# ============================================================================

class TokenLevelReconstructor(nn.Module):
    """
    Description -> activation trajectory reconstructor.

    Architecture:
        description
            -> frozen causal encoder
            -> encoder hidden states as memory
            -> learned trajectory queries
            -> causal TransformerDecoder
            -> projection head
            -> activation trajectory

    Output:
        [batch, seq_len, hidden_dim]
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        n_layers: int = 4,
        n_heads: int = 8,
        max_len: int = 64,
        encoder_name: str = "distilgpt2",
        dropout: float = 0.1,
        normalize_output: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.normalize_output = normalize_output

        # ------------------------------------------------------------------
        # Frozen causal encoder
        # ------------------------------------------------------------------

        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.encoder.config.pad_token_id = self.tokenizer.eos_token_id

        encoder_dim = self.encoder.config.hidden_size

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.encoder.eval()

        # ------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=encoder_dim,
            nhead=n_heads,
            dim_feedforward=encoder_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_layers,
        )

        # ------------------------------------------------------------------
        # Learned trajectory queries
        # ------------------------------------------------------------------

        self.positional_queries = nn.Parameter(
            torch.randn(1, max_len, encoder_dim) * 0.02
        )

        self.query_scale = nn.Parameter(torch.tensor(1.0))

        # ------------------------------------------------------------------
        # Output projection
        # ------------------------------------------------------------------

        self.output_proj = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, hidden_dim),
        )

    # ======================================================================
    # Encoder
    # ======================================================================

    def encode_text(
        self,
        texts: List[str],
        device: str,
    ):
        """
        Encode descriptions into contextual memory states.

        Returns:
            memory:         [batch, desc_seq_len, encoder_dim]
            attention_mask: [batch, desc_seq_len]
        """

        toks = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
        ).to(device)

        with torch.no_grad():
            out = self.encoder(**toks)

        memory = out.last_hidden_state

        return memory, toks["attention_mask"]

    # ======================================================================
    # Forward
    # ======================================================================

    def forward(
        self,
        texts: List[str],
        seq_len: int,
        device: str,
    ) -> torch.Tensor:
        """
        Generate activation trajectories.

        Args:
            texts:
                Description strings.

            seq_len:
                Number of trajectory positions to generate.

            device:
                Torch device string.

        Returns:
            Tensor:
                [batch, seq_len, hidden_dim]
        """

        seq_len = int(min(seq_len, self.max_len))

        memory, memory_attention_mask = self.encode_text(texts, device)

        batch_size = memory.shape[0]

        # ------------------------------------------------------------------
        # Learned target queries
        # ------------------------------------------------------------------

        queries = self.positional_queries[:, :seq_len]
        queries = queries.expand(batch_size, -1, -1)

        queries = queries * self.query_scale

        # ------------------------------------------------------------------
        # Causal decoding mask
        # ------------------------------------------------------------------

        tgt_mask = build_causal_mask(
            seq_len=seq_len,
            device=memory.device,
        )

        # memory_key_padding_mask expects True at PAD positions
        memory_key_padding_mask = ~memory_attention_mask.bool()

        decoded = self.decoder(
            tgt=queries,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        out = self.output_proj(decoded)

        if self.normalize_output:
            out = F.normalize(out, dim=-1)

        return out

    # ======================================================================
    # Pooled compatibility path
    # ======================================================================

    def forward_pooled(
        self,
        texts: List[str],
        device: str,
    ) -> torch.Tensor:
        """
        Compatibility mode for pooled legacy evaluation.

        Returns:
            [batch, hidden_dim]
        """

        seq = self.forward(
            texts=texts,
            seq_len=self.max_len,
            device=device,
        )

        pooled = seq.mean(dim=1)

        return F.normalize(pooled, dim=-1)


# ============================================================================
# Legacy pooled baseline
# ============================================================================

class ActivationReconstructor(nn.Module):
    """
    Legacy pooled reconstruction baseline.

    Retained for ablation comparisons.
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

    def encode_text(
        self,
        texts: List[str],
        device: str,
    ) -> torch.Tensor:

        toks = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        ).to(device)

        out = self.encoder(**toks)

        hidden = out.last_hidden_state

        attention_mask = toks["attention_mask"].unsqueeze(-1)

        masked_hidden = hidden * attention_mask

        pooled = masked_hidden.sum(dim=1)
        pooled = pooled / attention_mask.sum(dim=1).clamp(min=1)

        return pooled

    def forward(
        self,
        texts: List[str],
        device: str,
    ) -> torch.Tensor:
        """
        Returns:
            [batch, output_dim]
        """

        encoded = self.encode_text(texts, device)

        projected = self.projector(encoded)

        return F.normalize(projected, dim=-1)