"""
text
  ↓
transformer encoder
  ↓
mean pooling
  ↓
MLP projection
  ↓
activation vector
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer


class ActivationReconstructor(nn.Module):
    def __init__(
        self,
        encoder_name="distilbert-base-uncased",
        output_dim=768,
        hidden_dim=2048
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            encoder_name
        )

        self.encoder = AutoModel.from_pretrained(
            encoder_name
        )

        encoder_dim = self.encoder.config.hidden_size

        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def encode_text(self, texts, device):
        toks = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        outputs = self.encoder(**toks)

        hidden = outputs.last_hidden_state

        pooled = hidden.mean(dim=1)

        return pooled

    def forward(self, texts, device):
        pooled = self.encode_text(texts, device)

        vec = self.projector(pooled)

        vec = F.normalize(vec, dim=-1)

        return vec