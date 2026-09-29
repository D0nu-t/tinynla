"""
nla/ar.py

Activation Reconstructor (AR): explanation text -> activation vector.

As in the published NLA, the AR is the TARGET model truncated to blocks
0..K plus a Linear(d, d) head, read out at the final token of
    "<text>{explanation}</text> <summary>"
Because the body is the target's own first K+1 blocks, its hidden states
already live in the block-K coordinate system being reconstructed.

Deviation from kitft: the head is initialised to the identity (kitft uses
PyTorch's default random init). At step 0 the AR therefore outputs the
target's own block-K state at the final prompt token - an in-distribution
activation - which gives a small-compute run a sensible starting point.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from nla import model_adapter as ma
from nla.metrics import normalized_mse


class ActivationReconstructor(nn.Module):
    def __init__(
        self,
        target_name: str,
        layer: int,
        template: str,
        max_explanation_tokens: int = 80,
        lora: Optional[dict] = None,
        base_dtype: str = "float32",
    ):
        super().__init__()
        self.target_name = target_name
        self.layer = layer
        self.template = template
        self.max_explanation_tokens = max_explanation_tokens
        self.lora = lora
        self.base_dtype = base_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(target_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        body = AutoModel.from_pretrained(target_name, dtype=getattr(torch, base_dtype))
        d = ma.hidden_size(body)
        self.body = ma.truncate(body, layer)
        if lora:
            from peft import LoraConfig, get_peft_model
            self.body = get_peft_model(self.body, LoraConfig(**lora))

        self.head = nn.Linear(d, d)
        with torch.no_grad():
            self.head.weight.copy_(torch.eye(d))
            self.head.bias.zero_()

        # fixed template pieces around the explanation
        prefix, suffix = template.split("{explanation}")
        self._prefix_ids = self.tokenizer(prefix)["input_ids"]
        self._suffix_ids = self.tokenizer(suffix)["input_ids"]

    @property
    def device(self) -> torch.device:
        return self.head.weight.device

    def _encode(self, explanations: List[str]):
        """Truncate the explanation only, so the suffix (readout anchor) survives."""
        rows = []
        for e in explanations:
            body = self.tokenizer(
                e,
                truncation=True,
                max_length=self.max_explanation_tokens,
            )["input_ids"]
            rows.append(self._prefix_ids + body + self._suffix_ids)

        width = max(len(r) for r in rows)
        ids = torch.full((len(rows), width), self.tokenizer.pad_token_id)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = torch.tensor(r)
            mask[i, : len(r)] = 1
        return ids.to(self.device), mask.to(self.device)

    def forward(self, explanations: List[str]) -> torch.Tensor:
        """Returns [batch, d] reconstructions."""
        ids, mask = self._encode(explanations)
        hidden = self.body(input_ids=ids, attention_mask=mask).last_hidden_state
        last = mask.sum(dim=1) - 1                       # right padding
        readout = hidden[torch.arange(len(ids), device=ids.device), last]
        return self.head(readout.to(self.head.weight.dtype))

    def loss(
        self,
        explanations: List[str],
        targets: torch.Tensor,
        scale: Optional[float],
    ) -> torch.Tensor:
        pred = self(explanations)
        return normalized_mse(pred, targets.to(pred.device), scale).mean()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, out_dir: str) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), out / "ar.pt")
        with open(out / "ar_config.json", "w") as f:
            json.dump(
                {
                    "target_name": self.target_name,
                    "layer": self.layer,
                    "template": self.template,
                    "max_explanation_tokens": self.max_explanation_tokens,
                    "lora": self.lora,
                    "base_dtype": self.base_dtype,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, out_dir: str, map_location="cpu") -> "ActivationReconstructor":
        out = Path(out_dir)
        with open(out / "ar_config.json") as f:
            cfg = json.load(f)
        ar = cls(**cfg)
        ar.load_state_dict(torch.load(out / "ar.pt", map_location=map_location))
        return ar
