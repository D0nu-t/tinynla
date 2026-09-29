"""
nla/embed.py

Small sentence embedder (all-MiniLM-L6-v2, ~90 MB) used to compare meanings:
the pilot information check (study/simulate.py), the planted-concept and hint
benchmarks, and the meaning-level test in nla.contrast.
"""

from typing import Callable, List

import torch

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def minilm_embedder(device: str = "cpu") -> Callable[[List[str]], torch.Tensor]:
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    mod = AutoModel.from_pretrained(EMBED_MODEL).to(device).eval()

    @torch.no_grad()
    def embed(texts: List[str]) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), 64):
            b = tok(texts[i:i + 64], padding=True, truncation=True, max_length=256,
                    return_tensors="pt").to(device)
            h = mod(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            out.append(torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).cpu())
        return torch.cat(out)

    return embed
