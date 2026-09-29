"""
nla/lens.py

Logit lens: decode an intermediate residual state directly into next-token
probabilities with the model's own final norm + unembedding
(nostalgebraist 2020; formalised and improved as the tuned lens by
Belrose et al. 2023). No training, no decoder of ours involved.

Used as an INDEPENDENT second opinion next to the NLA: when a theme the
verbalizer reports is also among the words the lens says the model was
leaning towards, two methods agree - the corroboration the NLA paper
recommends before trusting a reading.

Known limitation: the plain logit lens is biased at middle layers (the
motivation for the tuned lens); treat its words as a rough view.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from nla import model_adapter as ma
from nla.claims import STOP


@torch.no_grad()
def logit_lens(model, hidden: torch.Tensor) -> torch.Tensor:
    """hidden [..., d] (a block output) -> log-probs over the vocabulary [..., V]."""
    norm = ma.final_norm(model)
    head = model.get_output_embeddings()
    dev = head.weight.device
    logits = head(norm(hidden.to(dev, head.weight.dtype)))
    return logits.float().log_softmax(dim=-1)


@torch.no_grad()
def difference_lens(model, diff: torch.Tensor) -> torch.Tensor:
    """
    Contrastive projection (arXiv 2609.09902): read a DIFFERENCE of hidden states
    [..., d] through the unembedding. The shared component cancels, so the top
    words show what separates the two states. The final norm's centring and bias
    would distort a difference, so only its per-dimension gain is applied.
    """
    norm = ma.final_norm(model)
    head = model.get_output_embeddings()
    x = diff.to(head.weight.device, head.weight.dtype)
    if getattr(norm, "weight", None) is not None:
        x = x * norm.weight
    return head(x).float().log_softmax(dim=-1)


TUNED_LENS_REPO ="AlignmentResearch/tuned-lens"   # a HF *space* holding lens/<model>/params.pt
_TUNED_CACHE: Dict[str, Optional[Dict[str, torch.Tensor]]] = {}


def load_tuned_lens(model_name: str) -> Optional[Dict[str, torch.Tensor]]:
    """
    Pretrained tuned-lens translators (Belrose et al. 2023) for `model_name`,
    or None if unavailable (offline, or no lens published for this model).
    Translator i is a residual affine map applied to hidden_states[i]
    (0 = embeddings), i.e. to the output of block i-1.
    """
    if model_name not in _TUNED_CACHE:
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(TUNED_LENS_REPO, f"lens/{model_name}/params.pt", repo_type="space")
            _TUNED_CACHE[model_name] = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            _TUNED_CACHE[model_name] = None
    return _TUNED_CACHE[model_name]


@torch.no_grad()
def tuned_lens(model, hidden: torch.Tensor, layer: int, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Tuned lens for the output of block `layer` -> log-probs [..., V].
    Uses translator layer+1 (verified empirically for GPT-2: it minimises KL to
    the model's final output; KL 0.80 vs 3.25 for the plain logit lens at block 8).
    """
    W, b = params[f"{layer + 1}.weight"], params[f"{layer + 1}.bias"]
    h = hidden.float()
    h = h + h @ W.to(h.device).T + b.to(h.device)
    return logit_lens(model, h)


class Lens:
    """Tuned lens when a pretrained one exists for the model, else the logit lens."""

    def __init__(self, model, model_name: str, layer: int):
        self.model, self.layer = model, layer
        self.params = load_tuned_lens(model_name)
        self.kind = "tuned lens" if self.params is not None else "logit lens"

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.params is not None:
            return tuned_lens(self.model, hidden, self.layer, self.params)
        return logit_lens(self.model, hidden)


def top_words(logprobs: torch.Tensor, tokenizer, k: int = 10, content_only: bool = True,
              pool: int = 200, whole_words: bool = False) -> List[Dict]:
    """Top-k decoded tokens for one position; optionally only content words.
    whole_words keeps only tokens that start a word (leading space), dropping
    fragments like 'nesday' that dominate difference reads."""
    probs, ids = logprobs.exp().topk(pool)
    out = []
    for p, i in zip(probs.tolist(), ids.tolist()):
        raw = tokenizer.decode([i])
        if whole_words and not raw.startswith(" "):
            continue
        word = raw.strip()
        if content_only and (len(word) < 3 or not word.isalpha() or word.lower() in STOP):
            continue
        out.append({"word": word, "prob": round(p, 4)})
        if len(out) >= k:
            break
    return out
