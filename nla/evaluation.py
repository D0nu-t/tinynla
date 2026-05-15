"""
nla/evaluation.py

Functional evaluation metrics for activation patching experiments.

Scalar metrics (shared between pooled and sequence modes):
  kl_divergence            — KL(original || patched) on last-token logits
  topk_overlap             — fraction of top-k tokens shared
  logit_cosine_similarity  — cosine between logit vectors

Sequence-aware evaluation (v3):
  evaluate_condition_sequence    — run one patching condition with a [seq_len, hidden_dim] patch
  evaluate_all_conditions_sequence — 4-condition table for sequence mode
  run_interpolation_sweep_sequence — alpha sweep for sequence patching

Legacy pooled evaluation (retained for ablation):
  evaluate_condition        — pooled [1, hidden_dim] patching
  evaluate_all_conditions   — pooled 4-condition table
  run_interpolation_sweep   — pooled alpha sweep

perplexity_shift            — PPL(patched) / PPL(original); works for both modes
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.patching import InterpolationPatcher, SequenceInterpolationPatcher


# ===========================================================================
# Scalar Metrics (shared)
# ===========================================================================

def kl_divergence(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    """KL(P_a || P_b) on last-token logits."""
    p = F.log_softmax(logits_a, dim=-1)
    q = F.softmax(logits_b, dim=-1)
    return F.kl_div(p, q, reduction="batchmean").item()


def topk_overlap(logits_a: torch.Tensor, logits_b: torch.Tensor, k: int = 10) -> float:
    """Fraction of tokens in top-k(a) that also appear in top-k(b)."""
    top_a = torch.topk(logits_a, k=k, dim=-1).indices
    top_b = torch.topk(logits_b, k=k, dim=-1).indices
    return (top_a == top_b).float().mean().item()


def logit_cosine_similarity(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    return F.cosine_similarity(logits_a, logits_b, dim=-1).mean().item()


def perplexity_shift(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    patch_tensor: Optional[torch.Tensor],   # [seq_len, hidden] or [1, hidden]
    layer_idx: int,
    device: str,
    max_length: int = 64,
    sequence_mode: bool = True,
) -> float:
    """
    PPL(patched) / PPL(original).

    1.0 = perfect preservation.  >> 1.0 = patching disrupts prediction.
    Accepts both pooled and sequence patch tensors.
    """
    toks = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(device)

    if toks["input_ids"].shape[1] < 2:
        return 1.0

    def _ppl(use_patch: bool) -> float:
        handle = None
        if use_patch and patch_tensor is not None:
            if sequence_mode:
                patcher = SequenceInterpolationPatcher(patch_tensor, alpha=1.0)
            else:
                patcher = InterpolationPatcher(patch_tensor, alpha=1.0)
            handle = model.transformer.h[layer_idx].register_forward_hook(
                patcher.hook_fn
            )
        with torch.no_grad():
            out = model(**toks, labels=toks["input_ids"])
            nll = out.loss.item()
        if handle is not None:
            handle.remove()
        return math.exp(nll)

    ppl_orig = _ppl(use_patch=False)
    ppl_patch = _ppl(use_patch=True)
    return ppl_patch / ppl_orig if ppl_orig > 0 else float("inf")


# ===========================================================================
# Sequence evaluation (v3)
# ===========================================================================

def evaluate_condition_sequence(
    model: AutoModelForCausalLM,
    toks: Dict,
    original_logits: torch.Tensor,
    layer_idx: int,
    patch_sequence: Optional[torch.Tensor],   # [seq_len, hidden_dim]
    topk: int,
    alpha: float = 1.0,
) -> Dict[str, float]:
    """
    Run one patching condition with a sequence reconstruction.
    patch_sequence=None returns the identity baseline.
    """
    if patch_sequence is None:
        return {"kl_divergence": 0.0, "topk_overlap": 1.0, "logit_cosine": 1.0}

    patcher = SequenceInterpolationPatcher(patch_sequence, alpha=alpha)
    handle = model.transformer.h[layer_idx].register_forward_hook(patcher.hook_fn)

    with torch.no_grad():
        out = model(**toks)
        patched_logits = out.logits[:, -1, :]

    handle.remove()

    return {
        "kl_divergence": kl_divergence(original_logits, patched_logits),
        "topk_overlap": topk_overlap(original_logits, patched_logits, k=topk),
        "logit_cosine": logit_cosine_similarity(original_logits, patched_logits),
    }


def evaluate_all_conditions_sequence(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    reconstructed_sequence: torch.Tensor,   # [seq_len, hidden_dim]
    layer_idx: int,
    device: str,
    topk: int = 10,
    max_length: int = 64,
) -> Dict[str, Dict[str, float]]:
    """
    4-condition table for sequence patching:
      reconstructed — AR output [seq_len, hidden_dim]
      random        — Gaussian noise, same shape
      zero          — zeros, same shape
    """
    toks = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(device)

    with torch.no_grad():
        original_logits = model(**toks).logits[:, -1, :]

    shared = dict(
        model=model,
        toks=toks,
        original_logits=original_logits,
        layer_idx=layer_idx,
        topk=topk,
    )

    random_seq = F.normalize(
        torch.randn_like(reconstructed_sequence), dim=-1
    )
    zero_seq = torch.zeros_like(reconstructed_sequence)

    return {
        "reconstructed": evaluate_condition_sequence(
            patch_sequence=reconstructed_sequence, **shared
        ),
        "random": evaluate_condition_sequence(
            patch_sequence=random_seq, **shared
        ),
        "zero": evaluate_condition_sequence(
            patch_sequence=zero_seq, **shared
        ),
    }


def run_interpolation_sweep_sequence(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: List[Dict],    # list of {"text": str, "reconstructed_sequence": Tensor}
    alphas: List[float],
    layer_idx: int,
    device: str,
    topk: int = 10,
    max_length: int = 64,
) -> Dict[float, Dict[str, float]]:
    """
    Sequence-mode interpolation sweep.
    h_patch[t] = α·h_recon[t] + (1-α)·h_orig[t]  per position.

    Returns per-alpha aggregated metrics.
    """
    results: Dict[float, Dict[str, List[float]]] = {
        a: {"kl_divergence": [], "topk_overlap": [], "logit_cosine": []}
        for a in alphas
    }

    for sample in samples:
        text = sample["text"]
        recon_seq = sample["reconstructed_sequence"]    # [seq_len, hidden]

        toks = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(device)

        with torch.no_grad():
            original_logits = model(**toks).logits[:, -1, :]

        for alpha in alphas:
            metrics = evaluate_condition_sequence(
                model=model,
                toks=toks,
                original_logits=original_logits,
                layer_idx=layer_idx,
                patch_sequence=recon_seq,
                topk=topk,
                alpha=alpha,
            )
            for k, v in metrics.items():
                results[alpha][k].append(v)

    return {
        alpha: {
            f"{k}_mean": sum(v) / len(v) if v else 0.0
            for k, v in metric_lists.items()
        }
        for alpha, metric_lists in results.items()
    }


# ===========================================================================
# Legacy pooled evaluation (retained for ablation)
# ===========================================================================

def evaluate_condition(
    model: AutoModelForCausalLM,
    toks: Dict,
    original_logits: torch.Tensor,
    layer_idx: int,
    patch_vector: Optional[torch.Tensor],
    topk: int,
    alpha: float = 1.0,
) -> Dict[str, float]:
    """Pooled patching condition runner (legacy)."""
    if patch_vector is None:
        return {"kl_divergence": 0.0, "topk_overlap": 1.0, "logit_cosine": 1.0}

    patcher = InterpolationPatcher(patch_vector, alpha=alpha)
    handle = model.transformer.h[layer_idx].register_forward_hook(patcher.hook_fn)

    with torch.no_grad():
        out = model(**toks)
        patched_logits = out.logits[:, -1, :]

    handle.remove()

    return {
        "kl_divergence": kl_divergence(original_logits, patched_logits),
        "topk_overlap": topk_overlap(original_logits, patched_logits, k=topk),
        "logit_cosine": logit_cosine_similarity(original_logits, patched_logits),
    }


def evaluate_all_conditions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    reconstructed: torch.Tensor,
    layer_idx: int,
    device: str,
    topk: int = 10,
    max_length: int = 64,
) -> Dict[str, Dict[str, float]]:
    """Pooled 4-condition table (legacy)."""
    toks = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(device)

    with torch.no_grad():
        original_logits = model(**toks).logits[:, -1, :]

    shared = dict(
        model=model, toks=toks, original_logits=original_logits,
        layer_idx=layer_idx, topk=topk,
    )

    random_vec = F.normalize(torch.randn_like(reconstructed), dim=-1)
    zero_vec = torch.zeros_like(reconstructed)

    return {
        "reconstructed": evaluate_condition(patch_vector=reconstructed, **shared),
        "random": evaluate_condition(patch_vector=random_vec, **shared),
        "zero": evaluate_condition(patch_vector=zero_vec, **shared),
    }


def run_interpolation_sweep(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: List[Dict],
    alphas: List[float],
    layer_idx: int,
    device: str,
    topk: int = 10,
    max_length: int = 64,
) -> Dict[float, Dict[str, float]]:
    """Pooled interpolation sweep (legacy)."""
    results = {
        a: {"kl_divergence": [], "topk_overlap": [], "logit_cosine": []}
        for a in alphas
    }

    for sample in samples:
        toks = tokenizer(
            sample["text"], return_tensors="pt",
            truncation=True, max_length=max_length
        ).to(device)

        with torch.no_grad():
            original_logits = model(**toks).logits[:, -1, :]

        for alpha in alphas:
            metrics = evaluate_condition(
                model=model, toks=toks, original_logits=original_logits,
                layer_idx=layer_idx, patch_vector=sample["reconstructed"],
                topk=topk, alpha=alpha,
            )
            for k, v in metrics.items():
                results[alpha][k].append(v)

    return {
        alpha: {f"{k}_mean": sum(v) / len(v) if v else 0.0 for k, v in ml.items()}
        for alpha, ml in results.items()
    }