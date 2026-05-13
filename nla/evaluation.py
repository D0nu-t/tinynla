"""
nla/evaluation.py

Functional evaluation metrics for activation patching experiments.

Functions:
  kl_divergence            — KL(original || patched) on last-token logits
  topk_overlap             — fraction of top-k tokens shared between conditions
  logit_cosine_similarity  — cosine between last-token logit vectors
  perplexity_shift         — ratio of sequence perplexity under patched vs original pass
  evaluate_condition       — run one patching condition and return all metrics
  evaluate_all_conditions  — full 4-condition table: original / reconstructed / random / zero
  run_interpolation_sweep  — KL and top-k vs alpha over a set of (text, description) pairs
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.patching import InterpolationPatcher


# ===========================================================================
# Scalar Metrics
# ===========================================================================

def kl_divergence(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    """KL(P_a || P_b) on last-token logits. Lower means more similar."""
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
    patch_vector: Optional[torch.Tensor],
    layer_idx: int,
    device: str,
    max_length: int = 64,
) -> float:
    """
    Ratio of sequence perplexity under patched conditions vs. original.

    perplexity_shift = PPL(patched) / PPL(original)

    Value of 1.0 → perfect behavioral preservation.
    Value >> 1.0 → patching disrupts next-token prediction.

    When patch_vector is None, returns 1.0 (identity baseline).
    """
    toks = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    input_ids = toks["input_ids"]
    if input_ids.shape[1] < 2:
        return 1.0

    def _get_ppl(use_patch: bool) -> float:
        handle = None
        if use_patch and patch_vector is not None:
            patcher = InterpolationPatcher(patch_vector, alpha=1.0)
            handle = model.transformer.h[layer_idx].register_forward_hook(
                patcher.hook_fn
            )
        with torch.no_grad():
            out = model(**toks, labels=input_ids)
            nll = out.loss.item()
        if handle is not None:
            handle.remove()
        return math.exp(nll)

    ppl_original = _get_ppl(use_patch=False)
    ppl_patched = _get_ppl(use_patch=True)

    if ppl_original == 0:
        return float("inf")
    return ppl_patched / ppl_original


# ===========================================================================
# Condition Runner
# ===========================================================================

def evaluate_condition(
    model: AutoModelForCausalLM,
    toks: Dict[str, torch.Tensor],
    original_logits: torch.Tensor,
    layer_idx: int,
    patch_vector: Optional[torch.Tensor],
    topk: int,
    alpha: float = 1.0,
) -> Dict[str, float]:
    """
    Run one patching condition and return a metrics dict.

    When patch_vector is None, returns metrics comparing original to itself
    (identity baseline; all metrics should be perfect).
    """
    if patch_vector is None:
        return {
            "kl_divergence": 0.0,
            "topk_overlap": 1.0,
            "logit_cosine": 1.0,
        }

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


# ===========================================================================
# Full 4-Condition Table
# ===========================================================================

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
    """
    Evaluate 4 patching conditions for a single text sample.

    Conditions:
      reconstructed — text-conditioned activation from AR
      random        — Gaussian noise vector (same shape as reconstructed)
      zero          — zero vector (tests model robustness)

    Returns nested dict:
      {
        "reconstructed": {"kl_divergence": ..., "topk_overlap": ..., "logit_cosine": ...},
        "random":        {...},
        "zero":          {...},
      }
    """
    toks = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
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

    random_vec = F.normalize(
        torch.randn_like(reconstructed), dim=-1
    )
    zero_vec = torch.zeros_like(reconstructed)

    return {
        "reconstructed": evaluate_condition(patch_vector=reconstructed, **shared),
        "random": evaluate_condition(patch_vector=random_vec, **shared),
        "zero": evaluate_condition(patch_vector=zero_vec, **shared),
    }


# ===========================================================================
# Interpolation Sweep
# ===========================================================================

def run_interpolation_sweep(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: List[Dict],   # list of {"text": ..., "reconstructed": Tensor}
    alphas: List[float],
    layer_idx: int,
    device: str,
    topk: int = 10,
    max_length: int = 64,
) -> Dict[float, Dict[str, float]]:
    """
    Run InterpolationPatcher at each alpha value across a set of samples.

    Returns:
        {
          alpha_0: {"kl_divergence_mean": ..., "topk_overlap_mean": ..., "logit_cosine_mean": ...},
          alpha_1: {...},
          ...
        }

    Expected pattern:
      - alpha=0.0 → KL≈0, top-k≈1.0 (identity)
      - alpha=1.0 → full replacement (matches standalone eval)
      - monotonic KL increase as alpha rises
      If KL is already high at low alpha, the reconstructed vector is
      geometrically far from what the layer expects at those positions.
    """
    results: Dict[float, Dict[str, List[float]]] = {
        a: {"kl_divergence": [], "topk_overlap": [], "logit_cosine": []}
        for a in alphas
    }

    for sample in samples:
        text = sample["text"]
        reconstructed = sample["reconstructed"]

        toks = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            original_logits = model(**toks).logits[:, -1, :]

        for alpha in alphas:
            metrics = evaluate_condition(
                model=model,
                toks=toks,
                original_logits=original_logits,
                layer_idx=layer_idx,
                patch_vector=reconstructed,
                topk=topk,
                alpha=alpha,
            )
            for k, v in metrics.items():
                results[alpha][k].append(v)

    # Aggregate
    summary = {}
    for alpha, metric_lists in results.items():
        summary[alpha] = {
            f"{k}_mean": (sum(v) / len(v) if v else 0.0)
            for k, v in metric_lists.items()
        }
    return summary