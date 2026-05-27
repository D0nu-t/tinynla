"""
nla/evaluation.py

Unified functional evaluation for TinyNLA v3.

Primary evaluation target:
    sequence-level trajectory reconstruction + trajectory patching

Core principle:
    pooled metrics alone are insufficient for NLA.
    Functional fidelity must be measured over token trajectories.

v3 additions:
    - trajectory-aware patching evaluation
    - sequence manifold diagnostics
    - hidden-state trajectory metrics
    - alpha interpolation sweeps
    - trajectory drift analysis
    - sequence perplexity preservation

Legacy pooled evaluation is retained for ablation only.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.patching import (
    InterpolationPatcher,
    SequenceInterpolationPatcher,
)

# ============================================================================
# Core Scalar Metrics
# ============================================================================


def kl_divergence(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
) -> float:
    """
    KL(P_a || P_b) on logits.

    Inputs:
        [batch, vocab]
    """
    p = F.log_softmax(logits_a, dim=-1)
    q = F.softmax(logits_b, dim=-1)

    return F.kl_div(
        p,
        q,
        reduction="batchmean",
    ).item()


def topk_overlap(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    k: int = 10,
) -> float:
    """
    Fraction of shared top-k tokens.
    """
    top_a = torch.topk(logits_a, k=k, dim=-1).indices
    top_b = torch.topk(logits_b, k=k, dim=-1).indices

    return (top_a == top_b).float().mean().item()


def logit_cosine_similarity(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
) -> float:
    """
    Cosine similarity between logits.
    """
    return F.cosine_similarity(
        logits_a,
        logits_b,
        dim=-1,
    ).mean().item()


# ============================================================================
# Sequence Geometry Metrics
# ============================================================================


def sequence_cosine_similarity(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> float:
    """
    Mean cosine similarity across sequence positions.

    Inputs:
        [seq_len, hidden_dim]
    """
    seq_len = min(
        original.shape[0],
        reconstructed.shape[0],
    )

    original = original[:seq_len]
    reconstructed = reconstructed[:seq_len]

    cos = F.cosine_similarity(
        original,
        reconstructed,
        dim=-1,
    )

    return cos.mean().item()


def trajectory_mse(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> float:
    """
    Mean trajectory MSE.
    """
    seq_len = min(
        original.shape[0],
        reconstructed.shape[0],
    )

    original = original[:seq_len]
    reconstructed = reconstructed[:seq_len]

    return F.mse_loss(
        original,
        reconstructed,
    ).item()


def trajectory_norm_difference(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> float:
    """
    Difference in trajectory norm statistics.

    Measures manifold distortion.
    """
    seq_len = min(
        original.shape[0],
        reconstructed.shape[0],
    )

    original = original[:seq_len]
    reconstructed = reconstructed[:seq_len]

    orig_norm = original.norm(dim=-1).mean()
    recon_norm = reconstructed.norm(dim=-1).mean()

    return abs(orig_norm - recon_norm).item()


def trajectory_drift(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> float:
    """
    Measures local transition preservation.

    Computes cosine similarity between:
        Δh_t = h_t+1 - h_t

    This is substantially more informative than
    static token cosine alone.
    """
    seq_len = min(
        original.shape[0],
        reconstructed.shape[0],
    )

    if seq_len < 2:
        return 1.0

    original = original[:seq_len]
    reconstructed = reconstructed[:seq_len]

    delta_orig = original[1:] - original[:-1]
    delta_recon = reconstructed[1:] - reconstructed[:-1]

    cos = F.cosine_similarity(
        delta_orig,
        delta_recon,
        dim=-1,
    )

    return cos.mean().item()


# ============================================================================
# Hidden-State Capture
# ============================================================================


class HiddenStateRecorder:
    """
    Records hidden trajectories from a transformer layer.
    """

    def __init__(self):
        self.hidden = None

    def hook_fn(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            self.hidden = outputs[0].detach()
        else:
            self.hidden = outputs.detach()


# ============================================================================
# Perplexity Shift
# ============================================================================


def perplexity_shift(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    patch_tensor: Optional[torch.Tensor],
    layer_idx: int,
    device: str,
    max_length: int = 128,
    sequence_mode: bool = True,
) -> float:
    """
    PPL(patched) / PPL(original)

    1.0 = perfect preservation.
    """

    toks = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    if toks["input_ids"].shape[1] < 2:
        return 1.0

    def _compute_ppl(use_patch: bool) -> float:
        handle = None

        if use_patch and patch_tensor is not None:
            if sequence_mode:
                patcher = SequenceInterpolationPatcher(
                    patch_tensor,
                    alpha=1.0,
                )
            else:
                patcher = InterpolationPatcher(
                    patch_tensor,
                    alpha=1.0,
                )

            handle = model.transformer.h[layer_idx].register_forward_hook(
                patcher.hook_fn
            )

        with torch.no_grad():
            out = model(
                **toks,
                labels=toks["input_ids"],
            )

            nll = out.loss.item()

        if handle is not None:
            handle.remove()

        return math.exp(nll)

    ppl_orig = _compute_ppl(False)
    ppl_patch = _compute_ppl(True)

    if ppl_orig <= 0:
        return float("inf")

    return ppl_patch / ppl_orig


# ============================================================================
# Sequence Functional Evaluation (PRIMARY)
# ============================================================================


def evaluate_condition_sequence(
    model: AutoModelForCausalLM,
    toks: Dict,
    original_logits: torch.Tensor,
    layer_idx: int,
    patch_sequence: Optional[torch.Tensor],
    topk: int,
    alpha: float = 1.0,
) -> Dict[str, float]:
    """
    Functional evaluation for sequence patching.
    """

    if patch_sequence is None:
        return {
            "kl_divergence": 0.0,
            "topk_overlap": 1.0,
            "logit_cosine": 1.0,
        }

    patcher = SequenceInterpolationPatcher(
        patch_sequence,
        alpha=alpha,
    )

    handle = model.transformer.h[layer_idx].register_forward_hook(
        patcher.hook_fn
    )

    with torch.no_grad():
        out = model(**toks)
        patched_logits = out.logits[:, -1, :]

    handle.remove()

    return {
        "kl_divergence": kl_divergence(
            original_logits,
            patched_logits,
        ),

        "topk_overlap": topk_overlap(
            original_logits,
            patched_logits,
            k=topk,
        ),

        "logit_cosine": logit_cosine_similarity(
            original_logits,
            patched_logits,
        ),
    }


def evaluate_hidden_trajectory(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    reconstructed_sequence: torch.Tensor,
    layer_idx: int,
    device: str,
    max_length: int = 128,
    alpha: float = 1.0,
) -> Dict[str, float]:
    """
    Direct trajectory-level evaluation.

    This is the critical v3 addition.

    Measures:
        - token trajectory cosine
        - transition preservation
        - manifold drift
        - hidden-state MSE
    """

    toks = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    # ----------------------------------------------------------------------
    # Capture original hidden states
    # ----------------------------------------------------------------------

    orig_recorder = HiddenStateRecorder()

    orig_handle = model.transformer.h[layer_idx].register_forward_hook(
        orig_recorder.hook_fn
    )

    with torch.no_grad():
        model(**toks)

    orig_handle.remove()

    original_hidden = orig_recorder.hidden.squeeze(0)

    # ----------------------------------------------------------------------
    # Capture patched hidden states
    # ----------------------------------------------------------------------

    patched_recorder = HiddenStateRecorder()

    patcher = SequenceInterpolationPatcher(
        reconstructed_sequence,
        alpha=alpha,
    )

    patch_handle = model.transformer.h[layer_idx].register_forward_hook(
        patcher.hook_fn
    )

    recorder_handle = model.transformer.h[layer_idx].register_forward_hook(
        patched_recorder.hook_fn
    )

    with torch.no_grad():
        model(**toks)

    patch_handle.remove()
    recorder_handle.remove()

    patched_hidden = patched_recorder.hidden.squeeze(0)

    # ----------------------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------------------

    return {
        "trajectory_cosine": sequence_cosine_similarity(
            original_hidden,
            patched_hidden,
        ),

        "trajectory_mse": trajectory_mse(
            original_hidden,
            patched_hidden,
        ),

        "trajectory_drift": trajectory_drift(
            original_hidden,
            patched_hidden,
        ),

        "norm_difference": trajectory_norm_difference(
            original_hidden,
            patched_hidden,
        ),
    }


def evaluate_all_conditions_sequence(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    reconstructed_sequence: torch.Tensor,
    layer_idx: int,
    device: str,
    topk: int = 10,
    max_length: int = 128,
) -> Dict[str, Dict[str, float]]:
    """
    4-condition sequence evaluation.
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

    random_seq = F.normalize(
        torch.randn_like(reconstructed_sequence),
        dim=-1,
    )

    zero_seq = torch.zeros_like(reconstructed_sequence)

    return {
        "reconstructed": evaluate_condition_sequence(
            patch_sequence=reconstructed_sequence,
            **shared,
        ),

        "random": evaluate_condition_sequence(
            patch_sequence=random_seq,
            **shared,
        ),

        "zero": evaluate_condition_sequence(
            patch_sequence=zero_seq,
            **shared,
        ),
    }


def run_interpolation_sweep_sequence(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: List[Dict],
    alphas: List[float],
    layer_idx: int,
    device: str,
    topk: int = 10,
    max_length: int = 128,
) -> Dict[float, Dict[str, float]]:
    """
    Sequence trajectory interpolation sweep.

    Primary v3 diagnostic.
    """

    results = {
        alpha: {
            "kl_divergence": [],
            "topk_overlap": [],
            "logit_cosine": [],
            "trajectory_cosine": [],
            "trajectory_drift": [],
            "norm_difference": [],
            "trajectory_mse": [],
        }
        for alpha in alphas
    }

    for sample in samples:
        text = sample["text"]
        recon_seq = sample["reconstructed_sequence"]

        toks = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            original_logits = model(**toks).logits[:, -1, :]

        for alpha in alphas:

            functional = evaluate_condition_sequence(
                model=model,
                toks=toks,
                original_logits=original_logits,
                layer_idx=layer_idx,
                patch_sequence=recon_seq,
                topk=topk,
                alpha=alpha,
            )

            trajectory = evaluate_hidden_trajectory(
                model=model,
                tokenizer=tokenizer,
                text=text,
                reconstructed_sequence=recon_seq,
                layer_idx=layer_idx,
                device=device,
                max_length=max_length,
                alpha=alpha,
                
            )

            merged = {
                **functional,
                **trajectory,
            }

            for k, v in merged.items():
                results[alpha][k].append(v)

    return {
        alpha: {
            f"{metric}_mean": (
                sum(values) / len(values)
                if values else 0.0
            )
            for metric, values in metrics.items()
        }
        for alpha, metrics in results.items()
    }


# ============================================================================
# Legacy Pooled Evaluation
# ============================================================================


def evaluate_condition(
    model: AutoModelForCausalLM,
    toks: Dict,
    original_logits: torch.Tensor,
    layer_idx: int,
    patch_vector: Optional[torch.Tensor],
    topk: int,
    alpha: float = 1.0,
) -> Dict[str, float]:
    """
    Legacy pooled evaluation.
    """

    if patch_vector is None:
        return {
            "kl_divergence": 0.0,
            "topk_overlap": 1.0,
            "logit_cosine": 1.0,
        }

    patcher = InterpolationPatcher(
        patch_vector,
        alpha=alpha,
    )

    handle = model.transformer.h[layer_idx].register_forward_hook(
        patcher.hook_fn
    )

    with torch.no_grad():
        out = model(**toks)
        patched_logits = out.logits[:, -1, :]

    handle.remove()

    return {
        "kl_divergence": kl_divergence(
            original_logits,
            patched_logits,
        ),

        "topk_overlap": topk_overlap(
            original_logits,
            patched_logits,
            k=topk,
        ),

        "logit_cosine": logit_cosine_similarity(
            original_logits,
            patched_logits,
        ),
    }