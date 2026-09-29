"""
nla/evaluate.py

Shared NLA evaluation: FVE plus the touchstone controls, used by the
trainers, training/eval_nla.py and the GUI so all report the same numbers.

Controls (see wiki concepts/nla_touchstone):
    mean        FVE of predicting the train mean      -> 0 by construction
    shuffled    FVE when each activation is paired with ANOTHER sample's
                explanation                          -> should be <= 0
    summary     FVE from the warm-start summary of the input text (the
                context-only baseline an AV must beat after RL)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from nla.metrics import fve, normalized_mse


@torch.no_grad()
def reconstruct(ar, explanations: Sequence[str], batch_size: int = 32) -> torch.Tensor:
    was_training = ar.training
    ar.eval()
    out = []
    for i in range(0, len(explanations), batch_size):
        with torch.autocast(ar.device.type, enabled=ar.device.type == "cuda"):
            out.append(ar(list(explanations[i:i + batch_size])).float().cpu())
    ar.train(was_training)
    return torch.cat(out)


def derange(n: int, seed: int = 0) -> torch.Tensor:
    """A permutation with no fixed points (i -> perm[i] != i)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    return perm.roll(1)[torch.argsort(perm)]


def ar_scores(
    ar,
    explanations: List[str],
    gold: torch.Tensor,
    mean: torch.Tensor,
    scale: Optional[float],
    controls: bool = True,
) -> Dict[str, float]:
    pred = reconstruct(ar, explanations)
    scores = {
        "fve": fve(pred, gold, mean, scale),
        "mse_nrm": float(normalized_mse(pred, gold, scale).mean()),
    }
    if controls:
        perm = derange(len(explanations))
        scores["fve_shuffled"] = fve(pred[perm], gold, mean, scale)
        scores["fve_mean"] = fve(mean.expand_as(gold), gold, mean, scale, normalize_pred=False)
        # The fair constant baseline for direction-normalised predictions:
        # the raw mean (FVE 0) has norm < scale, which an on-sphere
        # prediction can never produce. See wiki nla_touchstone check 1.
        scores["fve_mean_direction"] = fve(mean.expand_as(gold), gold, mean, scale)
    return scores
