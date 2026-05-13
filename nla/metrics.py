"""
nla/metrics.py

Geometric reconstruction metrics.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List


def cosine_similarity_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return F.cosine_similarity(pred, target).mean().item()


def aggregate_metrics(values: List[float]) -> Dict[str, float]:
    """Return mean and std for a list of scalar metric values."""
    if not values:
        return {"mean": 0.0, "std": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std())}