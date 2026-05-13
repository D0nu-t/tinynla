"""
nla/utils.py

Shared utilities used across training and evaluation scripts.

load_config     — resolves config path from TINYNLA_CONFIG env var or default
resolve_device  — "auto" → cuda > cpu
set_seed        — deterministic seeding across Python, NumPy, and PyTorch
"""

import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


_DEFAULT_CONFIG = "configs/base.yaml"


def load_config(override_path: str | None = None) -> Dict[str, Any]:
    """
    Load YAML config. Priority:
      1. override_path argument (from CLI or layer_sweep)
      2. TINYNLA_CONFIG environment variable
      3. configs/base.yaml default
    """
    path = (
        override_path
        or os.environ.get("TINYNLA_CONFIG")
        or _DEFAULT_CONFIG
    )
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_device(cfg: Dict[str, Any]) -> str:
    device = cfg.get("device", "auto")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False