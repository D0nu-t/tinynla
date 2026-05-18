"""
nla/utils.py

Shared utilities used across training and evaluation scripts.

Features:
  - Config loading with environment override
  - Recursive config merging
  - Device resolution
  - Deterministic seeding
  - Mixed precision helpers
  - Safe filesystem utilities
  - Run metadata helpers
"""

from __future__ import annotations

import json
import os
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


# ============================================================================
# Constants
# ============================================================================

_DEFAULT_CONFIG = "configs/base.yaml"


# ============================================================================
# Config utilities
# ============================================================================

def _deep_update(
    base: Dict[str, Any],
    override: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Recursively merge dictionaries.

    Values in override take precedence.
    """
    out = deepcopy(base)

    for k, v in override.items():

        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
        ):
            out[k] = _deep_update(out[k], v)

        else:
            out[k] = v

    return out


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """
    Load YAML safely.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(
    override_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load TinyNLA config.

    Priority:
      1. override_path argument
      2. TINYNLA_CONFIG env var
      3. configs/base.yaml

    Args:
        override_path:
            Explicit YAML path.

        overrides:
            Optional runtime overrides dictionary.

    Returns:
        Fully merged config dict.
    """
    path = (
        override_path
        or os.environ.get("TINYNLA_CONFIG")
        or _DEFAULT_CONFIG
    )

    cfg = load_yaml(path)

    if overrides:
        cfg = _deep_update(cfg, overrides)

    return cfg


def save_config(
    cfg: Dict[str, Any],
    path: str | Path,
) -> None:
    """
    Save config as JSON.
    """
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            cfg,
            f,
            indent=2,
        )


# ============================================================================
# Filesystem helpers
# ============================================================================

def ensure_dir(path: str | Path) -> Path:
    """
    Create directory if missing.
    """
    path = Path(path)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def timestamp() -> str:
    """
    Timestamp for experiment naming.
    """
    return datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


# ============================================================================
# Device helpers
# ============================================================================

def resolve_device(
    cfg: Dict[str, Any],
) -> str:
    """
    Resolve torch device.

    device:
      auto -> cuda > mps > cpu
    """
    device = "auto"

    if device != "auto":
        return device

    if torch.cuda.is_available():
        return "cuda"

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


def get_autocast_dtype(
    device: str,
):
    """
    Recommended autocast dtype.
    """
    if device == "cuda":
        return torch.float16

    if device == "mps":
        return torch.float16

    return torch.bfloat16


def use_amp(device: str) -> bool:
    """
    Whether AMP should be enabled.
    """
    return device in {"cuda", "mps"}


# ============================================================================
# Seeding
# ============================================================================

def set_seed(
    seed: int,
    deterministic: bool = True,
) -> None:
    """
    Global deterministic seeding.
    """
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


# ============================================================================
# Tensor utilities
# ============================================================================

def move_to_device(
    batch: Dict[str, Any],
    device: str,
) -> Dict[str, Any]:
    """
    Move tensor-valued batch fields to device.
    """
    out = {}

    for k, v in batch.items():

        if torch.is_tensor(v):
            out[k] = v.to(device)

        else:
            out[k] = v

    return out


def count_parameters(
    model: torch.nn.Module,
    trainable_only: bool = True,
) -> int:
    """
    Count model parameters.
    """
    if trainable_only:
        return sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

    return sum(
        p.numel()
        for p in model.parameters()
    )


# ============================================================================
# Logging helpers
# ============================================================================

def print_config(
    cfg: Dict[str, Any],
) -> None:
    """
    Pretty-print config.
    """
    print(
        yaml.dump(
            cfg,
            sort_keys=False,
            default_flow_style=False,
        )
    )


def log_header(title: str) -> None:
    """
    Standard console section header.
    """
    bar = "=" * 80

    print()
    print(bar)
    print(title)
    print(bar)


# ============================================================================
# Validation
# ============================================================================

def validate_config(
    cfg: Dict[str, Any],
) -> None:
    """
    Basic runtime config validation.
    """
    required_top_level = [
        "experiment",
        "model",
        "activation",
        "dataset",
        "training",
        "evaluation",
    ]

    for key in required_top_level:

        if key not in cfg:
            raise ValueError(
                f"Missing config section: {key}"
            )

    if cfg["activation"]["max_length"] <= 0:
        raise ValueError(
            "activation.max_length must be > 0"
        )

    if cfg["training"]["batch_size"] <= 0:
        raise ValueError(
            "training.batch_size must be > 0"
        )

    if cfg["training"]["epochs"] <= 0:
        raise ValueError(
            "training.epochs must be > 0"
        )

    if cfg["training"]["lr"] <= 0:
        raise ValueError(
            "training.lr must be > 0"
        )