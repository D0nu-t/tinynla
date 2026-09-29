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


from nla.config import (  # noqa: F401  (re-exported; torch-free)
    _DEFAULT_CONFIG,
    _deep_update,
    cli_config,
    load_config,
    load_yaml,
    parse_overrides,
    utf8_stdio,
)

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
    device = cfg.get("device", "auto")

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

# ============================================================================
# Surviving GPU memory spikes from other programs
# ============================================================================

def is_cuda_oom(err: BaseException) -> bool:
    """
    A CUDA out-of-memory error, from PyTorch's allocator (OutOfMemoryError) or
    from the driver ("CUDA error: out of memory", raised as AcceleratorError /
    RuntimeError). On Windows the GPU is shared with the desktop and browsers, so
    a job that fits can still hit either when another program briefly claims memory.
    """
    return isinstance(err, torch.OutOfMemoryError) or "out of memory" in str(err).lower()


def wait_for_gpu(attempt: int, wait: float = 30.0, label: str = "") -> None:
    """Free cached blocks and back off (wait x attempt seconds) after an OOM."""
    import gc
    import time

    print(f"\n[oom] {label or 'step'}: GPU out of memory (attempt {attempt}); "
          f"another program may be using the GPU. Retrying in {wait * attempt:.0f}s.", flush=True)
    gc.collect()
    time.sleep(wait * attempt)
    if torch.cuda.is_available():
        try:
            # when the card is fully exhausted even this call can fail with a driver
            # OOM; that must not end the run (the pipeline also retries whole stages)
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"[oom] could not free the cache yet: {str(e).splitlines()[0]}", flush=True)


def retry_on_cuda_oom(fn, tries: int = 6, wait: float = 30.0, label: str = ""):
    """
    Call fn(); on a CUDA OOM, free memory, wait and try again (up to `tries`).
    Only for work that is safe to repeat (it must not have half-applied an update).
    Total patience with the defaults: 30+60+...+150 s, about 7.5 minutes.
    """
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - re-raised unless it is an OOM
            if not is_cuda_oom(e) or attempt == tries:
                raise
            wait_for_gpu(attempt, wait, label)
