"""
nla/config.py

Config loading and CLI helpers with no torch import, so lightweight tools
(e.g. the training dashboard) can use them without committing GPU/BLAS
memory. nla.utils re-exports everything here.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

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


def parse_overrides(pairs) -> Dict[str, Any]:
    """
    ["ar_sft.epochs=1", "data.output_dir=data/smoke"] -> nested dict.
    Values are parsed as YAML, so numbers, bools and lists work.
    """
    out: Dict[str, Any] = {}
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        if not _:
            raise ValueError(f"--set expects key=value, got {pair!r}")
        node = out
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw)
    return out


def utf8_stdio() -> None:
    """
    Make print() safe for model text on Windows.

    Redirected stdout/stderr default to cp1252 there, and explanations
    routinely contain characters it cannot encode (primes, CJK, emoji).
    """
    import sys

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def cli_config(description: str = ""):
    """Standard --config / --set parsing shared by all v4 entry points."""
    import argparse

    utf8_stdio()

    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/gpt2_small.yaml")
    p.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="override a config value, e.g. --set ar_sft.epochs=1 (repeatable)",
    )
    args, extra = p.parse_known_args()
    return load_config(args.config, overrides=parse_overrides(args.set)), extra
