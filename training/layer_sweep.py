"""
training/layer_sweep.py

Layer Sweep Orchestrator.

Runs:
    build_buffer
    train_ar
    eval_patch
    eval_functional

for every layer in:
    cfg["activation"]["layer_indices"]

v3 upgrades:
  - Fully isolated per-layer experiment directories
  - Config snapshotting for reproducibility
  - Failure-tolerant execution
  - Runtime + status tracking
  - GPU memory cleanup between stages
  - Sweep-level summary + ranking
  - Deterministic subprocess execution
"""

import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from dotenv import load_dotenv

from nla.utils import (
    load_config,
    seed_worker,
)

load_dotenv()


# ============================================================================
# Helpers
# ============================================================================

def save_config(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(
            cfg,
            f,
            sort_keys=False,
        )


def run_step(
    name: str,
    cmd: str,
    env: dict,
) -> Dict:
    """
    Run one subprocess stage.

    Returns:
        {
            "success": bool,
            "runtime_sec": float,
            "returncode": int,
        }
    """
    print(f"\n  [{name}]")
    print(f"  cmd: {cmd}")

    start = time.time()

    result = subprocess.run(
        cmd,
        shell=True,
        env=env,
    )

    runtime = time.time() - start

    success = result.returncode == 0

    if success:
        print(f"  [OK] {name} ({runtime:.2f}s)")
    else:
        print(
            f"  [ERROR] {name} failed "
            f"(returncode={result.returncode})"
        )

    return {
        "success": success,
        "runtime_sec": runtime,
        "returncode": result.returncode,
    }


def cleanup_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_json(path: Path):
    if not path.exists():
        return None

    with open(path) as f:
        return json.load(f)


def format_metric(x):
    if x is None:
        return "nan"

    try:
        return f"{float(x):.4f}"
    except Exception:
        return "nan"


# ============================================================================
# Main
# ============================================================================

def main():

    base_cfg = load_config()

    layers = (
        base_cfg["activation"]
        .get("layer_indices", [])
    )

    if not layers:
        raise ValueError(
            "activation.layer_indices is empty."
        )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    sweep_root = (
        Path(base_cfg["paths"]["experiment_dir"])
        / f"layer_sweep_{timestamp}"
    )

    sweep_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 72)
    print("TinyNLA Layer Sweep")
    print("=" * 72)

    print(f"\nLayers: {layers}")
    print(f"Root:   {sweep_root}")

    summary: List[Dict] = []

    # ----------------------------------------------------------------------
    # Sweep loop
    # ----------------------------------------------------------------------

    for layer_idx in layers:

        print("\n" + "=" * 72)
        print(f"LAYER {layer_idx}")
        print("=" * 72)

        layer_start = time.time()

        # ------------------------------------------------------------------
        # Directories
        # ------------------------------------------------------------------

        exp_dir = sweep_root / f"layer_{layer_idx}"

        dataset_dir = exp_dir / "dataset"
        checkpoint_dir = exp_dir / "checkpoints"

        dataset_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ------------------------------------------------------------------
        # Config
        # ------------------------------------------------------------------

        cfg = deepcopy(base_cfg)

        cfg["activation"]["layer_idx"] = layer_idx

        cfg["dataset"]["output_dir"] = str(
            dataset_dir
        )

        cfg["training"]["save_dir"] = str(
            checkpoint_dir
        )

        cfg["tracking"]["run_name"] = (
            f"gpt2_layer_{layer_idx}"
        )

        cfg_path = exp_dir / "config.yaml"

        save_config(cfg, cfg_path)

        # ------------------------------------------------------------------
        # Environment
        # ------------------------------------------------------------------

        env = {
            **os.environ,
            "TINYNLA_CONFIG": str(cfg_path),
            "PYTHONHASHSEED": str(
                cfg["experiment"]["seed"]
            ),
        }

        # ------------------------------------------------------------------
        # Pipeline
        # ------------------------------------------------------------------

        steps = [
            (
                "build_buffer",
                "python -m training.build_buffer",
            ),
            (
                "train_ar",
                "python -m training.train_ar",
            ),
            (
                "eval_patch",
                "python -m training.eval_patch",
            ),
            (
                "eval_functional",
                "python -m training.eval_functional",
            ),
        ]

        step_results = []

        failed = False

        for step_name, cmd in steps:

            result = run_step(
                step_name,
                cmd,
                env,
            )

            step_results.append({
                "name": step_name,
                **result,
            })

            cleanup_cuda()

            if not result["success"]:
                failed = True
                break

        # ------------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------------

        metrics_path = (
            checkpoint_dir / "metrics.json"
        )

        interp_path = (
            checkpoint_dir / "interpolation.json"
        )

        metrics = load_json(metrics_path)
        interpolation = load_json(interp_path)

        layer_runtime = (
            time.time() - layer_start
        )

        layer_summary = {
            "layer": layer_idx,
            "success": not failed,
            "runtime_sec": layer_runtime,
            "steps": step_results,
            "metrics": metrics,
            "interpolation": interpolation,
            "paths": {
                "experiment_dir": str(exp_dir),
                "dataset_dir": str(dataset_dir),
                "checkpoint_dir": str(checkpoint_dir),
            },
        }

        summary.append(layer_summary)

        if failed:
            print(
                f"\n[WARN] Layer {layer_idx} failed."
            )
        else:
            print(
                f"\n[OK] Layer {layer_idx} complete "
                f"({layer_runtime:.2f}s)"
            )

    # ----------------------------------------------------------------------
    # Save summary
    # ----------------------------------------------------------------------

    summary_path = sweep_root / "summary.json"

    with open(summary_path, "w") as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # ----------------------------------------------------------------------
    # Ranking
    # ----------------------------------------------------------------------

    successful = [
        x for x in summary
        if x["success"]
        and x.get("metrics") is not None
    ]

    successful.sort(
        key=lambda x:
            x["metrics"].get(
                "reconstructed/kl_divergence_mean",
                float("inf"),
            )
    )

    # ----------------------------------------------------------------------
    # Console summary
    # ----------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("SWEEP SUMMARY")
    print("=" * 72)

    header = (
        f"{'Layer':>6}  "
        f"{'KL':>10}  "
        f"{'Top-k':>10}  "
        f"{'Cos':>10}  "
        f"{'PPL':>10}  "
        f"{'Status':>10}"
    )

    print("\n" + header)
    print("-" * len(header))

    for entry in summary:

        metrics = entry.get("metrics") or {}

        kl = metrics.get(
            "reconstructed/kl_divergence_mean"
        )

        topk = metrics.get(
            "reconstructed/topk_overlap_mean"
        )

        cos = metrics.get(
            "reconstructed/logit_cosine_mean"
        )

        ppl = metrics.get(
            "reconstructed/perplexity_shift_mean"
        )

        status = (
            "OK"
            if entry["success"]
            else "FAILED"
        )

        print(
            f"{entry['layer']:>6}  "
            f"{format_metric(kl):>10}  "
            f"{format_metric(topk):>10}  "
            f"{format_metric(cos):>10}  "
            f"{format_metric(ppl):>10}  "
            f"{status:>10}"
        )

    # ----------------------------------------------------------------------
    # Best layer
    # ----------------------------------------------------------------------

    if successful:

        best = successful[0]

        print("\n" + "=" * 72)
        print("BEST LAYER")
        print("=" * 72)

        best_metrics = best["metrics"]

        print(f"\nLayer: {best['layer']}")

        print(
            f"KL divergence: "
            f"{best_metrics['reconstructed/kl_divergence_mean']:.4f}"
        )

        print(
            f"Top-k overlap: "
            f"{best_metrics['reconstructed/topk_overlap_mean']:.4f}"
        )

        print(
            f"Logit cosine: "
            f"{best_metrics['reconstructed/logit_cosine_mean']:.4f}"
        )

    print("\nArtifacts:")
    print(summary_path)

    print("\n[OK] Sweep complete.")


if __name__ == "__main__":
    main()