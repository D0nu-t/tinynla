"""
training/layer_sweep.py

Layer Sweep Orchestrator.

Runs the full TinyNLA pipeline (buffer -> train -> eval_patch -> eval_functional)
for each layer in cfg["activation"]["layer_indices"].

Each layer gets:
  - isolated dataset directory
  - isolated checkpoint directory
  - a dedicated YAML config written to disk
  - per-layer metrics stored in summary.json

Config is passed to subprocesses via the TINYNLA_CONFIG env var,
which all training scripts read via nla.utils.load_config().

This replaces the previous pattern of modifying base.yaml in-place,
which caused races and made experiments non-reproducible.

Usage:
    python -m training.layer_sweep
    .\run_all.ps1 -Sweep
"""

import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml
import torch
from dotenv import load_dotenv
from nla.utils import load_config


def save_config(cfg: dict, path: Path) -> None:
    with open(path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)


def run_step(cmd: str, env: dict) -> None:
    result = subprocess.run(cmd, shell=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {cmd}")


def main():
    base_cfg = load_config()
    layers = base_cfg["activation"]["layer_indices"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = Path(base_cfg["paths"]["experiment_dir"]) / f"layer_sweep_{timestamp}"
    sweep_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"LAYER SWEEP — {len(layers)} layers")
    print(f"Experiment root: {sweep_root}")
    print(f"{'='*60}")

    summary: list = []

    for layer_idx in layers:
        print(f"\n{'='*60}")
        print(f"LAYER {layer_idx}")
        print(f"{'='*60}")

        # ------------------------------------------------------------------
        # Per-layer experiment paths
        # ------------------------------------------------------------------
        exp_dir = sweep_root / f"layer_{layer_idx}"
        dataset_dir = exp_dir / "dataset"
        checkpoint_dir = exp_dir / "checkpoints"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Build layer-specific config
        # ------------------------------------------------------------------
        cfg = deepcopy(base_cfg)
        cfg["activation"]["layer_idx"] = layer_idx
        cfg["dataset"]["output_dir"] = str(dataset_dir)
        cfg["training"]["save_dir"] = str(checkpoint_dir)
        cfg["tracking"]["run_name"] = f"gpt2_layer_{layer_idx}"

        cfg_path = exp_dir / "config.yaml"
        save_config(cfg, cfg_path)

        # Pass config path to all subprocesses via env var
        env = {**os.environ, "TINYNLA_CONFIG": str(cfg_path)}

        # ------------------------------------------------------------------
        # Pipeline stages
        # ------------------------------------------------------------------
        steps = [
            ("build_buffer",    "python -m training.build_buffer"),
            ("train_ar",        "python -m training.train_ar"),
            ("eval_patch",      "python -m training.eval_patch"),
            ("eval_functional", "python -m training.eval_functional"),
        ]

        failed = False
        for step_name, cmd in steps:
            print(f"\n  [{step_name}]")
            try:
                run_step(cmd, env)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                print(f"  [ERROR] {e}")
                failed = True
                break

        if failed:
            print(f"  Layer {layer_idx} failed; skipping to next.")
            continue

        # ------------------------------------------------------------------
        # Collect metrics
        # ------------------------------------------------------------------
        metrics_path = checkpoint_dir / "metrics.json"
        interp_path = checkpoint_dir / "interpolation.json"

        layer_summary: dict = {"layer": layer_idx}

        if metrics_path.exists():
            with open(metrics_path) as f:
                layer_summary["metrics"] = json.load(f)

        if interp_path.exists():
            with open(interp_path) as f:
                layer_summary["interpolation"] = json.load(f)

        summary.append(layer_summary)

        print(f"\n  [OK] Layer {layer_idx} complete.")

    # ------------------------------------------------------------------
    # Save sweep summary
    # ------------------------------------------------------------------
    summary_path = sweep_root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("LAYER SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Layer':>6}  {'KL (recon)':>12}  {'Top-k':>8}  {'PPL shift':>10}")
    print("-" * 44)
    for entry in summary:
        m = entry.get("metrics", {})
        kl  = m.get("reconstructed/kl_divergence_mean", float("nan"))
        ovr = m.get("reconstructed/topk_overlap_mean", float("nan"))
        ppl = m.get("reconstructed/perplexity_shift_mean", float("nan"))
        print(f"{entry['layer']:>6}  {kl:>12.4f}  {ovr:>8.4f}  {ppl:>10.4f}")

    print(f"\nArtifacts: {sweep_root}")
    print("[OK] Sweep complete.")


if __name__ == "__main__":
    main()