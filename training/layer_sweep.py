import os
import json
import yaml
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

from datetime import datetime


"""
Layer Sweep Orchestrator
========================

Purpose
-------
Runs the entire TinyNLA pipeline across multiple
transformer layers automatically.

For each layer:
    1. Build activation buffer
    2. Train activation reconstructor
    3. Evaluate reconstruction
    4. Evaluate functional preservation
    5. Save metrics + artifacts

This enables:
    - layer-wise representation analysis
    - recoverability studies
    - semantic depth analysis
    - causal preservation comparison

Recommended Usage
-----------------

python -m training.layer_sweep
"""


# =========================================================
# Utility Functions
# =========================================================

def load_base_config():

    with open("configs/base.yaml", "r") as f:
        return yaml.safe_load(f)


def save_config(cfg, path):

    with open(path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)


def run_command(cmd):

    result = subprocess.run(
        cmd,
        shell=True
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"Command failed:\n{cmd}"
        )


def ensure_dir(path):

    Path(path).mkdir(
        parents=True,
        exist_ok=True
    )


# =========================================================
# Main Sweep
# =========================================================

def main():

    # -----------------------------------------------------
    # Sweep Configuration
    # -----------------------------------------------------

    layers = [1, 3, 5, 7, 9, 11]

    base_cfg = load_base_config()

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    sweep_root = Path(
        f"experiments/layer_sweep_{timestamp}"
    )

    ensure_dir(sweep_root)

    summary_metrics = []

    # -----------------------------------------------------
    # Layer Loop
    # -----------------------------------------------------

    for layer_idx in layers:

        print("\n")
        print("=" * 60)
        print(f"RUNNING LAYER {layer_idx}")
        print("=" * 60)

        # -------------------------------------------------
        # Experiment Paths
        # -------------------------------------------------

        exp_dir = sweep_root / f"layer_{layer_idx}"

        ensure_dir(exp_dir)

        dataset_dir = (
            exp_dir / "dataset"
        )

        checkpoint_dir = (
            exp_dir / "checkpoints"
        )

        ensure_dir(dataset_dir)
        ensure_dir(checkpoint_dir)

        # -------------------------------------------------
        # Config Construction
        # -------------------------------------------------

        cfg = deepcopy(base_cfg)

        cfg["activation"]["layer_idx"] = (
            layer_idx
        )

        cfg["dataset"]["output_dir"] = str(
            dataset_dir
        )

        cfg["training"]["save_dir"] = str(
            checkpoint_dir
        )

        cfg["tracking"]["run_name"] = (
            f"gpt2_layer_{layer_idx}"
        )

        # Temporary config for this layer
        temp_cfg_path = (
            exp_dir / "config.yaml"
        )

        save_config(
            cfg,
            temp_cfg_path
        )

        # -------------------------------------------------
        # Environment Variable Override
        # -------------------------------------------------

        os.environ["TINYNLA_CONFIG"] = str(
            temp_cfg_path
        )

        # -------------------------------------------------
        # Stage 1
        # -------------------------------------------------

        print("\n[1/4] Building activation buffer")

        run_command(
            "python -m training.build_buffer"
        )

        # -------------------------------------------------
        # Stage 2
        # -------------------------------------------------

        print("\n[2/4] Training AR")

        run_command(
            "python -m training.train_ar"
        )

        # -------------------------------------------------
        # Stage 3
        # -------------------------------------------------

        print("\n[3/4] Evaluating reconstruction")

        run_command(
            "python -m training.eval_patch"
        )

        # -------------------------------------------------
        # Stage 4
        # -------------------------------------------------

        print("\n[4/4] Functional evaluation")

        run_command(
            "python -m training.eval_functional"
        )

        # -------------------------------------------------
        # Collect Metrics
        # -------------------------------------------------

        metrics_path = (
            checkpoint_dir / "metrics.json"
        )

        if metrics_path.exists():

            with open(metrics_path, "r") as f:

                metrics = json.load(f)

            metrics["layer"] = layer_idx

            summary_metrics.append(metrics)

        # -------------------------------------------------
        # Snapshot Current Outputs
        # -------------------------------------------------

        # Save current buffer
        src_buffer = (
            Path(
                cfg["dataset"]["output_dir"]
            ) / "buffer.pt"
        )

        dst_buffer = (
            exp_dir / "buffer.pt"
        )

        if src_buffer.exists():

            shutil.copy2(
                src_buffer,
                dst_buffer
            )

        # Save best model
        src_model = (
            Path(
                cfg["training"]["save_dir"]
            ) / "best_model.pt"
        )

        dst_model = (
            exp_dir / "best_model.pt"
        )

        if src_model.exists():

            shutil.copy2(
                src_model,
                dst_model
            )

        print("\n[OK] Layer completed")

    # =====================================================
    # Save Global Summary
    # =====================================================

    summary_path = (
        sweep_root / "summary.json"
    )

    with open(summary_path, "w") as f:

        json.dump(
            summary_metrics,
            f,
            indent=2
        )

    # =====================================================
    # Final Output
    # =====================================================

    print("\n")
    print("=" * 60)
    print("LAYER SWEEP COMPLETE")
    print("=" * 60)

    print("\nArtifacts saved to:\n")

    print(sweep_root)

    print("\nSummary metrics:")

    for item in summary_metrics:

        print(
            f"Layer {item['layer']} | "
            f"cosine={item.get('cosine', 'N/A')} | "
            f"kl={item.get('kl_divergence', 'N/A')}"
        )

    print("\n[OK] Sweep completed.")


if __name__ == "__main__":
    main()