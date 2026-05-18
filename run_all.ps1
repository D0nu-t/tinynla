# ==========================================================
# TinyNLA Research Pipeline Runner
# ==========================================================
#
# Full Sequential Pipeline
#
# Stages
# -------
#   1. Build activation buffer
#   2. Train activation reconstructor
#   3. Geometric reconstruction evaluation
#   3.5. Manifold fidelity evaluation
#   4. Functional activation patching evaluation
#
# Optional Modes
# --------------
#   Standard:
#
#       .\run_all.ps1
#
#   Layer Sweep:
#
#       .\run_all.ps1 -Sweep
#
# ==========================================================

param(
    [switch]$Sweep
)

# ==========================================================
# Helpers
# ==========================================================

function Print-Header($msg) {

    Write-Host ""
    Write-Host "=================================================="
    Write-Host $msg
    Write-Host "=================================================="
    Write-Host ""
}

function Run-Step($stepName, $command) {

    Write-Host ""
    Write-Host "--------------------------------------------------"
    Write-Host $stepName
    Write-Host "--------------------------------------------------"
    Write-Host ""

    Invoke-Expression $command

    if ($LASTEXITCODE -ne 0) {

        Write-Host ""
        Write-Host "[ERROR] Step failed:"
        Write-Host $stepName
        Write-Host ""

        exit 1
    }

    Write-Host ""
    Write-Host "[OK] Completed:"
    Write-Host $stepName
    Write-Host ""
}

# ==========================================================
# Startup
# ==========================================================

Print-Header "TinyNLA Research Pipeline"

Write-Host "Environment"
Write-Host "-----------"

Write-Host ("Python: " + (python --version))

if ($env:CUDA_VISIBLE_DEVICES) {

    Write-Host (
        "CUDA_VISIBLE_DEVICES=" +
        $env:CUDA_VISIBLE_DEVICES
    )
}

Write-Host ""

# ==========================================================
# Layer Sweep Mode
# ==========================================================

if ($Sweep) {

    Print-Header "Running Layer Sweep"

    Run-Step `
        "[Sweep] Multi-layer experiment orchestration" `
        "python -m training.layer_sweep"

    Print-Header "Layer Sweep Completed"

    exit 0
}

# ==========================================================
# STEP 1 — BUILD BUFFER
# ==========================================================

Run-Step `
    "[1/5] Building activation buffer" `
    "python -m training.build_buffer"

# ==========================================================
# STEP 2 — TRAIN AR
# ==========================================================

Run-Step `
    "[2/5] Training activation reconstructor" `
    "python -m training.train_ar"

# ==========================================================
# STEP 3 — GEOMETRIC RECONSTRUCTION EVAL
# ==========================================================

Run-Step `
    "[3/5] Evaluating reconstruction quality" `
    "python -m training.eval_patch"

# ==========================================================
# STEP 3.5 — MANIFOLD FIDELITY EVAL
# ==========================================================

Run-Step `
    "[4/5] Evaluating manifold fidelity" `
    "python -m training.eval_manifold"

# ==========================================================
# STEP 4 — FUNCTIONAL EVAL
# ==========================================================

Run-Step `
    "[5/5] Running functional activation evaluation" `
    "python -m training.eval_functional"

# ==========================================================
# Final Summary
# ==========================================================

Print-Header "TinyNLA Pipeline Completed"

Write-Host "Artifacts"
Write-Host "---------"
Write-Host ""

Write-Host "Dataset"
Write-Host "  activation_data/activation_buffer/buffer.pt"
Write-Host ""

Write-Host "Checkpoint"
Write-Host "  checkpoints/ar/best_model.pt"
Write-Host ""

Write-Host "Metrics"
Write-Host "  checkpoints/ar/metrics.json"
Write-Host ""

Write-Host "Interpolation"
Write-Host "  checkpoints/ar/interpolation.json"
Write-Host ""

# ==========================================================
# Optional W&B Notice
# ==========================================================

Write-Host "Tracking"
Write-Host "--------"

if ($env:WANDB_API_KEY) {

    Write-Host "W&B logging available"
}
else {

    Write-Host "W&B API key not detected"
}

Write-Host ""

# ==========================================================
# Recommended Next Experiments
# ==========================================================

Write-Host "Recommended Next Steps"
Write-Host "----------------------"
Write-Host ""

Write-Host "1. Run layer sweep"
Write-Host "     .\run_all.ps1 -Sweep"
Write-Host ""

Write-Host "2. Compare semantic recoverability by depth"
Write-Host ""

Write-Host "3. Analyze interpolation curves"
Write-Host ""

Write-Host "4. Compare manifold geometry across layers"
Write-Host ""

Write-Host "5. Upgrade reconstruction architecture"
Write-Host "     - deeper transformer decoder"
Write-Host "     - rotary positional embeddings"
Write-Host "     - latent bottleneck"
Write-Host "     - diffusion reconstruction"
Write-Host ""

# ==========================================================
# Done
# ==========================================================

Write-Host "[OK] Pipeline finished successfully."
Write-Host ""