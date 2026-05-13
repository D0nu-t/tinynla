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
#   3. Evaluate reconstruction quality
#   4. Run functional activation patching evaluation
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
    Write-Host $stepName
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
    "[1/4] Building activation buffer" `
    "python -m training.build_buffer"

# ==========================================================
# STEP 2 — TRAIN AR
# ==========================================================

Run-Step `
    "[2/4] Training activation reconstructor" `
    "python -m training.train_ar"

# ==========================================================
# STEP 3 — RECONSTRUCTION EVAL
# ==========================================================

Run-Step `
    "[3/4] Evaluating reconstruction quality" `
    "python -m training.eval_patch"

# ==========================================================
# STEP 4 — FUNCTIONAL EVAL
# ==========================================================

Run-Step `
    "[4/4] Running functional activation evaluation" `
    "python -m training.eval_functional"

# ==========================================================
# Final Summary
# ==========================================================

Print-Header "TinyNLA Pipeline Completed"

Write-Host "Artifacts"
Write-Host "---------"
Write-Host ""

Write-Host "Dataset"
Write-Host "  datasets/activation_buffer/buffer.pt"
Write-Host ""

Write-Host "Checkpoint"
Write-Host "  checkpoints/ar/model.pt"
Write-Host ""

Write-Host "Metrics"
Write-Host "  checkpoints/ar/metrics.json"
Write-Host ""

# ==========================================================
# Optional W&B Notice
# ==========================================================

Write-Host "Tracking"
Write-Host "--------"

if ($env:WANDB_API_KEY) {

    Write-Host "W&B logging enabled"
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

Write-Host "3. Add interpolation experiments"
Write-Host ""

Write-Host "4. Add perplexity-shift evaluation"
Write-Host ""

Write-Host "5. Upgrade AR architecture"
Write-Host "     - residual MLP"
Write-Host "     - transformer decoder"
Write-Host "     - diffusion latent model"
Write-Host ""

# ==========================================================
# Done
# ==========================================================

Write-Host "[OK] Pipeline finished successfully."
Write-Host ""