# run_all.ps1
#
# TinyNLA MVP Sequential Runner
#
# Runs:
#   1. Activation buffer creation
#   2. Activation reconstructor training
#   3. Reconstruction evaluation
#
# Usage:
#
#   powershell -ExecutionPolicy Bypass -File run_all.ps1
#
# OR inside PowerShell:
#
#   .\run_all.ps1
#

Write-Host ""
Write-Host "=================================================="
Write-Host "TinyNLA MVP Pipeline"
Write-Host "=================================================="
Write-Host ""

###############################################################################
# STEP 1 — BUILD ACTIVATION BUFFER
###############################################################################

Write-Host "[1/4] Building activation buffer..."
Write-Host ""

python -m training.build_buffer

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] build_buffer failed."
    exit 1
}

Write-Host ""
Write-Host "[OK] Activation buffer completed."
Write-Host ""

###############################################################################
# STEP 2 — TRAIN ACTIVATION RECONSTRUCTOR
###############################################################################

Write-Host "[2/4] Training activation reconstructor..."
Write-Host ""

python -m training.train_ar

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] train_ar failed."
    exit 1
}

Write-Host ""
Write-Host "[OK] Training completed."
Write-Host ""

###############################################################################
# STEP 3 — EVALUATE MODEL
###############################################################################

Write-Host "[3/4] Evaluating reconstruction quality..."
Write-Host ""

python -m training.eval_patch

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] eval_patch failed."
    exit 1
}

###############################################################################
# STEP 4 — FUNCTIONAL EVALUATION
###############################################################################

Write-Host "[4/4] Running activation patching evaluation..."
Write-Host ""

python -m training.eval_functional

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] eval_functional failed."
    exit 1
}

Write-Host ""
Write-Host "[OK] Functional evaluation completed."
Write-Host ""

Write-Host ""
Write-Host "=================================================="
Write-Host "TinyNLA MVP Pipeline Completed"
Write-Host "=================================================="
Write-Host ""

Write-Host "Artifacts:"
Write-Host ""

Write-Host "  Dataset:"
Write-Host "    datasets/activation_buffer/buffer.pt"
Write-Host ""

Write-Host "  Trained Model:"
Write-Host "    checkpoints/ar/model.pt"
Write-Host ""

Write-Host "Next Recommended Step:"
Write-Host "  Implement activation patching evaluation."
Write-Host ""