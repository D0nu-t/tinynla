# ==========================================================
# TinyNLA v4 pipeline runner
# ==========================================================
#   .\run_nla.ps1                                  # full GPT-2 pipeline
#   .\run_nla.ps1 -Config configs\qwen05b.yaml     # Qwen 0.5B (LoRA)
#   .\run_nla.ps1 -From av_sft                     # resume from a stage
#   .\run_nla.ps1 -Gui                             # open the thought reader afterwards
#
# Stages: datagen -> ar_sft -> av_sft -> rl -> eval
# Stops at the first failing stage.
# ==========================================================

param(
    [string]$Config = "configs\gpt2_small.yaml",
    [ValidateSet("datagen", "ar_sft", "av_sft", "rl", "eval")]
    [string]$From = "datagen",
    [switch]$Gui
)

$python = ".venv\Scripts\python.exe"
$stages = [ordered]@{
    "datagen" = "training.datagen"
    "ar_sft"  = "training.train_ar_sft"
    "av_sft"  = "training.train_av_sft"
    "rl"      = "training.train_rl"
    "eval"    = "training.eval_nla"
}

$running = $false
foreach ($name in $stages.Keys) {
    if ($name -eq $From) { $running = $true }
    if (-not $running) { continue }

    Write-Host ""
    Write-Host "=================================================="
    Write-Host "[$name] python -m $($stages[$name]) --config $Config"
    Write-Host "=================================================="
    & $python -m $stages[$name] --config $Config
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] stage $name failed" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "[OK] Pipeline complete. Eval report: <rl.save_dir>\av\nla_eval.json" -ForegroundColor Green

if ($Gui) {
    & $python -m nla.gui --config $Config
}
