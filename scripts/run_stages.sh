#!/usr/bin/env bash
# ==========================================================
# TinyNLA pipeline runner with stage-level retries (Git Bash)
# ==========================================================
#   scripts/run_stages.sh configs/qwen05b.yaml [first_stage] [log]
#
# Each stage runs as a fresh Python process. If it fails (e.g. another program
# exhausted the shared GPU and even CUDA clean-up failed), it is retried up to
# MAX_TRIES times, WAIT seconds apart. Stages are cheap to repeat:
#   datagen    reuses activations.pt and the explanation cache
#   rl         a retry resumes from the last saved RL checkpoint (rl.init_from)
# The log uses the section headers the training dashboard parses.
# ==========================================================
set -u
CONFIG="${1:-configs/gpt2_small.yaml}"
FROM="${2:-datagen}"
LOG="${3:-logs/pipeline_$(basename "$CONFIG" .yaml).log}"
MAX_TRIES="${MAX_TRIES:-3}"
WAIT="${WAIT:-120}"
PY=".venv/Scripts/python.exe"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

STAGES=(datagen ar_sft av_sft rl eval calibrate anticipation unusual)
declare -A MOD=([datagen]=training.datagen [ar_sft]=training.train_ar_sft [av_sft]=training.train_av_sft
                [rl]=training.train_rl [eval]=training.eval_nla [calibrate]=training.calibrate_nla
                [anticipation]=training.anticipation_nla [unusual]=training.unusual_nla)

rl_dir=$("$PY" -c "from nla.config import load_config; print(load_config('$CONFIG')['rl']['save_dir'])")

running=0
for stage in "${STAGES[@]}"; do
  [ "$stage" = "$FROM" ] && running=1
  [ $running -eq 0 ] && continue
  for try in $(seq 1 "$MAX_TRIES"); do
    extra=()
    if [ "$stage" = "rl" ] && [ "$try" -gt 1 ] && [ -f "$rl_dir/av/av_config.json" ]; then
      extra=(--set "rl.init_from=$rl_dir")          # resume, don't start RL over
    fi
    printf '==================================================\n[%s] python -m %s --config %s\n==================================================\n' \
      "$stage" "${MOD[$stage]}" "$CONFIG" >> "$LOG"
    if "$PY" -m "${MOD[$stage]}" --config "$CONFIG" "${extra[@]}" >> "$LOG" 2>&1; then
      break
    fi
    if [ "$try" -eq "$MAX_TRIES" ]; then
      echo "[ERROR] stage $stage failed after $MAX_TRIES tries" >> "$LOG"
      exit 1
    fi
    echo "[retry] stage $stage failed (try $try of $MAX_TRIES); retrying in ${WAIT}s" >> "$LOG"
    sleep "$WAIT"
  done
done
echo "[OK] Pipeline complete." >> "$LOG"
