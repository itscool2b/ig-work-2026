#!/usr/bin/env bash
# Run the full pass: 170M x 4 released tasks x 2 seeds x 50 eps @ m=64.
# TurnFaucet-v1 was attempted in the original pass and dropped (no released
# data). Add it back to TASKS to repeat that attempt.
# Idempotent via --resume; re-running picks up where a crash left off.
# On 24 GB GPUs, --no-checkpoint is the default (fits and runs ~1.4x faster).
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-170m}"
EPISODES="${EPISODES:-50}"
M="${M:-64}"
TASKS="${TASKS:-PickCube-v1 StackCube-v1 PegInsertionSide-v1 PickSingleYCB-v1}"
SEEDS="${SEEDS:-42 142}"
VIDEO_ROOT="${VIDEO_ROOT:-output/videos/${MODEL}}"

for task in $TASKS; do
  for seed in $SEEDS; do
    out="data/metrics_${task}_${MODEL}_seed${seed}.jsonl"
    videos="${VIDEO_ROOT}/${task}-s${seed}"
    mkdir -p "$(dirname "$videos")"
    echo "=== ${task} seed=${seed} -> ${out} ==="
    .venv/bin/python per_step_ig.py \
      --task "$task" \
      --model "$MODEL" \
      --episodes "$EPISODES" \
      --m "$M" \
      --seed-base "$seed" \
      --out "$out" \
      --resume \
      --no-checkpoint \
      --video-dir "$videos"
  done
done

echo "full pass done."
