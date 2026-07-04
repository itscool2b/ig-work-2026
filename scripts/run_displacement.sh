#!/usr/bin/env bash
# ============================================================================
# Month 5 action-displacement experiment, as executed on the RunPod RTX PRO
# 6000 Blackwell pod. GPU required.
#
# Prerequisite: a base per-step IG run with attribution sidecars for each
# task and scale (per_step_ig.py at m=64; see run_paper_experiments.sh for
# the 170M regeneration). The 1B base runs are generated below if their
# JSONLs are missing (the StackCube 1B base record is not committed; only
# its displacement output is). The 1B rows carry reference action norms
# below the signal threshold, so the 1B runs pass --no-signal-filter.
#
# Sweeps solver steps T in {1,2,3,5,10,20} and deletion fractions
# {0,1,5,10,20}% on vision and language, ranking from the T=5 attributions.
#
# Usage:
#   bash scripts/run_displacement.sh
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

LIMIT_170="${LIMIT_170:-50}"
LIMIT_1B="${LIMIT_1B:-60}"

base_run_1b() {
  local task="$1" episodes="$2" out="$3"
  if [ ! -f "$out" ]; then
    echo "=== 1B base run for ${task} (sidecars) -> ${out} ==="
    .venv/bin/python per_step_ig.py \
      --task "$task" \
      --model 1b \
      --episodes "$episodes" \
      --m 64 \
      --seed-base 42 \
      --out "$out" \
      --no-checkpoint
  fi
}

base_run_1b PickCube-v1  20 data/m5_metrics_PickCube-v1_1b_seed42.jsonl
base_run_1b StackCube-v1 5  data/m5_metrics_StackCube-v1_1b_seed42.jsonl

run_disp() {
  local task="$1" model="$2" metrics="$3" limit="$4"
  shift 4
  out="data/m5_metrics_displacement_${task}_${model}.jsonl"
  echo "=== displacement: ${task} ${model} -> ${out} ==="
  .venv/bin/python displacement.py \
    --metrics "$metrics" \
    --task "$task" \
    --model "$model" \
    --solver-steps "1,2,3,5,10,20" \
    --del-grid "0,1,5,10,20" \
    --modality both \
    --limit "$limit" \
    --out "$out" \
    --no-checkpoint "$@"
}

run_disp PickCube-v1  1b   data/m5_metrics_PickCube-v1_1b_seed42.jsonl  "$LIMIT_1B" --no-signal-filter
run_disp PickCube-v1  170m data/metrics_PickCube-v1_170m_seed42.jsonl   "$LIMIT_170"
run_disp StackCube-v1 1b   data/m5_metrics_StackCube-v1_1b_seed42.jsonl "$LIMIT_1B"  --no-signal-filter
run_disp StackCube-v1 170m data/metrics_StackCube-v1_170m_seed42.jsonl  "$LIMIT_170"

echo "Displacement runs complete. Regenerate the figure and CSV with:"
echo "  python3 make_month5_figs.py"
