#!/usr/bin/env bash
# ============================================================================
# Target-ablation and cascade-C1 stages, as executed on the Month 4 pod.
# GPU required; not runnable on a laptop.
#
# T1: per-step IG with alternative targets (l2, maxdev) on PickCube
# T2: faithfulness post-processing per alternative target
# T3: cascade C1 sanity (full-backbone re-init, frozen target) on the
#     regeneration base runs from run_paper_experiments.sh step 0
#
# Usage:
#   bash scripts/run_target_ablation.sh            # run all stages
#   STAGE=T3 bash scripts/run_target_ablation.sh   # cascade C1 only
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${STAGE:-all}"
MODEL_170="${MODEL_170:-170m}"

run_stage() { [[ "$STAGE" == "all" || "$STAGE" == "$1" ]]; }

# ---- T1: alternative-target IG runs (PickCube, 2 seeds, 15 eps, m=64) ----
if run_stage T1; then
  for tgt in l2 maxdev; do
    for seed in 42 142; do
      out="data/metrics_PickCube-v1_${MODEL_170}_seed${seed}_${tgt}.jsonl"
      echo "=== PickCube target=${tgt} seed=${seed} -> ${out} ==="
      .venv/bin/python per_step_ig.py \
        --task PickCube-v1 \
        --model "$MODEL_170" \
        --episodes 15 \
        --m 64 \
        --seed-base "$seed" \
        --target "$tgt" \
        --out "$out" \
        --resume \
        --no-checkpoint
    done
  done
  echo "T1 done."
fi

# ---- T2: faithfulness per alternative target ----
if run_stage T2; then
  shopt -s nullglob
  for tgt in l2 maxdev; do
    for jsonl in data/metrics_PickCube-v1_${MODEL_170}_seed*_${tgt}.jsonl; do
      echo "=== faithfulness target=${tgt} jsonl=${jsonl} ==="
      .venv/bin/python faithfulness.py \
        --metrics "$jsonl" \
        --task PickCube-v1 \
        --model "$MODEL_170" \
        --target "$tgt" \
        --no-checkpoint
    done
  done
  echo "T2 done."
fi

# ---- T3: cascade C1 (full backbone re-init, frozen target) ----
if run_stage T3; then
  SANITY_LIMIT="${SANITY_LIMIT:-50}"
  shopt -s nullglob
  for jsonl in data/metrics_*_${MODEL_170}_seed*.jsonl; do
    #_logpi excluded like _l2/_maxdev: the Month 6 matched-run files are
    #near-duplicates of the base runs and would double-count PickCube.
    case "$(basename "$jsonl")" in
      metrics_faithfulness_*|metrics_sanity_*|metrics_baseline_*|*_m128*|*_l2*|*_maxdev*|*_logpi*) continue ;;
    esac
    base=$(basename "$jsonl")
    task=$(echo "$base" | sed -E 's/^metrics_(.+)_(170m|1b)_seed[0-9]+.*/\1/')
    echo "=== cascade C1: ${task} jsonl=${jsonl} ==="
    .venv/bin/python sanity.py --phase C1_cascade \
      --metrics "$jsonl" --task "$task" --model "$MODEL_170" --m 16 \
      --limit "$SANITY_LIMIT" --no-checkpoint
  done
  echo "T3 done."
fi

echo "Target-ablation stages complete."
