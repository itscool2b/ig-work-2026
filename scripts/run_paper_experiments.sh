#!/usr/bin/env bash
# ============================================================================
# Paper-strengthening experiments. Run on a 5090/4090 RunPod pod.
#
# E1: 1B faithfulness on PickCube         (Pod B only)
# E2: m=128 on PegInsertion+PickSingleYCB (Pod A)
# E3: Frozen-target C1 sanity             (Pod A, needs sidecars from step 1)
# E4: Baseline sensitivity, vision only   (Pod A, needs sidecars from step 1)
# E5: m=128 faithfulness                  (Pod A, needs E2 outputs)
#
# Usage:
#   bash scripts/run_paper_experiments.sh           # run all stages (E1 expects the 1B pod)
#   STAGE=E1 bash scripts/run_paper_experiments.sh  # 1B faithfulness (Pod B)
#   STAGE=E2 bash scripts/run_paper_experiments.sh  # m=128 remaining tasks
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${STAGE:-all}"
MODEL_170="${MODEL_170:-170m}"

run_stage() { [[ "$STAGE" == "all" || "$STAGE" == "$1" ]]; }

# ---- Step 0: Sidecar regeneration (Pod A prerequisite for E3/E4) ----
if run_stage sidecars || run_stage all; then
  echo ""
  echo "================================================================"
  echo "Step 0: Regenerate m=64 sidecars (15 eps/task-seed)"
  echo "================================================================"
  echo ""
  for task in PickCube-v1 StackCube-v1 PegInsertionSide-v1 PickSingleYCB-v1; do
    for seed in 42 142; do
      out="data/metrics_${task}_${MODEL_170}_seed${seed}.jsonl"
      echo "=== ${task} seed=${seed} -> ${out} ==="
      .venv/bin/python per_step_ig.py \
        --task "$task" \
        --model "$MODEL_170" \
        --episodes 15 \
        --m 64 \
        --seed-base "$seed" \
        --out "$out" \
        --resume \
        --no-checkpoint
    done
  done
  echo "sidecar regen done."
fi

# ---- E1: 1B faithfulness on PickCube (Pod B) ----
if run_stage E1; then
  echo ""
  echo "================================================================"
  echo "E1: 1B faithfulness on PickCube (authors' weights)"
  echo "================================================================"
  echo ""
  JSONL_1B="data/metrics_PickCube-v1_1b_seed42.jsonl"
  if [ ! -f "$JSONL_1B" ]; then
    echo "1B PickCube JSONL not found, running 1B per-step IG first..."
    .venv/bin/python per_step_ig.py \
      --task PickCube-v1 \
      --model 1b \
      --episodes 20 \
      --m 64 \
      --seed-base 42 \
      --out "$JSONL_1B" \
      --no-checkpoint
  fi
  echo "Running 1B faithfulness..."
  .venv/bin/python faithfulness.py \
    --metrics "$JSONL_1B" \
    --task PickCube-v1 \
    --model 1b \
    --no-checkpoint
  echo "E1 done."
fi

# ---- E2: m=128 on PegInsertion + PickSingleYCB ----
if run_stage E2 || run_stage all; then
  echo ""
  echo "================================================================"
  echo "E2: m=128 per-step IG on remaining 2 tasks"
  echo "================================================================"
  echo ""
  for task in PegInsertionSide-v1 PickSingleYCB-v1; do
    for seed in 42 142; do
      out="data/metrics_${task}_${MODEL_170}_seed${seed}_m128.jsonl"
      echo "=== ${task} seed=${seed} m=128 -> ${out} ==="
      .venv/bin/python per_step_ig.py \
        --task "$task" \
        --model "$MODEL_170" \
        --episodes 8 \
        --m 128 \
        --seed-base "$seed" \
        --out "$out" \
        --resume \
        --no-checkpoint
    done
  done
  echo "E2 done."
fi

# ---- E3: Frozen-target C1 sanity check ----
if run_stage E3 || run_stage all; then
  echo ""
  echo "================================================================"
  echo "E3: Frozen-target C1 sanity (correct Adebayo-style test)"
  echo "================================================================"
  echo ""
  SANITY_LIMIT="${SANITY_LIMIT:-50}"
  shopt -s nullglob
  for jsonl in data/metrics_*_${MODEL_170}_seed*.jsonl; do
    #Also skip the l2/maxdev target-ablation step files (their sidecars were
    #produced under alternative targets and E3 builds a logpi context) and the
    #Month 6 _logpi matched-run files (near-duplicates of the base runs; sweeping
    #them in would double-count PickCube in the pooled sanity medians).
    case "$(basename "$jsonl")" in
      metrics_faithfulness_*|metrics_sanity_*|metrics_baseline_*|*_m128*) continue ;;
      *_l2.jsonl|*_maxdev.jsonl|*_logpi.jsonl) continue ;;
    esac
    base=$(basename "$jsonl")
    task=$(echo "$base" | sed -E 's/^metrics_(.+)_(170m|1b)_seed[0-9]+.*/\1/')
    echo "=== frozen-target C1: ${task} jsonl=${jsonl} ==="
    .venv/bin/python sanity.py --phase C1_frozen \
      --metrics "$jsonl" --task "$task" --model "$MODEL_170" --m 16 \
      --limit "$SANITY_LIMIT" --no-checkpoint
  done
  echo "E3 done."
fi

# ---- E4: Baseline sensitivity (vision only) ----
if run_stage E4 || run_stage all; then
  echo ""
  echo "================================================================"
  echo "E4: Baseline sensitivity sweep (black vs gray vs blur)"
  echo "================================================================"
  echo ""
  for task in PickCube-v1 StackCube-v1; do
    jsonl="data/metrics_${task}_${MODEL_170}_seed42.jsonl"
    if [ ! -f "$jsonl" ]; then
      echo "skip: no seed42 JSONL for $task"
      continue
    fi
    echo "=== baseline sensitivity: ${task} ==="
    .venv/bin/python baseline_sensitivity.py \
      --metrics "$jsonl" \
      --task "$task" \
      --model "$MODEL_170" \
      --m 64 \
      --limit 30 \
      --no-checkpoint
  done
  echo "E4 done."
fi

# ---- E5: m=128 faithfulness on E2's new JSONLs ----
if run_stage E5 || run_stage all; then
  echo ""
  echo "================================================================"
  echo "E5: Faithfulness on m=128 PegInsertion + PickSingleYCB"
  echo "================================================================"
  echo ""
  shopt -s nullglob
  for jsonl in data/metrics_*_${MODEL_170}_seed*_m128.jsonl; do
    case "$(basename "$jsonl")" in
      metrics_faithfulness_*|metrics_sanity_*|metrics_baseline_*) continue ;;
    esac
    base=$(basename "$jsonl")
    task=$(echo "$base" | sed -E 's/^metrics_(.+)_(170m|1b)_seed[0-9]+.*/\1/')
    echo "=== faithfulness (m=128): ${task} jsonl=${jsonl} ==="
    .venv/bin/python faithfulness.py \
      --metrics "$jsonl" \
      --task "$task" \
      --model "$MODEL_170" \
      --no-checkpoint
  done
  echo "E5 done."
fi

echo ""
echo "================================================================"
echo "All paper experiments complete."
echo "================================================================"
