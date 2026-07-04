#!/usr/bin/env bash
# Run sanity passes (C1 model re-init, C2 input randomization) over every
# metrics_*_seed*.jsonl in data/. One output JSONL per input per phase.
# Prerequisite: attribution sidecars under data/per_step_attr/ from a base
# per-step IG run. The sidecars are not redistributed, and regeneration against
# the committed JSONLs is skipped by --resume, so regenerate in a clean checkout
# with an emptied data/ (see the README's Reproducing notes), e.g. via
# STAGE=sidecars scripts/run_paper_experiments.sh, before running this script.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-170m}"
M="${M:-16}"
SHUFFLE_SEED="${SHUFFLE_SEED:-777}"
#The sanity stage needs only a subset of the pass. Our full pass has 50 eps/task,
#so bound via --limit to keep sanity wall in check (~7h vs ~35h unbounded).
#--limit 50 = first 50 step rows per JSONL = ~12 eps on 4-call tasks
#(PickCube/StackCube/YCB), ~7 eps on PegInsertion (7 calls/ep).
SANITY_LIMIT="${SANITY_LIMIT:-50}"

#C2: sanity.py now does runtime post-T5 embedding shuffle if no pre-shuffled .pt
#exists, so we don't need T5-XXL on the pod. Skipping the encode step entirely.
echo "=== C2 prep: runtime shuffle in sanity.py (no T5 needed) ==="
shopt -s nullglob

for jsonl in data/metrics_*_seed*.jsonl; do
  #Skip derived outputs (faithfulness/sanity/baseline-sensitivity), the
  #displacement measurement records, the l2/maxdev target-ablation step files
  #(handled by run_target_ablation.sh), and the Month 6 _logpi matched-run
  #files (near-duplicates of the base runs; new sanity records over them would
  #double-count PickCube in the pooled medians).
  case "$(basename "$jsonl")" in
    metrics_faithfulness_*|metrics_sanity_*|metrics_baseline_sensitivity_*) continue ;;
    metrics_displacement_*) continue ;;
    *_l2.jsonl|*_maxdev.jsonl|*_logpi.jsonl) continue ;;
  esac
  base=$(basename "$jsonl")
  task=$(echo "$base" | sed -E 's/^metrics_(.+)_(170m|1b)_seed[0-9]+.*/\1/')
  if [[ ! "$base" =~ _${MODEL}_ ]]; then
    continue
  fi
  echo "=== sanity C1: ${task} jsonl=${jsonl} ==="
  .venv/bin/python sanity.py --phase C1 \
    --metrics "$jsonl" --task "$task" --model "$MODEL" --m "$M" \
    --limit "$SANITY_LIMIT" --no-checkpoint

  echo "=== sanity C2: ${task} jsonl=${jsonl} ==="
  .venv/bin/python sanity.py --phase C2 \
    --metrics "$jsonl" --task "$task" --model "$MODEL" --m "$M" \
    --shuffle-seed "$SHUFFLE_SEED" --limit "$SANITY_LIMIT" --no-checkpoint
done

echo "sanity passes done."
