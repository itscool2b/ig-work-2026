#!/usr/bin/env bash
# Run faithfulness post-processing (B1 Δlog p + B2 Insertion/Deletion AUC)
# over every metrics_*_seed*.jsonl in data/. One output JSONL per input.
# Prerequisite: attribution sidecars under data/per_step_attr/ from a base
# per-step IG run. The sidecars are not redistributed, and regeneration against
# the committed JSONLs is skipped by --resume, so regenerate in a clean checkout
# with an emptied data/ (see the README's Reproducing notes), e.g. via
# STAGE=sidecars scripts/run_paper_experiments.sh, before running this script.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-170m}"

#RDT-1B is the scale check; the O2 thresholds and Table 2
#are checked on the 170M primary. 1B faithfulness is ~9.5 h pod time on our setup
#(1B forwards are ~5x slower than 170M and a typical scale-check has ~440 sidecars).
#We skip by default; set FORCE_1B_FAITH=1 to override and run anyway.
if [ "$MODEL" = "1b" ] && [ -z "${FORCE_1B_FAITH:-}" ]; then
  echo "skipping 1B faithfulness (scale check only). Set FORCE_1B_FAITH=1 to override."
  exit 0
fi

shopt -s nullglob
for jsonl in data/metrics_*_seed*.jsonl; do
  #Skip the faithfulness/sanity outputs themselves, the baseline-sensitivity
  #outputs, the displacement measurement records (not base runs), the l2/maxdev
  #target-ablation step files (those are processed by run_target_ablation.sh
  #with the matching --target flag), the Month 6 _logpi matched-run files
  #(their faithfulness records are committed; re-walking them here would pool
  #PickCube twice downstream), and the m=128 step files (their faithfulness
  #records are also committed, and a re-walk would write a second timestamped
  #copy that the downstream m128 globs would double-pool).
  case "$(basename "$jsonl")" in
    metrics_faithfulness_*|metrics_sanity_*|metrics_baseline_sensitivity_*) continue ;;
    metrics_displacement_*) continue ;;
    *_l2.jsonl|*_maxdev.jsonl|*_logpi.jsonl) continue ;;
    *_m128.jsonl) continue ;;
  esac
  base=$(basename "$jsonl")
  #Extract task from filename pattern: metrics_<task>_<model>_seed<N>.jsonl
  task=$(echo "$base" | sed -E 's/^metrics_(.+)_(170m|1b)_seed[0-9]+.*/\1/')
  if [[ ! "$base" =~ _${MODEL}_ ]]; then
    echo "skip (wrong model): $jsonl"
    continue
  fi
  echo "=== faithfulness: task=${task} jsonl=${jsonl} ==="
  .venv/bin/python faithfulness.py \
    --metrics "$jsonl" \
    --task "$task" \
    --model "$MODEL" \
    --no-checkpoint
done

echo "faithfulness pass done."
