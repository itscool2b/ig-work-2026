#!/usr/bin/env bash
# Walk every metrics_*_seed*.jsonl and generate the qualitative artifacts.
# Writes out/{overlays,tokens,episodes,figures}/<task>/... PNGs.
# Prerequisite: attribution sidecars under data/per_step_attr/ from a base
# per-step IG run. The sidecars are not redistributed, and regeneration against
# the committed JSONLs is skipped by --resume, so regenerate in a clean checkout
# with an emptied data/ (see the README's Reproducing notes), e.g. via
# STAGE=sidecars scripts/run_paper_experiments.sh, before running this script.
set -euo pipefail
cd "$(dirname "$0")/.."

shopt -s nullglob
for jsonl in data/metrics_*_seed*.jsonl; do
  #Skip derived outputs, the displacement measurement records (their filenames
  #do not parse as base runs), the l2/maxdev target-ablation step files (their
  #sidecars carry alternative-target attributions), and the Month 6 _logpi
  #matched-run files (near-duplicate overlays of the base runs).
  case "$(basename "$jsonl")" in
    metrics_faithfulness_*|metrics_sanity_*|metrics_baseline_sensitivity_*) continue ;;
    metrics_displacement_*) continue ;;
    *_l2.jsonl|*_maxdev.jsonl|*_logpi.jsonl) continue ;;
  esac
  base=$(basename "$jsonl")
  task=$(echo "$base" | sed -E 's/^metrics_(.+)_(170m|1b)_seed[0-9]+.*/\1/')
  echo "=== overlays: task=${task} jsonl=${jsonl} ==="
  .venv/bin/python generate_overlays.py \
    --metrics "$jsonl" \
    --task "$task" \
    --out-root out \
    --three-panel
done

echo "overlays done."
