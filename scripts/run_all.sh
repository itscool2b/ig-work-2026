#!/usr/bin/env bash
# One-command reproduction:
#   1. Full pass (per_step_ig, 4 tasks x 2 seeds x 50 eps @ m=64)
#   2. Faithfulness post-processing (B1 + B2)
#   3. Sanity C1 + C2
#   4. Overlay generation (qualitative artifacts)
#   5. Notebook execution (local; requires jupyter)
#
# The first 4 steps are pod-heavy and designed for the RTX 4090 cloud setup.
# Step 5 runs locally and doesn't need a GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${STAGE:-all}"

run_stage() { [[ "$STAGE" == "all" || "$STAGE" == "$1" ]]; }

if run_stage full_pass; then
  echo "note: run_full_pass appends episodes 15-49 to the committed data/*.jsonl records (they ship with the first 15). To avoid growing the released files, run into a fresh --out or an emptied data/ per the README Reproducing section."
  bash scripts/run_full_pass.sh
fi
if run_stage faithfulness; then bash scripts/run_faithfulness.sh; fi
if run_stage sanity;      then bash scripts/run_sanity.sh;       fi
if run_stage overlays;    then bash scripts/run_overlays.sh;     fi

if run_stage report; then
  echo "=== notebook (local) ==="
  #Note: executing the notebook overwrites out/figures/fig_c_auc_curves.png,
  #the preserved main-pass render that cannot be regenerated (see README
  #Reproducing), so it is opt-in via RUN_NOTEBOOK=1 and the default one-command
  #run never clobbers it.
  if [[ "${RUN_NOTEBOOK:-0}" == "1" && -x .venv/bin/jupyter ]]; then
    .venv/bin/jupyter nbconvert --to notebook --execute notebooks/metrics_report.ipynb \
      --output metrics_report_executed.ipynb
  elif [[ "${RUN_NOTEBOOK:-0}" != "1" ]]; then
    echo "skipping notebook execute: set RUN_NOTEBOOK=1 to run it (it overwrites the preserved fig_c_auc_curves.png)"
  else
    echo "skipping notebook execute: .venv/bin/jupyter not installed"
  fi
fi

echo "run_all done."
