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

if run_stage full_pass;   then bash scripts/run_full_pass.sh;    fi
if run_stage faithfulness; then bash scripts/run_faithfulness.sh; fi
if run_stage sanity;      then bash scripts/run_sanity.sh;       fi
if run_stage overlays;    then bash scripts/run_overlays.sh;     fi

if run_stage report; then
  echo "=== notebook (local) ==="
  #Note: executing the notebook overwrites out/figures/fig_c_auc_curves.png,
  #the preserved main-pass render (see README Reproducing).
  if [[ -x .venv/bin/jupyter ]]; then
    .venv/bin/jupyter nbconvert --to notebook --execute notebooks/metrics_report.ipynb \
      --output metrics_report_executed.ipynb
  else
    echo "skipping notebook execute: .venv/bin/jupyter not installed"
  fi
fi

echo "run_all done."
