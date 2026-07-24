# AGENTS.md

## Cursor Cloud specific instructions

This is a research/analysis repository (per-step Integrated Gradients for the RDT
diffusion-policy VLA model), not a web app or long-running service. There is
nothing to "serve"; work happens by running Python scripts that reproduce the
paper's metrics and figures from the committed JSONL records under `data/`.

### Environment
- The startup update script creates a virtualenv at `.venv` (repo root) and
  installs the CPU analysis layer (`numpy pandas matplotlib scipy Pillow jupyter`).
  The shell scripts in `scripts/` hard-code `.venv/bin/python`, so keep the venv
  at the repo root.
- Two layers exist (see `README.md` "Setup"):
  - CPU analysis layer — runnable here, no GPU needed.
  - GPU pipeline (`per_step_ig.py`, `faithfulness.py`, `sanity.py`,
    `displacement.py`, the `ig_*.py` demos, `verify_models.py`, and the
    `scripts/run_*.sh` stages). These need a CUDA GPU plus a clone of the RDT
    authors' repo at `~/rdt-repo`, ManiSkill3/SAPIEN, and Hugging Face
    checkpoints/language embeddings that are NOT redistributed. None of this is
    available in the cloud VM, so treat the GPU stages as not runnable here.

### Running / verifying the analysis layer (all CPU-only, seconds each)
- `.venv/bin/python bootstrap_ci.py --selftest` — closest thing to a test;
  confirms the un-resampled points match committed medians. Prints `SELFTEST PASS`.
- `.venv/bin/python audit.py` — independently recomputes released medians/AUCs
  from `data/`; the 1B AUC recompute should show `maxdiff=0.000000`.
- `.venv/bin/python analyze_month4.py` — Month 4 aggregate numbers.
- `.venv/bin/python bootstrap_ci.py --table all` — all CI tables (default
  `--B 10000`; pass a smaller `--B` for a fast smoke run).
- `.venv/bin/python make_paper_figs.py` / `make_month4_figs.py` /
  `make_month5_figs.py` — regenerate figures and CSV tables.

### Lint / tests
There is no lint config and no automated test suite. `bootstrap_ci.py --selftest`
and `audit.py` are the reproduction checks that stand in for tests.

### Gotchas
- Do NOT run `notebooks/metrics_report.ipynb` with Run-All: it overwrites the
  preserved `out/figures/fig_c_auc_curves.png`, which cannot be regenerated
  (see `README.md` "Reproducing"). `notebooks/month4_report.ipynb` is read-only
  over `data/` and safe to execute.
- The `make_*_figs.py` generators overwrite committed PNGs under `out/` and
  `paper/figures/`. The CSV tables regenerate byte-identically, but the PNGs
  differ at the byte level across matplotlib versions (committed renders used
  matplotlib 3.11). If you only ran them to verify, restore with
  `git checkout -- out/ paper/` so you don't commit spurious binary diffs.
- `data/*` is gitignored except `data/README.md` and `data/*.jsonl`; the
  `scripts/run_*.sh` stages append to those records, so re-running GPU stages
  against the committed `data/` mutates the released files (see `README.md`).
