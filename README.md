# The Readout, Not the Denoiser

Per-step, per-modality Integrated Gradients for diffusion-policy vision-language-action models. This repository holds the code, the released metrics records, and the analysis notebooks behind the paper.

The paper source and PDF live in `paper/` (`paper/paper.tex`, `paper/paper.pdf`). The method attributes every control decision of the Robotics Diffusion Transformer (RDT) to its vision, language, and state inputs across ManiSkill3 manipulation episodes, and the paper reports two structural findings about evaluating attribution on diffusion policies.

## Layout

- `paper/` contains the paper source, figures, and bibliography.
- `data/` contains the released per-step metrics JSONL records. `data/README.md` maps every file to the run that produced it, including which runs were regenerated after the original cloud pods were decommissioned.
- `notebooks/` contains the executed analysis notebooks. `metrics_report.ipynb` builds the main-pass tables and figures. `month4_report.ipynb` is the Month 4 analysis record and opens with a dated note on two interpretations the paper later superseded.
- `out/figures*/` contains the generated figures and the CSV tables behind them, including the bootstrap interval tables.
- `scripts/` contains the single-command stages. `run_full_pass.sh` runs per-step IG, `run_faithfulness.sh` and `run_sanity.sh` run the metric stages, `run_target_ablation.sh` and `run_displacement.sh` record the exact Month 4 and Month 5 commands.
- `docs/` documents each pipeline component. `docs/per_step_ig.md` and `docs/ig_rdt.md` cover the RDT attribution path used by the paper. The other files document the Month 2 single-model IG studies.
- Top-level Python files are the pipeline itself. `per_step_ig.py` and `per_step_attribution.py` run episode attribution, `faithfulness.py`, `sanity.py`, `baseline_sensitivity.py`, and `displacement.py` run the evaluations, and `make_paper_figs.py`, `make_month4_figs.py`, and `make_month5_figs.py` regenerate the paper figures. `ig_resnet.py`, `ig_vit.py`, `ig_tinyllama.py`, and `ig_llava.py` are the Month 2 single-model studies that preceded the RDT work (`image.jpg` is their third-party demo input photo, which is not covered by the repository license). `audit.py` independently recomputes the released medians and AUCs from `data/`, and `analyze_month4.py` and `patch_nb_alttarget.py` are retained one-off helpers from the Month 4 analysis, kept for the record.

## Setup

The analysis layer needs only Python 3.12 or newer and the packages in `requirements.txt` (the byte-identical regeneration below was last verified on Python 3.14). With those installed, the notebooks, `audit.py`, `bootstrap_ci.py --selftest`, and the `make_*_figs.py` generators reproduce every committed CSV table, and every committed figure except three preserved artifacts, from the committed JSONL records plus a few preserved decommissioned-era constants embedded in the generators, which `data/README.md` documents. The CSV tables regenerate byte-identically. The figure PNGs re-render with identical data but can differ at the byte level across matplotlib versions, and the committed renders used matplotlib 3.11. The exceptions are `fig_c_auc_curves.png` (a preserved render of the decommissioned main-pass curves, which `make_paper_figs.py` only re-tiles), `out/figures/fig_b_overlays_grid.png` (the preserved Month 3 overlay grid, whose source overlays are not redistributed), and `out/figures/fig_b_overlays_grid_1b.png` (the RDT-1B seed 242 overlay grid, identical to the paper's Figure 3 file `paper/figures/fig_b_overlays_grid.png`, whose source overlays are likewise not redistributed). The two `fig_b_overlays_grid` files share a stem but differ, with the Month 3 grid under `out/figures/` and the 1B grid as the paper figure.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The GPU stages (`per_step_ig.py` and everything downstream of it) additionally require:

- The RDT authors' source repository cloned at `~/rdt-repo`. The pipeline imports `models.rdt_runner`, the SigLIP encoder, `configs/state_vec.py`, and `scripts/maniskill_model.py` from it, and the 1B path reads `~/rdt-repo/configs/base.yaml`. Clone `thu-ml/RoboticsDiffusionTransformer` from GitHub to that path. The recorded runs used the repository as of spring 2026. Small local adjustments made during the recorded runs are documented in the authors' research notes, which are kept privately.
- ManiSkill3 with a working Vulkan/SAPIEN rendering stack (installed via `mani_skill` in `requirements.txt`).
- A CUDA GPU with at least 24 GB to match the recorded RDT-170M runs, while development is workable on 12 GB with gradient checkpointing. The recorded runs used PyTorch 2.6 to 2.8 across the machines listed in the paper's Hardware paragraph.

The shell scripts invoke `.venv/bin/python`, so they expect the virtualenv above at the repo root.

## Reproducing

Episode seeds are fixed at 42 and 142, with 242 added for the third RDT-1B evaluation seed, and the diffusion noise is seeded per forward pass. The scripts in `scripts/` record each stage with the commands as executed on the original GPUs, except the Month 6 strengthening runs (the seed 142 and 242 RDT-1B faithfulness runs, the matched-population logpi ablation runs, and the seed 142 displacement runs), which used the same entry points and flags at the seeds and scopes recorded in `data/README.md`.

Two things to know before re-running anything:

- The committed records partially defeat naive re-runs. `per_step_ig.py --resume` skips every episode whose `episode_end` row already exists in the output file, and `run_displacement.sh` skips its two 1B base runs when their outputs exist, so the regeneration stage of `run_paper_experiments.sh` and the PickCube 1B base run are no-ops against the committed `data/`, while the StackCube 1B base output is not committed and regenerates when missing, as `data/README.md` records. `run_full_pass.sh` defaults to 50 episodes per task and seed while the committed records hold 15, so against the committed `data/` it would append episodes 15 to 49 rather than skip. The displacement measurements themselves, however, open their outputs in append mode with no skip logic, so re-running `run_displacement.sh` appends duplicate rows to the committed displacement files. To regenerate raw records from scratch, run the stages in a clean checkout with an emptied `data/` directory (keep `data/README.md`), or point `--out` at fresh paths.
- Re-executing `notebooks/metrics_report.ipynb` end to end overwrites `out/figures/fig_c_auc_curves.png`, which is the only preserved render of the decommissioned main-pass faithfulness curves and cannot be regenerated, with a version computed from the committed subset records only. It also overwrites `out/figures/fig_a_completeness.png`, which is harmless because `make_paper_figs.py` rebuilds that figure from committed records. A re-executed notebook would also pool the later-added ablation, m=128, and RDT-1B records into its table populations, so its saved outputs are the preserved record rather than a re-run target. Do not run that notebook with Run-All unless you intend this.

Model checkpoints download from the `robotics-diffusion-transformer` Hugging Face repositories and are not redistributed here. The RDT-1B results additionally require the RDT authors' ManiSkill fine-tuned checkpoint, published in the `robotics-diffusion-transformer/maniskill-model` Hugging Face repository as `rdt/mp_rank_00_model_states.pt`. Place it at `checkpoints/rdt_maniskill_authors/mp_rank_00_model_states.pt` before running any 1B stage. If it is absent the loader falls back to the Hugging Face base weights, plus the project LoRA only when a local `checkpoints/rdt_maniskill_lora/final.pt` exists, and neither fallback reproduces the paper's 1B numbers.

The precomputed instruction embeddings under `data/lang_embeds/` are not redistributed. Either generate them with `encode_task_lang.py` (requires hosting T5-XXL) or fetch the authors' precomputed `text_embed_*.pt` files from the `maniskill-model` Hugging Face repository, noting that the latter covers five tasks that do not include PickSingleYCB-v1, which is why the Month 4 follow-up runs reused the PickCube embedding for that task as the paper's Experimental Setup section records. The authors' files are raw embedding tensors, and converting them to the padded dictionary format the pipeline loads and producing the zeroed `baseline_bos_eos.pt` were manual steps in the recorded runs with no committed converter, so `encode_task_lang.py` is the only path runnable from this repository alone. The committed Month 4 follow-up records used those precomputed embeddings with a zeroed language baseline, while the Month 5 displacement and reproduction records used real T5-XXL embeddings hosted on the Month 5 machine, so which path reproduces which record is set by the deviations recorded in the paper.

The reproducibility statement in the paper records which original records were decommissioned and which committed records corroborate them.

## Citing

See `CITATION.cff`, or use:

```bibtex
@article{bajpai2026readout,
  title  = {The Readout, Not the Denoiser: Per-Step Integrated Gradients
            for Diffusion-Policy Vision-Language-Action Models},
  author = {Bajpai, Arjun and Jha, Sumit K. and Pramanik, Vishal and Rodriguez, Brian},
  year   = {2026},
}
```

## License

MIT. See `LICENSE`. The license covers the code and records in this repository. `image.jpg` is a third-party demo photograph retained for the Month 2 study record and is not relicensed by this repository.
