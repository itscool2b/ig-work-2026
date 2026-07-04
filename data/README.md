# Metrics records: filename to run mapping

Every file is JSONL, one record per line. Step records carry per-modality
completeness error, IG sums, reference action norm, and wall time, and the step
files also interleave one `episode_end` record per episode carrying the
episode's success flag. Those rows are the source of the success counts quoted
below, and loaders should dispatch on the `event` field. Faithfulness
and sanity records are keyed to the same policy calls. Episode `e` uses
environment seed `42+e` or `142+e`, plus `242+e` for the third RDT-1B evaluation
seed. The diffusion noise is re-seeded per forward pass. The signal-bearing filter (reference action norm >= 15) is applied when
computing the published table values, not stored in the files.

## What these files are (and are not)

The committed records are **not a byte-level reproduction of every paper table.**
The original main evaluation pass (4 tasks x 2 seeds x 50 episodes at m=64), the
original m=128 PickCube and StackCube subset, the standard sanity runs (C1
last-layer and C2 input randomization), the main-pass faithfulness records, the
initial RDT-1B scale check, the ~3,881 overlay PNGs, and the per-step attribution
sidecar tensors were produced on cloud GPUs that have since been decommissioned and are
not redistributed here. The headline completeness, faithfulness, and
standard-sanity tables (Tables 1 to 3) were computed from those main-pass records and
are preserved in the analysis notebooks (saved outputs)
rather than regenerable from the files below. Note in particular that the
notebook's faithfulness and standard-sanity cells compute over an unfiltered
(all-calls) population, so they do not reproduce the signal-bearing table values on
their own. The files below are the regeneration and follow-up runs that are
shippable. They reproduce the target-ablation table and corroborate the protocol at
a smaller episode budget.

## Step / completeness records

| File pattern | Run | Scope |
|---|---|---|
| `metrics_<task>_170m_seed{42,142}.jsonl` | **Regeneration run** (Sec 4), NOT the 50-episode main pass despite the name | 15 episodes x 25 policy calls per seed, post-fix episode ceiling, 0 successes |
| `metrics_{PegInsertionSide,PickSingleYCB}-v1_170m_seed{42,142}_m128.jsonl` | m=128 subset, later-added tasks only | 8 episodes per seed (`PegInsertionSide seed142` includes one duplicated partial episode from a `--resume` restart) |
| `metrics_PickCube-v1_170m_seed{42,142}_{l2,maxdev}.jsonl` | Target ablation, alternative targets | regeneration scope; the `seed42` l2/maxdev files contain a duplicated partial episode 12 from a `--resume` restart, and `..._seed42_maxdev.jsonl` has one NUL-corrupted line (the analysis loader skips it) |
| `metrics_PickCube-v1_170m_seed{42,142}_logpi.jsonl` | Matched-population logpi target step records, Month 6 (the per-step IG run behind the Table 4 logpi faithfulness column) | regeneration scope, 15 episodes per seed |
| `metrics_PickCube-v1_1b_seed42.jsonl` | RDT-1B scale check (authors' fine-tuned weights) | 20 episodes, 9 successes; all rows below the signal threshold |
| `metrics_PickCube-v1_1b_seed{142,242}.jsonl` | RDT-1B scale check, Month 6 evaluation seeds. Environment and IG protocol matched to the committed seed 42 run, executed on a fresh RTX PRO 6000 pod that encoded its own T5-XXL instruction embeddings and two-token language baseline, so the language rows span a different baseline provenance than the seed 42 record (see the paper's Setup deviations) | 20 episodes each, 1 and 5 successes; all rows below the signal threshold |
| `metrics_StackCube-v1_1b_seed142.jsonl` | RDT-1B StackCube base run for the Month 6 displacement second seed. The 1B is not fine-tuned for StackCube and succeeds on 7 of 20 episodes here; this run only sources the displacement re-run and is not a faithfulness result | 20 episodes, 7 successes; all rows below the signal threshold |

## Faithfulness records (`metrics_faithfulness_*`)

| File pattern | Run |
|---|---|
| `..._PickCube-v1_1b_seed42_*.jsonl` | RDT-1B faithfulness (Table 2 bottom block; 320 rows) |
| `..._PickCube-v1_1b_seed{142,242}_*.jsonl` | RDT-1B faithfulness, Month 6 evaluation seeds (Table 2 bottom block, per-seed plus pooled; 480 and 399 rows). Vision passes both bars at every seed; language insertion passes at 42 and 142 and misses at 242. |
| `..._{PegInsertionSide,PickSingleYCB}-v1_170m_seed*_m128_*.jsonl` | m=128 faithfulness, later-added tasks |
| `..._PickCube-v1_170m_seed*_{l2,maxdev}_*.jsonl` | Target-ablation faithfulness (Table 4 l2 / maxdev columns) |
| `..._PickCube-v1_170m_seed*_logpi_*.jsonl` | Matched-population logpi faithfulness, Month 6 (Table 4 logpi column, same re-run population as l2/maxdev, ~750 rows). Deletion AUC 0.448 misses the 0.40 bar on this 0-success population, where l2 and maxdev pass. |

(The faithfulness files over the duplicated step runs noted above, PegInsertionSide
seed142 m128 and the seed42 l2/maxdev pair, mirror those replayed rows. The
bootstrap loader's dedup applies to them the same way.)

## Sanity records (`metrics_sanity_*`)

| File pattern | Run |
|---|---|
| `metrics_sanity_C1_frozen_*` | Frozen-target C1 variant (Table 3 bottom block), m=16, first 50 steps per task-seed, 4 tasks x 2 seeds |
| `metrics_sanity_C1_cascade_*` | Full-backbone cascade C1 variant, same scope |

(The standard last-layer C1 and the C2 input-randomization runs behind the
per-task rows of Table 3 are part of the main pass and are not in this release.)

## Baseline sensitivity (`metrics_baseline_sensitivity_*`)

PickCube and StackCube, seed 42, first 30 steps each (Table 5).

## Month 5 records (`m5_*`, `disp_validate.jsonl`)

| File | Run |
|---|---|
| `m5_metrics_displacement_{PickCube,StackCube}-v1_{170m,1b}.jsonl` | Action-displacement experiment behind the displacement figure. Vision and language, solver steps T in {1,2,3,5,10,20}, deletion fractions {0,1,5,10,20}%, rankings from the T=5 attributions, seed 42. 49 to 50 policy calls per run (60 for PickCube 1B); episode counts 8/20/8/5 for PickCube-170m/PickCube-1b/StackCube-170m/StackCube-1b. The 1B runs are unfiltered because 1B action norms sit below the signal threshold. The PickCube 170m file contains 12 `--resume` replay duplicate rows at (episode 1, call 0), one per modality x solver-step pair, identical except wall time; the bootstrap loader's dedup keeps the last. |
| `metrics_displacement_metrics_{PickCube,StackCube}-v1_1b_seed142_*.jsonl` | Displacement second seed, Month 6 (seed 142, vision modality, `--limit 60` stratified, `--no-signal-filter` since 1B norms sit below the threshold). Pooled with the seed-42 `m5` files for the two-seed 1B curves in the displacement figure, 40 episode groups on PickCube and 25 on StackCube. |
| `disp_validate.jsonl` | Validation record for the displacement identity. Contains displacement-side rows only: two policy calls at each of solver steps {1, 5, 20} (six rows, deletion grid {0,5,20}%, rankings from the T=5 attributions, with l2/rms/relative fields). The identity check uses the two solver-steps-5 rows, joining their nonzero deletion fractions against the matching `vision_deletion_curve` values in `m5_metrics_faithfulness_PickCube-v1_1b_seed42.jsonl` (which ran the production five-step chain), where delta log pi equals minus the squared displacement over (2 x 512), reproducing at correlation ~0.9999997 (recomputing the join from the committed files gives 0.9999998). The other-T rows have no faithfulness counterpart and are not part of the check. |
| `m5_metrics_PickCube-v1_1b_seed42.jsonl` | Month 5 1B reproduction base run on the RunPod RTX PRO 6000 with real T5-XXL embeddings. 20 episodes, 8 successes (the Month 4 run on identical seeds had 9; per-forward noise seeding does not bit-reproduce across machines and library versions). |
| `m5_metrics_faithfulness_PickCube-v1_1b_seed42.jsonl` | Reproduction-gate faithfulness over that base run (334 rows). Vision insertion median 0.919 and deletion 0.264, inside the episode-bootstrap intervals of the committed Month 4 run (0.926 and 0.248). Language insertion median 0.515, below the 0.55 bar, which the paper's Table 2 caption records as a baseline-provenance caveat on the two of three count. |
| `m5_metrics_PickCube-v1_170m_seed42.jsonl` | Partial 170M smoke record from the pod bootstrap. Contains a re-executed (episode 1, call 0) row pair that shares vision/language errors but differs in state error and wall time, plus a dangling partial episode from an interrupted run. Used by no table; the analysis loader's dedup keeps the last row. |

The StackCube 1B base run (5 episodes, seed 42) that supplied sidecars for
`m5_metrics_displacement_StackCube-v1_1b.jsonl` is not committed. Only its
displacement output is. `scripts/run_displacement.sh` regenerates it when
missing.

## Interval tables

`out/figures_m5/table_*_ci.csv` carry 95% episode-bootstrap intervals
(`bootstrap_ci.py`, B=10000, seed 0, episode-level resampling). The bootstrap
loader deduplicates the documented `--resume` replay rows, so its point
estimates and `n` can differ from the duplicate-inclusive
`out/figures_m4/table_*.csv` values that the paper tables print, in the third
decimal for most statistics and by up to 0.8 points for the m=128 pass-rate
cells. Both populations are derivable from the files above.

One derived table needs a view caveat. `out/figures_m4/table_m128.csv` mixes
two comparison views in its m128 column, the preserved matched-view PickCube
and StackCube constants next to seed-averaged full-view PegInsertionSide and
PickSingleYCB values. The paper's Table 6 keeps the views separate and reports
the released tasks per seed, so quote that table rather than the CSV.
