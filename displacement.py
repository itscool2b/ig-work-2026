"""Direct contraction-mechanism measurement (Month 5).

For each signal-bearing policy-call row of a per_step_ig.py JSONL, measure the
action displacement ||a_perturbed - a_original|| as a function of
  (a) top-k input deletion fraction k in --del-grid, and
  (b) the number of DPM-Solver++ denoising steps T in --solver-steps.

The Month-4 hypothesis held that the iterative denoiser contracts perturbed
conditioning back toward the action manifold, predicting a stronger policy
"contracts more, not less." This script tests that claim directly: if
contraction were the mechanism, the displacement should shrink as T grows and
as the policy gets stronger. The measurement refuted it. The displacement holds
or grows between the two-step and twenty-step endpoints on every curve, and the
paper ("The Readout, Not the Denoiser", Sec. 6.1) traces the small log-density
magnitude to the quadratic per-entry readout instead.

a_original = ctx["ref_action"], the seeded conditional_sample on the real adapted
             inputs at the CURRENT T (re-sampled per T, NOT the sidecar's T=5
             ref_action, so original and perturbed share the same chain length).
a_perturbed = same chain, but with the top-k |IG| ext-cam tokens (vision) or the
             top-k real language tokens replaced by the baseline embeddings,
             reusing faithfulness.py's perturb_image / perturb_lang verbatim.
The tokens deleted are chosen by the canonical T=5 sidecar attribution (rank_T=5)
so the deletion SET is held fixed and only the chain length T varies.

Displacement is reported over the 8 active ManiSkill dims and over the full
(64,128) chunk, as raw L2, per-element RMS, and relative-to-reference L2.

Output: data/metrics_displacement_{source_metrics_basename}_{ts}.jsonl, one row
per (episode, policy_call, T, modality). Downstream: make_month5_figs.py renders
fig_displacement_vs_T.png and bootstrap_ci.py adds episode-bootstrap CI bands.
"""
import argparse
import json
import os
import time

import torch
from PIL import Image

from per_step_attribution import prepare_ig_context, MANISKILL_INDICES
from pipeline import load_pipeline, load_lang
from faithfulness import (
    topk_mask, perturb_image, perturb_lang, EXT_CAM_START, EXT_CAM_END,
    load_step_rows,
)

RANK_T = 5  # the canonical T at which the sidecar attribution was computed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True,
                   help="Path to a per_step_ig.py JSONL (its sidecars supply the "
                        "T=5 attribution ranking and the obs/proprio to replay).")
    p.add_argument("--task", required=True, help="ManiSkill task id.")
    p.add_argument("--model", choices=["170m", "1b"], default="170m")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="Disable gradient checkpointing on the pipeline load.")
    p.add_argument("--out", default=None,
                   help="Output JSONL. Auto-named from metrics if omitted.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N signal-bearing step rows.")
    p.add_argument("--solver-steps", default="1,2,3,5,10,20",
                   help="Comma list of DPM-Solver++ step counts T to sweep.")
    p.add_argument("--del-grid", default="0,1,5,10,20",
                   help="Comma list of top-k deletion percentages. 0 is the "
                        "zero-displacement anchor.")
    p.add_argument("--modality", choices=["vision", "language", "both"],
                   default="vision")
    p.add_argument("--no-signal-filter", action="store_true",
                   help="Process all rows, not just |ref action| >= 15. Use for 1B "
                        "(its fine-tuned action norms sit below 15 throughout).")
    return p.parse_args()


def displacement(a_pert, a_orig):
    """All quantities under no_grad. a_pert, a_orig are (1, 64, 128) bf16."""
    d8 = (a_pert[..., MANISKILL_INDICES] - a_orig[..., MANISKILL_INDICES]).float()
    full = (a_pert - a_orig).float()
    ref8 = a_orig[..., MANISKILL_INDICES].float()
    return {
        "l2_active": d8.norm().item(),                       # over 64*8 active entries
        "rms_active": d8.pow(2).mean().sqrt().item(),        # per-entry RMS
        "l2_full": full.norm().item(),                       # over 64*128
        "rel_active": (d8.norm() / (ref8.norm() + 1e-9)).item(),
    }


def main():
    args = parse_args()
    solver_grid = [int(x) for x in args.solver_steps.split(",")]
    del_grid = [int(x) for x in args.del_grid.split(",")]
    modalities = ["vision", "language"] if args.modality == "both" else [args.modality]

    rows = load_step_rows(args.metrics)
    if not args.no_signal_filter:
        rows = [r for r in rows if r.get("ref_norm_maniskill", -1) >= 15]
    if args.limit is not None and len(rows) > args.limit:
        #Stratify the sample across episodes (round-robin by policy call) so the
        #downstream episode-bootstrap has many groups, not the 2-3 you get by
        #taking the first N rows of long failure trajectories.
        from collections import OrderedDict
        byep = OrderedDict()
        for r in rows:
            byep.setdefault(r["episode"], []).append(r)
        picked, idx = [], 0
        while len(picked) < args.limit:
            added = False
            for ers in byep.values():
                if idx < len(ers):
                    picked.append(ers[idx]); added = True
                    if len(picked) >= args.limit:
                        break
            if not added:
                break
            idx += 1
        rows = picked

    if args.out is None:
        base = os.path.splitext(os.path.basename(args.metrics))[0]
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out = f"data/metrics_displacement_{base}_{ts}.jsonl"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"\n=== displacement over {len(rows)} rows x {len(solver_grid)} T values, "
          f"writing {args.out} ===")
    print(f"    T grid: {solver_grid}   del grid: {del_grid}   modalities: {modalities}\n")
    jsonl = open(args.out, "a")

    #Outer loop over T: the DPM-Solver++ step count is baked into the scheduler at
    #RDTRunner construction, so each T needs its own pipeline load. The 96 GB card
    #makes the reload (~30-60s) cheap relative to the per-row forwards.
    for T in solver_grid:
        print(f"--- loading pipeline at T={T} ---")
        pipe = load_pipeline(args.model, enable_checkpoint=not args.no_checkpoint,
                             solver_steps=T)
        lang = load_lang(args.task)

        for i, row in enumerate(rows):
            t0 = time.time()
            sidecar = torch.load(row["attr_file"], weights_only=False, map_location="cpu")
            obs_image = Image.fromarray(sidecar["obs_image"])
            proprio = sidecar["proprio"]

            #ref_action is recomputed at THIS T inside prepare_ig_context, so
            #a_original and every a_perturbed share the same chain length.
            ctx = prepare_ig_context(
                pipe["runner"], pipe["vision_model"],
                obs_image, proprio,
                lang["lang_tokens"], lang["lang_attn_mask"], lang["lang_tokens_baseline"],
                pipe["bg_image_encoded"], pipe["img_tokens_baseline"],
                pipe["action_mask"], pipe["ctrl_freqs"],
                seed=row["seed"], sigma_sq=1.0, target="logpi",
            )
            a_orig = ctx["ref_action"]

            for modality in modalities:
                #Rank positions with the same |IG|-sum reduction as faithfulness.py,
                #using the canonical T=5 sidecar attribution (held fixed across T).
                if modality == "vision":
                    attr = sidecar["vision_attr"].to("cuda", dtype=torch.bfloat16)
                    abs_per_pos = attr.abs().sum(dim=-1).squeeze(0)
                    ranked_scores = abs_per_pos[EXT_CAM_START:EXT_CAM_END]
                    n_ranked = 729
                    real_mask = real_idx = None
                else:
                    attr = sidecar["lang_attr"].to("cuda", dtype=torch.bfloat16)
                    abs_per_pos = attr.abs().sum(dim=-1).squeeze(0)
                    real_mask = ctx["lang_attn_mask"].squeeze(0)
                    real_idx = torch.nonzero(real_mask, as_tuple=False).squeeze(-1)
                    ranked_scores = abs_per_pos[real_idx]
                    n_ranked = int(real_mask.sum().item())

                disp_lists = {"l2_active": [], "rms_active": [], "l2_full": [],
                              "rel_active": []}
                with torch.no_grad():
                    for k in del_grid:
                        if k == 0:
                            #Zero-deletion anchor: a_perturbed == a_original exactly.
                            for key in disp_lists:
                                disp_lists[key].append(0.0)
                            continue
                        ranked_mask = topk_mask(ranked_scores, k, n_ranked)
                        if modality == "vision":
                            cond = perturb_image(ctx["img_adapted"], ctx["img_adapted_bl"],
                                                 ranked_mask)
                            a_pert = ctx["seeded_conditional_sample"](
                                ctx["lang_adapted"], cond, ctx["state_traj_actual"]).detach()
                        else:
                            full_mask = torch.zeros(ctx["lang_adapted"].shape[1],
                                                    dtype=torch.bool, device=ranked_scores.device)
                            full_mask[real_idx] = ranked_mask
                            cond = perturb_lang(ctx["lang_adapted"], ctx["lang_adapted_bl"],
                                                real_mask, full_mask)
                            a_pert = ctx["seeded_conditional_sample"](
                                cond, ctx["img_adapted"], ctx["state_traj_actual"]).detach()
                        d = displacement(a_pert, a_orig)
                        for key in disp_lists:
                            disp_lists[key].append(d[key])
                        del a_pert, cond

                ref_norm_active = a_orig[..., MANISKILL_INDICES].float().norm().item()
                out_row = {
                    "event": "displacement",
                    "task": args.task, "model": args.model,
                    "episode": row["episode"], "seed": row["seed"],
                    "policy_call_idx": row["policy_call_idx"],
                    "attr_file": row["attr_file"],
                    "ref_norm_maniskill": row["ref_norm_maniskill"],
                    "modality": modality,
                    "solver_steps": T, "rank_T": RANK_T,
                    "del_grid": del_grid,
                    "l2_active": disp_lists["l2_active"],
                    "rms_active": disp_lists["rms_active"],
                    "l2_full": disp_lists["l2_full"],
                    "rel_active": disp_lists["rel_active"],
                    "ref_action_norm_active": ref_norm_active,
                    "wall_seconds": time.time() - t0,
                }
                jsonl.write(json.dumps(out_row) + "\n")
                jsonl.flush()
                del attr

            #Progress: show the largest-deletion displacement of the last modality
            #written (out_row survives from the final loop iteration).
            big = out_row["l2_active"][-1]
            print(f"  T={T} {i+1}/{len(rows)} ep{row['episode']:03d}t{row['policy_call_idx']:02d} "
                  f"{out_row['modality']}: l2_active@k{del_grid[-1]}={big:.4f} "
                  f"|ref|active={ref_norm_active:.3f}")
            del ctx, sidecar
            torch.cuda.empty_cache()

        del pipe, lang
        torch.cuda.empty_cache()

    jsonl.close()
    print(f"\ndone. wrote {args.out}")


if __name__ == "__main__":
    main()
