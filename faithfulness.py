"""
Faithfulness post-processing.

Reads a per-step JSONL + its sidecar .pt files, re-loads the model once, and
for every policy-call row computes:

  B1. Δlog p under top-k deletion for k in {1, 5, 10} percent, per modality.
      F(deleted) minus F(input). F(input) = 0 by construction (log π of the
      Gaussian's own mean), so Δlog p = F(deleted). Expected sign: <= 0.
      Threshold: image median @ k=5%  <= -0.5 nats,
                       tokens median @ k=5% <= -0.5 nats.

  B2. Insertion / Deletion AUC over k-grid {0, 1, 5, 10, 20, 30, 50, 75, 100}.
      Deletion: start from input, progressively replace top-|IG| positions with
                baseline, record log π at each k. Normalized AUC close to 0
                means attribution ranked the causally-important inputs well.
      Insertion: start from baseline, progressively add top-|IG| positions back,
                record log π at each k. Normalized AUC close to 1 is good.
      Thresholds: image Insertion AUC >= 0.60, Deletion AUC <= 0.40;
                        tokens Insertion AUC >= 0.55, Deletion AUC <= 0.45.

Normalization:
  AUC_norm = trapz(y_k over k/100) with y linearly renormalized so
  f_input -> 1 and f_baseline -> 0. Random ranking gives AUC_norm ~= 0.5.

Writes one JSONL row per (episode, policy_call) to:
    data/metrics_faithfulness_{source_metrics_basename}_{timestamp}.jsonl

Usage:
    .venv/bin/python faithfulness.py \\
        --metrics data/metrics_PickCube-v1_170m_seed42_20260421_182705.jsonl \\
        --task PickCube-v1 --model 170m
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from per_step_attribution import (
    prepare_ig_context, build_forward_fns, MANISKILL_INDICES,
)
from pipeline import load_pipeline, load_lang


#k-grid for B2 AUC (percent). 0 and 100 are the endpoints of the sweep.
AUC_K_GRID = [0, 1, 5, 10, 20, 30, 50, 75, 100]
#k values at which B1 Δlog p is reported.
B1_K = [1, 5, 10]
#slot 3 of the 6-slot image condition = external camera at time t; indices 2187..2915.
#Only this slot has a non-trivial IG path, the other 5 slots have input == baseline.
EXT_CAM_START = 3 * 729
EXT_CAM_END = 4 * 729


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True,
                   help="Path to per_step_ig.py JSONL output.")
    p.add_argument("--task", required=True, help="ManiSkill task id.")
    p.add_argument("--model", choices=["170m", "1b"], default="170m")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="Disable gradient checkpointing on the pipeline load.")
    p.add_argument("--out", default=None,
                   help="Output JSONL. Auto-named from metrics if omitted.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N step rows. For smoke tests.")
    p.add_argument("--target", default="logpi",
                   choices=["logpi", "l2", "l2sq", "maxdev", "cosine"],
                   help="IG target the source JSONL used. Must match per_step_ig --target "
                        "so the perturbation Δ is measured on the same scalar.")
    return p.parse_args()


def load_step_rows(metrics_path):
    """Return list of step rows (those without event='episode_end')."""
    rows = []
    with open(metrics_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") == "episode_end":
                continue
            if "attr_file" not in r:
                continue
            rows.append(r)
    return rows


def topk_mask(abs_scores, k_pct, n_total):
    """
    Boolean (n_total,) mask selecting the top k_pct% of positions by abs_scores.
    abs_scores is the ranked population (e.g. 729 ext-cam tokens, or the
    lang_attn_mask.sum() real tokens); n_total is its length.
    """
    k_count = max(1, int(round(n_total * k_pct / 100.0)))
    if k_count >= n_total:
        return torch.ones(n_total, dtype=torch.bool, device=abs_scores.device)
    _, idx = torch.topk(abs_scores, k=k_count)
    mask = torch.zeros(n_total, dtype=torch.bool, device=abs_scores.device)
    mask[idx] = True
    return mask


def perturb_image(img_adapted, img_adapted_bl, ext_mask):
    """
    Return img_adapted with ext-cam token positions selected by `ext_mask`
    (shape (729,), bool) replaced by the corresponding baseline positions.
    """
    out = img_adapted.clone()
    start, end = EXT_CAM_START, EXT_CAM_END
    replace_idx = start + torch.nonzero(ext_mask, as_tuple=False).squeeze(-1)
    out[:, replace_idx, :] = img_adapted_bl[:, replace_idx, :]
    return out


def perturb_image_insertion(img_adapted, img_adapted_bl, ext_mask):
    """
    Symmetric to perturb_image: start from baseline, add input values at
    positions selected by `ext_mask`. Only the ext-cam slot differs; the other
    five slots already agree between input and baseline.
    """
    out = img_adapted_bl.clone()
    start, end = EXT_CAM_START, EXT_CAM_END
    replace_idx = start + torch.nonzero(ext_mask, as_tuple=False).squeeze(-1)
    out[:, replace_idx, :] = img_adapted[:, replace_idx, :]
    return out


def perturb_lang(lang_adapted, lang_adapted_bl, real_token_mask, full_mask):
    """
    Replace positions selected by `full_mask` (shape (1024,), bool) in
    `lang_adapted` with the corresponding baseline positions. `real_token_mask`
    is unused here but indicates which positions are real vs padding (for the
    caller to scope its top-k selection correctly).
    """
    out = lang_adapted.clone()
    replace_idx = torch.nonzero(full_mask, as_tuple=False).squeeze(-1)
    out[:, replace_idx, :] = lang_adapted_bl[:, replace_idx, :]
    return out


def perturb_lang_insertion(lang_adapted, lang_adapted_bl, full_mask):
    out = lang_adapted_bl.clone()
    replace_idx = torch.nonzero(full_mask, as_tuple=False).squeeze(-1)
    out[:, replace_idx, :] = lang_adapted[:, replace_idx, :]
    return out


def auc_normalized(k_grid, y_values, f_input, f_baseline):
    """
    Trapezoidal AUC of y_values over k_grid/100, with y linearly renormalized
    so f_input -> 1 and f_baseline -> 0. Returns AUC in [0, 1] ideally (outside
    that range if y went past the endpoints, which can happen from noise).
    """
    denom = f_input - f_baseline
    if abs(denom) < 1e-9:
        return float("nan")
    y_norm = (np.asarray(y_values, dtype=np.float64) - f_baseline) / denom
    x = np.asarray(k_grid, dtype=np.float64) / 100.0
    #numpy 2.0 renamed np.trapz to np.trapezoid; fall back to np.trapz on numpy 1.x.
    trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trapezoid(y_norm, x))


def compute_modality_metrics(
    fwd_fn, input_tensor, baseline_tensor, attribution,
    modality, real_mask=None,
):
    """
    Run all B1 + B2 forward passes for one modality and one step.

    Args:
        fwd_fn: forward_fn_vision or forward_fn_language from build_forward_fns
        input_tensor: the modality's "real" input to fwd_fn (adapted embeddings)
        baseline_tensor: same-shape baseline to fwd_fn
        attribution: IG output, same shape as input_tensor
        modality: "vision" | "language"
        real_mask: (seq_len,) bool on GPU, for language; positions that are real
            tokens (rest are padding and must not be ranked by |IG|).
    Returns: dict with dlogp_k1/5/10, deletion_auc, insertion_auc,
             deletion_curve, insertion_curve
    """
    device = input_tensor.device

    #Reduce attribution to a per-position score by summing |IG| over hidden dim.
    abs_per_pos = attribution.abs().sum(dim=-1).squeeze(0)  # (seq_len,)

    if modality == "vision":
        #Rank only within the 729 ext-cam tokens; the other 5 slots are
        #attribution-zero by construction (input == baseline on them).
        ranked_scores = abs_per_pos[EXT_CAM_START:EXT_CAM_END]
        n_ranked = 729
    elif modality == "language":
        real_idx = torch.nonzero(real_mask, as_tuple=False).squeeze(-1)
        ranked_scores = abs_per_pos[real_idx]
        n_ranked = int(real_mask.sum().item())
    else:
        raise ValueError(f"unsupported modality: {modality}")

    #f_input and f_baseline, cached once.
    with torch.no_grad():
        f_input = fwd_fn(input_tensor).item()
        f_baseline = fwd_fn(baseline_tensor).item()

    #B1: Δlog p at k in {1, 5, 10}.
    dlogp = {}
    for k in B1_K:
        ranked_mask = topk_mask(ranked_scores, k, n_ranked)
        if modality == "vision":
            perturbed = perturb_image(input_tensor, baseline_tensor, ranked_mask)
        else:
            full_mask = torch.zeros(input_tensor.shape[1], dtype=torch.bool, device=device)
            full_mask[real_idx] = ranked_mask
            perturbed = perturb_lang(input_tensor, baseline_tensor, real_mask, full_mask)
        with torch.no_grad():
            f_del = fwd_fn(perturbed).item()
        dlogp[k] = f_del - f_input
        del perturbed

    #B2: AUC curves. Share the ranked_mask computation with B1 where k overlaps.
    deletion_curve = []
    insertion_curve = []
    for k in AUC_K_GRID:
        if k == 0:
            deletion_curve.append(f_input)
            insertion_curve.append(f_baseline)
            continue
        if k == 100:
            deletion_curve.append(f_baseline)
            insertion_curve.append(f_input)
            continue
        ranked_mask = topk_mask(ranked_scores, k, n_ranked)
        if modality == "vision":
            p_del = perturb_image(input_tensor, baseline_tensor, ranked_mask)
            p_ins = perturb_image_insertion(input_tensor, baseline_tensor, ranked_mask)
        else:
            full_mask = torch.zeros(input_tensor.shape[1], dtype=torch.bool, device=device)
            full_mask[real_idx] = ranked_mask
            p_del = perturb_lang(input_tensor, baseline_tensor, real_mask, full_mask)
            p_ins = perturb_lang_insertion(input_tensor, baseline_tensor, full_mask)
        with torch.no_grad():
            deletion_curve.append(fwd_fn(p_del).item())
            insertion_curve.append(fwd_fn(p_ins).item())
        del p_del, p_ins

    deletion_auc = auc_normalized(AUC_K_GRID, deletion_curve, f_input, f_baseline)
    insertion_auc = auc_normalized(AUC_K_GRID, insertion_curve, f_input, f_baseline)

    return {
        "dlogp_k1": dlogp[1],
        "dlogp_k5": dlogp[5],
        "dlogp_k10": dlogp[10],
        "deletion_auc": deletion_auc,
        "insertion_auc": insertion_auc,
        "deletion_curve": deletion_curve,
        "insertion_curve": insertion_curve,
        "f_input": f_input,
        "f_baseline": f_baseline,
    }


def main():
    args = parse_args()

    rows = load_step_rows(args.metrics)
    if args.limit is not None:
        rows = rows[:args.limit]
    print(f"loaded {len(rows)} step rows from {args.metrics}")

    if args.out is None:
        base = os.path.basename(args.metrics).replace(".jsonl", "")
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out = f"data/metrics_faithfulness_{base}_{ts}.jsonl"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    pipe = load_pipeline(args.model, enable_checkpoint=not args.no_checkpoint)
    lang = load_lang(args.task)

    print(f"\n=== faithfulness over {len(rows)} steps, writing {args.out} ===\n")
    jsonl = open(args.out, "a")

    for i, row in enumerate(rows):
        t0 = time.time()
        sidecar = torch.load(row["attr_file"], weights_only=False, map_location="cpu")

        #Reconstruct context using the sidecar's proprio and obs_image.
        obs_image = Image.fromarray(sidecar["obs_image"])
        proprio = sidecar["proprio"]

        ctx = prepare_ig_context(
            pipe["runner"], pipe["vision_model"],
            obs_image, proprio,
            lang["lang_tokens"], lang["lang_attn_mask"], lang["lang_tokens_baseline"],
            pipe["bg_image_encoded"], pipe["img_tokens_baseline"],
            pipe["action_mask"], pipe["ctrl_freqs"],
            seed=row["seed"], sigma_sq=1.0, target=args.target,
        )
        fwd_v, fwd_l, _ = build_forward_fns(ctx)

        vision_attr = sidecar["vision_attr"].to("cuda", dtype=torch.bfloat16)
        lang_attr = sidecar["lang_attr"].to("cuda", dtype=torch.bfloat16)

        v_metrics = compute_modality_metrics(
            fwd_v, ctx["img_adapted"], ctx["img_adapted_bl"], vision_attr,
            modality="vision")
        l_metrics = compute_modality_metrics(
            fwd_l, ctx["lang_adapted"], ctx["lang_adapted_bl"], lang_attr,
            modality="language", real_mask=ctx["lang_attn_mask"].squeeze(0))

        wall = time.time() - t0
        out_row = {
            "event": "faithfulness",
            "task": args.task,
            "model": args.model,
            "episode": row["episode"],
            "seed": row["seed"],
            "policy_call_idx": row["policy_call_idx"],
            "attr_file": row["attr_file"],
            "ref_norm_maniskill": row["ref_norm_maniskill"],
            "wall_seconds": wall,
            #vision
            "vision_dlogp_k1": v_metrics["dlogp_k1"],
            "vision_dlogp_k5": v_metrics["dlogp_k5"],
            "vision_dlogp_k10": v_metrics["dlogp_k10"],
            "vision_deletion_auc": v_metrics["deletion_auc"],
            "vision_insertion_auc": v_metrics["insertion_auc"],
            "vision_deletion_curve": v_metrics["deletion_curve"],
            "vision_insertion_curve": v_metrics["insertion_curve"],
            "vision_f_input": v_metrics["f_input"],
            "vision_f_baseline": v_metrics["f_baseline"],
            #language
            "lang_dlogp_k1": l_metrics["dlogp_k1"],
            "lang_dlogp_k5": l_metrics["dlogp_k5"],
            "lang_dlogp_k10": l_metrics["dlogp_k10"],
            "lang_deletion_auc": l_metrics["deletion_auc"],
            "lang_insertion_auc": l_metrics["insertion_auc"],
            "lang_deletion_curve": l_metrics["deletion_curve"],
            "lang_insertion_curve": l_metrics["insertion_curve"],
            "lang_f_input": l_metrics["f_input"],
            "lang_f_baseline": l_metrics["f_baseline"],
            #k-grids used (record for downstream sanity)
            "k_grid_auc": AUC_K_GRID,
            "k_grid_b1": B1_K,
        }
        jsonl.write(json.dumps(out_row) + "\n")
        jsonl.flush()
        print(f"  step {i+1}/{len(rows)} ep{row['episode']:03d}t{row['policy_call_idx']:02d}: "
              f"v_dlogp@5={v_metrics['dlogp_k5']:+.3f} "
              f"v_ins={v_metrics['insertion_auc']:.3f} v_del={v_metrics['deletion_auc']:.3f} | "
              f"l_dlogp@5={l_metrics['dlogp_k5']:+.3f} "
              f"l_ins={l_metrics['insertion_auc']:.3f} l_del={l_metrics['deletion_auc']:.3f} | "
              f"wall={wall:.1f}s")

        del ctx, vision_attr, lang_attr, sidecar
        torch.cuda.empty_cache()

    jsonl.close()
    print(f"\ndone. wrote {args.out}")


if __name__ == "__main__":
    main()
