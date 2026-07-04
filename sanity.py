"""
Sanity check post-processing.

Re-runs IG on the same observations used by a completed per_step_ig.py pass,
with a controlled perturbation applied, and reports Spearman rank correlation
between the perturbed attribution and the original. Threshold:
Spearman rho <= 0.2 means attributions did not survive the perturbation, which
is the intended negative result (attribution should not be robust to nulling
out whatever signal it was supposed to attribute).

Two phases:

  C1  Model randomization. Xavier-uniform re-init of
      runner.model.final_layer.ffn_final.fc2 (the final action head). Follows
      Adebayo et al. 2018 "Sanity Checks for Saliency Maps": the paper asks
      that saliency methods fail the sanity check when the last layer is
      re-initialized because the network is no longer the same function.

  C2  Input randomization. Pixel shuffle of the 384x384 obs image (deterministic
      permutation keyed on --shuffle-seed) AND token shuffle of the language
      instruction (consumed via data/lang_embeds/{task}_shuffled_s{seed}.pt
      produced offline by encode_task_lang.py --task X --shuffle-seed N).

Spearman rho is computed over modality-specific vectors:
  - vision: 729 ext-cam patch scores (slot 3 only; other slots have zero attr
    by baseline construction)
  - language: real-token positions only (lang_attn_mask.sum())
  - state: 8 MANISKILL_INDICES dims (the 120 padding dims are zero either way)

Writes one JSONL row per (episode, policy_call) to:
    data/metrics_sanity_{phase}_{source_metrics_basename}_{timestamp}.jsonl

Usage:
    .venv/bin/python sanity.py --phase C1 \\
        --metrics data/metrics_PickCube-v1_170m_seed42_20260421_182705.jsonl \\
        --task PickCube-v1 --model 170m --m 16

    .venv/bin/python sanity.py --phase C2 \\
        --metrics data/metrics_PickCube-v1_170m_seed42_20260421_182705.jsonl \\
        --task PickCube-v1 --model 170m --m 16 --shuffle-seed 777
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr

from per_step_attribution import (
    prepare_ig_context, build_forward_fns, MANISKILL_INDICES,
)
from integrated_gradients import integrated_gradients
from pipeline import load_pipeline, load_lang


EXT_CAM_START = 3 * 729
EXT_CAM_END = 4 * 729


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["C1", "C2", "C1_frozen", "C1_cascade"], required=True)
    p.add_argument("--metrics", required=True,
                   help="Path to the original per_step_ig.py JSONL.")
    p.add_argument("--task", required=True)
    p.add_argument("--model", choices=["170m", "1b"], default="170m")
    p.add_argument("--m", type=int, default=16,
                   help="Integration steps for the sanity re-run. Dev default.")
    p.add_argument("--shuffle-seed", type=int, default=None,
                   help="C2 only. Seed controlling pixel shuffle order and loading the "
                        "matching data/lang_embeds/{task}_shuffled_s{seed}.pt.")
    p.add_argument("--no-checkpoint", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def reinit_last_action_layer(runner, seed):
    """
    C1: Xavier-uniform re-init of runner.model.final_layer.ffn_final.fc2.

    The RDT paper zero-initializes this layer's weight+bias at training time.
    For the sanity check we use Xavier uniform (Adebayo 2018 uses standard
    random re-init, not zero): zero would collapse log π to a constant and
    trivially produce zero attribution, which is a degenerate pass, not the
    "does saliency survive a randomized model" question.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    fc2 = runner.model.final_layer.ffn_final.fc2
    w = torch.empty_like(fc2.weight, device="cpu").to(torch.float32)
    torch.nn.init.xavier_uniform_(w, generator=g)
    fc2.weight.data.copy_(w.to(fc2.weight.device, dtype=fc2.weight.dtype))
    if fc2.bias is not None:
        fc2.bias.data.zero_()


def reinit_full_backbone(runner, seed):
    """
    C1_cascade: Xavier-uniform re-init of EVERY 2D+ weight matrix in the DiT
    backbone (runner.model: all 14 blocks' attention qkv/proj + cross-attn +
    MLP fc1/fc2, plus the final layer and embedders). 1D params (norm scales,
    biases) are left intact.

    This is the proper Adebayo et al. 2018 "cascading randomization": destroying
    only the final layer (reinit_last_action_layer) leaves the feature extractor
    intact, so saliency barely moves (we measured rho~0.96 on both moving- and
    frozen-target last-layer variants). Randomizing the whole backbone scrambles
    the attention pattern that drives spatial attribution, so a faithful saliency
    method should decorrelate from the original (rho -> low). Used with a frozen
    target (the original model's ref_action) so the only thing changing between
    the original and perturbed run is the model's learned weights.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = 0
    for name, p in runner.model.named_parameters():
        if p.dim() >= 2:
            w = torch.empty_like(p, device="cpu").to(torch.float32)
            torch.nn.init.xavier_uniform_(w, generator=g)
            p.data.copy_(w.to(p.device, dtype=p.dtype))
            n += 1
    print(f"re-initialized {n} weight matrices across the full DiT backbone")


def pixel_shuffle_image(obs_image, shuffle_seed):
    """
    Deterministic pixel permutation of an HxWx3 uint8 array. Returns a PIL Image
    of the same dimensions. The permutation applies identically to all three
    color channels so the marginal color distribution is preserved.
    """
    arr = np.asarray(obs_image)
    h, w, c = arr.shape
    rng = np.random.default_rng(shuffle_seed)
    perm = rng.permutation(h * w)
    flat = arr.reshape(h * w, c)[perm]
    return Image.fromarray(flat.reshape(h, w, c))


def reduce_vision_for_spearman(attr_tensor):
    """(1, 4374, H) -> (729,) of |sum over H| on the ext-cam slot."""
    per_pos = attr_tensor.abs().sum(dim=-1).squeeze(0)  # (4374,)
    return per_pos[EXT_CAM_START:EXT_CAM_END].float().cpu().numpy()


def reduce_language_for_spearman(attr_tensor, real_mask):
    per_pos = attr_tensor.abs().sum(dim=-1).squeeze(0)  # (1024,)
    #attr_tensor may be on CPU (from sidecar) while real_mask is on GPU (from ctx);
    #unify on CPU for indexing.
    real_idx = torch.nonzero(real_mask.cpu(), as_tuple=False).squeeze(-1)
    per_pos = per_pos.cpu()
    return per_pos[real_idx].float().numpy()


def reduce_state_for_spearman(attr_tensor):
    flat = attr_tensor.abs().squeeze(0).squeeze(0).float().cpu().numpy()  # (128,)
    return flat[MANISKILL_INDICES]


def safe_spearman(a, b):
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    r, _ = spearmanr(a, b)
    return float(r)


def main():
    args = parse_args()

    if args.phase == "C2" and args.shuffle_seed is None:
        raise SystemExit("ERROR: --phase C2 requires --shuffle-seed.")

    #Load original metrics rows.
    rows = []
    with open(args.metrics) as f:
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
    if args.limit is not None:
        rows = rows[:args.limit]
    print(f"loaded {len(rows)} step rows from {args.metrics}")

    if args.out is None:
        base = os.path.basename(args.metrics).replace(".jsonl", "")
        ts = time.strftime("%Y%m%d_%H%M%S")
        tag = args.phase + (f"_s{args.shuffle_seed}" if args.phase == "C2" else "")
        args.out = f"data/metrics_sanity_{tag}_{base}_{ts}.jsonl"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    pipe = load_pipeline(args.model, enable_checkpoint=not args.no_checkpoint)

    #Model-randomization variants, re-init weights once before the loop:
    #  C1          last-layer reinit, target recomputed under re-init'd model (moving target)
    #  C1_frozen   last-layer reinit, target = original model's ref_action (frozen target)
    #  C1_cascade  FULL backbone reinit + frozen target — the proper Adebayo cascading test
    #C1_frozen and C1_cascade load sidecar ref_action so log_pi measures distance from the
    #ORIGINAL policy's action; only the model weights change between original and perturbed.
    if args.phase in ("C1", "C1_frozen"):
        reinit_last_action_layer(pipe["runner"], seed=12345)
        print("re-initialized runner.model.final_layer.ffn_final.fc2 (Xavier uniform)")
        if args.phase == "C1_frozen":
            print("C1_frozen: will use sidecar ref_action as frozen target")
    elif args.phase == "C1_cascade":
        reinit_full_backbone(pipe["runner"], seed=12345)
        print("C1_cascade: full-backbone reinit + sidecar ref_action as frozen target")

    #C2: language comes from the pre-shuffled .pt. For C1/C1_frozen or clean baseline,
    #use the task's normal .pt.
    if args.phase == "C2":
        #Two ways to do "token shuffle" for C2:
        #  (a) shuffle SentencePiece tokens before T5, re-encode through T5-XXL.
        #      Requires T5-XXL (~22 GB) on the pod and pre-generated *_shuffled_s{N}.pt
        #      files via `encode_task_lang.py --task X --shuffle-seed N`.
        #  (b) shuffle the saved post-T5 embeddings along the real-token axis at
        #      runtime here. Equivalent for the sanity check's purpose (destroys
        #      input the policy sees; ρ should still collapse to ≤ 0.2) and skips
        #      the T5 dependency entirely. Documented as a small deviation from
        #      C2's strict reading.
        from pipeline import load_lang as _ll
        base = _ll(args.task)
        shuffled_lang_path = f"data/lang_embeds/{args.task}_shuffled_s{args.shuffle_seed}.pt"
        if os.path.exists(shuffled_lang_path):
            data = torch.load(shuffled_lang_path, weights_only=True)
            lang = {
                "lang_tokens": data["embeds"].to("cuda", dtype=torch.bfloat16),
                "lang_attn_mask": data["attn_mask"].to("cuda", dtype=torch.bool),
                "lang_tokens_baseline": base["lang_tokens_baseline"],
                "tokens": data.get("tokens", None),
            }
            print(f"C2: loaded pre-shuffled lang from {shuffled_lang_path}")
        else:
            #Runtime post-T5 shuffle: permute real-token positions among themselves.
            attn = base["lang_attn_mask"]            # (1, 1024) bool
            tok = base["lang_tokens"].clone()        # (1, 1024, H) bf16
            real_n = int(attn.sum().item())
            rng = torch.Generator(device="cpu").manual_seed(args.shuffle_seed)
            perm = torch.randperm(real_n, generator=rng)
            tok[0, :real_n] = tok[0, perm]
            lang = {
                "lang_tokens": tok,
                "lang_attn_mask": base["lang_attn_mask"],
                "lang_tokens_baseline": base["lang_tokens_baseline"],
                "tokens": base.get("tokens", None),
            }
            print(f"C2: runtime post-T5 shuffle (seed={args.shuffle_seed}, n_real={real_n})")
    else:
        lang = load_lang(args.task)

    print(f"\n=== sanity {args.phase} over {len(rows)} steps, writing {args.out} ===\n")
    jsonl = open(args.out, "a")

    for i, row in enumerate(rows):
        t0 = time.time()
        sidecar = torch.load(row["attr_file"], weights_only=False, map_location="cpu")

        obs_image = Image.fromarray(sidecar["obs_image"])
        if args.phase == "C2":
            obs_image = pixel_shuffle_image(obs_image, args.shuffle_seed)
        proprio = sidecar["proprio"]

        frozen_ref = None
        if args.phase in ("C1_frozen", "C1_cascade"):
            frozen_ref = sidecar["ref_action"]

        ctx = prepare_ig_context(
            pipe["runner"], pipe["vision_model"],
            obs_image, proprio,
            lang["lang_tokens"], lang["lang_attn_mask"], lang["lang_tokens_baseline"],
            pipe["bg_image_encoded"], pipe["img_tokens_baseline"],
            pipe["action_mask"], pipe["ctrl_freqs"],
            seed=row["seed"], sigma_sq=1.0,
            frozen_ref_action=frozen_ref,
        )
        fwd_v, fwd_l, fwd_s = build_forward_fns(ctx)

        vision_attr_new = integrated_gradients(
            fwd_v, ctx["img_adapted"], ctx["img_adapted_bl"], m=args.m)
        lang_attr_new = integrated_gradients(
            fwd_l, ctx["lang_adapted"], ctx["lang_adapted_bl"], m=args.m)
        state_attr_new = integrated_gradients(
            fwd_s, ctx["state_input_actual"], ctx["state_input_baseline"], m=args.m)

        #Originals from the sidecar (the baseline run we are comparing against).
        vision_attr_orig = sidecar["vision_attr"]
        lang_attr_orig = sidecar["lang_attr"]
        state_attr_orig = sidecar["state_attr"]

        rho_v = safe_spearman(
            reduce_vision_for_spearman(vision_attr_orig),
            reduce_vision_for_spearman(vision_attr_new))
        rho_l = safe_spearman(
            reduce_language_for_spearman(lang_attr_orig,
                                         ctx["lang_attn_mask"].squeeze(0)),
            reduce_language_for_spearman(lang_attr_new,
                                         ctx["lang_attn_mask"].squeeze(0)))
        rho_s = safe_spearman(
            reduce_state_for_spearman(state_attr_orig),
            reduce_state_for_spearman(state_attr_new))

        wall = time.time() - t0
        out_row = {
            "event": f"sanity_{args.phase}",
            "task": args.task,
            "model": args.model,
            "episode": row["episode"],
            "seed": row["seed"],
            "policy_call_idx": row["policy_call_idx"],
            "attr_file": row["attr_file"],
            "ref_norm_maniskill": row["ref_norm_maniskill"],
            "wall_seconds": wall,
            "spearman_vision": rho_v,
            "spearman_language": rho_l,
            "spearman_state": rho_s,
            "m": args.m,
        }
        if args.phase == "C2":
            out_row["shuffle_seed"] = args.shuffle_seed
        jsonl.write(json.dumps(out_row) + "\n")
        jsonl.flush()
        print(f"  step {i+1}/{len(rows)} ep{row['episode']:03d}t{row['policy_call_idx']:02d}: "
              f"rho_v={rho_v:+.3f} rho_l={rho_l:+.3f} rho_s={rho_s:+.3f} wall={wall:.1f}s")

        del ctx, vision_attr_new, lang_attr_new, state_attr_new, sidecar
        torch.cuda.empty_cache()

    jsonl.close()
    print(f"\ndone. wrote {args.out}")


if __name__ == "__main__":
    main()
