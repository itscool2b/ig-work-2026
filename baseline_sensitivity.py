"""
Baseline sensitivity sweep for vision IG (Fig D).

Runs vision IG on the same observations with three different baselines:
  - black: (0,0,0) solid image
  - gray:  (128,128,128) solid image (the current production baseline)
  - blur:  Gaussian-blurred version of the actual observation (content-aware)

For each step row, computes Spearman rank correlation between every pair of
baselines' attribution vectors (729 ext-cam patches). High rho (>0.8) means
attribution rankings are robust to baseline choice.

Usage:
    .venv/bin/python baseline_sensitivity.py \
        --metrics data/metrics_PickCube-v1_170m_seed42.jsonl \
        --task PickCube-v1 --model 170m --m 64
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy.stats import spearmanr

from per_step_attribution import prepare_ig_context, build_forward_fns
from integrated_gradients import integrated_gradients
from pipeline import load_pipeline, load_lang

EXT_CAM_START = 3 * 729
EXT_CAM_END = 4 * 729


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--model", choices=["170m", "1b"], default="170m")
    p.add_argument("--m", type=int, default=64)
    p.add_argument("--no-checkpoint", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def encode_6slot_baseline(vision_model, baseline_image, bg_image_encoded):
    proc = vision_model.image_processor
    px = proc.preprocess(baseline_image, return_tensors="pt")["pixel_values"][0]
    px = px.unsqueeze(0).to("cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        tokens = vision_model(px).detach()
    bg = bg_image_encoded.squeeze(0)
    bl = tokens.squeeze(0)
    return torch.stack([bg, bg, bg, bl, bg, bg]).reshape(1, -1, vision_model.hidden_size)


def reduce_vision(attr_tensor):
    per_pos = attr_tensor.abs().sum(dim=-1).squeeze(0)
    return per_pos[EXT_CAM_START:EXT_CAM_END].float().cpu().numpy()


def safe_spearman(a, b):
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    r, _ = spearmanr(a, b)
    return float(r)


def main():
    args = parse_args()

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
        args.out = f"data/metrics_baseline_sensitivity_{base}_{ts}.jsonl"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    pipe = load_pipeline(args.model, enable_checkpoint=not args.no_checkpoint)
    lang = load_lang(args.task)

    pairs = [("black", "gray"), ("black", "blur"), ("gray", "blur")]

    print(f"\n=== baseline sensitivity over {len(rows)} steps, m={args.m} ===")
    print(f"=== writing {args.out} ===\n")
    jsonl = open(args.out, "a")

    for i, row in enumerate(rows):
        t0 = time.time()
        sidecar = torch.load(row["attr_file"], weights_only=False, map_location="cpu")
        obs_image = Image.fromarray(sidecar["obs_image"])
        proprio = sidecar["proprio"]

        black_img = Image.new("RGB", (384, 384), (0, 0, 0))
        gray_img = Image.new("RGB", (384, 384), (128, 128, 128))
        blur_img = obs_image.filter(ImageFilter.GaussianBlur(radius=30))

        ctx = prepare_ig_context(
            pipe["runner"], pipe["vision_model"],
            obs_image, proprio,
            lang["lang_tokens"], lang["lang_attn_mask"], lang["lang_tokens_baseline"],
            pipe["bg_image_encoded"], pipe["img_tokens_baseline"],
            pipe["action_mask"], pipe["ctrl_freqs"],
            seed=row["seed"], sigma_sq=1.0,
        )
        fwd_v, _, _ = build_forward_fns(ctx)

        attrs = {}
        errs = {}
        for bname, bimg in [("black", black_img), ("gray", gray_img), ("blur", blur_img)]:
            bl_tokens = encode_6slot_baseline(
                pipe["vision_model"], bimg, pipe["bg_image_encoded"])
            with torch.no_grad():
                bl_adapted = pipe["runner"].img_adaptor(bl_tokens)

            attr = integrated_gradients(fwd_v, ctx["img_adapted"], bl_adapted, m=args.m)

            with torch.no_grad():
                f_input = fwd_v(ctx["img_adapted"]).item()
                f_base = fwd_v(bl_adapted).item()
            expected = f_input - f_base
            actual = attr.sum().item()
            err = abs(expected - actual) / abs(expected) * 100 if expected != 0 else float("inf")

            attrs[bname] = reduce_vision(attr)
            errs[bname] = err
            del attr, bl_tokens, bl_adapted
            torch.cuda.empty_cache()

        rhos = {}
        for a, b in pairs:
            rhos[f"rho_{a}_{b}"] = safe_spearman(attrs[a], attrs[b])

        wall = time.time() - t0
        out_row = {
            "event": "baseline_sensitivity",
            "task": args.task,
            "model": args.model,
            "episode": row["episode"],
            "seed": row["seed"],
            "policy_call_idx": row["policy_call_idx"],
            "m": args.m,
            "wall_seconds": wall,
            "err_black": errs["black"],
            "err_gray": errs["gray"],
            "err_blur": errs["blur"],
            **rhos,
        }
        jsonl.write(json.dumps(out_row) + "\n")
        jsonl.flush()
        print(f"  step {i+1}/{len(rows)} ep{row['episode']:03d}t{row['policy_call_idx']:02d}: "
              f"bk-gr={rhos['rho_black_gray']:.3f} "
              f"bk-bl={rhos['rho_black_blur']:.3f} "
              f"gr-bl={rhos['rho_gray_blur']:.3f} "
              f"wall={wall:.1f}s")

        del attrs, ctx, sidecar
        torch.cuda.empty_cache()

    jsonl.close()
    print(f"\ndone. wrote {args.out}")


if __name__ == "__main__":
    main()
