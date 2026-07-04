"""
Walk a per_step_ig.py JSONL and generate three artifact classes:

    out/overlays/<task>/ep{EP:03d}_t{T:02d}.png    (vision heatmap over obs)
    out/tokens/<task>/ep{EP:03d}_t{T:02d}.png      (language bar chart)
    out/episodes/<task>/ep{EP:03d}_summary.png     (4-6 key-frame stitch)

Also optionally writes the 3-panel combined figure used by ig_rdt.py to
    out/figures/<task>/ep{EP:03d}_t{T:02d}_3panel.png
    (controlled by --three-panel, off by default since it duplicates the
    information in the three separate PNGs).

Usage:
    .venv/bin/python generate_overlays.py \\
        --metrics data/metrics_PickCube-v1_170m_seed42_20260421_182705.jsonl \\
        --task PickCube-v1 \\
        --out-root out
"""

import argparse
import json
import os
from collections import defaultdict

import torch

import overlays


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--out-root", default="out")
    p.add_argument("--lang-embeds", default=None,
                   help="Path to data/lang_embeds/{task}.pt. If omitted, uses the "
                        "matching file in data/lang_embeds/.")
    p.add_argument("--three-panel", action="store_true",
                   help="Also write the 3-panel combined figure per step.")
    p.add_argument("--episode-frames", type=int, default=4,
                   help="Number of key frames per episode-summary PNG.")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    #Load language tokens (for token-bar labels) and attention mask.
    lang_path = args.lang_embeds or f"data/lang_embeds/{args.task}.pt"
    lang_data = torch.load(lang_path, weights_only=True)
    token_labels = lang_data.get("tokens", None)
    lang_attn_mask = lang_data.get("attn_mask", None)

    #Parse metrics rows.
    step_rows = []
    episode_outcomes = {}
    with open(args.metrics) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") == "episode_end":
                episode_outcomes[r["episode"]] = {
                    "success": bool(r.get("success", False)),
                    "env_steps": r.get("env_steps", 0),
                    "policy_calls": r.get("policy_calls", 0),
                }
                continue
            if "attr_file" not in r:
                continue
            step_rows.append(r)

    if args.limit is not None:
        step_rows = step_rows[:args.limit]
    print(f"rendering {len(step_rows)} step rows from {args.metrics}")

    overlay_dir = os.path.join(args.out_root, "overlays", args.task)
    tokens_dir  = os.path.join(args.out_root, "tokens", args.task)
    episode_dir = os.path.join(args.out_root, "episodes", args.task)
    figures_dir = os.path.join(args.out_root, "figures", args.task)
    for d in (overlay_dir, tokens_dir, episode_dir, figures_dir):
        os.makedirs(d, exist_ok=True)

    #Group by episode for per-episode summary at the end.
    by_episode = defaultdict(list)
    for row in step_rows:
        by_episode[row["episode"]].append(row)

    #Per-step artifacts.
    for i, row in enumerate(step_rows):
        sidecar = torch.load(row["attr_file"], weights_only=False, map_location="cpu")
        ep, t = row["episode"], row["policy_call_idx"]
        stem = f"ep{ep:03d}_t{t:02d}"

        overlays.render_overlay_only_png(
            sidecar, os.path.join(overlay_dir, f"{stem}.png"))
        overlays.render_tokens_only_figure(
            sidecar, token_labels, os.path.join(tokens_dir, f"{stem}.png"))
        if args.three_panel:
            outcome = episode_outcomes.get(ep, {})
            title = (f"{args.task} ep{ep} call{t} | "
                     f"|ref|={row['ref_norm_maniskill']:.1f} | "
                     f"success={outcome.get('success', '?')}")
            overlays.render_step_figure(
                sidecar, lang_attn_mask, token_labels,
                os.path.join(figures_dir, f"{stem}_3panel.png"),
                title=title)
        print(f"  [{i+1}/{len(step_rows)}] wrote {stem}.png")
        del sidecar

    #Per-episode summaries, loaded after single-step renders to batch the I/O.
    for ep, rows in sorted(by_episode.items()):
        rows = sorted(rows, key=lambda r: r["policy_call_idx"])
        sidecars = [torch.load(r["attr_file"], weights_only=False, map_location="cpu")
                    for r in rows]
        out_path = os.path.join(episode_dir, f"ep{ep:03d}_summary.png")
        overlays.render_episode_summary(
            sidecars, out_path, n_frames=args.episode_frames)
        print(f"  episode {ep} summary -> {out_path}")

    print(f"done. rendered to {args.out_root}/")


if __name__ == "__main__":
    main()
