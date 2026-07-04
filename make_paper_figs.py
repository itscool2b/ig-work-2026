"""Generate the paper's figure PNGs from committed records.

Outputs (written into paper/figures/, fig_a also into out/figures/):
  fig_a_completeness.png   2x2 completeness-error histograms from the committed
                           regeneration run, with m=128 overlays on the two tasks
                           that have committed m=128 records and an RDT-1B overlay
                           on the PickCube panel.
  fig_1b_vs_170m.png       restyled 300-DPI render of out/figures_m4/table_e1.csv.
  fig_c1_variants.png      restyled 300-DPI render of out/figures_m4/table_c1.csv.
  fig_c_auc_curves.png     legibility re-tile (1x2 strip -> 2x1 stack) of the
                           preserved main-pass artifact. The source data for this
                           figure is decommissioned, so the pixels are rearranged
                           but never regenerated.
  fig_displacement_vs_T.png  paper-sized 2x1 re-render of the displacement curves
                           from out/figures_m5/table_displacement.csv (medians and
                           95% episode-bootstrap bands, same values as the pipeline
                           figure that make_month5_figs.py writes).

Every other input is a committed JSONL or CSV, so each paper figure has a
committed generator. CPU-only, no pandas.
"""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from bootstrap_ci import dedup

REPO = os.path.dirname(os.path.abspath(__file__))
PAPER_FIGS = os.path.join(REPO, "paper", "figures")
OUT_FIGS = os.path.join(REPO, "out", "figures")
M4 = os.path.join(REPO, "out", "figures_m4")

TASKS = ["PegInsertionSide-v1", "PickCube-v1", "PickSingleYCB-v1", "StackCube-v1"]
ERR_FIELDS = [("vision_err", "vision"), ("lang_err", "language"), ("state_err", "state")]


def load_steps(path):
    rows = []
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("event") == "episode_end":
                continue
            rows.append(r)
    return rows


def clipped(rows, field):
    vals = [r[field] for r in rows if r.get(field) is not None]
    return np.clip(np.asarray(vals, dtype=float), 0.0, 0.5)


def fig_a():
    bins = np.linspace(0, 0.5, 30)
    fig, axes = plt.subplots(2, 2, figsize=(5.2, 4.4), dpi=200)
    for ax, task in zip(axes.flat, TASKS):
        regen = []
        for seed in (42, 142):
            regen += load_steps(os.path.join(REPO, "data", f"metrics_{task}_170m_seed{seed}.jsonl"))
        regen = dedup(regen)
        for (field, label), color in zip(ERR_FIELDS, ("C0", "C1", "C2")):
            ax.hist(clipped(regen, field), bins=bins, alpha=0.5, color=color,
                    label=f"{label} (m=64)")
        overlay, tag = [], None
        m128_files = sorted(glob.glob(os.path.join(REPO, "data", f"metrics_{task}_170m_seed*_m128.jsonl")))
        if m128_files:
            for f in m128_files:
                overlay += load_steps(f)
            tag = "m=128"
        elif task == "PickCube-v1":
            overlay = load_steps(os.path.join(REPO, "data", "metrics_PickCube-v1_1b_seed42.jsonl"))
            tag = "RDT-1B"
        if overlay:
            overlay = dedup(overlay)
            for (field, label), color in zip(ERR_FIELDS, ("C0", "C1", "C2")):
                ax.hist(clipped(overlay, field), bins=bins, histtype="step",
                        linewidth=1.5, color=color, label=f"{label} ({tag})")
        ax.axvline(0.03, color="k", ls="--", lw=0.9)
        ax.set_title(task.replace("-v1", ""), fontsize=10)
        ax.set_xlabel("completeness error (clipped at 0.5)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, framealpha=0.85)
    fig.tight_layout()
    for dest in (OUT_FIGS, PAPER_FIGS):
        path = os.path.join(dest, "fig_a_completeness.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  wrote {path}")
    plt.close(fig)


def fig_e1():
    with open(os.path.join(M4, "table_e1.csv")) as f:
        rows = {r["metric"]: r for r in csv.DictReader(f)}
    order = ["vis_ins", "vis_del", "lang_ins", "lang_del"]
    labels = ["vision\ninsertion", "vision\ndeletion", "language\ninsertion", "language\ndeletion"]
    v170 = [float(rows[k]["170M"]) for k in order]
    v1b = [float(rows[k]["1B"]) for k in order]
    x = np.arange(4)
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    ax.bar(x - w / 2, v170, w, color="#8888aa", label="RDT-170M")
    ax.bar(x + w / 2, v1b, w, color="#dd8855", label="RDT-1B (authors' weights)")
    ax.axhline(0.60, color="green", ls="--", lw=1.2, label="insertion bar 0.60")
    ax.axhline(0.40, color="red", ls=":", lw=1.4, label="deletion bar 0.40")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("normalized AUC")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    path = os.path.join(PAPER_FIGS, "fig_1b_vs_170m.png")
    fig.savefig(path, dpi=300)
    print(f"  wrote {path}")
    plt.close(fig)


def fig_c1():
    with open(os.path.join(M4, "table_c1.csv")) as f:
        rows = list(csv.DictReader(f))
    labels = {
        "C1 moving (last layer)": "C1 moving\n(last layer)",
        "C1 frozen (last layer)": "C1 frozen\n(last layer)",
        "C1 cascade (backbone)": "C1 cascade\n(backbone)",
        "C2 (input shuffle)": "C2\n(input shuffle)",
    }
    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.bar(x - w / 2, [float(r["rho_vision"]) for r in rows], w,
           color="#4488aa", label=r"$\rho$ vision")
    ax.bar(x + w / 2, [float(r["rho_language"]) for r in rows], w,
           color="#aa8844", label=r"$\rho$ language")
    ax.axhline(0.2, color="red", ls="--", lw=1.2, label=r"bar $\rho \leq 0.2$")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[r["variant"]] for r in rows])
    ax.set_ylabel(r"Spearman $\rho$ (original vs perturbed)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(PAPER_FIGS, "fig_c1_variants.png")
    fig.savefig(path, dpi=300)
    print(f"  wrote {path}")
    plt.close(fig)


def fig_displacement():
    with open(os.path.join(REPO, "out", "figures_m5", "table_displacement.csv")) as f:
        rows = [r for r in csv.DictReader(f)]
    series = sorted({(r["task"], r["model"]) for r in rows})
    fig, axes = plt.subplots(2, 1, figsize=(4.6, 6.0), dpi=200)
    panels = [
        ("l2_active_at_k5", r"median $\|a_{\mathrm{pert}}-a_{\mathrm{orig}}\|_2$ (active dims)"),
        ("rel_active_at_k5", "median relative displacement"),
    ]
    for ax, (field, ylab) in zip(axes, panels):
        for idx, (task, model) in enumerate(series):
            sel = sorted((r for r in rows if r["task"] == task and r["model"] == model
                          and r["field"] == field), key=lambda r: int(r["solver_steps"]))
            xs = [int(r["solver_steps"]) for r in sel]
            ys = [float(r["point"]) for r in sel]
            los = [float(r["ci_lo"]) for r in sel]
            his = [float(r["ci_hi"]) for r in sel]
            label = f"{task.replace('-v1', '')} {model.replace('170m', '170M').replace('1b', '1B')}"
            line, = ax.plot(xs, ys, marker="o", markersize=4, color=f"C{idx}", label=label)
            ax.fill_between(xs, los, his, alpha=0.15, color=line.get_color())
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 3, 5, 10, 20])
        ax.set_xticklabels(["1", "2", "3", "5", "10", "20"], fontsize=9)
        ax.set_xlabel("DPM-Solver++ steps T", fontsize=10)
        ax.set_ylabel(ylab, fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    path = os.path.join(PAPER_FIGS, "fig_displacement_vs_T.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  wrote {path}")
    plt.close(fig)


def fig_c_retile():
    src = os.path.join(OUT_FIGS, "fig_c_auc_curves.png")
    img = Image.open(src).convert("RGB")
    split = 892
    left = img.crop((0, 0, split, img.height))
    right = img.crop((split, 0, img.width, img.height))
    width = max(left.width, right.width)
    canvas = Image.new("RGB", (width, left.height + right.height), "white")
    canvas.paste(left, ((width - left.width) // 2, 0))
    canvas.paste(right, ((width - right.width) // 2, left.height))
    path = os.path.join(PAPER_FIGS, "fig_c_auc_curves.png")
    canvas.save(path)
    print(f"  wrote {path} ({canvas.width}x{canvas.height})")


def main():
    print("=== paper figures ===")
    fig_a()
    fig_e1()
    fig_c1()
    fig_displacement()
    fig_c_retile()
    print("done.")


if __name__ == "__main__":
    main()
