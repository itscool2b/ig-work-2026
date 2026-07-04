"""Render Month 5 figures and CI-augmented tables.

Primary output: out/figures_m5/fig_displacement_vs_T.png, the direct
contraction-mechanism figure. It plots the median action displacement
||a_perturbed - a_original|| against the DPM-Solver++ step count T, one line per
(task, model), with 95% episode-bootstrap confidence bands. If contraction were
the mechanism, the curves would fall as T grows and the competent 1B would sit
below the 170M. The committed data refuted that prediction: the relative curves
do not fall with T and the 1B sits at or above the 170M at matched T (see the
displacement.py docstring and the paper's Section 6.1).

Companion: fig_displacement_vs_delfrac.png (displacement vs deletion fraction,
one line per T). Tables: table_displacement.csv plus the CI-augmented copies of
the Month 4 tables (delegated to bootstrap_ci.py --table all).

Reads data/metrics_displacement_*.jsonl. Reuses bootstrap_ci's verified loaders
(NUL-robust load, (task,model,seed,episode,call) dedup, episode grouping, the
episode-resampled percentile bootstrap). matplotlib Agg, no display needed.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import bootstrap_ci as B

REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(REPO, "data")
OUT = os.path.join(REPO, "out", "figures_m5")

VISION_DEL_FOR_T = 5      # the deletion fraction at which we plot displacement vs T
T_FOR_DELFRAC = 5         # the T at which we plot displacement vs deletion fraction


def load_displacement(modality="vision"):
    rows = B.dedup(B.load_glob(["*metrics_displacement_*.jsonl"]))
    return [r for r in rows if r.get("event") == "displacement"
            and r.get("modality") == modality]


def scalar_at_k(rows, k, field="l2_active"):
    """Project each displacement row to the scalar field value at deletion pct k,
    storing it back as r[f'{field}_at_k{k}'] so the generic bootstrap can resample
    it. Rows whose del_grid lacks k are dropped."""
    out = []
    key = f"{field}_at_k{k}"
    for r in rows:
        grid = r.get("del_grid", [])
        vals = r.get(field, [])
        if k in grid and len(vals) == len(grid):
            rr = dict(r)
            rr[key] = vals[grid.index(k)]
            out.append(rr)
    return out, key


def median_ci(rows, field, signal, B_n, seed):
    """Episode-bootstrap median + 95% CI of a scalar field over the given rows."""
    rows = B.signal_filter(rows, on=signal)
    groups = B.group_by_episode(rows)
    point, lo, hi, n = B.bootstrap_groups(groups, B.make_stat(field, "median"),
                                          B=B_n, seed=seed)
    return point, lo, hi, n


def groups_present(rows):
    keys = set()
    for r in rows:
        keys.add((r.get("task"), r.get("model")))
    # stable order: 170m before 1b, tasks alphabetical
    return sorted(keys, key=lambda tm: (tm[0] or "", tm[1] or ""))


def fig_displacement_vs_T(modality, B_n, seed):
    rows_all = load_displacement(modality)
    if not rows_all:
        print(f"  (no displacement data for modality={modality}; skipping fig)")
        return []
    series = groups_present(rows_all)
    Ts = sorted({r["solver_steps"] for r in rows_all})
    csv_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, field, ylab in [(axes[0], "l2_active", r"median $\|a_{pert}-a_{orig}\|_2$ (active dims)"),
                            (axes[1], "rel_active", "median relative displacement")]:
        for (task, model) in series:
            xs, ys, los, his = [], [], [], []
            for T in Ts:
                sub = [r for r in rows_all
                       if r.get("task") == task and r.get("model") == model
                       and r.get("solver_steps") == T]
                proj, key = scalar_at_k(sub, VISION_DEL_FOR_T, field)
                signal = (model != "1b")
                point, lo, hi, n = median_ci(proj, key, signal, B_n, seed)
                if not np.isnan(point):
                    xs.append(T); ys.append(point); los.append(lo); his.append(hi)
                    csv_rows.append(dict(table="displacement", task=task, model=model,
                                         field=f"{field}_at_k{VISION_DEL_FOR_T}", stat="median",
                                         point=point, ci_lo=lo, ci_hi=hi, n_groups=n,
                                         n_rows=len(proj), solver_steps=T))
            if xs:
                label = f"{task} {model}"
                line, = ax.plot(xs, ys, marker="o", label=label)
                ax.fill_between(xs, los, his, alpha=0.18, color=line.get_color())
        ax.set_xlabel("DPM-Solver++ steps T")
        ax.set_ylabel(ylab)
        ax.set_xscale("log")
        ax.set_xticks(Ts); ax.set_xticklabels([str(t) for t in Ts])
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle(f"Action displacement under top-{VISION_DEL_FOR_T}% {modality} deletion "
                 f"vs denoising steps", fontsize=10)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    sfx = "" if modality == "vision" else f"_{modality}"
    path = os.path.join(OUT, f"fig_displacement_vs_T{sfx}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return csv_rows


def fig_displacement_vs_delfrac(modality, B_n, seed):
    rows_all = load_displacement(modality)
    if not rows_all:
        return
    series = groups_present(rows_all)
    ks = sorted(set().union(*[set(r.get("del_grid", [])) for r in rows_all]))
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for (task, model) in series:
        xs, ys = [], []
        for k in ks:
            sub = [r for r in rows_all
                   if r.get("task") == task and r.get("model") == model
                   and r.get("solver_steps") == T_FOR_DELFRAC]
            proj, key = scalar_at_k(sub, k, "l2_active")
            signal = (model != "1b")
            point, _, _, _ = median_ci(proj, key, signal, max(1000, B_n // 4), seed)
            if not np.isnan(point):
                xs.append(k); ys.append(point)
        if xs:
            ax.plot(xs, ys, marker="o", label=f"{task} {model}")
    ax.set_xlabel("top-k deletion (%)")
    ax.set_ylabel(r"median $\|a_{pert}-a_{orig}\|_2$ (active dims)")
    ax.set_title(f"Displacement vs deletion fraction at T={T_FOR_DELFRAC}", fontsize=10)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    sfx = "" if modality == "vision" else f"_{modality}"
    path = os.path.join(OUT, f"fig_displacement_vs_delfrac{sfx}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def write_displacement_csv(csv_rows, sfx=""):
    #Non-vision modalities write suffixed paths so a language re-run cannot
    #clobber the committed vision figure and CSV that back the paper.
    if not csv_rows:
        return
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"table_displacement{sfx}.csv")
    cols = ["table", "task", "model", "field", "stat", "solver_steps",
            "point", "ci_lo", "ci_hi", "n_groups", "n_rows"]
    with open(path, "w") as o:
        o.write(",".join(cols) + "\n")
        for r in csv_rows:
            o.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modality", default="vision", choices=["vision", "language"])
    ap.add_argument("--skip-ci-tables", action="store_true",
                    help="Skip regenerating the Month 4 CI-augmented CSVs.")
    args = ap.parse_args()

    print("=== Month 5 figures ===")
    csv_rows = fig_displacement_vs_T(args.modality, args.B, args.seed)
    fig_displacement_vs_delfrac(args.modality, args.B, args.seed)
    write_displacement_csv(csv_rows,
                           "" if args.modality == "vision" else f"_{args.modality}")

    if not args.skip_ci_tables:
        print("=== CI-augmented Month 4 tables (via bootstrap_ci) ===")
        for table in ["faithfulness1b", "targets", "sanity", "baseline", "m128"]:
            try:
                rows = B.PRESETS[table](args.B, args.seed)
                B.write_csv(rows, os.path.join(OUT, f"table_{table}_ci.csv"))
                print(f"  wrote table_{table}_ci.csv ({len(rows)} cells)")
            except Exception as e:
                print(f"  ({table} skipped: {e})")
    print("done.")


if __name__ == "__main__":
    main()
