"""Episode-resampled percentile bootstrap 95% CIs for the paper tables (Month 5).

numpy only (no scipy). Loads any metrics / faithfulness / sanity / displacement
JSONL, dedups (episode, policy_call_idx) keeping the LAST occurrence (the committed
data carries --resume duplicate rows), applies the signal-bearing filter where
appropriate, groups rows by (episode, seed), resamples GROUPS (episodes) with
replacement B times, and reports point + [2.5, 97.5] percentile CI of the
statistic computed on each resample.

The resampling unit is the EPISODE, because rows within an episode are correlated
and the row-level n overstates the effective sample size (paper Section Discussion,
Limitations). Fixed RNG seed makes every CI reproducible and quotable.

Self-test (no pod needed): `python bootstrap_ci.py --selftest` checks that the
un-resampled point equals the committed medians that audit.py / the paper report.
"""
import argparse
import glob
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(REPO, "data")


# ----------------------------------------------------------------------------- IO
def load_jsonl(path):
    """Binary read, skip empty/NUL-corrupted lines (mirrors audit.py)."""
    rows = []
    with open(path, "rb") as f:
        for raw in f:
            if raw.strip() == b"":
                continue
            try:
                rows.append(json.loads(raw))
            except Exception:
                continue
    return rows


def load_glob(patterns):
    rows = []
    for pat in patterns:
        full = pat if os.path.isabs(pat) else os.path.join(DATA, pat)
        for fn in sorted(glob.glob(full)):
            rows += load_jsonl(fn)
    return rows


def dedup(rows):
    """Keep the LAST row per (task, model, seed, episode, policy_call_idx). The key
    includes task/model/seed so pooling across task-seed files does NOT collapse
    distinct rows that happen to share (episode, policy_call_idx); the --resume
    duplicates we must drop are always WITHIN one task-seed file. Rows lacking
    policy_call_idx (e.g. episode_end) pass through unchanged, in order."""
    seen = {}
    passthrough = []
    for r in rows:
        if "policy_call_idx" in r and "episode" in r:
            # Include solver_steps and modality so displacement rows (12 per call:
            # 6 T x 2 modalities) are not collapsed; they default to None for
            # step/faithfulness/sanity rows so --resume duplicates still dedup.
            key = (r.get("task"), r.get("model"), r.get("seed"),
                   r["episode"], r["policy_call_idx"],
                   r.get("solver_steps"), r.get("modality"))
            seen[key] = r
        else:
            passthrough.append(r)
    return list(seen.values()) + passthrough


def signal_filter(rows, on=True):
    if not on:
        return [r for r in rows if r.get("event") != "episode_end"]
    return [
        r
        for r in rows
        if r.get("event") != "episode_end" and r.get("ref_norm_maniskill", -1) >= 15
    ]


def group_by_episode(rows):
    """Group rows into bootstrap resampling units. An episode (the correlated unit
    per the paper) is identified by (task, seed, episode) so pooled multi-task
    tables resample whole episodes across all tasks and seeds."""
    groups = {}
    for r in rows:
        key = (r.get("task"), r.get("seed"), r.get("episode"))
        groups.setdefault(key, []).append(r)
    return list(groups.values())


# -------------------------------------------------------------------------- stats
def _vals(rows, field):
    out = []
    for r in rows:
        v = r.get(field)
        if v is not None:
            out.append(v)
    return np.asarray(out, dtype=float)


def make_stat(field, stat):
    """Return a function pooled_rows -> float."""
    if stat == "median":
        return lambda rows: float(np.median(_vals(rows, field))) if len(_vals(rows, field)) else float("nan")
    if stat == "median_abs":
        return lambda rows: float(np.median(np.abs(_vals(rows, field)))) if len(_vals(rows, field)) else float("nan")
    if stat == "mean":
        return lambda rows: float(np.mean(_vals(rows, field))) if len(_vals(rows, field)) else float("nan")
    if stat == "pct_le3":
        return lambda rows: float(np.mean(_vals(rows, field) <= 0.03) * 100.0) if len(_vals(rows, field)) else float("nan")
    raise ValueError(f"unknown stat {stat}")


def bootstrap_groups(groups, stat_fn, B=10000, seed=0):
    """groups: list of row-lists (one per (episode,seed)). Resample episodes."""
    rng = np.random.default_rng(seed)
    n = len(groups)
    pooled_all = [r for g in groups for r in g]
    point = stat_fn(pooled_all)
    if n < 2:
        return point, point, point, n
    boots = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        pooled = [r for i in idx for r in groups[i]]
        boots[b] = stat_fn(pooled)
    lo, hi = np.percentile(boots[~np.isnan(boots)], [2.5, 97.5])
    return point, float(lo), float(hi), n


def ci(patterns, field, stat="median", signal=True, B=10000, seed=0):
    rows = dedup(load_glob(patterns))
    rows = signal_filter(rows, on=signal)
    groups = group_by_episode(rows)
    n_rows = sum(len(g) for g in groups)
    point, lo, hi, n_groups = bootstrap_groups(groups, make_stat(field, stat), B=B, seed=seed)
    return dict(field=field, stat=stat, point=point, ci_lo=lo, ci_hi=hi,
                n_groups=n_groups, n_rows=n_rows)


# ------------------------------------------------------------------------- presets
FAITH_FIELDS = ["vision_insertion_auc", "vision_deletion_auc", "vision_dlogp_k5",
                "lang_insertion_auc", "lang_deletion_auc", "lang_dlogp_k5"]


ONEB_SEEDS = (42, 142, 242)


def _faith1b_pats(s):
    return [f"metrics_faithfulness_metrics_PickCube-v1_1b_seed{s}_*.jsonl"]


def table_faithfulness1b(B, seed):
    """RDT-1B PickCube faithfulness, per seed plus a pooled row across the seeds
    that have committed files. Seed 42 is the original main-pass scale check; 142
    and 242 are the Month 6 evaluation-seed additions. Pooling resamples whole
    episodes keyed on (task, seed, episode), so the seeds do not collide. The
    pooled interval measures rollout/sampling variance on this one checkpoint, not
    generalization across training initializations."""
    seeds = [s for s in ONEB_SEEDS if load_glob(_faith1b_pats(s))]
    out = []
    for s in seeds:
        for f in FAITH_FIELDS:
            out.append(dict(table="faithfulness1b", task="PickCube", model=f"1b_s{s}",
                            **ci(_faith1b_pats(s), f, "median", signal=False, B=B, seed=seed)))
    if len(seeds) > 1:
        pooled = [p for s in seeds for p in _faith1b_pats(s)]
        for f in FAITH_FIELDS:
            out.append(dict(table="faithfulness1b", task="PickCube", model="1b_pooled",
                            **ci(pooled, f, "median", signal=False, B=B, seed=seed)))
    return out


def table_targets(B, seed):
    out = []
    for suffix, label in [("", "logpi"), ("_l2", "l2"), ("_maxdev", "maxdev")]:
        base = [f"metrics_PickCube-v1_170m_seed{s}{suffix}.jsonl" for s in (42, 142)]
        out.append(dict(table="targets", task="PickCube", model=f"170m_{label}",
                        **ci(base, "vision_gap", "median_abs", signal=True, B=B, seed=seed)))
        out.append(dict(table="targets", task="PickCube", model=f"170m_{label}",
                        **ci(base, "vision_err", "pct_le3", signal=True, B=B, seed=seed)))
        faith = [f"metrics_faithfulness_metrics_PickCube-v1_170m_seed*{suffix}_*.jsonl"]
        if label == "logpi":
            # logpi faithfulness on the regen population is the Month 6 matched run; skip if absent
            faith = [f"metrics_faithfulness_metrics_PickCube-v1_170m_seed*_logpi_*.jsonl"]
        for fld in ["vision_insertion_auc", "vision_deletion_auc", "vision_dlogp_k5"]:
            if load_glob(faith):
                out.append(dict(table="targets", task="PickCube", model=f"170m_{label}",
                                **ci(faith, fld, "median", signal=False, B=B, seed=seed)))
    return out


def table_sanity(B, seed):
    out = []
    for kind in ["frozen", "cascade"]:
        pats = [f"metrics_sanity_C1_{kind}_*.jsonl"]
        for fld in ["spearman_vision", "spearman_language", "spearman_state"]:
            out.append(dict(table="sanity", task="pooled", model=kind,
                            **ci(pats, fld, "median", signal=False, B=B, seed=seed)))
    return out


def table_baseline(B, seed):
    out = []
    for task in ["PickCube", "StackCube"]:
        pats = [f"metrics_baseline_sensitivity_metrics_{task}-v1_*.jsonl"]
        for fld in ["rho_black_gray", "rho_black_blur", "rho_gray_blur"]:
            out.append(dict(table="baseline", task=task, model="170m",
                            **ci(pats, fld, "median", signal=False, B=B, seed=seed)))
    return out


def table_m128(B, seed):
    out = []
    for task in ["PegInsertionSide", "PickSingleYCB"]:
        for s in (42, 142):
            pats = [f"metrics_{task}-v1_170m_seed{s}_m128.jsonl"]
            out.append(dict(table="m128", task=f"{task}_s{s}", model="170m",
                            **ci(pats, "vision_err", "pct_le3", signal=True, B=B, seed=seed)))
    return out


def table_displacement(B, seed):
    pats = ["*metrics_displacement_*.jsonl"]
    if not load_glob(pats):
        return []
    # displacement rows carry list fields; CI on the median final-T l2_active is
    # computed in make_month5_figs.py where the (T,k) structure is unpacked. Here
    # we only emit a presence note.
    return [dict(table="displacement", task="-", model="-", field="(see make_month5_figs)",
                 stat="-", point=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                 n_groups=0, n_rows=len(load_glob(pats)))]


PRESETS = {
    "faithfulness1b": table_faithfulness1b,
    "targets": table_targets,
    "sanity": table_sanity,
    "baseline": table_baseline,
    "m128": table_m128,
    "displacement": table_displacement,
}


def write_csv(rows, path):
    cols = ["table", "task", "model", "field", "stat", "point", "ci_lo", "ci_hi",
            "n_groups", "n_rows"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as o:
        o.write(",".join(cols) + "\n")
        for r in rows:
            o.write(",".join(str(r.get(c, "")) for c in cols) + "\n")


def selftest():
    """Confirm the un-resampled point reproduces the committed/audited medians.
    The committed truth is the seed-42 1B row, so the checks look it up by model
    label rather than by position, since the table now also carries 142/242 and a
    pooled row when those files exist."""
    rows = table_faithfulness1b(200, 0)

    def s42(field):
        for r in rows:
            if r["model"] == "1b_s42" and r["field"] == field:
                return r
        raise AssertionError(f"no seed-42 1B row for {field}")

    checks = [
        ("faithfulness1b vision_ins", s42("vision_insertion_auc"), 0.926063, "point"),
        ("faithfulness1b vision_del", s42("vision_deletion_auc"), 0.247621, "point"),
    ]
    ok = True
    for name, row, expected, key in checks:
        got = row[key]
        good = abs(got - expected) < 1e-4
        ok = ok and good
        print(f"  [{'OK' if good else 'FAIL'}] {name}: point={got:.6f} expected={expected:.6f} "
              f"(95% CI [{row['ci_lo']:.4f}, {row['ci_hi']:.4f}], n_groups={row['n_groups']})")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", choices=list(PRESETS) + ["all"])
    ap.add_argument("--glob", action="append", help="JSONL glob(s) for manual mode")
    ap.add_argument("--field", action="append", help="metric field(s) for manual mode")
    ap.add_argument("--stat", default="median", choices=["median", "median_abs", "mean", "pct_le3"])
    ap.add_argument("--no-signal-filter", action="store_true")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        ok = selftest()
        raise SystemExit(0 if ok else 1)

    rows = []
    if args.table == "all":
        for name, fn in PRESETS.items():
            rows += fn(args.B, args.seed)
    elif args.table:
        rows = PRESETS[args.table](args.B, args.seed)
    elif args.glob and args.field:
        for fld in args.field:
            rows.append(dict(table="manual", task="-", model="-",
                             **ci(args.glob, fld, args.stat, signal=not args.no_signal_filter,
                                  B=args.B, seed=args.seed)))
    else:
        ap.error("provide --table, or --glob with --field, or --selftest")

    for r in rows:
        print(f"  {r['table']:16s} {r.get('task','-'):16s} {r.get('model','-'):12s} "
              f"{r['field']:22s} {r['stat']:10s} point={r['point']:.5f} "
              f"CI=[{r['ci_lo']:.5f}, {r['ci_hi']:.5f}] n_grp={r['n_groups']} n={r['n_rows']}")
    if args.out:
        write_csv(rows, args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
