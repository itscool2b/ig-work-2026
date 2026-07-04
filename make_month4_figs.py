"""Generate Month 4 paper figures + table CSVs from data/. Run on the pod.
Writes to out/figures_m4/. Read-only on the JSONLs.

Figures:
  fig_e1_1b_vs_170m.png   grouped bars, vision/lang insertion & deletion AUC, 1B vs 170M
  fig_c1_variants.png     C1 moving/frozen/cascade vs C2, rho_vision per variant
  fig_baseline_sens.png   pairwise baseline rho (black-gray/black-blur/gray-blur) per task
  fig_m128_tail.png       vision %<=3% at m=64 vs m=128, all 4 tasks
Tables (CSV):
  table_e1.csv, table_c1.csv, table_baseline.csv, table_m128.csv, table_targets.csv
"""
import json, glob, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "out/figures_m4"
os.makedirs(OUT, exist_ok=True)


def load(p):
    rows = []
    for l in open(p):
        l = l.strip()
        if not l:
            continue
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            pass  # skip partial/corrupt lines (e.g. interrupted writes)
    return rows


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else float("nan")


def pct_le(xs, thr=0.03):
    xs = [x for x in xs if x is not None]
    return 100 * sum(1 for x in xs if x <= thr) / len(xs) if xs else float("nan")


def first(pat):
    g = glob.glob(pat)
    return g[0] if g else None


# ---------- E1: 1B faithfulness vs 170M ----------
def fig_e1():
    # Pool all committed 1B PickCube faithfulness seeds (42 main-pass + the Month 6
    # evaluation seeds 142/242). The 1B bars show the pooled median; the 170M bars
    # are the decommissioned main-pass record and stay hard-coded below.
    files = sorted(glob.glob("data/metrics_faithfulness_metrics_PickCube-v1_1b_seed*.jsonl"))
    if not files:
        print("E1: no 1B faithfulness file"); return
    r = []
    for f in files:
        r += [x for x in load(f) if x.get("event") == "faithfulness"]
    print(f"E1: pooled {len(r)} 1B faithfulness rows from {len(files)} seed file(s)")
    onebig = {
        "vis_ins": med([x["vision_insertion_auc"] for x in r]),
        "vis_del": med([x["vision_deletion_auc"] for x in r]),
        "lang_ins": med([x["lang_insertion_auc"] for x in r]),
        "lang_del": med([x["lang_deletion_auc"] for x in r]),
    }
    # 170M PickCube baseline from the Month-3 record, Table 2 (hard-coded)
    m170 = {"vis_ins": 0.856, "vis_del": 0.317, "lang_ins": 0.504, "lang_del": 0.477}
    labels = ["vision\nins", "vision\ndel", "lang\nins", "lang\ndel"]
    keys = ["vis_ins", "vis_del", "lang_ins", "lang_del"]
    x = range(len(keys)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - w/2 for i in x], [m170[k] for k in keys], w, label="170M", color="#88a")
    ax.bar([i + w/2 for i in x], [onebig[k] for k in keys], w, label="1B (authors')", color="#d85")
    ax.axhline(0.60, ls="--", c="g", lw=0.8, label="ins bar 0.60")
    ax.axhline(0.40, ls=":", c="r", lw=0.8, label="del bar 0.40")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("normalized AUC"); ax.set_title("E1: Faithfulness AUC, 170M vs 1B (PickCube)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_e1_1b_vs_170m.png", dpi=130); plt.close(fig)
    with open(f"{OUT}/table_e1.csv", "w") as o:
        o.write("metric,170M,1B\n")
        for k in keys:
            o.write(f"{k},{m170[k]:.3f},{onebig[k]:.3f}\n")
    print("E1 fig + table done")


# ---------- C1 variants ----------
def fig_c1():
    def pooled(pat, field):
        xs = []
        for f in glob.glob(pat):
            xs += [r.get(field) for r in load(f)]
        return med(xs)
    variants = {
        "C1 frozen\n(last layer)": ("data/metrics_sanity_C1_frozen_*.jsonl",),
        "C1 cascade\n(backbone)": ("data/metrics_sanity_C1_cascade_*.jsonl",),
    }
    vis = {k: pooled(p[0], "spearman_vision") for k, p in variants.items()}
    lang = {k: pooled(p[0], "spearman_language") for k, p in variants.items()}
    # moving-target C1 and C2 from the Month-3 record; C2 JSONLs were a
    # Month-3 run, not regenerated this round (pooled vision ~0.13, language ~0.12).
    vis = {"C1 moving\n(last layer)": 0.95, **vis, "C2\n(input shuffle)": 0.13}
    lang = {"C1 moving\n(last layer)": 0.95, **lang, "C2\n(input shuffle)": 0.12}
    ks = list(vis.keys()); x = range(len(ks)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - w/2 for i in x], [vis[k] for k in ks], w, label="ρ vision", color="#48a")
    ax.bar([i + w/2 for i in x], [lang[k] for k in ks], w, label="ρ language", color="#a84")
    ax.axhline(0.2, ls="--", c="r", lw=0.9, label="bar ρ≤0.2")
    ax.set_xticks(list(x)); ax.set_xticklabels(ks, fontsize=8)
    ax.set_ylabel("Spearman ρ (orig vs perturbed)")
    ax.set_title("C1 model-randomization: input-anchoring keeps ρ high; only C2 collapses")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_c1_variants.png", dpi=130); plt.close(fig)
    with open(f"{OUT}/table_c1.csv", "w") as o:
        o.write("variant,rho_vision,rho_language\n")
        for k in ks:
            o.write(f"{k.replace(chr(10),' ')},{vis[k]:.3f},{lang[k]:.3f}\n")
    print("C1 fig + table done")


# ---------- baseline sensitivity ----------
def fig_baseline():
    files = sorted(glob.glob("data/metrics_baseline_sensitivity_*.jsonl"))
    if not files:
        print("baseline: none"); return
    tasks, bg, bb, gb = [], [], [], []
    for f in files:
        r = load(f); tasks.append(r[0]["task"].replace("-v1", ""))
        bg.append(med([x.get("rho_black_gray") for x in r]))
        bb.append(med([x.get("rho_black_blur") for x in r]))
        gb.append(med([x.get("rho_gray_blur") for x in r]))
    x = range(len(tasks)); w = 0.26
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - w for i in x], bg, w, label="black–gray", color="#4a8")
    ax.bar(list(x), bb, w, label="black–blur", color="#a48")
    ax.bar([i + w for i in x], gb, w, label="gray–blur", color="#84a")
    ax.set_xticks(list(x)); ax.set_xticklabels(tasks)
    ax.set_ylabel("Spearman ρ (patch ranking)")
    ax.set_title("E4: baseline sensitivity — solid baselines agree, blur diverges")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_baseline_sens.png", dpi=130); plt.close(fig)
    with open(f"{OUT}/table_baseline.csv", "w") as o:
        o.write("task,black_gray,black_blur,gray_blur\n")
        for i, t in enumerate(tasks):
            o.write(f"{t},{bg[i]:.3f},{bb[i]:.3f},{gb[i]:.3f}\n")
    print("baseline fig + table done")


# ---------- m=128 tail ----------
def fig_m128():
    # Month-3 record m=64 vision %<=3% (Table 1) + Month-4 m=128
    m64 = {"PegInsertionSide": 75.8, "PickCube": 68.6, "PickSingleYCB": 69.8, "StackCube": 59.9}
    m128 = {"PickCube": 80.5, "StackCube": 89.8}
    for f in glob.glob("data/metrics_*_m128.jsonl"):
        if any(x in f for x in ["faithfulness", "sanity", "baseline"]):
            continue
        rows = [r for r in load(f)
                if r.get("event") == "step" and r.get("ref_norm_maniskill", 0) >= 15]
        if not rows:
            continue
        t = rows[0]["task"].replace("-v1", "")
        v = pct_le([r["vision_err"] for r in rows])
        m128[t] = (m128.get(t) + v) / 2 if t in m128 else v  # avg seeds
    tasks = ["PickCube", "StackCube", "PegInsertionSide", "PickSingleYCB"]
    x = range(len(tasks)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - w/2 for i in x], [m64.get(t, 0) for t in tasks], w, label="m=64", color="#88a")
    ax.bar([i + w/2 for i in x], [m128.get(t, 0) for t in tasks], w, label="m=128", color="#d85")
    ax.axhline(90, ls="--", c="g", lw=0.9, label="bar 90%")
    ax.set_xticks(list(x)); ax.set_xticklabels([t.replace("Side", "") for t in tasks], fontsize=8)
    ax.set_ylabel("vision % cases ≤3% completeness err")
    ax.set_title("E2: completeness tail at m=64 vs m=128 (m=64 from Month-3 record)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_m128_tail.png", dpi=130); plt.close(fig)
    with open(f"{OUT}/table_m128.csv", "w") as o:
        o.write("task,m64_vis_pct_le3,m128_vis_pct_le3\n")
        for t in tasks:
            o.write(f"{t},{m64.get(t,'NA')},{round(m128.get(t,float('nan')),1)}\n")
    print("m128 fig + table done")


# ---------- alt targets table: completeness + faithfulness ----------
def table_targets():
    def faith(tgt):
        # median vision Ins/Del AUC + dlogp@k5 over a target's faithfulness files
        vi, vd, vdl = [], [], []
        for f in glob.glob(f"data/metrics_faithfulness_*_{tgt}_*.jsonl"):
            for r in load(f):
                vi.append(r.get("vision_insertion_auc"))
                vd.append(r.get("vision_deletion_auc"))
                vdl.append(r.get("vision_dlogp_k5"))
        n = len([x for x in vi if x is not None])
        return med(vi), med(vd), med(vdl), n

    with open(f"{OUT}/table_targets.csv", "w") as o:
        o.write("target,median_abs_vision_gap,vis_pct_le3,n_compl,"
                "vision_ins_auc,vision_del_auc,vision_dlogp_k5,n_faith\n")
        for tgt in ["logpi", "l2", "maxdev"]:
            if tgt == "logpi":
                # logpi completeness comes from the base regeneration run
                # (seed*.jsonl), not the *_logpi matched faithfulness run. Exclude
                # every variant suffix, including _logpi, so the base files are not
                # double-counted with the later-added matched run.
                files = [f for f in glob.glob("data/metrics_PickCube-v1_170m_seed*.jsonl")
                         if not any(x in f for x in ["faithfulness", "sanity", "baseline",
                                                     "m128", "_l2", "_maxdev", "_cosine",
                                                     "_logpi"])]
            else:
                files = glob.glob(f"data/metrics_PickCube-v1_170m_seed*_{tgt}.jsonl")
            gaps, vpct = [], []
            for f in files:
                rows = [r for r in load(f)
                        if r.get("event") == "step" and r.get("ref_norm_maniskill", 0) >= 15]
                gaps += [abs(r.get("vision_gap", 0)) for r in rows]
                vpct += [r["vision_err"] for r in rows]
            fi, fd, fdl, fn = faith(tgt) if tgt != "logpi" else (float("nan"),) * 4
            if gaps:
                o.write(f"{tgt},{med(gaps):.4f},{pct_le(vpct):.1f},{len(gaps)},"
                        f"{fi:.3f},{fd:.3f},{fdl:.4f},{fn}\n")
            else:
                o.write(f"{tgt},NA,NA,0,NA,NA,NA,0\n")
    print("targets table done")


if __name__ == "__main__":
    fig_e1(); fig_c1(); fig_baseline(); fig_m128(); table_targets()
    print(f"\nall Month-4 figures + tables in {OUT}/")
