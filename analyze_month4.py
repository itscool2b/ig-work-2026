"""Aggregate Month 4 experiment numbers. Run on the pod
against data/. Read-only. Prints exact medians/percentages per experiment."""
import json, glob, statistics as st


def load(path):
    rows = []
    for l in open(path):
        l = l.strip()
        if not l:
            continue
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            continue
    return rows


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else float("nan")


def pct_le(xs, thr=0.03):
    xs = [x for x in xs if x is not None]
    return 100 * sum(1 for x in xs if x <= thr) / len(xs) if xs else float("nan")


def main():
    print("=" * 70)
    print("E1: 1B FAITHFULNESS (PickCube)")
    print("=" * 70)
    for f in glob.glob("data/metrics_faithfulness_*1b*.jsonl"):
        rows = load(f)
        n = len(rows)
        vi = med([r["vision_insertion_auc"] for r in rows])
        vd = med([r["vision_deletion_auc"] for r in rows])
        vdl = med([r["vision_dlogp_k5"] for r in rows])
        li = med([r["lang_insertion_auc"] for r in rows])
        ld = med([r["lang_deletion_auc"] for r in rows])
        ldl = med([r["lang_dlogp_k5"] for r in rows])
        print(f"  n={n}")
        print(f"  vision: insertion={vi:.3f} deletion={vd:.3f} dlogp@5={vdl:.5f}")
        print(f"  lang:   insertion={li:.3f} deletion={ld:.3f} dlogp@5={ldl:.5f}")

    print("\n" + "=" * 70)
    print("E3/cascade: C1 MODEL-RANDOMIZATION (pooled over tasks)")
    print("=" * 70)
    for variant, pat in [
        ("frozen  (last-layer reinit)", "data/metrics_sanity_C1_frozen_*.jsonl"),
        ("cascade (full backbone)    ", "data/metrics_sanity_C1_cascade_*.jsonl"),
    ]:
        av, al, as_ = [], [], []
        for f in glob.glob(pat):
            rows = load(f)
            av += [r.get("spearman_vision") for r in rows]
            al += [r.get("spearman_language") for r in rows]
            as_ += [r.get("spearman_state") for r in rows]
        nv = len([x for x in av if x is not None])
        print(f"  {variant}: rho_v={med(av):.3f} rho_l={med(al):.3f} "
              f"rho_s={med(as_):.3f} (n={nv})")

    print("\n" + "=" * 70)
    print("E4: BASELINE SENSITIVITY (per task)")
    print("=" * 70)
    for f in sorted(glob.glob("data/metrics_baseline_sensitivity_*.jsonl")):
        rows = load(f)
        task = rows[0]["task"]
        bg = med([r.get("rho_black_gray") for r in rows])
        bb = med([r.get("rho_black_blur") for r in rows])
        gb = med([r.get("rho_gray_blur") for r in rows])
        print(f"  {task}: black-gray={bg:.3f} black-blur={bb:.3f} "
              f"gray-blur={gb:.3f} (n={len(rows)})")

    print("\n" + "=" * 70)
    print("E2: m=128 COMPLETENESS (signal-bearing |ref|>=15)")
    print("=" * 70)
    for f in sorted(glob.glob("data/metrics_*_m128.jsonl")):
        if any(x in f for x in ["faithfulness", "sanity", "baseline"]):
            continue
        rows = [r for r in load(f)
                if r.get("event") == "step" and r.get("ref_norm_maniskill", 0) >= 15]
        if not rows:
            continue
        task = rows[0]["task"]
        seed = rows[0].get("seed")
        vmed = med([r["vision_err"] for r in rows])
        vpct = pct_le([r["vision_err"] for r in rows])
        lpct = pct_le([r["lang_err"] for r in rows])
        print(f"  {task} s{seed}: vis_med={vmed * 100:.2f}% "
              f"vis%<=3={vpct:.1f} lang%<=3={lpct:.1f} (n={len(rows)})")

    print("\n" + "=" * 70)
    print("TARGET-B: ALTERNATIVE IG TARGETS (PickCube, median expected_gap |ref|>=15)")
    print("=" * 70)
    for tgt in ["logpi", "l2", "l2sq", "maxdev", "cosine"]:
        if tgt == "logpi":
            # _logpi excluded so the Month 6 matched-run files are not
            # double-counted with the base logpi step records.
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
        if gaps:
            print(f"  {tgt:8s}: median |vision_gap|={med(gaps):.4f} "
                  f"vis%<=3={pct_le(vpct):.1f} (n={len(gaps)})")
        else:
            print(f"  {tgt:8s}: (no data yet)")


if __name__ == "__main__":
    main()
