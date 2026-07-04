"""
Build out/annotations.csv (V2 annotation output).

16 rows = 2 PickCube successes (1B authors' weights) + 2 PickCube failures (170M)
        + 4 failures × 3 other tasks (170M).

Source data:
  data/metrics_*_170m_seed*.jsonl       # step rows for 170M failures
  data/metrics_PickCube-v1_1b_seed42.jsonl  # step + episode_end rows for 1B successes
  data/metrics_faithfulness_*.jsonl     # dlogp_top5_img (vision_dlogp_k5)

Columns:
  task, episode, t, category, note, dlogp_top5_img, complete_err

Categories (5 values: 4 failure modes + 1 success tag for 1B rows):
  vision_mis_focus       - vision_err dominates, vision IG signature noisy
  language_mis_grounding - lang_err dominates
  state_misreading       - state_err dominates
  delayed_credit         - all three errs similar in magnitude (IG diffuse, no clear modality)
  success_well_grounded  - 1B success row, low completeness across modalities

The category column is assigned algorithmically from the JSONL completeness fields
(rather than by manual overlay inspection) because sidecars are not on the local
machine and the overlay PNGs exist for every row anyway.

complete_err = vision_err (the headline O2 modality; the report scaffold names it
complete_err_img).
"""

from __future__ import annotations

import csv
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA = REPO / "data"
OUT = REPO / "out"
OUT_CSV = OUT / "annotations.csv"

TASKS = ["PickCube-v1", "StackCube-v1", "PegInsertionSide-v1", "PickSingleYCB-v1"]
SIGNAL_BEARING_REF = 15.0  # |ref| threshold for failure picks (170M)
# 1B success picks intentionally drop this filter; 1B successes have small action
# magnitudes throughout (the policy executes precise, low-magnitude moves), so all
# 1B success steps are sub-15 by the 170M-tuned threshold.

# Failure-mode picking weights: each category's defining metric is the modality
# whose completeness error is the largest. delayed_credit is "no single modality
# dominates" - quantified as min(v,l,s)/max(v,l,s) > 0.5 (errors are within 2x).


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                #data/README documents one NUL-corrupted line in the released
                #records; skip rather than crash.
                continue
    return rows


def collect_step_rows(model: str, task: str) -> list[dict]:
    rows: list[dict] = []
    if model == "1b" and task == "PickCube-v1":
        for r in load_jsonl(str(DATA / "metrics_PickCube-v1_1b_seed42.jsonl")):
            if r.get("event") == "step":
                rows.append(r)
    else:
        for jf in sorted(glob.glob(str(DATA / f"metrics_{task}_{model}_seed*.jsonl"))):
            #Skip derived/ablation records: this taxonomy reads only the base
            #step runs (the _l2/_maxdev target-ablation files and the Month 6
            #_logpi matched-run files were committed after this script's
            #Month 3 run; sweeping them in would duplicate near-identical rows).
            if any(tag in jf for tag in ("_m128", "faithfulness", "sanity",
                                         "_l2", "_maxdev", "_logpi")):
                continue
            for r in load_jsonl(jf):
                if r.get("event") == "step":
                    rows.append(r)
    return rows


def collect_episode_end_rows(jsonl_path: Path) -> list[dict]:
    return [r for r in load_jsonl(str(jsonl_path)) if r.get("event") == "episode_end"]


def collect_faithfulness_index() -> dict[tuple[str, str, int, int, int], dict]:
    """Index faithfulness rows by (task, model, seed, episode, policy_call_idx).

    Skips the l2/maxdev target-ablation, m=128, and Month 6 _logpi matched-run
    faithfulness files: their dlogp values are on different target scales,
    integration budgets, or IG runs than the base step rows this taxonomy
    annotates, and seed is part of the key so seed-42 and seed-142 runs never
    collide.
    """
    idx: dict[tuple[str, str, int, int, int], dict] = {}
    for jf in sorted(glob.glob(str(DATA / "metrics_faithfulness_*.jsonl"))):
        if any(tag in jf for tag in ("_l2", "_maxdev", "_m128", "_logpi")):
            continue
        for r in load_jsonl(jf):
            if r.get("event") == "faithfulness":
                key = (r["task"], r["model"], r["seed"],
                       r["episode"], r["policy_call_idx"])
                idx[key] = r
    return idx


def categorize_failure(step: dict) -> str:
    """Pick the failure category for a step row from its 3 completeness errors."""
    v, l, s = step["vision_err"], step["lang_err"], step["state_err"]
    errs = [("vision_mis_focus", v), ("language_mis_grounding", l), ("state_misreading", s)]
    errs.sort(key=lambda x: -x[1])
    top_name, top_val = errs[0]
    bot_val = errs[2][1]
    if top_val > 0 and bot_val / top_val > 0.5:
        # All three errors within 2x of each other - no single modality dominates.
        return "delayed_credit"
    return top_name


def write_note(step: dict, category: str, faith: dict | None) -> str:
    v = step["vision_err"]
    l = step["lang_err"]
    s = step["state_err"]
    ref = step["ref_norm_maniskill"]
    if category == "vision_mis_focus":
        return f"vision_err={v:.2f} dominant (l={l:.2f} s={s:.2f}); |ref|={ref:.1f}"
    if category == "language_mis_grounding":
        return f"lang_err={l:.2f} dominant (v={v:.2f} s={s:.2f}); short instruction; |ref|={ref:.1f}"
    if category == "state_misreading":
        return f"state_err={s:.2f} dominant (v={v:.2f} l={l:.2f}); 8-dim noise plausible; |ref|={ref:.1f}"
    if category == "delayed_credit":
        return f"all modalities similar (v={v:.2f} l={l:.2f} s={s:.2f}); IG diffuse; |ref|={ref:.1f}"
    if category == "success_well_grounded":
        return f"1B authors' weights success; v={v:.3f} l={l:.3f} s={s:.3f} |ref|={ref:.1f}"
    return "uncategorized"


def pick_failures(step_rows: list[dict], n: int = 4) -> list[tuple[dict, str]]:
    """Pick n failure rows, one per category where possible, prefer signal-bearing."""
    signal_bearing = [r for r in step_rows if r["ref_norm_maniskill"] >= SIGNAL_BEARING_REF]
    pool = signal_bearing or step_rows  # fallback if no signal-bearing exist
    picks_by_cat: dict[str, dict] = {}
    cat_priority = ["vision_mis_focus", "language_mis_grounding", "state_misreading", "delayed_credit"]
    # Sort each category's candidates by the defining metric (descending), then pick top.
    metric_for_cat = {
        "vision_mis_focus": "vision_err",
        "language_mis_grounding": "lang_err",
        "state_misreading": "state_err",
    }
    for cat in cat_priority:
        if cat in metric_for_cat:
            metric = metric_for_cat[cat]
            cands = sorted(
                (r for r in pool if categorize_failure(r) == cat and r.get("attr_file")),
                key=lambda r: -r[metric],
            )
        else:  # delayed_credit
            cands = sorted(
                (r for r in pool if categorize_failure(r) == cat),
                key=lambda r: -max(r["vision_err"], r["lang_err"], r["state_err"]),
            )
        # Avoid duplicate (episode, t) across categories.
        for c in cands:
            key = (c["episode"], c["policy_call_idx"])
            if key in {(p["episode"], p["policy_call_idx"]) for p in picks_by_cat.values()}:
                continue
            picks_by_cat[cat] = c
            break
    # If we have fewer than n, fill with the next-largest-err rows we haven't picked.
    out: list[tuple[dict, str]] = [(picks_by_cat[c], c) for c in cat_priority if c in picks_by_cat]
    if len(out) < n:
        used = {(p[0]["episode"], p[0]["policy_call_idx"]) for p in out}
        backup = sorted(
            (r for r in pool if (r["episode"], r["policy_call_idx"]) not in used),
            key=lambda r: -max(r["vision_err"], r["lang_err"], r["state_err"]),
        )
        for r in backup:
            if len(out) >= n:
                break
            cat = categorize_failure(r)
            out.append((r, cat))
    return out[:n]


def pick_1b_pickcube_successes(n: int = 2) -> list[tuple[dict, str]]:
    """Pick n 1B PickCube success-episode rows. Prefer mid-trajectory (t > 0)."""
    ep_end = collect_episode_end_rows(DATA / "metrics_PickCube-v1_1b_seed42.jsonl")
    success_eps = {r["episode"] for r in ep_end if r.get("success") is True}
    step_rows = collect_step_rows("1b", "PickCube-v1")
    candidates = [r for r in step_rows if r["episode"] in success_eps]
    # Prefer t >= 1 (mid-trajectory, robot in motion toward cube) and lowest completeness err.
    mid = [r for r in candidates if r["policy_call_idx"] >= 1]
    pool = mid or candidates
    pool.sort(key=lambda r: max(r["vision_err"], r["lang_err"], r["state_err"]))

    # Pick from different episodes to give a more representative pair.
    out: list[tuple[dict, str]] = []
    seen_eps: set[int] = set()
    for r in pool:
        if r["episode"] in seen_eps:
            continue
        out.append((r, "success_well_grounded"))
        seen_eps.add(r["episode"])
        if len(out) == n:
            break
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    faith_idx = collect_faithfulness_index()
    print(f"loaded {len(faith_idx)} faithfulness rows")

    rows: list[dict] = []

    # PickCube: 2 1B successes + 2 170M failures.
    pc_success = pick_1b_pickcube_successes(n=2)
    pc_fail = pick_failures(collect_step_rows("170m", "PickCube-v1"), n=2)
    for step, cat in pc_success + pc_fail:
        model = "1b" if cat == "success_well_grounded" else "170m"
        key = (step["task"], model, step["seed"],
               step["episode"], step["policy_call_idx"])
        faith = faith_idx.get(key)
        rows.append(_emit(step, cat, faith))

    # Other 3 tasks: 4 170M failures each.
    for task in ["StackCube-v1", "PegInsertionSide-v1", "PickSingleYCB-v1"]:
        fails = pick_failures(collect_step_rows("170m", task), n=4)
        for step, cat in fails:
            key = (step["task"], "170m", step["seed"],
                   step["episode"], step["policy_call_idx"])
            faith = faith_idx.get(key)
            rows.append(_emit(step, cat, faith))

    print(f"selected {len(rows)} rows total")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["task", "episode", "t", "category", "note", "dlogp_top5_img", "complete_err"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CSV}")

    # Echo a per-task summary.
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    for task in TASKS:
        cats = [r["category"] for r in by_task[task]]
        print(f"  {task}: {len(by_task[task])} rows  {cats}")


def _emit(step: dict, category: str, faith: dict | None) -> dict:
    if faith:
        dlogp_top5_str = f"{float(faith['vision_dlogp_k5']):.6f}"
    else:
        # No logpi faithfulness coverage for this policy call in the released
        # records (the original m=64 faithfulness files are not redistributed).
        dlogp_top5_str = ""
    return {
        "task": step["task"],
        "episode": step["episode"],
        "t": step["policy_call_idx"],
        "category": category,
        "note": write_note(step, category, faith),
        "dlogp_top5_img": dlogp_top5_str,
        "complete_err": f"{step['vision_err']:.4f}",
    }


if __name__ == "__main__":
    main()
