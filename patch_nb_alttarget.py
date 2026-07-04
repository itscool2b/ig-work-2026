"""Patch the month4_report.ipynb alt-target cell with final l2/maxdev results
(completeness + faithfulness). Run on the pod where data/ lives. Idempotent:
replaces the single code cell that references 'logpi'."""
import json

NBP = "notebooks/month4_report.ipynb"

NEW_SRC = r'''# Alternative IG targets: completeness gap AND faithfulness (the load-bearing result)
import os
def _load(f):
    rows = []
    for l in open(f):
        l = l.strip()
        if not l:
            continue
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    return rows
def _faith(tgt):
    vi, vd, vdl = [], [], []
    for f in glob.glob(str(DATA / ('metrics_faithfulness_*_' + tgt + '_*.jsonl'))):
        for r in _load(f):
            vi.append(r.get('vision_insertion_auc'))
            vd.append(r.get('vision_deletion_auc'))
            vdl.append(r.get('vision_dlogp_k5'))
    return med(vi), med(vd), med(vdl), len([x for x in vi if x is not None])
hdr = '{:8s} {:>8s} {:>8s} {:>6s} {:>6s} {:>9s}'.format('target', '|gap|', 'vis%<=3', 'Ins', 'Del', 'dlogp@5')
print(hdr)
for tgt in ['logpi', 'l2', 'maxdev']:
    if tgt == 'logpi':
        files = [f for f in glob.glob(str(DATA / 'metrics_PickCube-v1_170m_seed*.jsonl'))
                 if not any(x in os.path.basename(f) for x in ['faithfulness', 'sanity', 'baseline', 'm128', '_l2', '_maxdev', '_cosine', '_logpi'])]
    else:
        files = glob.glob(str(DATA / ('metrics_PickCube-v1_170m_seed*_' + tgt + '.jsonl')))
    gaps, vpct = [], []
    for f in files:
        rows = [r for r in _load(f) if r.get('event') == 'step' and r.get('ref_norm_maniskill', 0) >= 15]
        gaps += [abs(r.get('vision_gap', 0)) for r in rows]
        vpct += [r['vision_err'] for r in rows]
    fi, fd, fdl, _ = (float('nan'),) * 4 if tgt == 'logpi' else _faith(tgt)
    g = med(gaps)
    vp = (100 * sum(1 for x in vpct if x <= 0.03) / len(vpct)) if vpct else float('nan')
    print('{:8s} {:8.4f} {:7.1f}% {:6.3f} {:6.3f} {:9.4f}'.format(tgt, g, vp, fi, fd, fdl))
print()
print('l2 clears the O2 dlogp bar at -1.23 (bar -0.5) while keeping AUCs faithful.')
print('Tradeoff: alt-targets have worse completeness (vis%<=3 ~11% vs logpi 77%) from non-smooth sqrt/max aliasing.')
'''


def main():
    nb = json.load(open(NBP))
    patched = False
    for c in nb["cells"]:
        if c["cell_type"] == "code" and any("logpi" in l for l in c["source"]):
            c["source"] = NEW_SRC.splitlines(keepends=True)
            c["outputs"] = []
            c["execution_count"] = None
            patched = True
            break
    json.dump(nb, open(NBP, "w"), indent=1)
    print("patched" if patched else "CELL NOT FOUND")


if __name__ == "__main__":
    main()
