#!/usr/bin/env python3
"""Recompute the two statistics the review could not reproduce."""
import glob, json
import numpy as np
from scipy import stats

def load(pats, floor):
    rows = []
    for p in pats:
        for f in sorted(glob.glob(f"runs/_d3il/{p}")): rows += json.load(open(f))
    return [r for r in rows if r["SR"] >= floor]
def tvu(r):
    q = np.asarray(r.get("C_ref", []), float)
    return float(0.5*np.abs(q-1.0/q.size).sum()) if q.size else np.nan

print("1  Section 4.5 regression, on the CURRENT endpoint tv_to_ref")
A = load(["mb_skew*.json"], 0.15)
bbs = sorted({r.get("bb_seed") for r in A} - {None}); sk = sorted({r["skew"] for r in A})
xs, ys = [], []
for s_ in sk:
    for bb in bbs:
        v = [r for r in A if r["skew"]==s_ and r.get("bb_seed")==bb and r["arm"]=="mop"]
        w = [r for r in A if r["skew"]==s_ and r.get("bb_seed")==bb and r["arm"]=="mi"]
        if len(v)>1 and len(w)>1:
            xs.append(np.mean([tvu(r) for r in v]))
            ys.append(np.mean([r["tv_to_ref"] for r in w]) - np.mean([r["tv_to_ref"] for r in v]))
sl, ic, r_, p_, se = stats.linregress(xs, ys)
print(f"   slope={sl:+.3f}+-{se:.3f}  intercept={ic:+.3f}  r={r_:+.3f}  p={p_:.5f}  n={len(xs)}")
ise = se * np.sqrt(np.mean(np.square(xs)))
print(f"   intercept 95% CI approx [{ic-1.96*ise:+.3f}, {ic+1.96*ise:+.3f}]")
print(f"   lowest x observed = {min(xs):.3f}\n")

print("2  Section 4.9, JS vs TV on the CURRENT endpoint")
for task, tvp, jsp, fl in (("Avoiding", ["mb_skew*.json"], ["js_skew*.json"], 0.15),
                           ("Pushing", ["pushrw_skew*.json"], ["jsP_skew*.json"], 0.10)):
    tvr = np.array([r["tv_to_ref"] for r in load(tvp, fl) if r["arm"]=="mop"])
    jsr = np.array([r["tv_to_ref"] for r in load(jsp, fl) if r["arm"]=="mop"])
    t, p2 = stats.ttest_ind(jsr, tvr, equal_var=False)
    pooled = np.sqrt((tvr.var(ddof=1)+jsr.var(ddof=1))/2)
    print(f"   {task:<9} TV={tvr.mean():.3f} (n={len(tvr)})   JS={jsr.mean():.3f} (n={len(jsr)})   "
          f"d(JS-TV)={(jsr.mean()-tvr.mean())/pooled:+.2f}  p={p2:.4f}")
print()

print("3  Divergence ablation (skew 0.60), on the CURRENT endpoint")
for key, lab in (("klraw","KL, raw reference"), ("klsmooth","KL + smoothing 0.20"),
                 ("tvraw","TV"), ("jsraw","JS")):
    try: rs = [r for r in json.load(open(f"runs/_d3il/div_{key}.json")) if r["arm"]=="mop"]
    except FileNotFoundError: continue
    d = np.array([r["tv_to_ref"] for r in rs])
    col = sum(1 for r in rs if r.get("collapsed"))
    cm = max(r["coef_absmax"] for r in rs)
    print(f"   {lab:<22} TV(Cphi,Cref)={d.mean():.3f}  collapsed={col}/{len(rs)}  max|coef|={cm:.2f}  SR={np.mean([r['SR'] for r in rs]):.3f}")
