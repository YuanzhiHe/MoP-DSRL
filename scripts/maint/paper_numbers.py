#!/usr/bin/env python3
"""Every number the manuscript reports, computed from the run files in one place.

Primary endpoint is the raw total variation between the fine-tuned policy's mode
histogram and the frozen policy's, which is the quantity the method targets and
which is monotone in it. The normalised form used earlier is not monotone: it can
award its maximum to a policy that has moved further from uniform than the
reference, past the reference rather than onto it.
"""
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
ARMS = ("dsrl", "mi", "bmd", "mop")
TASKS = [("Avoiding", ["mb_skew*.json"], 0.15), ("Pushing", ["pushrw_skew*.json","bmdP_skew*.json"], 0.10)]

def se(v): return v.std(ddof=1)/np.sqrt(len(v)) if len(v) > 1 else float("nan")

for name, pats, floor in TASKS:
    rows = load(pats, floor)
    skews = sorted({r["skew"] for r in rows}, key=lambda s: np.nanmean([tvu(r) for r in rows if r["skew"]==s]))
    print(f"########## {name} ##########")
    print("TABLE 1  TV(C_phi, C_ref), mean +- SE  (n)")
    print(f"  {'TV(Cref,u)':>10}  " + "  ".join(f"{a:>20}" for a in ARMS))
    for s_ in skews:
        sr = [r for r in rows if r["skew"]==s_]
        cells = []
        for a in ARMS:
            v = np.array([r["tv_to_ref"] for r in sr if r["arm"]==a])
            cells.append(f"{v.mean():.3f}+-{se(v):.3f} ({len(v)})".rjust(20) if len(v)>1 else f"{'--':>20}")
        print(f"  {np.nanmean([tvu(r) for r in sr]):>10.3f}  " + "  ".join(cells))
    print("\nTABLE 2  success rate, mean +- SE  (n)")
    print(f"  {'TV(Cref,u)':>10}  " + "  ".join(f"{a:>20}" for a in ARMS))
    for s_ in skews:
        sr = [r for r in rows if r["skew"]==s_]
        cells = []
        for a in ARMS:
            v = np.array([r["SR"] for r in sr if r["arm"]==a])
            cells.append(f"{v.mean():.3f}+-{se(v):.3f} ({len(v)})".rjust(20) if len(v)>1 else f"{'--':>20}")
        print(f"  {np.nanmean([tvu(r) for r in sr]):>10.3f}  " + "  ".join(cells))
    print("\nPOOLED  mop advantage on TV(C_phi,C_ref), one-sided")
    x = np.array([r["tv_to_ref"] for r in rows if r["arm"]=="mop"])
    for other in ("mi","dsrl","bmd"):
        y = np.array([r["tv_to_ref"] for r in rows if r["arm"]==other])
        if len(y)<2: continue
        t,p2 = stats.ttest_ind(x,y,equal_var=False); p1 = p2/2 if t<0 else 1-p2/2
        pooled = np.sqrt((x.var(ddof=1)+y.var(ddof=1))/2)
        print(f"  vs {other:<5} d={(y.mean()-x.mean())/pooled:+.2f}  p={p1:.2e}  n={len(x)}/{len(y)}")
    print("\nPER-LEVEL one-sided p, mop vs each baseline")
    for s_ in skews:
        sr = [r for r in rows if r["skew"]==s_]
        xs = np.array([r["tv_to_ref"] for r in sr if r["arm"]=="mop"])
        out=[]
        for other in ("mi","dsrl","bmd"):
            ys = np.array([r["tv_to_ref"] for r in sr if r["arm"]==other])
            if len(ys)<2: continue
            t,p2 = stats.ttest_ind(xs,ys,equal_var=False); p1 = p2/2 if t<0 else 1-p2/2
            out.append(f"{other}:{p1:.4f}")
        print(f"  skew {np.nanmean([tvu(r) for r in sr]):.3f}  " + "  ".join(out))
    print("\nUNFILTERED pooled")
    allr = load(pats, 0.0)
    x = np.array([r["tv_to_ref"] for r in allr if r["arm"]=="mop"])
    for other in ("mi","dsrl","bmd"):
        y = np.array([r["tv_to_ref"] for r in allr if r["arm"]==other])
        if len(y)<2: continue
        t,p2 = stats.ttest_ind(x,y,equal_var=False); p1 = p2/2 if t<0 else 1-p2/2
        pooled = np.sqrt((x.var(ddof=1)+y.var(ddof=1))/2)
        print(f"  vs {other:<5} d={(y.mean()-x.mean())/pooled:+.2f}  p={p1:.4f}")
    print("\nVALID-MODE FRACTION per arm (classifiable evaluation episodes)")
    for a in ARMS:
        v = [r for r in rows if r["arm"]==a]
        fr = np.mean([sum(1 for c in r["C_phi"] if c>0)>0 for r in v]) if v else 0
        print(f"  {a:<5} runs={len(v)}")
    print()

print("########## backbone-clustered paired analysis (Avoiding) ##########")
A = load(["mb_skew*.json"], 0.15)
bbs = sorted({r.get("bb_seed") for r in A} - {None}); skews = sorted({r["skew"] for r in A})
for other in ("mi","dsrl","bmd"):
    diffs=[]
    for s_ in skews:
        for bb in bbs:
            v=[r["tv_to_ref"] for r in A if r["skew"]==s_ and r.get("bb_seed")==bb and r["arm"]=="mop"]
            w=[r["tv_to_ref"] for r in A if r["skew"]==s_ and r.get("bb_seed")==bb and r["arm"]==other]
            if len(v)>1 and len(w)>1: diffs.append(np.mean(w)-np.mean(v))
    d=np.array(diffs); t,p2=stats.ttest_1samp(d,0); p1=p2/2 if t>0 else 1-p2/2
    lo,hi = d.mean()-1.96*se(d), d.mean()+1.96*se(d)
    print(f"  vs {other:<5} mean diff={d.mean():+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  dz={d.mean()/d.std(ddof=1):+.2f}  p={p1:.5f}  units={len(d)}  wins={int((d>0).sum())}/{len(d)}")
