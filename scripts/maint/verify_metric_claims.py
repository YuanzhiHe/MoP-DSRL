#!/usr/bin/env python3
"""Check the reviewer's claims about the fidelity metric and the JS bound."""
import glob, json
import numpy as np
from scipy import stats

print("CLAIM A  F awards 1.0 to a policy that doubled the pretrained skew")
u = np.array([.5, .5]); ref = np.array([.7, .3]); phi = np.array([.9, .1])
tv = lambda a, b: 0.5 * np.abs(a - b).sum()
F = (tv(phi, u) - tv(phi, ref)) / tv(ref, u)
print(f"  TV(phi,u)={tv(phi,u):.3f}  TV(phi,ref)={tv(phi,ref):.3f}  TV(ref,u)={tv(ref,u):.3f}  F={F:.3f}")
print(f"  -> {'CONFIRMED' if abs(F-1.0)<1e-9 else 'not reproduced'}\n")

def load(pats, floor):
    rows = []
    for p in pats:
        for f in sorted(glob.glob(f"runs/_d3il/{p}")): rows += json.load(open(f))
    return [r for r in rows if r["SR"] >= floor]
def tvu(r):
    q = np.asarray(r.get("C_ref", []), float); return float(0.5*np.abs(q-1.0/q.size).sum()) if q.size else np.nan
def gn(r):
    d = tvu(r); return (r["tv_to_uniform"]-r["tv_to_ref"])/d if d and np.isfinite(d) and d>1e-9 else np.nan

print("CLAIM B  F reverses a ranking that raw TV-to-reference gets right (Avoiding)")
A = load(["mb_skew*.json"], 0.15)
for arm in ("mop","bmd","mi","dsrl"):
    g = np.array([gn(r) for r in A if r["arm"]==arm]); g = g[np.isfinite(g)]
    raw = np.array([r["tv_to_ref"] for r in A if r["arm"]==arm])
    print(f"  {arm:<5} F={g.mean():+.3f}   TV(Cphi,Cref)={raw.mean():.3f}  (lower is better)")
print()

print("CLAIM C  raw TV-to-reference is a stronger result for us")
for task, pats, floor in (("Avoiding", ["mb_skew*.json"], 0.15),
                          ("Pushing", ["pushrw_skew*.json","bmdP_skew*.json"], 0.10)):
    rows = load(pats, floor)
    x = np.array([r["tv_to_ref"] for r in rows if r["arm"]=="mop"])
    out = []
    for other in ("mi","dsrl","bmd"):
        y = np.array([r["tv_to_ref"] for r in rows if r["arm"]==other])
        if len(y)<2: continue
        t,p2 = stats.ttest_ind(x,y,equal_var=False); p1 = p2/2 if t<0 else 1-p2/2
        pooled = np.sqrt((x.var(ddof=1)+y.var(ddof=1))/2)
        out.append(f"vs {other}: d={(y.mean()-x.mean())/pooled:+.2f} p={p1:.5f}")
    # per level wins
    skews = sorted({r["skew"] for r in rows})
    wins = sum(1 for s in skews
               if np.mean([r["tv_to_ref"] for r in rows if r["skew"]==s and r["arm"]=="mop"])
               < min(np.mean([r["tv_to_ref"] for r in rows if r["skew"]==s and r["arm"]==o])
                     for o in ("mi","dsrl","bmd")
                     if any(r["skew"]==s and r["arm"]==o for r in rows)))
    print(f"  {task:<9} " + "  ".join(out) + f"   wins {wins}/{len(skews)} levels")
print()

print("CLAIM D  JS coefficient exceeds the 0.5*log2 bound stated in the paper")
js = []
for f in sorted(glob.glob("runs/_d3il/js_skew*.json")): js += json.load(open(f))
cm = np.array([r["coef_absmax"] for r in js if r["arm"]=="mop"])
print(f"  stated bound 0.5*log2 = {0.5*np.log(2):.3f}   observed max = {cm.max():.3f}  mean = {cm.mean():.3f}")
print()

print("CLAIM E  total runs behind the reported experiments")
tot = 0
for pat in ("mb_skew*.json","pushrw_skew*.json","bmdP_skew*.json","js_skew*.json","jsP_skew*.json"):
    n = sum(len(json.load(open(f))) for f in sorted(glob.glob(f"runs/_d3il/{pat}")))
    print(f"  {pat:<22} {n}")
    tot += n
print(f"  total reported {tot}")
