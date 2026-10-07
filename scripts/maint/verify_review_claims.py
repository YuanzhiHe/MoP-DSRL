#!/usr/bin/env python3
"""Check the reviewer's four load-bearing factual claims against the artefacts."""
import glob, json
import numpy as np
from scipy import stats

def load(pats, floor=None):
    rows=[]
    for p in pats:
        for f in sorted(glob.glob(f"runs/_d3il/{p}")): rows+=json.load(open(f))
    return rows if floor is None else [r for r in rows if r["SR"]>=floor]
def tvu(r):
    q=np.asarray(r.get("C_ref",[]),float); return float(0.5*np.abs(q-1.0/q.size).sum()) if q.size else np.nan
def gn(r):
    d=tvu(r); return (r["tv_to_uniform"]-r["tv_to_ref"])/d if d and np.isfinite(d) and d>1e-9 else np.nan
def cmp(rows,a,b):
    x=np.array([gn(r) for r in rows if r["arm"]==a]); y=np.array([gn(r) for r in rows if r["arm"]==b])
    x,y=x[np.isfinite(x)],y[np.isfinite(y)]
    t,p2=stats.ttest_ind(x,y,equal_var=False); p1=p2/2 if t>0 else 1-p2/2
    pooled=np.sqrt((x.var(ddof=1)+y.var(ddof=1))/2)
    return (x.mean()-y.mean())/pooled, p1, len(x), len(y)

print("UNFILTERED EFFECT SIZES, measured here\n")
import sys
print("CLAIM 1  Pushing mop-vs-bmd loses significance without the SR filter")
P=["pushrw_skew*.json","bmdP_skew*.json"]
for lab,fl in (("filtered SR>=0.10",0.10),("unfiltered",None)):
    d,p,na,nb=cmp(load(P,fl),"mop","bmd")
    print(f"  {lab:<18} d={d:+.2f}  p={p:.4f}  n={na}/{nb}")

print("\nCLAIM 2  the anchor retains the FEWEST modes at the highest Avoiding skew")
A=load(["mb_skew*.json"],0.15)
hi=max({r['skew'] for r in A}, key=lambda s: np.nanmean([tvu(r) for r in A if r['skew']==s]))
for a in ("dsrl","mop","mi","bmd"):
    v=[r for r in A if r["skew"]==hi and r["arm"]==a]
    print(f"  {a:<5} n_modes={np.mean([r['n_modes'] for r in v]):.2f}  "
          f"H={np.mean([r['H_modes'] for r in v]):.3f}  mc@0.8={np.mean([r['mc_at_08'] for r in v]):.2f}")

print("\nCLAIM 3  JS beats TV in the unreported divergence ablation")
for key,lab in (("klraw","KL raw"),("klsmooth","KL+smooth"),("tvraw","TV"),("jsraw","JS")):
    try: rs=[r for r in json.load(open(f"runs/_d3il/div_{key}.json")) if r["arm"]=="mop"]
    except FileNotFoundError: continue
    g=np.array([gn(r) for r in rs]); g=g[np.isfinite(g)]
    print(f"  {lab:<10} F={g.mean():+.3f}  SR={np.mean([r['SR'] for r in rs]):.3f}  n={len(rs)}")

print("\nCLAIM 4  the figure caption's 'means over 12 to 30 runs' is wrong")
for lab,rows,fl in (("Avoiding",load(["mb_skew*.json"],0.15),0.15),
                    ("Pushing",load(P,0.10),0.10)):
    ns=[len([r for r in rows if r["skew"]==s and r["arm"]==a])
        for s in {r["skew"] for r in rows} for a in ("dsrl","mop","mi","bmd")]
    ns=[n for n in ns if n>0]
    print(f"  {lab:<9} smallest cell n={min(ns)}  largest n={max(ns)}")
