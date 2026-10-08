#!/usr/bin/env python3
"""Re-score Pushing against one reference per level, and read skew from the runs.

The BMD arm was run as a separate job and estimated its own C_ref, so the four
arms at a level were scored against references differing by up to 0.072 in total
variation. Re-scoring every arm against the reference the majority of arms used
removes that inconsistency.
"""
import glob, json
import numpy as np
from scipy import stats

def rows(pat):
    out = []
    for f in sorted(glob.glob(f"runs/_d3il/{pat}")): out += json.load(open(f))
    return out

print("=== 1  Pushing: how far apart are the per-arm references? ===")
P = rows("pushrw_skew*.json"); B = rows("bmdP_skew*.json")
refs = {}
for lab, rs in (("main", P), ("bmd", B)):
    for r in rs:
        refs.setdefault((r["skew"], lab), []).append(np.asarray(r["C_ref"], float))
for sk in sorted({k[0] for k in refs}):
    a = np.mean(refs[(sk, "main")], axis=0); b = np.mean(refs[(sk, "bmd")], axis=0)
    print(f"  skew {sk}: TV between the two references = {0.5*np.abs(a-b).sum():.3f}")

print("\n=== 2  Pushing re-scored against the main reference ===")
def tv(p, q): return float(0.5*np.abs(np.asarray(p,float)-np.asarray(q,float)).sum())
common = {sk: np.mean(refs[(sk,"main")], axis=0) for sk in {k[0] for k in refs}}
allr = [dict(r) for r in P + B]
for r in allr:
    r["tv_to_ref"] = tv(r["C_phi"], common[r["skew"]])
allr = [r for r in allr if r["SR"] >= 0.10]
REF = {"0.00":0.327, "0.10":0.120, "0.25":0.537, "0.50":0.770}
sks = sorted({r["skew"] for r in allr}, key=lambda s: REF[s])
print(f"  {'level':>6}  " + "  ".join(f"{a:>18}" for a in ("dsrl","mi","bmd","mop")))
for sk in sks:
    sr = [r for r in allr if r["skew"]==sk]
    cells=[]
    for a in ("dsrl","mi","bmd","mop"):
        v=np.array([r["tv_to_ref"] for r in sr if r["arm"]==a])
        cells.append(f"{v.mean():.3f}+-{v.std(ddof=1)/np.sqrt(len(v)):.3f} ({len(v)})".rjust(18) if len(v)>1 else f"{'--':>18}")
    print(f"  {sk:>6}  " + "  ".join(cells))
x=np.array([r["tv_to_ref"] for r in allr if r["arm"]=="mop"])
for o in ("mi","dsrl","bmd"):
    y=np.array([r["tv_to_ref"] for r in allr if r["arm"]==o])
    t,p2=stats.ttest_ind(x,y,equal_var=False); p1=p2/2 if t<0 else 1-p2/2
    pooled=np.sqrt((x.var(ddof=1)+y.var(ddof=1))/2)
    print(f"  pooled vs {o:<5} d={(y.mean()-x.mean())/pooled:+.2f}  p={p1:.4f}")

print("\n=== 3  reference skew, from the run files (not the plotting script) ===")
A = [r for r in rows("mb_skew*.json")]
kl = {}
for r in A:
    if "C_ref_kl_to_uniform" in r:
        kl.setdefault((r["skew"], r.get("bb_seed")), r["C_ref_kl_to_uniform"])
vals = np.array(sorted(kl.values()))
unmod = np.array(sorted(v for (s,_),v in kl.items() if s=="0.00"))
print(f"  all twenty: {vals.min():.3f} to {vals.max():.3f} nats")
print(f"  unmodified demonstrations: {unmod.min():.3f} to {unmod.max():.3f}  ({unmod.max()/unmod.min():.1f}-fold)")

print("\n=== 4  sampling floor on the endpoint ===")
rng=np.random.default_rng(0)
for lab, K, nphi, nref in (("Avoiding", 6, 330, 650), ("Pushing", 4, 95, 255)):
    p = np.array([0.4,0.25,0.15,0.1,0.06,0.04])[:K]; p = p/p.sum()
    d=[tv(np.bincount(rng.choice(K,nphi,p=p),minlength=K)/nphi,
          np.bincount(rng.choice(K,nref,p=p),minlength=K)/nref) for _ in range(2000)]
    print(f"  {lab}: expected TV between two samples of one distribution = {np.mean(d):.3f}")
