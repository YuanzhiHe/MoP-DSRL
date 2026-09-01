#!/usr/bin/env python3
"""Report the behavioural-mode distribution of D3IL's Avoiding demonstrations.

This single number decides the experiment design. The toy established that
KL(C_phi || C_ref) and an entropy/MI objective are the same objective when the
reference is uniform -- KL(p || uniform) = ln K - H(p) -- so if Avoiding's
demonstrations really are uniform, running it as shipped could not separate the
two arms, and the skew has to be constructed (scripts/make_skewed_demos.py).
D3IL's paper claims balanced collection and 96 = 24 x 4 implies exactly uniform;
this verifies it against the files rather than trusting the claim.
"""
from __future__ import annotations

import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

DATA = Path("/shared/scratch/SCWF00111/d3il_data/data/avoiding/data")


def main() -> int:
    files = sorted(DATA.glob("*.pkl"))
    print(f"demo files: {len(files)}  in {DATA}")
    if not files:
        print("no demonstrations found")
        return 1

    with open(files[0], "rb") as f:
        d = pickle.load(f)
    print(f"top-level keys: {list(d)[:10]}")
    for k in list(d)[:6]:
        v = d[k]
        if isinstance(v, dict):
            inner = {kk: getattr(vv, "shape", type(vv).__name__) for kk, vv in list(v.items())[:6]}
            print(f"  {k}: {inner}")
        else:
            print(f"  {k}: {getattr(v, 'shape', type(v).__name__)}")

    # The filename encodes the context/rep: env_<ctx>_<rep>.pkl. D3IL labels the
    # behavioural mode offline via environments/dataset/data/*/check_mode.py; the
    # per-file context id is the closest label available without replaying, so
    # report its distribution plus trajectory lengths as a sanity check.
    ctx = Counter(int(f.stem.split("_")[1]) for f in files)
    reps = Counter(f.stem.split("_")[2] for f in files)
    counts = np.array(sorted(ctx.values(), reverse=True), dtype=float)
    p = counts / counts.sum()
    K = len(counts)
    kl_unif = float(np.sum(p * np.log(p * K + 1e-12)))
    print(f"\ndistinct contexts: {K}  reps per context: {dict(reps)}")
    print(f"per-context counts: min={counts.min():.0f} max={counts.max():.0f} "
          f"max/min={counts.max()/counts.min():.2f}x")
    print(f"KL(context distribution || uniform) = {kl_unif:.4f} nats "
          f"(0 = exactly balanced)")

    lens = []
    for f in files[:20]:
        with open(f, "rb") as fh:
            dd = pickle.load(fh)
        robot = dd.get("robot", {})
        arr = robot.get("des_c_pos", robot.get("c_pos"))
        if arr is not None:
            lens.append(len(arr))
    if lens:
        print(f"trajectory length over {len(lens)} demos: "
              f"mean={np.mean(lens):.1f} min={min(lens)} max={max(lens)}")

    print("\nCONCLUSION:", "uniform -- skew must be constructed"
          if kl_unif < 1e-6 else f"non-uniform (KL={kl_unif:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
