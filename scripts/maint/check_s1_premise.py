#!/usr/bin/env python3
"""Test the premise of pre-registered check S1, which failed.

S1 asserted that mop and mi must coincide at skew 0.00 because
KL(p || uniform) = ln K - H(p) makes the two objectives identical. That identity
requires the REFERENCE to be uniform. "skew 0.00" only states that the
DEMONSTRATION weighting is uniform, which is not the same thing -- behaviour
cloning learns its own preferences over routes, so the backbone's mode
distribution can be skewed even from perfectly balanced demonstrations.

If C_ref is measurably non-uniform at skew 0.00, S1 was mis-specified and its
failure says nothing about the implementation. If C_ref really is near-uniform,
the arms differ where they provably cannot and the replication is not
trustworthy. This measures which, rather than arguing for either.
"""
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.run_d3il_headtohead import reference_profile  # noqa: E402

K = 6
args = types.SimpleNamespace(workers=8, ref_episodes=1200, M=10,
                             device="cpu", mode_groups=K)

print("C_ref measured by rolling out each frozen backbone (6-group partition)\n")
for skew in ("0.00", "0.15", "0.30", "0.60"):
    p = reference_profile(skew, args)
    nz = p > 0
    kl = float(np.sum(p[nz] * np.log(p[nz] * K)))
    H = float(-(p[nz] * np.log(p[nz])).sum())
    print(f"  demo-skew {skew}:  KL(C_ref || uniform) = {kl:.4f} nats   "
          f"H = {H:.4f} (max {np.log(K):.4f})   groups={int(nz.sum())}/{K}")
    print(f"      C_ref = {np.round(p, 3)}")
print("\nUniform => KL = 0. A clearly non-zero KL at skew 0.00 means S1's premise")
print("fails: the identity needs a uniform REFERENCE, not uniform DEMONSTRATIONS.")
