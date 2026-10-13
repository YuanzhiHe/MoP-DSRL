#!/usr/bin/env python3
"""Time the reference rollout for each Pushing backbone at one weighting.

Two jobs spent 36 hours at skew 0.50 without finishing the 600-episode reference
estimate for backbones 3 and 4, while backbones 1 and 2 finished theirs. Either
a worker died and the pool hung, or those policies are simply far slower to roll
out. A handful of episodes, timed in this process with no pool involved,
separates the two.
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from run_d3il_headtohead import _ref_chunk, backbone_path

SKEW, N = "0.50", 6
print(f"{'backbone':>9}{'checkpoint':>12}{'episodes':>10}{'seconds':>10}{'per ep':>9}")
for bb in range(5):
    p = backbone_path("pushing", SKEW, bb)
    t0 = time.time()
    try:
        counts = _ref_chunk((SKEW, N, 999 + bb, 4, "cpu", "pushing", bb))
        dt = time.time() - t0
        print(f"{bb:>9}{'ok' if p.exists() else 'MISSING':>12}{N:>10}"
              f"{dt:>10.1f}{dt / N:>9.2f}   modes={np.asarray(counts).sum():.0f}")
    except Exception as exc:                                     # noqa: BLE001
        print(f"{bb:>9}{'ok':>12}{N:>10}{time.time() - t0:>10.1f}"
              f"   RAISED {type(exc).__name__}: {str(exc)[:50]}")
