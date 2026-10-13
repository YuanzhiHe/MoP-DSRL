#!/usr/bin/env python3
"""Time the pooled reference estimate, which the thread cap is meant to fix.

Before the cap each of N worker processes inherited OMP_NUM_THREADS=N, so the
pool ran N-squared threads against N cores. This measures the pooled path end to
end; the per-worker timing of time_ref_profile.py called the worker directly and
so never saw the contention.
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from run_d3il_headtohead import _ref_chunk

SKEW, BB, WORKERS, PER = "0.50", 3, 12, 4
payloads = [(SKEW, PER, 4000 + i, 4, "cpu", "pushing", BB) for i in range(WORKERS)]
t0 = time.time()
with ProcessPoolExecutor(max_workers=WORKERS) as ex:
    counts = sum(ex.map(_ref_chunk, payloads))
dt = time.time() - t0
n = WORKERS * PER
print(f"pooled: {WORKERS} workers x {PER} episodes = {n} episodes in {dt:.1f}s "
      f"({dt / n:.2f}s per episode)")
print(f"labelled episodes: {np.asarray(counts).sum():.0f}")
print(f"extrapolated to the 600-episode reference estimate: "
      f"{600 / WORKERS * (dt / PER) / 60:.1f} min")
