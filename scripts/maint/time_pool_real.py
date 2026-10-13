#!/usr/bin/env python3
"""Time the reference estimate at the width the sweeps actually use.

The earlier pool test used four workers and passed; the sweeps use twenty-four,
where each worker's BLAS pool took the core count and the node ran hundreds of
threads. This repeats the real shape so the thread cap is measured where it
matters, and prints what a 600-episode estimate would cost.
"""
import os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from run_d3il_headtohead import _ref_chunk

W = int(os.environ.get("SLURM_CPUS_PER_TASK", 24))
PER = 4
print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}  workers={W}  "
      f"episodes={W * PER}", flush=True)
payloads = [("0.50", PER, 6000 + i, 4, "cpu", "pushing", 3) for i in range(W)]
t0 = time.time()
with ProcessPoolExecutor(max_workers=W) as ex:
    counts = sum(ex.map(_ref_chunk, payloads))
dt = time.time() - t0
print(f"{W * PER} episodes in {dt:.1f}s  ({dt / (W * PER):.2f}s per episode)")
print(f"labelled {np.asarray(counts).sum():.0f}")
print(f"600-episode reference estimate would take {dt * 600 / (W * PER) / 60:.1f} min")
print(f"one (level, policy) group of 20 cells: roughly "
      f"{dt * (600 + 20 * (60 * 16 + 300)) / (W * PER) / 3600:.1f} h")
