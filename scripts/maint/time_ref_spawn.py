#!/usr/bin/env python3
"""Does the Pushing reference estimate return under spawn at full width?

The sweeps print nothing after their resume lines, which places the stall in
reference_profile. The width smoke test passed on Avoiding, so the difference is
either the task or the per-worker episode count. This times both, each under a
deadline, and says which returns.
"""
import os, signal, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from run_d3il_headtohead import _ref_chunk

W = int(os.environ.get("SLURM_CPUS_PER_TASK", 24))


def run(label, task, per, limit):
    def fire(*_):
        raise TimeoutError(f"no result in {limit}s")
    payloads = [("0.50" if task == "pushing" else "0.00", per, 7000 + i, 4,
                 "cpu", task, 3) for i in range(W)]
    t0 = time.time()
    signal.signal(signal.SIGALRM, fire); signal.alarm(limit)
    try:
        with ProcessPoolExecutor(max_workers=W, mp_context=mp.get_context("spawn")) as ex:
            counts = sum(ex.map(_ref_chunk, payloads))
        signal.alarm(0)
        dt = time.time() - t0
        print(f"  {label:<34} {dt:7.1f}s  ({dt/(W*per):.2f}s/ep)  "
              f"labelled={np.asarray(counts).sum():.0f}", flush=True)
    except Exception as exc:                                    # noqa: BLE001
        signal.alarm(0)
        print(f"  {label:<34} {time.time()-t0:7.1f}s  {type(exc).__name__}: "
              f"{str(exc)[:50]}", flush=True)


print(f"workers={W}  OMP={os.environ.get('OMP_NUM_THREADS')}", flush=True)
run("avoiding, 2 episodes each", "avoiding", 2, 600)
run("pushing, 2 episodes each", "pushing", 2, 600)
run("pushing, 25 each (the real size)", "pushing", 25, 1500)
