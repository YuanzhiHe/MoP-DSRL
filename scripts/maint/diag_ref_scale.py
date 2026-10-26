#!/usr/bin/env python3
"""At what size does the Pushing reference estimate stop returning?

Twenty-four episodes come back in thirty seconds; six hundred have not come back
in twenty-four hours. That is not a slope, so this walks the size up under a
deadline and reports the last one that returned.
"""
import os, signal, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from run_d3il_headtohead import _ref_chunk

W = int(os.environ.get("SLURM_CPUS_PER_TASK", 16))
print(f"workers={W}", flush=True)

for total in (24, 96, 240, 600):
    per = max(1, total // W)
    payloads = [("0.50", per, 8000 + i, 4, "cpu", "pushing", 3) for i in range(W)]
    def fire(*_):
        raise TimeoutError("deadline")
    signal.signal(signal.SIGALRM, fire); signal.alarm(900)
    t = time.time()
    try:
        with ProcessPoolExecutor(max_workers=W,
                                 mp_context=mp.get_context("spawn")) as ex:
            c = sum(ex.map(_ref_chunk, payloads))
        signal.alarm(0)
        print(f"  total={total:4d} per={per:3d}  {time.time()-t:7.1f}s  "
              f"labelled={np.asarray(c).sum():.0f}", flush=True)
    except Exception as exc:                                     # noqa: BLE001
        signal.alarm(0)
        print(f"  total={total:4d} per={per:3d}  {time.time()-t:7.1f}s  "
              f"{type(exc).__name__}", flush=True)
        break
