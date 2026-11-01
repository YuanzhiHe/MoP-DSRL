#!/usr/bin/env python3
"""Does a pool worker return from a second task?

Every size that worked gave each worker at most one chunk; every size that hung
gave some worker more than one. This holds the total work fixed and varies only
how many chunks each worker takes.
"""
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from run_d3il_headtohead import _ref_chunk


def trial(label, workers, chunks, per, limit=420):
    payloads = [("0.50", per, 9000 + i, 4, "cpu", "pushing", 3)
                for i in range(chunks)]

    def fire(*_):
        raise TimeoutError("deadline")
    signal.signal(signal.SIGALRM, fire)
    signal.alarm(limit)
    t = time.time()
    try:
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=mp.get_context("spawn")) as ex:
            c = sum(ex.map(_ref_chunk, payloads))
        signal.alarm(0)
        print(f"  {label:<40}{time.time() - t:7.1f}s  "
              f"labelled={np.asarray(c).sum():.0f}", flush=True)
        return True
    except Exception as exc:                                     # noqa: BLE001
        signal.alarm(0)
        print(f"  {label:<40}{time.time() - t:7.1f}s  {type(exc).__name__}",
              flush=True)
        return False


def main() -> int:
    print("same total work, different chunks per worker", flush=True)
    trial("8 workers, 8 chunks of 2   (1 each)", 8, 8, 2)
    trial("4 workers, 8 chunks of 2   (2 each)", 4, 8, 2)
    trial("2 workers, 8 chunks of 2   (4 each)", 2, 8, 2)
    trial("1 worker,  8 chunks of 2   (8 in one)", 1, 8, 2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
