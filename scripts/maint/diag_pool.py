#!/usr/bin/env python3
"""Where does the pooled rollout stall?

Calling the worker directly runs at seven seconds an episode, while the same
work inside a ProcessPoolExecutor produced nothing in thirty minutes. Three
candidates: the simulator does not survive fork, worker threads contend, or the
pool is fine and something else is slow. This times the same tiny workload
served four ways, each under a watchdog, and prints which ones return.
"""
import os, sys, time, signal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

W, PER = 4, 2
PAYLOADS = [("0.50", PER, 5000 + i, 4, "cpu", "pushing", 3) for i in range(W)]


def watchdog(seconds):
    def fire(*_):
        raise TimeoutError(f"no result within {seconds}s")
    signal.signal(signal.SIGALRM, fire)
    signal.alarm(seconds)


def run(label, fn, limit=300):
    t0 = time.time()
    try:
        watchdog(limit)
        out = fn()
        signal.alarm(0)
        print(f"  {label:<34} {time.time() - t0:7.1f}s   counts={int(sum(out))}",
              flush=True)
    except Exception as exc:                                     # noqa: BLE001
        signal.alarm(0)
        print(f"  {label:<34} {time.time() - t0:7.1f}s   {type(exc).__name__}: "
              f"{str(exc)[:60]}", flush=True)


def main():
    from run_d3il_headtohead import _ref_chunk
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
          f"cpus={os.cpu_count()}  {W} workers x {PER} episodes")

    run("serial, in this process",
        lambda: sum(_ref_chunk(p) for p in PAYLOADS), limit=300)

    def pooled(ctx_name):
        ctx = mp.get_context(ctx_name)
        with ProcessPoolExecutor(max_workers=W, mp_context=ctx) as ex:
            return sum(ex.map(_ref_chunk, PAYLOADS))

    run("pool, fork (the current path)", lambda: pooled("fork"), limit=300)
    run("pool, spawn", lambda: pooled("spawn"), limit=420)
    run("pool, forkserver", lambda: pooled("forkserver"), limit=420)


if __name__ == "__main__":
    main()
