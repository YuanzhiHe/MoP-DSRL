#!/usr/bin/env python3
"""Which Pushing arm hangs?

The sweep writes one cell and then stalls. ex.map yields in order, so a stall
after the first result means the second cell never returned, not that every cell
failed; the first cell is dsrl. This runs each arm on its own, one seed, under a
deadline, and names the one that does not come back.

The namespace comes from the driver's own parser, so it cannot drift from what
the sweep actually passes.
"""
import argparse
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_d3il_headtohead import build_parser, reference_profile, run_arm

DEADLINE = int(os.environ.get("ARM_DEADLINE", 420))


def main() -> int:
    cli = argparse.ArgumentParser()
    cli.add_argument("--task", default="pushing")
    cli.add_argument("--skew", default="0.50")
    cli.add_argument("--bb", type=int, default=3)
    c = cli.parse_args()

    args = build_parser().parse_args([
        "--task", c.task, "--skews", c.skew, "--backbone-seeds", str(c.bb),
        "--arms", "mop", "--seeds", "1", "--seed-offset", "7600",
        "--mode-groups", "4" if c.task == "pushing" else "6",
        "--divergence", "tv", "--n-rewarded", "1", "--bmd-prior", "fitted",
        "--iters", "2", "--episodes-per-iter", "4", "--eval-episodes", "10",
        "--ref-episodes", "24", "--workers", "8",
    ])
    print(f"task={c.task} skew={c.skew} bb={c.bb}", flush=True)

    t0 = time.time()
    C_ref = reference_profile(c.skew, args, c.bb)
    print(f"reference estimate returned in {time.time() - t0:.1f}s", flush=True)

    worst = 0
    for arm in ("dsrl", "mop", "mi", "bmd"):
        def fire(*_):
            raise TimeoutError(f"did not return in {DEADLINE}s")
        signal.signal(signal.SIGALRM, fire)
        signal.alarm(DEADLINE)
        t = time.time()
        try:
            r = run_arm(arm, 0.0 if arm == "dsrl" else 1.0, c.skew, 7600,
                        args, C_ref, c.bb)
            signal.alarm(0)
            print(f"  {arm:<5} returned in {time.time() - t:6.1f}s  "
                  f"SR={r['SR']:.3f}", flush=True)
        except Exception as exc:                                 # noqa: BLE001
            signal.alarm(0)
            worst = 1
            print(f"  {arm:<5} {time.time() - t:6.1f}s  {type(exc).__name__}: "
                  f"{str(exc)[:60]}", flush=True)
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
