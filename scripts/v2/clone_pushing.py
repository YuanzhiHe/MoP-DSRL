#!/usr/bin/env python3
"""Re-clone one Pushing backbone on the sine-cosine observation.

The twenty Pushing backbones were cloned on the benchmark's ``tan(yaw)``, whose
pole at a quarter turn carried the normalised observation past 1e12 in every
run of the earlier matrix and engaged the observation bound in 95 of the frozen
policy's 100 evaluations. The observation now carries the sine and cosine of
the same angle, which the old checkpoints cannot be rolled out against at all:
they hold ten channels of normalisation statistics where the environment
presents twelve, and a probe against them stops with a shape error.

Each backbone is independent and takes about three minutes, so this runs as one
array task per backbone and the whole set finishes in roughly the time of one.
The task index selects its (weighting, seed) pair from the plan the launcher
wrote, which keeps the mapping from index to backbone on disk rather than in
the submission line.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(os.environ.get("MOP_DSRL_ROOT",
                           Path(__file__).resolve().parents[2]))
PLAN = ROOT / "runs" / "_d3il_v2" / "_clone_plan.json"
SKEW = Path("/shared/scratch/SCWF00111/d3il_data/data/pushing/skew")
OUT = Path("/shared/scratch/SCWF00111/d3il_backbones/pushing")
# Outside range(5), so the weight is never chosen on a matrix policy.
CALIBRATION_SEED = 5
# Steps of behaviour cloning. Set on the frozen policy alone, with no arm and
# no contrast in view: the frozen completion rate on Pushing runs 0.036 at
# 20000 steps, 0.185 at 60000 and 0.345 at 120000, and doubling again to
# 240000 leaves it at 0.350, so the budget is the point past which the policy
# the whole matrix starts from stops improving. The cloning loss keeps falling
# after that point on both tasks and is therefore not what selects it.
CLONE_STEPS = 120000


def one(i: int) -> int:
    """Clone the backbone the plan's ``i``-th entry names."""
    plan = json.loads(PLAN.read_text())
    weighting, bb = plan[i]
    # bb None is the calibration checkpoint, which carries no seed suffix and
    # is trained at a seed outside the matrix's own range.
    out = (OUT / f"pushing_skew{weighting}.pt" if bb is None
           else OUT / f"pushing_skew{weighting}_bb{bb}.pt")
    seed = CALIBRATION_SEED if bb is None else bb
    weights = SKEW / f"demo_weights_skew{weighting}.pkl"
    if not weights.exists():
        print(f"missing {weights}", flush=True)
        return 1
    cmd = [sys.executable, str(ROOT / "scripts" / "train_d3il_backbone.py"),
           "--task", "pushing", "--weights", str(weights),
           "--out", str(out), "--seed", str(seed), "--device", "cpu",
           "--steps", str(CLONE_STEPS)]
    # One thread per clone. Twenty of these run at once, and torch otherwise
    # sizes its pool to the whole allocation.
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    t0 = time.time()
    print(f"  start {out.name}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    ok = r.returncode == 0 and out.exists()
    # The final cloning loss belongs in the log whether or not the clone
    # succeeded: a backbone that trains without raising and still cannot do
    # the task leaves no other trace of how its fit ended.
    last = [l for l in r.stdout.splitlines() if "flow-matching" in l]
    print(f"  [{'ok' if ok else 'FAIL'}] {out.name} "
          f"{(time.time() - t0) / 60:.1f} min"
          + (f" {out.stat().st_size}B" if out.exists() else "")
          + (f" | {last[-1].strip()}" if last else " | no loss line")
          + ("" if ok else "\n" + r.stdout[-800:] + r.stderr[-800:]),
          flush=True)
    return 0 if ok else 1


def main() -> int:
    """Clone every Pushing backbone, in parallel inside one job.

    The queue allows ten submitted jobs, so twenty array tasks are not
    available; each clone is a separate process here instead, which gets the
    same parallelism out of one allocation.
    """
    plan = json.loads(PLAN.read_text())
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        return one(int(os.environ["SLURM_ARRAY_TASK_ID"]))
    n = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    print(f"cloning {len(plan)} Pushing backbones, {min(n, len(plan))} at once",
          flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=min(n, len(plan))) as ex:
        bad = sum(ex.map(one, range(len(plan))))
    print(f"{len(plan) - bad} of {len(plan)} cloned", flush=True)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
