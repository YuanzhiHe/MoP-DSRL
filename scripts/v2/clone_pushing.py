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
from pathlib import Path

ROOT = Path(os.environ.get("MOP_DSRL_ROOT",
                           Path(__file__).resolve().parents[2]))
PLAN = ROOT / "runs" / "_d3il_v2" / "_clone_plan.json"
SKEW = Path("/shared/scratch/SCWF00111/d3il_data/data/pushing/skew")
OUT = Path("/shared/scratch/SCWF00111/d3il_backbones/pushing")


def main() -> int:
    i = int(os.environ.get("SLURM_ARRAY_TASK_ID",
                           sys.argv[1] if len(sys.argv) > 1 else 0))
    plan = json.loads(PLAN.read_text())
    weighting, bb = plan[i]
    out = OUT / f"pushing_skew{weighting}_bb{bb}.pt"
    weights = SKEW / f"demo_weights_skew{weighting}.pkl"
    if not weights.exists():
        print(f"missing {weights}", flush=True)
        return 1

    # The seed names the cloning run, exactly as it did for the cohort being
    # replaced, so a backbone keeps its identity across the observation change
    # and only the observation it was cloned on differs.
    cmd = [sys.executable, str(ROOT / "scripts" / "train_d3il_backbone.py"),
           "--task", "pushing", "--weights", str(weights),
           "--out", str(out), "--seed", str(bb), "--device", "cpu"]
    print(f"array {i}: weighting {weighting} bb{bb} -> {out.name}", flush=True)
    print("  " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode == 0:
        print(f"  wrote {out} ({out.stat().st_size} bytes)", flush=True)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
