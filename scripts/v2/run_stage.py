#!/usr/bin/env python3
"""One stage of the v2.2 cohort, as a sequence of driver calls in one job.

The queue allows ten submitted jobs and four running, so the thirty-two
invocations the cohort needs cannot each be their own job. They are grouped
here into eight stages that SLURM chains with afterok, each stage taking a
whole node and running its calls one after another. A call that fails stops its
stage, and the stages that depend on it never start.

    python scripts/v2/run_stage.py prep            gates, re-clone, preflight
    python scripts/v2/run_stage.py calib-avoiding
    python scripts/v2/run_stage.py calib-pushing
    python scripts/v2/run_stage.py main-avoiding   selects the weight first
    python scripts/v2/run_stage.py main-pushing
    python scripts/v2/run_stage.py frozen
    python scripts/v2/run_stage.py abl-avoiding
    python scripts/v2/run_stage.py abl-pushing
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("MOP_DSRL_ROOT",
                           Path(__file__).resolve().parents[2]))
SRC = Path(__file__).resolve().parent
RUNS = "runs/_d3il_v2"
PY = sys.executable

AVOIDING_W = ["0.00", "0.15", "0.30", "0.60"]
PUSHING_W = ["0.00", "0.10", "0.25", "0.50"]
BB = ["0", "1", "2", "3", "4"]
LAM_FROM = ["--lam-from", f"{RUNS}/v22_lambda_choice.json"]
# The divergence comparison's most skewed level on each task.
ABL_W = {"avoiding": "0.60", "pushing": "0.50"}


def drv(*a):
    return [PY, str(SRC / "run_d3il_v2.py"), *[str(x) for x in a]]


def workers():
    """Leave a core for the parent, which is doing the writing."""
    return str(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", 8)) - 1))


def calibration(task, ws):
    """Three candidate weights on a policy the matrix excludes."""
    return [drv("--task", task, "--weightings", *ws, "--seeds", 3,
                "--seed-offset", 500, "--arms", "mi", "mop", "bmd",
                "--lam", lam, "--workers", workers(),
                "--out", f"{RUNS}/v22_calibration_{task}_{lam}.jsonl")
            for lam in ("0.25", "0.50", "1.00")]


def main_matrix(task, ws):
    return [drv("--task", task, "--weightings", w, "--backbone-seeds", *BB,
                "--seeds", 5, "--seed-offset", 700, "--workers", workers(),
                *LAM_FROM, "--out", f"{RUNS}/v22_main_{task}_{w}.jsonl")
            for w in ws]


def ablations(task):
    w = ABL_W[task]
    out = []
    # The smoothed form is the same divergence with the reference floored.
    for name, extra in (("kl", []), ("kl-smooth", ["--ref-smooth", "0.05"]),
                        ("js", []), ("l2", [])):
        out.append(drv(
            "--task", task, "--weightings", w, "--backbone-seeds", *BB,
            "--seeds", 5, "--seed-offset", 700, "--arms", "mop",
            "--divergence", "kl" if name.startswith("kl") else name,
            *extra, "--workers", workers(), *LAM_FROM,
            "--out", f"{RUNS}/v22_ablation_divergence_{task}_{name}.jsonl"))
    out.append(drv(
        "--task", task, "--weightings", w, "--backbone-seeds", *BB,
        "--seeds", 5, "--seed-offset", 700, "--arms", "bmd",
        "--bmd-prior", "fitted", "--workers", workers(), *LAM_FROM,
        "--out", f"{RUNS}/v22_ablation_fitted_prior_{task}.jsonl"))
    if task == "pushing":
        # The weight grid lives on the two lowest Pushing skews, which is
        # where the identity places the two objectives closest together.
        for lam in ("0.50", "1.00"):
            out.append(drv(
                "--task", "pushing", "--weightings", "0.00", "0.10",
                "--backbone-seeds", *BB, "--seeds", 5, "--seed-offset", 700,
                "--arms", "mi", "mop", "--lam", lam, "--workers", workers(),
                "--out",
                f"{RUNS}/v22_ablation_weight_pushing_{lam}.jsonl"))
    return out


def frozen():
    return [[PY, str(SRC / "frozen_baseline.py"), "--task", task,
             "--weightings", *ws, "--backbone-seeds", *BB, "--seeds", "5",
             "--seed-offset", "700", "--workers", workers(),
             "--out", f"{RUNS}/v22_frozen_{task}.jsonl"]
            for task, ws in (("avoiding", AVOIDING_W),
                             ("pushing", PUSHING_W))]


STAGES = {
    "prep": lambda: [
        [PY, str(SRC / "gates.py")],
        [PY, str(SRC / "clone_pushing.py")],
        [PY, str(SRC / "preflight.py")],
    ],
    "calib-avoiding": lambda: calibration("avoiding", AVOIDING_W),
    "calib-pushing": lambda: calibration("pushing", PUSHING_W),
    # The rule runs here, so the matrix in this stage reads a file that was
    # just written from the calibration both stages above produced.
    "main-avoiding": lambda: [
        [PY, str(SRC / "select_lambda.py"),
         "--calibration", f"{RUNS}/v22_calibration_*.jsonl",
         "--out", f"{RUNS}/v22_lambda_choice.json"],
    ] + main_matrix("avoiding", AVOIDING_W),
    "main-pushing": lambda: main_matrix("pushing", PUSHING_W),
    "frozen": frozen,
    "abl-avoiding": lambda: ablations("avoiding"),
    "abl-pushing": lambda: ablations("pushing"),
}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in STAGES:
        print("stages: " + ", ".join(STAGES))
        return 2
    name = sys.argv[1]
    cmds = STAGES[name]()
    print(f"stage {name}: {len(cmds)} calls, {workers()} workers each",
          flush=True)
    for i, c in enumerate(cmds, 1):
        print(f"\n[{i}/{len(cmds)}] " + " ".join(c), flush=True)
        t0 = time.time()
        rc = subprocess.run(c, cwd=ROOT).returncode
        print(f"[{i}/{len(cmds)}] exit {rc} after {(time.time() - t0) / 60:.1f}"
              f" min", flush=True)
        if rc != 0:
            print(f"STAGE {name}: FAIL at call {i}", flush=True)
            return rc
    print(f"\nSTAGE {name}: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
