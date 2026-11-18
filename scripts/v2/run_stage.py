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

sys.path.insert(0, str(SRC))
import launch_plan as LP  # noqa: E402

AVOIDING_W = ["0.00", "0.15", "0.30", "0.60"]
PUSHING_W = ["0.00", "0.10", "0.25", "0.50"]
BB = ["0", "1", "2", "3", "4"]
LAM_FROM = ["--lam-from", f"{RUNS}/v22_lambda_choice.json"]
# The divergence comparison's most skewed level on each task.
ABL_W = {"avoiding": "0.60", "pushing": "0.50"}


def drv(*a):
    return [PY, str(SRC / "run_d3il_v2.py"), *[str(x) for x in a]]


def cfg(task):
    """The settings that describe the task rather than the stage.

    How many routes the task has, how many of them carry reward, and how long
    an evaluation runs are properties of the task, and the driver's defaults
    are Avoiding's. A stage that leaves them out therefore runs Pushing as if
    it were Avoiding: six route groups where the task has four, which puts
    every episode in one bin and makes the fidelity distance identically zero,
    and four rewarded routes where the task rewards one, which is a different
    objective. They come from the same table the launch plan uses so that the
    two launchers cannot drift apart.
    """
    t = LP.TASK[task]
    return ["--mode-groups", str(t["mode_groups"]),
            "--n-rewarded", str(t["n_rewarded"]),
            "--eval-episodes", str(t["eval_episodes"]),
            "--ref-episodes", str(t["ref_episodes"])]


def seed_offset(task, stage):
    """The seed block the plan gives this task and stage."""
    return str(LP.TASK[task]["calib_seed" if stage == "calibration"
                             else "main_seed"])


def workers():
    """Leave a core for the parent, which is doing the writing."""
    return str(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", 8)) - 1))


def calibration(task, ws):
    """Three candidate weights on a policy the matrix excludes."""
    return [drv("--task", task, *cfg(task), "--weightings", *ws, "--seeds", 3,
                "--seed-offset", seed_offset(task, "calibration"),
                "--arms", "mi", "mop", "bmd",
                "--lam", lam, "--workers", workers(),
                "--out", f"{RUNS}/v22_calibration_{task}_{lam}.jsonl")
            for lam in ("0.25", "0.50", "1.00")]


def main_matrix(task, ws):
    return [drv("--task", task, *cfg(task), "--weightings", w,
                "--backbone-seeds", *BB, "--seeds", 5,
                "--seed-offset", seed_offset(task, "main"), "--workers", workers(),
                *LAM_FROM, "--out", f"{RUNS}/v22_main_{task}_{w}.jsonl")
            for w in ws]


def ablations(task):
    w = ABL_W[task]
    out = []
    # The smoothed form is the same divergence with the reference floored.
    for name, extra in (("kl", []), ("kl-smooth", ["--ref-smooth", "0.05"]),
                        ("js", []), ("l2", [])):
        out.append(drv(
            "--task", task, *cfg(task), "--weightings", w,
            "--backbone-seeds", *BB, "--seeds", 5,
            "--seed-offset", seed_offset(task, "main"), "--arms", "mop",
            "--divergence", "kl" if name.startswith("kl") else name,
            *extra, "--workers", workers(), *LAM_FROM,
            "--out", f"{RUNS}/v22_ablation_divergence_{task}_{name}.jsonl"))
    out.append(drv(
        "--task", task, *cfg(task), "--weightings", w,
        "--backbone-seeds", *BB, "--seeds", 5,
        "--seed-offset", seed_offset(task, "main"), "--arms", "bmd",
        "--bmd-prior", "fitted", "--workers", workers(), *LAM_FROM,
        "--out", f"{RUNS}/v22_ablation_fitted_prior_{task}.jsonl"))
    if task == "pushing":
        # The weight grid lives on the two lowest Pushing skews, which is
        # where the identity places the two objectives closest together.
        # Every candidate weight the plan names, not the two that were not
        # the chosen one when the plan was written: the rule can choose any of
        # them, and it chose 0.50 here, which left the grid's third weight
        # unrun while the accounting still planned for it.
        for lam in ("0.25", "0.50", "1.00"):
            out.append(drv(
                "--task", "pushing", *cfg("pushing"),
                "--weightings", "0.00", "0.10",
                "--backbone-seeds", *BB, "--seeds", 5,
                "--seed-offset", seed_offset("pushing", "main"),
                "--arms", "mi", "mop", "--lam", lam, "--workers", workers(),
                "--out",
                f"{RUNS}/v22_ablation_weight_pushing_{lam}.jsonl"))
    return out


def frozen():
    return [[PY, str(SRC / "frozen_baseline.py"), "--task", task, *cfg(task),
             "--weightings", *ws, "--backbone-seeds", *BB, "--seeds", "5",
             "--seed-offset", seed_offset(task, "main"), "--workers", workers(),
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
    "lambda": lambda: [
        [PY, str(SRC / "select_lambda.py"),
         # Named one by one: a glob over this directory also matches the
         # calibration of a superseded cloning cohort, whose files sit here
         # under a suffix and are not the cohort the matrix runs on.
         "--calibration",
         *[f"{RUNS}/v22_calibration_{t}_{lam}.jsonl"
           for t in ("avoiding", "pushing")
           for lam in ("0.25", "0.50", "1.00")],
         "--out", f"{RUNS}/v22_lambda_choice.json"],
    ],
    "main-avoiding": lambda: main_matrix("avoiding", AVOIDING_W),
    "main-pushing": lambda: main_matrix("pushing", PUSHING_W),
    "frozen": frozen,
    "abl-avoiding": lambda: ablations("avoiding"),
    "abl-pushing": lambda: ablations("pushing"),
}


def main() -> int:
    argv = [a for a in sys.argv if a != "--dry"]
    dry = "--dry" in sys.argv
    sys.argv = argv
    if not 2 <= len(sys.argv) <= 3 or sys.argv[1] not in STAGES:
        print("stages: " + ", ".join(STAGES))
        print("usage: run_stage.py <stage> [first:last]")
        return 2
    name = sys.argv[1]
    cmds = STAGES[name]()
    # A stage runs its calls in series, so a long one is a long job however
    # many cores it holds. A slice lets the calls of one stage run as separate
    # jobs side by side; the calls are independent and each writes its own
    # file, so the union of the slices is the stage.
    span = ""
    if len(sys.argv) == 3:
        a, _, b = sys.argv[2].partition(":")
        lo = int(a) if a else 0
        hi = int(b) if b else len(cmds)
        if not 0 <= lo < hi <= len(cmds):
            print(f"slice {sys.argv[2]} is outside the {len(cmds)} calls"
                  f" of {name}")
            return 2
        cmds, span = cmds[lo:hi], f" [{lo}:{hi}] of {len(cmds)}"
    print(f"stage {name}{span}: {len(cmds)} calls, {workers()} workers each",
          flush=True)
    for i, c in enumerate(cmds, 1):
        print(f"\n[{i}/{len(cmds)}] " + " ".join(c), flush=True)
        if dry:
            continue
        t0 = time.time()
        rc = subprocess.run(c, cwd=ROOT).returncode
        print(f"[{i}/{len(cmds)}] exit {rc} after {(time.time() - t0) / 60:.1f}"
              f" min", flush=True)
        if rc != 0:
            print(f"STAGE {name}{span}: FAIL at call {i}", flush=True)
            return rc
    print(f"\nSTAGE {name}{span}: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
