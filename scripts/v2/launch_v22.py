#!/usr/bin/env python3
"""Submit the v2.2 cohort, in an order where each step can stop the next.

The v2.1 cohort was launched, reported, and then withdrawn three times, each
time for something checkable before it ran. This driver puts those checks in
front of the submission rather than beside it: the gates run first and nothing
is submitted while one of them fails, the preflight slice runs the whole
pipeline on forty cells and the matrix waits on it, and each stage names the
job it depends on so SLURM holds it until that job has succeeded.

    python scripts/v2/launch_v22.py plan          what would be submitted
    python scripts/v2/launch_v22.py gates         the checks, on their own
    python scripts/v2/launch_v22.py preflight     gates, clone, slice
    python scripts/v2/launch_v22.py full          everything, chained

Nothing is submitted by ``plan``. Every other mode prints the job it created
and the dependency it carries, so the chain can be read back from the log.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOB = "scripts/v2/cpu_job_v2.sh"

AVOIDING_W = ["0.00", "0.15", "0.30", "0.60"]
PUSHING_W = ["0.00", "0.10", "0.25", "0.50"]


def sbatch(name, cmd, cpus, mem, time, dep=None, array=None, dry=False):
    """Submit one job, optionally held until ``dep`` has succeeded."""
    a = ["sbatch", "--parsable", f"--job-name={name}",
         f"--cpus-per-task={cpus}", f"--mem={mem}", f"--time={time}"]
    if dep:
        a.append(f"--dependency=afterok:{dep}")
    if array:
        a.append(f"--array={array}")
    a += [JOB] + cmd
    if dry:
        print(f"  [plan] {name:<26} dep={dep or '-':<10} {' '.join(cmd)}")
        return None
    jid = subprocess.run(a, cwd=ROOT, capture_output=True, text=True,
                         check=True).stdout.strip()
    print(f"  submitted {jid}  {name:<26} dep={dep or '-'}", flush=True)
    return jid


def stage_gates(dep=None, dry=False):
    """The checks. Everything else depends on this one succeeding."""
    return sbatch("v22-gates", ["python", "scripts/v2/gates.py"],
                  4, "16G", "0:45:00", dep, dry=dry)


def stage_clone(dep=None, dry=False):
    """Re-clone the twenty Pushing backbones on the new observation.

    The old ones were cloned on tan(yaw) and present ten channels where the
    environment now presents twelve, so they cannot be rolled out at all: the
    parity probe against them stopped with a shape error, which is the
    observation version refusing to be mixed. Each backbone is independent and
    takes about three minutes, so the array runs them at once.
    """
    jobs = []
    for i, w in enumerate(PUSHING_W):
        for bb in range(5):
            jobs.append((w, bb))
    spec = json.dumps(jobs)
    (ROOT / "runs" / "_d3il_v2" / "_clone_plan.json").write_text(spec)
    return sbatch("v22-clone-pushing",
                  ["python", "scripts/v2/clone_pushing.py"],
                  2, "12G", "1:00:00", dep, array=f"0-{len(jobs) - 1}", dry=dry)


def stage_preflight(dep=None, dry=False):
    """Forty cells end to end, before the matrix is allowed to start."""
    return sbatch("v22-preflight",
                  ["python", "scripts/v2/preflight.py"],
                  20, "40G", "6:00:00", dep, dry=dry)


def stage_calibration(dep=None, dry=False):
    """The cohort the weight is selected on, on a policy the matrix excludes.

    The rule reads only how often a weight drives success below the task floor,
    never fidelity, and it refuses to run on a main-cohort backbone. It has to
    finish before the matrix starts, because the matrix carries what it picks.
    """
    out = []
    for task, ws, hours in (("avoiding", AVOIDING_W, "8:00:00"),
                            ("pushing", PUSHING_W, "1-12:00:00")):
        for lam in ("0.25", "0.50", "1.00"):
            out.append(sbatch(
                f"v22-calib-{task}-{lam}",
                ["python", "scripts/v2/run_d3il_v2.py", "--task", task,
                 "--weightings"] + ws + ["--seeds", "3", "--seed-offset", "500",
                 "--arms", "mi", "mop", "bmd", "--lam", lam,
                 "--workers", "24",
                 "--out", f"runs/_d3il_v2/v22_calibration_{task}_{lam}.jsonl"],
                25, "50G", hours, dep, dry=dry))
    return out


def stage_select(dep=None, dry=False):
    """Apply the rule and write the weights the matrix will read."""
    return sbatch("v22-select-lambda",
                  ["python", "scripts/v2/select_lambda.py",
                   "--calibration",
                   "runs/_d3il_v2/v22_calibration_*.jsonl",
                   "--out", "runs/_d3il_v2/v22_lambda_choice.json"],
                  2, "8G", "0:20:00", dep, dry=dry)


LAM_FROM = ["--lam-from", "runs/_d3il_v2/v22_lambda_choice.json"]


def stage_matrix(dep=None, dry=False):
    """The cohort itself, one job per weighting so a failure is local."""
    out = []
    for task, ws, hours in (("avoiding", AVOIDING_W, "12:00:00"),
                            ("pushing", PUSHING_W, "2-00:00:00")):
        for w in ws:
            out.append(sbatch(
                f"v22-main-{task}-{w}",
                ["python", "scripts/v2/run_d3il_v2.py", "--task", task,
                 "--weightings", w, "--backbone-seeds", "0", "1", "2", "3", "4",
                 "--seeds", "5", "--seed-offset", "700", "--workers", "24"]
                + LAM_FROM
                + ["--out", f"runs/_d3il_v2/v22_main_{task}_{w}.jsonl"],
                25, "50G", hours, dep, dry=dry))
    return out


def stage_ablations(dep=None, dry=False):
    """The three ablations, on both tasks, at the selected weight."""
    out = []
    for task, w, hours in (("avoiding", "0.60", "8:00:00"),
                           ("pushing", "0.50", "1-12:00:00")):
        # The smoothed form is the Kullback-Leibler one with the reference
        # floored, which is how the comparison was run before: it is a guard on
        # the same divergence rather than a fifth one.
        for div, extra in (("kl", []), ("kl-smooth", ["--ref-smooth", "0.05"]),
                           ("js", []), ("l2", [])):
            out.append(sbatch(
                f"v22-div-{task}-{div}",
                ["python", "scripts/v2/run_d3il_v2.py", "--task", task,
                 "--weightings", w, "--backbone-seeds", "0", "1", "2", "3", "4",
                 "--seeds", "5", "--seed-offset", "700", "--arms", "mop",
                 "--divergence", "kl" if div.startswith("kl") else div,
                 "--workers", "24"] + extra + LAM_FROM
                + ["--out",
                   f"runs/_d3il_v2/v22_ablation_divergence_{task}_{div}.jsonl"],
                25, "40G", hours, dep, dry=dry))
        out.append(sbatch(
            f"v22-prior-{task}",
            ["python", "scripts/v2/run_d3il_v2.py", "--task", task,
             "--weightings", w, "--backbone-seeds", "0", "1", "2", "3", "4",
             "--seeds", "5", "--seed-offset", "700", "--arms", "bmd",
             "--bmd-prior", "fitted", "--workers", "24"] + LAM_FROM
            + ["--out",
               f"runs/_d3il_v2/v22_ablation_fitted_prior_{task}.jsonl"],
            25, "40G", hours, dep, dry=dry))
    for lam in ("0.50", "1.00"):
        out.append(sbatch(
            f"v22-weight-pushing-{lam}",
            ["python", "scripts/v2/run_d3il_v2.py", "--task", "pushing",
             "--weightings", "0.00", "0.10", "--backbone-seeds",
             "0", "1", "2", "3", "4", "--seeds", "5", "--seed-offset", "700",
             "--arms", "mi", "mop", "--lam", lam, "--workers", "24",
             "--out", f"runs/_d3il_v2/v22_ablation_weight_pushing_{lam}.jsonl"],
            25, "50G", "2-00:00:00", dep, dry=dry))
    return out


def stage_frozen(dep=None, dry=False):
    """Each frozen policy under its own latent prior, on the same protocol.

    This is the stage the environment rule hit hardest: it rolls an untrained
    policy out, which is where the corrupted environment showed itself, so it
    is measured again from scratch rather than carried over.
    """
    out = []
    for task, ws, hours in (("avoiding", AVOIDING_W, "4:00:00"),
                            ("pushing", PUSHING_W, "10:00:00")):
        out.append(sbatch(
            f"v22-frozen-{task}",
            ["python", "scripts/v2/frozen_baseline.py", "--task", task,
             "--weightings"] + ws
            + ["--backbone-seeds", "0", "1", "2", "3", "4",
               "--seeds", "5", "--seed-offset", "700", "--workers", "24",
               "--out", f"runs/_d3il_v2/v22_frozen_{task}.jsonl"],
            25, "50G", hours, dep, dry=dry))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "gates", "preflight", "full"])
    a = ap.parse_args()
    dry = a.mode == "plan"

    print("gates")
    g = stage_gates(dry=dry)
    if a.mode == "gates":
        return 0

    print("\nre-clone the Pushing backbones on the new observation")
    c = stage_clone(dep=g, dry=dry)

    print("\npreflight: forty cells through the whole pipeline")
    pf = stage_preflight(dep=c, dry=dry)
    if a.mode == "preflight":
        print("\nthe matrix is not submitted; run 'full' once the preflight "
              "has been read")
        return 0

    print("\ncalibration, which the weight is selected on")
    cal = stage_calibration(dep=pf, dry=dry)
    print("\napply the pre-registered rule")
    sel = stage_select(dep=":".join(j for j in cal if j) if not dry else None,
                       dry=dry)

    print("\nthe matrix, at the weight the rule picked")
    m = stage_matrix(dep=sel, dry=dry)
    print("\nthe frozen policies, on the same protocol")
    stage_frozen(dep=pf, dry=dry)
    print("\nthe ablations")
    stage_ablations(dep=sel, dry=dry)
    print("\nnothing above starts until the gates and the preflight have "
          "returned success, and nothing in the matrix starts until the "
          "weight has been selected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
