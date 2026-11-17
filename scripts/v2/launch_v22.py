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


def sbatch(name, cmd, cpus, time, dep=None, array=None, dry=False):
    """Submit one job, optionally held until ``dep`` has succeeded.

    Memory comes from the wrapper, which asks for it per core because that is
    how these stages use it. A job carrying both forms is refused outright, so
    there is nothing to pass here.
    """
    a = ["sbatch", "--parsable", f"--job-name={name}",
         f"--cpus-per-task={cpus}", f"--time={time}"]
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


# The queue allows ten submitted jobs and four running, so the cohort's
# thirty-two driver calls are grouped into eight stages that each take a node
# and run their calls in series. run_stage.py holds the grouping.
# A stage runs its calls one after another and each call fans out over its own
# cells, so the cores a stage can use are bounded by the widest call it makes,
# not by the node. Asking for all 192 made every stage wait for a whole node to
# drain: the scheduler put calib-pushing, which never has more than 36 cells in
# flight, a day out. Each stage now asks for its widest call plus a core for the
# parent, which is doing the writing.
STAGES = [
    # name,            cpus, walltime,     depends on      widest call
    ("prep",            24, "12:00:00",    None),          # 8 cells
    ("calib-avoiding",  48, "12:00:00",    "prep"),        # 36
    ("calib-pushing",   48, "1-00:00:00",  "prep"),        # 36
    ("main-avoiding",  104, "12:00:00",    "calib-avoiding,calib-pushing"),
    ("main-pushing",   104, "2-00:00:00",  "main-avoiding"),      # 100
    ("frozen",          96, "8:00:00",     "prep"),
    ("abl-avoiding",    32, "12:00:00",    "main-avoiding"),      # 25
    ("abl-pushing",    104, "2-00:00:00",  "main-avoiding"),      # 25 and 100
]


def write_clone_plan():
    """The twenty Pushing backbones the prep stage re-clones."""
    import json
    # The matrix reads bb0..bb4. The calibration stage passes no backbone
    # seed, which resolves to the unsuffixed checkpoint, so that one has to be
    # re-cloned as well or the calibration opens a ten-channel file against a
    # twelve-channel environment. It is trained at a seed the matrix never
    # uses, which is what keeps the weight from being chosen on a policy the
    # matrix then reports.
    jobs = [(w, bb) for w in PUSHING_W for bb in range(5)]
    jobs += [(w, None) for w in PUSHING_W]
    (ROOT / "runs" / "_d3il_v2" / "_clone_plan.json").write_text(
        json.dumps(jobs))
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "full"])
    ap.add_argument("--only", nargs="*", default=None,
                    help="submit just these stages, with their dependencies "
                         "taken from whatever is already in the queue")
    a = ap.parse_args()
    dry = a.mode == "plan"
    jobs = write_clone_plan()
    print(f"clone plan: {len(jobs)} Pushing backbones\n")

    ids: dict[str, str] = {}
    for name, cpus, wall, dep in STAGES:
        if a.only and name not in a.only:
            continue
        d = None
        if dep:
            got = [ids[x] for x in dep.split(",") if x in ids]
            d = ":".join(got) if got else None
        jid = sbatch(f"v22-{name}", ["python", "scripts/v2/run_stage.py", name],
                     cpus, wall, d, dry=dry)
        if jid:
            ids[name] = jid
    print("\neight stages. prep runs the gates, the re-clone and the forty"
          "-cell slice, and nothing downstream starts unless it returns"
          " success.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
