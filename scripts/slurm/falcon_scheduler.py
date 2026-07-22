#!/usr/bin/env python3
"""Falcon top-up scheduler for the trimmed MoP-DSRL matrix.

The Falcon QOS caps concurrent jobs per partition (h100: 1 run/2 submit,
l40s: 2/4, v100: 2/3), so the 30-run trimmed plan cannot all be queued at once.
This script is IDEMPOTENT and STATELESS: run it periodically (cron / loop). Each
invocation:

  1. enumerates the fixed trimmed plan (partition assigned per run),
  2. classifies every run as done (final_summary.json) / active (a job with its
     name is in my squeue) / pending,
  3. for each partition, submits pending runs until that partition's live-job
     count reaches its submit cap.

Continuation across the 3-day wall is handled by train_job.sh's self-resubmit
(a run stays "active" while its chain runs), so this scheduler only ever STARTS
new runs as slots free. When every run has final_summary.json it submits nothing.
"""
from __future__ import annotations
import subprocess, sys, os, time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))
from scripts.submit_all import (  # noqa: E402
    plan_tier1_main, plan_tier1_baselines, plan_robustness, plan_ablations,
    OPTIMAL_DEFAULTS, DEFAULT_RUNS_ROOT, DEFAULT_LOGS_ROOT,
)

RUNS_ROOT = Path(DEFAULT_RUNS_ROOT)
LOGS_ROOT = Path(DEFAULT_LOGS_ROOT)
TOTAL_ENV_STEPS = 250_000
TIME = "72:00:00"

# Per-partition submit caps (QOSMaxSubmitJobPerUser). Only h100 + l40s are
# usable: V100 (sm_70) can't run the cu13 torch build (no kernel image), and
# gpu_h200 submits are rejected by a submit-count limit. Filling both to their
# caps gives ~3 concurrently RUNNING jobs (h100:1, l40s:2).
PARTITIONS = {
    "gpu_h100": 2,
    "gpu_l40s": 4,
}


def build_plan() -> list:
    OPTIMAL_DEFAULTS["total_env_steps"] = TOTAL_ENV_STEPS
    plan = []
    # main: workstation owns seed 0; Falcon does seeds 1,2 on goal/spatial/object
    # (no libero_10/long vocab on Falcon). -> h100.
    for r in plan_tier1_main(["pi05"], ["libero_goal", "libero_spatial", "libero_object"], [1, 2]):
        r.partition = "gpu_h100"
        plan.append(r)
    # baselines: 5 x seeds 0,1 -> l40s
    for r in plan_tier1_baselines([0, 1]):
        r.partition = "gpu_l40s"
        plan.append(r)
    # lambda ablation only, seeds 0,1 -> l40s (v100 unusable with cu13 torch)
    for r in plan_ablations([0, 1]):
        if r.group == "ablation_lambda":
            r.partition = "gpu_l40s"
            plan.append(r)
    # robustness: ood + sparse, seeds 0,1 -> l40s
    for r in plan_robustness([0, 1]):
        if r.group in ("robustness_ood", "robustness_sparse"):
            r.partition = "gpu_l40s"
            plan.append(r)
    # common SLURM resources
    for r in plan:
        r.cpus, r.mem, r.gres, r.time = 8, "64G", "gpu:1", TIME
        r.job_name = f"mop-{r.group}-{r.name}"[:60]
    return plan


def my_active_job_names() -> set[str]:
    """Job names currently in my squeue (running or pending)."""
    out = subprocess.run(
        ["squeue", "-u", os.environ["USER"], "-h", "-o", "%j|%P"],
        capture_output=True, text=True,
    ).stdout
    names, per_part = set(), {}
    for line in out.splitlines():
        if not line.strip():
            continue
        name, part = (line.split("|") + [""])[:2]
        names.add(name)
        per_part[part] = per_part.get(part, 0) + 1
    return names, per_part


def main() -> int:
    plan = build_plan()
    active_names, live_per_part = my_active_job_names()

    done = sum(1 for r in plan if (r.run_dir(RUNS_ROOT) / "final_summary.json").exists())
    print(f"[scheduler] plan={len(plan)} done={done} "
          f"live={ {k: live_per_part.get(k, 0) for k in PARTITIONS} }")

    submitted = 0
    def ckpt_step(rd) -> int:
        """Latest checkpoint step for a run (0 if none / not started)."""
        best = 0
        for c in rd.glob("checkpoint_step*.pt"):
            try:
                best = max(best, int(c.stem.replace("checkpoint_step", "")))
            except ValueError:
                pass
        return best

    for part, cap in PARTITIONS.items():
        live = live_per_part.get(part, 0)
        # Build the pending list for this partition, then order it so IN-PROGRESS
        # runs (have a checkpoint) resume BEFORE fresh runs start — highest step
        # first. This prevents the priority inversion where a run that just
        # wall-stopped at high step waits behind freshly-started 0-step runs.
        pend = []
        for r in [x for x in plan if x.partition == part]:
            rd = r.run_dir(RUNS_ROOT)
            if (rd / "final_summary.json").exists():
                continue                          # done
            if r.job_name in active_names:
                continue                          # already running / queued
            step = ckpt_step(rd)
            # Tiny freshness guard: skip a checkpoint written seconds ago (job may
            # still be finalizing). No self-resubmit races anymore, so 90s is plenty.
            if step > 0:
                newest = max(c.stat().st_mtime for c in rd.glob("checkpoint_step*.pt"))
                if (time.time() - newest) < 90:
                    continue
            pend.append((step, r))
        pend.sort(key=lambda t: -t[0])            # in-progress (high step) first
        for step, r in pend:
            if live >= cap:
                break
            rd = r.run_dir(RUNS_ROOT)
            rd.mkdir(parents=True, exist_ok=True)
            cmd = r.sbatch_command(RUNS_ROOT, LOGS_ROOT)
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                tag = f"resume@{step}" if step else "start"
                print(f"[scheduler] SUBMIT {part} {r.group}/{r.name} ({tag}) -> {res.stdout.strip()}")
                live += 1
                submitted += 1
                active_names.add(r.job_name)
            else:
                err = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "?"
                print(f"[scheduler] reject {part} {r.group}/{r.name}: {err}")
                break   # partition full or policy block; stop trying this partition

    print(f"[scheduler] submitted {submitted} new job(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
