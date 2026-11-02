#!/usr/bin/env python3
"""Emit the launch commands for the corrected programme.

Every command that produces a reported number comes from here, so the record of
how the experiments were started is a file rather than shell history. Nothing is
submitted unless --submit is given.

Stages:

  calibration  the weight grid on the unnumbered checkpoints and the
               calibration seed block, which the main analysis excludes;
  main         the full matrix at the weights select_lambda.py chose, one job
               per task and demonstration weighting;
  ablation     the divergence comparison at each task's highest weighting, and
               the Pushing weight comparison on its two lowest.

Reference measurement is shared: a job that needs a reference the cache already
holds reads it instead of measuring it again, so splitting the matrix by
weighting costs nothing extra.
"""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = "runs/_d3il_v2"

TASK = {
    "avoiding": dict(weightings=["0.00", "0.15", "0.30", "0.60"],
                     mode_groups=6, n_rewarded=4, eval_episodes=500,
                     ref_episodes=900, main_seed=700, calib_seed=500),
    "pushing": dict(weightings=["0.00", "0.10", "0.25", "0.50"],
                    mode_groups=4, n_rewarded=1, eval_episodes=300,
                    ref_episodes=600, main_seed=900, calib_seed=850),
}
CANDIDATES = (0.25, 0.50, 1.00)
BUDGET = ["--iters", "60", "--episodes-per-iter", "16", "--lr", "0.003",
          "--M", "10"]


def base(task, out, workers):
    t = TASK[task]
    return (["python", "-u", "scripts/v2/run_d3il_v2.py", "--task", task,
             "--mode-groups", str(t["mode_groups"]),
             "--n-rewarded", str(t["n_rewarded"]),
             "--eval-episodes", str(t["eval_episodes"]),
             "--ref-episodes", str(t["ref_episodes"])]
            + BUDGET
            + ["--ref-workers", str(workers), "--workers", str(workers),
               "--out", f"{RUNS}/{out}"])


def calibration(workers):
    jobs = []
    for task, t in TASK.items():
        for lam in CANDIDATES:
            # One file per job. Several jobs appending to one file on a
            # shared filesystem is a way to lose a row, and a lost row is a
            # cell that silently never happened.
            cmd = base(task, f"calibration_{task}_{lam:.2f}.jsonl", workers) + [
                "--weightings", *t["weightings"],
                "--arms", "mi", "mop", "bmd",
                "--seeds", "3", "--seed-offset", str(t["calib_seed"]),
                "--lam", f"{lam:.2f}"]
            jobs.append((f"cal-{task[:2]}-{lam:.2f}", "12:00:00", cmd))
    return jobs


def main_matrix(workers, choice):
    jobs = []
    for task, t in TASK.items():
        lam = choice[task]
        for w in t["weightings"]:
            cmd = base(task, f"main_{task}_{w}.jsonl", workers) + [
                "--weightings", w, "--backbone-seeds", "0", "1", "2", "3", "4",
                "--arms", "dsrl", "mi", "mop", "bmd",
                "--seeds", "5", "--seed-offset", str(t["main_seed"]),
                "--lam", "1.0",
                "--lam-mop", f"{lam['mop']['lam']:.2f}",
                "--lam-mi", f"{lam['mi']['lam']:.2f}",
                "--lam-bmd", f"{lam['bmd']['lam']:.2f}"]
            jobs.append((f"main-{task[:2]}-{w}", "2-00:00:00", cmd))
    return jobs


def ablation(workers, choice):
    """Divergence comparison, and the Pushing weight comparison."""
    jobs = []
    for task, t in TASK.items():
        top = t["weightings"][-1]
        lam = choice[task]["mop"]["lam"]
        for div, extra in (("js", []), ("kl", ["--kl-eps", "1e-6"]),
                           ("kl", ["--kl-eps", "1e-8", "--ref-smooth", "0.05"])):
            tag = div + ("-smooth" if "--ref-smooth" in extra else "")
            cmd = base(task, f"ablation_divergence_{task}_{tag}.jsonl",
                       workers) + [
                "--weightings", top, "--backbone-seeds", "0", "1", "2", "3", "4",
                "--arms", "mop", "--seeds", "5",
                "--seed-offset", str(t["main_seed"]),
                "--lam", f"{lam:.2f}", "--divergence", div] + extra
            jobs.append((f"abl-{task[:2]}-{tag}", "1-00:00:00", cmd))

    t = TASK["pushing"]
    for lam in CANDIDATES:
        cmd = base("pushing", f"ablation_weight_pushing_{lam:.2f}.jsonl",
                   workers) + [
            "--weightings", "0.00", "0.10",
            "--backbone-seeds", "0", "1", "2", "3", "4",
            "--arms", "mi", "mop", "--seeds", "5",
            "--seed-offset", str(t["main_seed"]), "--lam", f"{lam:.2f}"]
        jobs.append((f"abl-pu-lam{lam:.2f}", "1-00:00:00", cmd))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("calibration", "main", "ablation"))
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--cpus", type=int, default=96)
    ap.add_argument("--mem", default="160G")
    ap.add_argument("--choice", type=Path,
                    default=ROOT / "runs" / "_d3il_v2" / "lambda_choice.json")
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    if args.stage == "calibration":
        jobs = calibration(args.workers)
    else:
        if not args.choice.exists():
            raise SystemExit(f"{args.choice} does not exist: run the "
                             f"calibration and select_lambda.py first")
        choice = json.loads(args.choice.read_text())
        jobs = (main_matrix(args.workers, choice) if args.stage == "main"
                else ablation(args.workers, choice))

    for name, walltime, cmd in jobs:
        sbatch = ["sbatch", "--parsable", f"--job-name={name}",
                  f"--time={walltime}", f"--cpus-per-task={args.cpus}",
                  f"--mem={args.mem}", "scripts/v2/cpu_job_v2.sh"] + cmd
        print(" ".join(sbatch))
        if args.submit:
            jid = subprocess.run(sbatch, cwd=ROOT, capture_output=True,
                                 text=True, check=True).stdout.strip()
            print(f"  -> {jid}")


if __name__ == "__main__":
    main()
