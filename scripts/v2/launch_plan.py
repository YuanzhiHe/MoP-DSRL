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
    """The settings every job of a task shares.

    The driver measures every reference first and then runs one pool over all
    the cells the job owns, so sizing the pool to the cell count puts the whole
    job in a single wave. Reference measurement gets the same width.
    """
    t = TASK[task]
    return (["python", "-u", "scripts/v2/run_d3il_v2.py", "--task", task,
             "--mode-groups", str(t["mode_groups"]),
             "--n-rewarded", str(t["n_rewarded"]),
             "--eval-episodes", str(t["eval_episodes"]),
             "--ref-episodes", str(t["ref_episodes"])]
            + BUDGET
            + ["--ref-workers", str(workers), "--workers", str(workers),
               "--out", f"{RUNS}/{out}"])


def calibration(_):
    jobs = []
    for task, t in TASK.items():
        workers = 4 * 3 * 3                     # weightings x seeds x arms
        for lam in CANDIDATES:
            # One file per job. Several jobs appending to one file on a
            # shared filesystem is a way to lose a row, and a lost row is a
            # cell that silently never happened.
            cmd = base(task, f"calibration_{task}_{lam:.2f}.jsonl", workers) + [
                "--weightings", *t["weightings"],
                "--arms", "mi", "mop", "bmd",
                "--seeds", "3", "--seed-offset", str(t["calib_seed"]),
                "--lam", f"{lam:.2f}"]
            jobs.append((f"cal-{task[:2]}-{lam:.2f}", "1-00:00:00",
                         workers, cmd))
    return jobs


def main_matrix(_, choice):
    jobs = []
    for task, t in TASK.items():
        workers = 5 * 5 * 4                     # policies x seeds x arms
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
            jobs.append((f"main-{task[:2]}-{w}", "2-00:00:00",
                         workers, cmd))
    return jobs


def ablation(_, choice):
    """Divergence comparison, skill prior, and the Pushing weight comparison."""
    jobs = []
    workers = 5 * 5                             # policies x seeds, one arm
    for task, t in TASK.items():
        top = t["weightings"][-1]
        lam = choice[task]["mop"]["lam"]
        # l2 is here to separate two explanations of any advantage: matching a
        # reference at all, against the particular choice of total variation.
        # Its derivative p - q is bounded and smooth, so it isolates the first.
        for div, extra in (("js", []), ("l2", []),
                           ("kl", ["--kl-eps", "1e-6"]),
                           ("kl", ["--kl-eps", "1e-8", "--ref-smooth", "0.05"])):
            tag = div + ("-smooth" if "--ref-smooth" in extra else "")
            cmd = base(task, f"ablation_divergence_{task}_{tag}.jsonl",
                       workers) + [
                "--weightings", top, "--backbone-seeds", "0", "1", "2", "3", "4",
                "--arms", "mop", "--seeds", "5",
                "--seed-offset", str(t["main_seed"]),
                "--lam", f"{lam:.2f}", "--divergence", div] + extra
            jobs.append((f"abl-{task[:2]}-{tag}", "1-00:00:00",
                         workers, cmd))

    # The learned-skill arm with its prior fitted to the reference. It is the
    # stronger control, since fitting hands it the target the proposed arm is
    # matching, and it is reported under its own label rather than folded into
    # the main table.
    for task, t in TASK.items():
        cmd = base(task, f"ablation_fitted_prior_{task}.jsonl", workers) + [
            "--weightings", t["weightings"][-1],
            "--backbone-seeds", "0", "1", "2", "3", "4",
            "--arms", "bmd", "--seeds", "5",
            "--seed-offset", str(t["main_seed"]),
            "--lam", f"{choice[task]['bmd']['lam']:.2f}",
            "--bmd-prior", "fitted"]
        jobs.append((f"abl-{task[:2]}-fitted", "1-00:00:00",
                     workers, cmd))

    t = TASK["pushing"]
    for lam in CANDIDATES:
        cmd = base("pushing", f"ablation_weight_pushing_{lam:.2f}.jsonl",
                   2 * 5 * 5 * 2) + [
            "--weightings", "0.00", "0.10",
            "--backbone-seeds", "0", "1", "2", "3", "4",
            "--arms", "mi", "mop", "--seeds", "5",
            "--seed-offset", str(t["main_seed"]), "--lam", f"{lam:.2f}",
            # Where the main matrix already ran this exact configuration the
            # cell key matches and the run is reused rather than repeated. A
            # weight that differs from the main one has a different hash and
            # runs here.
            "--resume", "--resume-also",
            f"{RUNS}/main_pushing_0.00.jsonl",
            f"{RUNS}/main_pushing_0.10.jsonl"]
        jobs.append((f"abl-pu-lam{lam:.2f}", "2-00:00:00",
                     2 * 5 * 5 * 2, cmd))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("calibration", "main", "ablation"))
    ap.add_argument("--headroom", type=int, default=4,
                    help="cores above the worker count, for the parent")
    ap.add_argument("--mem-per-worker-mb", type=int, default=1400,
                    help="measured peak was about 890 MB per worker")
    ap.add_argument("--choice", type=Path,
                    default=ROOT / "runs" / "_d3il_v2" / "lambda_choice.json")
    ap.add_argument("--only", default=None,
                    help="submit only jobs whose name contains this")
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    if args.stage == "calibration":
        jobs = calibration(None)
    else:
        if not args.choice.exists():
            raise SystemExit(f"{args.choice} does not exist: run the "
                             f"calibration and select_lambda.py first")
        choice = json.loads(args.choice.read_text())
        jobs = (main_matrix(None, choice) if args.stage == "main"
                else ablation(None, choice))

    for name, walltime, workers, cmd in jobs:
        if args.only and args.only not in name:
            continue
        cpus = workers + args.headroom
        mem = max(16, (workers * args.mem_per_worker_mb) // 1024 + 8)
        sbatch = ["sbatch", "--parsable", f"--job-name={name}",
                  f"--time={walltime}", f"--cpus-per-task={cpus}",
                  f"--mem={mem}G", "scripts/v2/cpu_job_v2.sh"] + cmd
        print(" ".join(sbatch))
        if args.submit:
            jid = subprocess.run(sbatch, cwd=ROOT, capture_output=True,
                                 text=True, check=True).stdout.strip()
            print(f"  -> {jid}")


if __name__ == "__main__":
    main()
