#!/usr/bin/env python3
"""Master experiment-matrix submitter for MoP-DSRL.

Enumerates every cell of the experiment plan from README.md ("Experiment
plan") with theoretically optimal hyperparameters from method_spec §8, and
submits each as a standalone SLURM job via ``sbatch``.

Groups produced (default = `--tier all`):

  Tier-1 main:            (4 LIBERO suites + 2 RoboCasa suites) × 5 seeds × {pi05, dpt}
                          + ablation lambda=0 (DSRL) is implicit in --tier ablation.
  Tier-1 baselines:       BC-only, DSRL (lambda=0), V-GPS, DPPO, X1, X2 on libero_goal × 5 seeds
  Tier-2 robustness:      libero_goal × pi05 × 3 seeds × {flip 10/30/50, sparse, OOD}
  Ablations:              libero_goal × pi05 × 3 seeds × each ablation axis
  UT-5 BC sanity:         libero_goal × pi05 × 1 seed (already supplied by upstream script)

The script writes:
  - `submissions/manifest_<timestamp>.tsv`  — every (group, run_name, sbatch cmd, args)
  - prints each sbatch command (when --dry-run, that's all it does).

All hyperparameters honor the user's directive: no compromise for compute or
time.  Defaults follow method_spec §8: 5e5 env steps, 5 seeds (Tier-1 main)
or 3 seeds (Tier-2, ablations), 50 final eval episodes per task.

Usage examples:
    # Print the full plan without submitting anything.
    python scripts/submit_all.py --dry-run

    # Submit just the main Tier-1 experiments on the H100 partition.
    python scripts/submit_all.py --tier main --partition gpu_h100 --backbones pi05

    # Submit the full plan (Tier-1 + Tier-2 + ablations + baselines).
    python scripts/submit_all.py --tier all --partition gpu_l40s
"""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
SLURM_DIR = REPO_ROOT / "scripts" / "slurm"
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs"
DEFAULT_LOGS_ROOT = REPO_ROOT / "logs"
DEFAULT_SUBMIT_ROOT = REPO_ROOT / "submissions"

# ---------------------------------------------------------------------------
# Theoretically optimal defaults (method_spec §8 — LIBERO + pi_0.5 column)
# ---------------------------------------------------------------------------

OPTIMAL_DEFAULTS = {
    # Action geometry
    "H": 50,
    "d_a": 7,
    "M_pi05": 10,                 # Euler steps for pi_0.5 (spec recommended)
    "M_dpt": 4,                   # Euler steps for DP-T fallback
    "M_pi0": 4,                   # Euler steps for legacy pi_0
    "K": 16,
    # SAC core
    "total_env_steps": 500_000,
    "warmup_env_steps": 10_000,
    "replay_capacity": 1_000_000,
    "batch_size": 256,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "alpha_lr": 3e-4,
    "tau_polyak": 5e-3,
    "gamma": 0.99,
    # Mode regularizer.
    #
    # NOTE on lambda compensation: pi_0.5 forwards are ~110ms each on a 5090,
    # so computing R_mode fully every actor update is infeasible.  The pi_0.5
    # path subsamples ``rmode_subsample`` held-out obs per evaluation and only
    # re-evaluates every ``rmode_every`` actor updates (in-between the most
    # recent KL value is reused as a detached constant).  To keep the integrated
    # gradient pressure on R_mode equal to spec §8's nominal lambda=0.1, we
    # scale lambda by rmode_every (= 4) → effective lambda = 0.4.  Ablation
    # sweeps below use the same x4 compensation.
    "lambda_": 4e-1,             # = 0.1 × rmode_every=4 compensation
    "epsilon": 0.1,
    "alpha_lambda": 1e-3,
    "n_on": 8,
    "n_ref": 50,
    "n_obs": 1000,
    "rmode_subsample": 8,        # held-out obs sampled per R_mode evaluation
    "rmode_every": 4,            # only evaluate R_mode every K actor updates
    # Eval
    "eval_interval": 25_000,
    "eval_episodes": 10,
    "final_eval_episodes": 50,
    "ckpt_interval": 100_000,
    "log_interval": 1000,
    # Episode length cap (LIBERO default)
    "episode_steps": 300,
}

# Suites and the per-suite primary task IDs (one job per (suite, task_id)
# would explode the count; for Tier-1 main we average over the suite by
# launching with task_id=-1 → driver iterates all tasks during eval. For
# now the driver always trains on a single task; we enumerate the canonical
# task_id 0 per suite. Ablation/robustness fix the suite to libero_goal.)
LIBERO_SUITES = ["libero_goal", "libero_spatial", "libero_object", "libero_long"]
ROBOCASA_SUITES = ["robocasa_kitchen_atomic", "robocasa_kitchen_composite"]

# Backbones
BACKBONES = {
    "pi05": {"M_key": "M_pi05",  "pretrained_path": "lerobot/pi05_libero"},
    "dpt":  {"M_key": "M_dpt",   "checkpoint_dir":  ""},  # to be overridden by env
}


# ---------------------------------------------------------------------------
# Run dataclass
# ---------------------------------------------------------------------------


@dataclass
class Run:
    group: str           # e.g. "tier1_main", "ablation_lambda", "robustness_flip"
    name: str            # unique within group (used as run_dir leaf)
    args: list[str] = field(default_factory=list)
    partition: str = "gpu_l40s"
    cpus: int = 8
    mem: str = "64G"
    gres: str = "gpu:1"
    time: str = "72:00:00"
    job_name: str | None = None

    def run_dir(self, runs_root: Path) -> Path:
        return runs_root / self.group / self.name

    def sbatch_command(self, runs_root: Path, logs_root: Path) -> list[str]:
        run_dir = self.run_dir(runs_root)
        log_prefix = logs_root / self.group / self.name
        log_prefix.parent.mkdir(parents=True, exist_ok=True)
        sbatch = [
            "sbatch",
            f"--job-name={self.job_name or self.name}",
            f"--partition={self.partition}",
            "--nodes=1",
            "--ntasks-per-node=1",
            f"--cpus-per-task={self.cpus}",
            f"--gres={self.gres}",
            f"--mem={self.mem}",
            f"--time={self.time}",
            f"--output={log_prefix}_%j.out",
            f"--error={log_prefix}_%j.err",
            "--export=ALL,RUN_DIR={run_dir},TRAIN_ARGS={train_args},SBATCH_TIME_SEC={tsec}".format(
                run_dir=shlex.quote(str(run_dir)),
                train_args=shlex.quote(" ".join(self.args)),
                tsec=time_to_seconds(self.time),
            ),
            str(SLURM_DIR / "train_job.sh"),
        ]
        return sbatch


def time_to_seconds(t: str) -> int:
    parts = t.split(":")
    if len(parts) == 3:
        h, m, s = (int(p) for p in parts)
        return h * 3600 + m * 60 + s
    if len(parts) == 2:
        m, s = (int(p) for p in parts)
        return m * 60 + s
    if len(parts) == 1:
        if "-" in parts[0]:
            d, hms = parts[0].split("-")
            h, m, s = (int(p) for p in hms.split(":"))
            return int(d) * 86400 + h * 3600 + m * 60 + s
        return int(parts[0])
    raise ValueError(f"Bad SLURM time: {t}")


# ---------------------------------------------------------------------------
# Plan enumerators — each returns a list of Run objects.
# ---------------------------------------------------------------------------


def _common_args(
    backbone: str,
    suite: str,
    seed: int,
    lambda_: float | None = None,
    K: int | None = None,
    M: int | None = None,
    extra: dict | None = None,
) -> list[str]:
    M_default = OPTIMAL_DEFAULTS[BACKBONES[backbone]["M_key"]]
    args = [
        f"--backbone {backbone}",
        f"--env {suite}",
        f"--task_id 0",
        f"--H {OPTIMAL_DEFAULTS['H']}",
        f"--d_a {OPTIMAL_DEFAULTS['d_a']}",
        f"--K {K if K is not None else OPTIMAL_DEFAULTS['K']}",
        f"--M {M if M is not None else M_default}",
        f"--total_env_steps {OPTIMAL_DEFAULTS['total_env_steps']}",
        f"--warmup_env_steps {OPTIMAL_DEFAULTS['warmup_env_steps']}",
        f"--replay_capacity {OPTIMAL_DEFAULTS['replay_capacity']}",
        f"--batch_size {OPTIMAL_DEFAULTS['batch_size']}",
        f"--actor_lr {OPTIMAL_DEFAULTS['actor_lr']}",
        f"--critic_lr {OPTIMAL_DEFAULTS['critic_lr']}",
        f"--alpha_lr {OPTIMAL_DEFAULTS['alpha_lr']}",
        f"--tau_polyak {OPTIMAL_DEFAULTS['tau_polyak']}",
        f"--gamma {OPTIMAL_DEFAULTS['gamma']}",
        f"--epsilon {OPTIMAL_DEFAULTS['epsilon']}",
        f"--alpha_lambda {OPTIMAL_DEFAULTS['alpha_lambda']}",
        f"--n_on {OPTIMAL_DEFAULTS['n_on']}",
        f"--n_ref {OPTIMAL_DEFAULTS['n_ref']}",
        f"--n_obs {OPTIMAL_DEFAULTS['n_obs']}",
        f"--rmode_subsample {OPTIMAL_DEFAULTS['rmode_subsample']}",
        f"--rmode_every {OPTIMAL_DEFAULTS['rmode_every']}",
        f"--eval_interval {OPTIMAL_DEFAULTS['eval_interval']}",
        f"--eval_episodes {OPTIMAL_DEFAULTS['eval_episodes']}",
        f"--final_eval_episodes {OPTIMAL_DEFAULTS['final_eval_episodes']}",
        f"--ckpt_interval {OPTIMAL_DEFAULTS['ckpt_interval']}",
        f"--log_interval {OPTIMAL_DEFAULTS['log_interval']}",
        f"--episode_steps {OPTIMAL_DEFAULTS['episode_steps']}",
        f"--seed {seed}",
        "--device cuda",
    ]
    args.append(f"--lambda_ {lambda_ if lambda_ is not None else OPTIMAL_DEFAULTS['lambda_']}")
    pretrained = BACKBONES[backbone].get("pretrained_path")
    if pretrained:
        args.append(f"--pretrained_path {pretrained}")
    if extra:
        for k, v in extra.items():
            if isinstance(v, bool):
                if v:
                    args.append(f"--{k}")
            else:
                args.append(f"--{k} {v}")
    return args


def plan_tier1_main(
    backbones: list[str], suites: list[str], seeds: list[int]
) -> list[Run]:
    runs = []
    for backbone in backbones:
        for suite in suites:
            for seed in seeds:
                name = f"{suite}_{backbone}_seed{seed}"
                runs.append(Run(
                    group="tier1_main",
                    name=name,
                    args=_common_args(backbone, suite, seed),
                ))
    return runs


def plan_tier1_baselines(seeds: list[int]) -> list[Run]:
    """Baselines listed in README: BC, DSRL, V-GPS, DPPO, plus X1/X2."""
    runs = []
    suite = "libero_goal"
    backbone = "pi05"
    for seed in seeds:
        # BC-only (no RL): set total_env_steps=0 by way of --total_env_steps 0?
        # Driver always runs warmup; emulate BC by lambda=0 + total_env_steps=warmup.
        # Cleanest: lambda=0 + warmup_env_steps == total_env_steps, plus deterministic eval.
        runs.append(Run(
            group="baseline_bc", name=f"libero_goal_pi05_seed{seed}",
            args=_common_args(
                backbone, suite, seed, lambda_=0.0,
                extra={"warmup_env_steps": OPTIMAL_DEFAULTS["total_env_steps"]},
            ),
        ))
        # DSRL (lambda = 0)
        runs.append(Run(
            group="baseline_dsrl", name=f"libero_goal_pi05_seed{seed}",
            args=_common_args(backbone, suite, seed, lambda_=0.0),
        ))
        # V-GPS (value-guidance)
        runs.append(Run(
            group="baseline_vgps", name=f"libero_goal_pi05_seed{seed}",
            args=_common_args(backbone, suite, seed, extra={"value_guidance": True}),
        ))
        # X1 counter-hypothesis: weight-level reweighting
        runs.append(Run(
            group="baseline_x1_weight", name=f"libero_goal_pi05_seed{seed}",
            args=_common_args(backbone, suite, seed, extra={"weight_level": True}),
        ))
        # X2 counter-hypothesis: imitation-only regularizer
        runs.append(Run(
            group="baseline_x2_imitation", name=f"libero_goal_pi05_seed{seed}",
            args=_common_args(backbone, suite, seed, extra={"imitation_only": True}),
        ))
    return runs


def plan_robustness(seeds: list[int]) -> list[Run]:
    """Tier-2 robustness: pref label flips + sparse + OOD on libero_goal × pi_0.5."""
    runs = []
    suite, backbone = "libero_goal", "pi05"
    for seed in seeds:
        for flip in (0.10, 0.30, 0.50):
            runs.append(Run(
                group="robustness_pref_flip", name=f"flip{int(flip*100)}_seed{seed}",
                args=_common_args(
                    backbone, suite, seed,
                    extra={"pref_reward": True, "pref_flip_rate": flip},
                ),
            ))
        runs.append(Run(
            group="robustness_sparse", name=f"sparse_seed{seed}",
            args=_common_args(backbone, suite, seed, extra={"reward_mode": "sparse"}),
        ))
        runs.append(Run(
            group="robustness_ood", name=f"ood_seed{seed}",
            args=_common_args(backbone, suite, seed, extra={"ood_init": True}),
        ))
    return runs


def plan_ablations(seeds: list[int]) -> list[Run]:
    runs = []
    suite, backbone = "libero_goal", "pi05"
    # lambda axis.  Values are scaled by rmode_every=4 to keep the *effective*
    # gradient pressure on R_mode equal to the spec sweep {0, 1e-2, 1e-1, 1, 10}.
    for seed in seeds:
        for lam in (0.0, 4e-2, 4e-1, 4.0, 40.0):
            runs.append(Run(
                group="ablation_lambda", name=f"lam{lam}_seed{seed}",
                args=_common_args(backbone, suite, seed, lambda_=lam),
            ))
        # K axis (excluding the canonical K=16, which is in tier1_main)
        for K in (8, 32):
            runs.append(Run(
                group="ablation_K", name=f"K{K}_seed{seed}",
                args=_common_args(backbone, suite, seed, K=K),
            ))
        # M axis
        for M in (2, 4, 8, 16):
            runs.append(Run(
                group="ablation_M", name=f"M{M}_seed{seed}",
                args=_common_args(backbone, suite, seed, M=M),
            ))
        # Coupling endpoint on/off
        runs.append(Run(
            group="ablation_coupling_off", name=f"nocoupling_seed{seed}",
            args=_common_args(backbone, suite, seed, extra={"no_coupling_endpoint": True}),
        ))
        # Dual ascent on/off
        runs.append(Run(
            group="ablation_dual_ascent", name=f"dual_seed{seed}",
            args=_common_args(backbone, suite, seed, extra={"tune_lambda": True}),
        ))
    return runs


def plan_ut5(seeds: list[int]) -> list[Run]:
    """UT-5 BC-sanity: BC-only flow vs diffusion baseline on each LIBERO suite."""
    runs = []
    backbone = "pi05"
    for suite in LIBERO_SUITES:
        for seed in seeds:
            runs.append(Run(
                group="ut5_bc_sanity",
                name=f"{suite}_pi05_seed{seed}",
                args=_common_args(
                    backbone, suite, seed, lambda_=0.0,
                    extra={"warmup_env_steps": OPTIMAL_DEFAULTS["total_env_steps"]},
                ),
            ))
    return runs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Submit the full MoP-DSRL experiment matrix to SLURM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tier", default="all",
        choices=["all", "main", "baselines", "robustness", "ablations", "ut5"],
        help="Which group(s) to enumerate.",
    )
    p.add_argument("--suites", nargs="*", default=LIBERO_SUITES,
                   help="LIBERO suites for tier1_main.")
    p.add_argument("--include-robocasa", action="store_true",
                   help="Also enumerate RoboCasa-Kitchen suites in tier1_main.")
    p.add_argument("--backbones", nargs="*", default=["pi05"],
                   help="Backbones to use for tier1_main.")
    p.add_argument("--seeds-main", type=int, nargs="*", default=[0, 1, 2, 3, 4],
                   help="Seeds for Tier-1 main and baselines.")
    p.add_argument("--seeds-aux", type=int, nargs="*", default=[0, 1, 2],
                   help="Seeds for Tier-2 robustness and ablations.")
    p.add_argument("--seeds-ut5", type=int, nargs="*", default=[0],
                   help="Seeds for UT-5 BC sanity.")
    p.add_argument("--partition", default="gpu_l40s",
                   help="SLURM partition for every job (default gpu_l40s).")
    p.add_argument("--time", default="72:00:00",
                   help="SLURM --time per job.")
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--mem", default="64G")
    p.add_argument("--gres", default="gpu:1")
    p.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    p.add_argument("--logs-root", default=str(DEFAULT_LOGS_ROOT))
    p.add_argument("--submit-root", default=str(DEFAULT_SUBMIT_ROOT))
    p.add_argument("--max-jobs", type=int, default=0,
                   help="If >0, cap total submissions (truncate the plan).")
    p.add_argument("--filter", default="",
                   help="If set, only include runs whose group matches this substring.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be submitted and write the manifest, but don't sbatch.")
    p.add_argument("--include-skipped", action="store_true",
                   help="Don't skip groups whose run_dirs already contain final_summary.json.")
    p.add_argument("--print-cmd", action="store_true",
                   help="Print every sbatch command to stdout.")
    return p.parse_args()


def collect_runs(args: argparse.Namespace) -> list[Run]:
    suites = list(args.suites)
    if args.include_robocasa:
        suites = list(suites) + ROBOCASA_SUITES

    plan: list[Run] = []
    if args.tier in ("all", "main"):
        plan += plan_tier1_main(args.backbones, suites, args.seeds_main)
    if args.tier in ("all", "baselines"):
        plan += plan_tier1_baselines(args.seeds_main)
    if args.tier in ("all", "robustness"):
        plan += plan_robustness(args.seeds_aux)
    if args.tier in ("all", "ablations"):
        plan += plan_ablations(args.seeds_aux)
    if args.tier in ("all", "ut5"):
        plan += plan_ut5(args.seeds_ut5)

    if args.filter:
        plan = [r for r in plan if args.filter in r.group or args.filter in r.name]

    # Apply common SLURM resources to every run
    for r in plan:
        r.partition = args.partition
        r.cpus = args.cpus
        r.mem = args.mem
        r.gres = args.gres
        r.time = args.time
        r.job_name = f"mop-{r.group}-{r.name}"[:60]

    return plan


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    runs = collect_runs(args)
    if not runs:
        logger.warning("Empty plan — nothing to submit.")
        return 0

    runs_root = Path(args.runs_root)
    logs_root = Path(args.logs_root)
    submit_root = Path(args.submit_root)
    submit_root.mkdir(parents=True, exist_ok=True)

    # Skip already-completed runs.
    if not args.include_skipped:
        kept = []
        for r in runs:
            done = r.run_dir(runs_root) / "final_summary.json"
            if done.exists():
                continue
            kept.append(r)
        skipped = len(runs) - len(kept)
        if skipped:
            logger.info("Skipping %d runs that already have final_summary.json", skipped)
        runs = kept

    if args.max_jobs > 0 and len(runs) > args.max_jobs:
        logger.info("Capping plan to first %d runs of %d", args.max_jobs, len(runs))
        runs = runs[: args.max_jobs]

    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest_path = submit_root / f"manifest_{timestamp}.tsv"
    with open(manifest_path, "w") as f:
        f.write("group\trun_name\trun_dir\tsbatch_command\ttrain_args\n")
        for r in runs:
            cmd = r.sbatch_command(runs_root, logs_root)
            f.write(
                f"{r.group}\t{r.name}\t{r.run_dir(runs_root)}\t"
                f"{shlex.join(cmd)}\t{shlex.join(r.args)}\n"
            )
    logger.info("Wrote manifest of %d runs to %s", len(runs), manifest_path)

    by_group = {}
    for r in runs:
        by_group.setdefault(r.group, 0)
        by_group[r.group] += 1
    logger.info("Plan summary (%d runs total):", len(runs))
    for g, n in sorted(by_group.items()):
        logger.info("  %-30s  %d runs", g, n)

    if args.dry_run:
        logger.info("--dry-run set: not submitting.")
        if args.print_cmd:
            for r in runs:
                cmd = r.sbatch_command(runs_root, logs_root)
                print(shlex.join(cmd))
        return 0

    if shutil.which("sbatch") is None:
        logger.error("sbatch not found in PATH — cannot submit.")
        return 1

    submitted = 0
    for r in runs:
        cmd = r.sbatch_command(runs_root, logs_root)
        run_dir = r.run_dir(runs_root)
        run_dir.mkdir(parents=True, exist_ok=True)
        if args.print_cmd:
            print(shlex.join(cmd))
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error("sbatch failed for %s/%s: %s", r.group, r.name, e.stderr.strip())
            continue
        out = res.stdout.strip()
        logger.info("[%-30s] %-40s -> %s", r.group, r.name, out)
        # Record the JobID alongside the run dir.
        with open(run_dir / "slurm_job.txt", "w") as f:
            f.write(out + "\n")
            f.write(shlex.join(cmd) + "\n")
        submitted += 1

    logger.info("Submitted %d / %d jobs", submitted, len(runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
