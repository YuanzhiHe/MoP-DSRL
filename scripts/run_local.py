#!/usr/bin/env python3
"""Local multi-GPU runner for the MoP-DSRL experiment matrix.

Reuses the experiment-plan enumeration from ``scripts/submit_all.py`` but
launches every run as a local subprocess instead of an sbatch job.  Useful
on workstations / single-node servers with N GPUs (e.g. 2×RTX 5090) where
SLURM is not available.

Behaviour
---------
- Enumerates the same experiment matrix as ``submit_all.py``.
- Maintains a pool of ``--n-gpus`` worker slots, each pinned to its own
  ``CUDA_VISIBLE_DEVICES`` index.
- Pops one run at a time from the queue and launches it on the first free
  slot; the slot becomes available again when the subprocess exits.
- Writes per-run stdout/stderr to ``logs/<group>/<run_name>.log``.
- Writes ``submissions/local_manifest_<timestamp>.tsv`` with one row per
  launched run (PID, GPU, exit code, wall seconds).

Usage
-----
    # Dry-run: print the plan, no subprocesses launched.
    python scripts/run_local.py --dry-run --tier all

    # Run the full plan on 2 GPUs concurrently.
    python scripts/run_local.py --tier all --n-gpus 2

    # Run just tier1_main with seeds 0–2 (3 seeds) on 2 GPUs.
    python scripts/run_local.py --tier main --seeds-main 0 1 2 --n-gpus 2

    # Resume after a crash — skips any run with final_summary.json.
    python scripts/run_local.py --tier all --n-gpus 2  # (default behaviour)

Use ``--gpu-ids 0 1`` to pick specific CUDA device IDs (default = first N).
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

# Reuse the plan enumeration from the SLURM submitter.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from scripts.submit_all import (  # noqa: E402
    DEFAULT_LOGS_ROOT, DEFAULT_RUNS_ROOT, DEFAULT_SUBMIT_ROOT,
    LIBERO_SUITES, ROBOCASA_SUITES, collect_runs,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Local multi-GPU runner for the MoP-DSRL experiment matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ---- plan selection (mirrors submit_all.py) ----
    p.add_argument("--tier", default="all",
                   choices=["all", "main", "baselines", "robustness", "ablations", "ut5"])
    p.add_argument("--suites", nargs="*", default=LIBERO_SUITES)
    p.add_argument("--include-robocasa", action="store_true")
    p.add_argument("--backbones", nargs="*", default=["pi05"])
    p.add_argument("--seeds-main", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    p.add_argument("--seeds-aux", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--seeds-ut5", type=int, nargs="*", default=[0])
    p.add_argument("--filter", default="")
    p.add_argument("--max-jobs", type=int, default=0)
    p.add_argument("--include-skipped", action="store_true")
    # The submit_all plan also takes these resource args; we keep them so the
    # plan enumeration is bit-identical but they are unused locally.
    p.add_argument("--partition", default="local")
    p.add_argument("--time", default="00:00:00")
    p.add_argument("--cpus", type=int, default=4)
    p.add_argument("--mem", default="-")
    p.add_argument("--gres", default="-")

    # ---- local launcher specifics ----
    p.add_argument("--n-gpus", type=int, default=2,
                   help="Number of concurrent subprocesses (1 per GPU).")
    p.add_argument("--gpu-ids", type=int, nargs="*", default=None,
                   help="Specific CUDA device IDs (default = 0..n_gpus-1).")
    p.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    p.add_argument("--logs-root", default=str(DEFAULT_LOGS_ROOT))
    p.add_argument("--submit-root", default=str(DEFAULT_SUBMIT_ROOT))
    p.add_argument("--python", default=sys.executable,
                   help="Python interpreter for the child processes.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan + the local command for each run; don't launch.")
    p.add_argument("--print-cmd", action="store_true")
    p.add_argument("--poll-interval", type=float, default=2.0)
    return p.parse_args()


def child_command(python_exe: str, args_str: list[str], run_dir: Path) -> list[str]:
    cmd = [python_exe, "-u", str(_HERE / "train.py")]
    cmd.extend(shlex.split(" ".join(args_str)))
    cmd.extend(["--run_dir", str(run_dir)])
    return cmd


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    runs = collect_runs(args)
    if not runs:
        logger.warning("Empty plan."); return 0

    runs_root = Path(args.runs_root)
    logs_root = Path(args.logs_root)
    submit_root = Path(args.submit_root)
    submit_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    if not args.include_skipped:
        kept = []
        for r in runs:
            if (r.run_dir(runs_root) / "final_summary.json").exists():
                continue
            kept.append(r)
        skipped = len(runs) - len(kept)
        if skipped:
            logger.info("Skipping %d runs that already have final_summary.json", skipped)
        runs = kept

    if args.max_jobs > 0 and len(runs) > args.max_jobs:
        runs = runs[: args.max_jobs]

    by_group = {}
    for r in runs:
        by_group[r.group] = by_group.get(r.group, 0) + 1
    logger.info("Plan: %d runs across %d groups", len(runs), len(by_group))
    for g, n in sorted(by_group.items()):
        logger.info("  %-30s  %d runs", g, n)

    gpu_ids = args.gpu_ids if args.gpu_ids is not None else list(range(args.n_gpus))
    if len(gpu_ids) != args.n_gpus:
        logger.error("--gpu-ids count (%d) != --n-gpus (%d)", len(gpu_ids), args.n_gpus)
        return 2
    logger.info("Concurrency: %d worker(s) on GPU(s) %s", args.n_gpus, gpu_ids)

    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest_path = submit_root / f"local_manifest_{timestamp}.tsv"
    manifest_f = open(manifest_path, "w")
    manifest_f.write("group\trun_name\tgpu\tpid\texit_code\twall_sec\tlog_path\n")

    if args.dry_run:
        for r in runs:
            run_dir = r.run_dir(runs_root)
            cmd = child_command(args.python, r.args, run_dir)
            print(shlex.join(cmd))
        logger.info("--dry-run: no subprocesses launched."); manifest_f.close(); return 0

    if args.print_cmd:
        for r in runs:
            run_dir = r.run_dir(runs_root)
            cmd = child_command(args.python, r.args, run_dir)
            print(shlex.join(cmd))

    # ------------------------------------------------------------------
    # Pool of worker slots — index into gpu_ids.  None = free.
    # ------------------------------------------------------------------
    workers = [None] * args.n_gpus  # list[(Run, subprocess.Popen, log_file_handle, t_start) | None]
    queue = list(runs)
    completed = 0
    failed = 0
    t_session_start = time.time()

    def launch(gpu_slot: int, r) -> None:
        nonlocal workers
        run_dir = r.run_dir(runs_root)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_dir = logs_root / r.group
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{r.run_name if False else r.name}.log"
        cmd = child_command(args.python, r.args, run_dir)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[gpu_slot])
        log_f = open(log_path, "w")
        log_f.write(f"# cmd: {shlex.join(cmd)}\n# CUDA_VISIBLE_DEVICES={gpu_ids[gpu_slot]}\n")
        log_f.flush()
        proc = subprocess.Popen(
            cmd, stdout=log_f, stderr=subprocess.STDOUT,
            env=env, cwd=str(_HERE.parent),
        )
        workers[gpu_slot] = (r, proc, log_f, time.time(), log_path)
        logger.info("[GPU%d] launch %-30s %-40s (PID %d) → %s",
                    gpu_ids[gpu_slot], r.group, r.name, proc.pid, log_path)

    try:
        while queue or any(w is not None for w in workers):
            # 1. Reap finished workers.
            for slot, w in enumerate(workers):
                if w is None:
                    continue
                r, proc, log_f, t_start, log_path = w
                if proc.poll() is None:
                    continue
                exit_code = proc.returncode
                wall = time.time() - t_start
                log_f.flush(); log_f.close()
                completed += 1
                if exit_code != 0:
                    failed += 1
                    logger.error("[GPU%d] EXIT %d (%s/%s) wall=%.0fs  log=%s",
                                 gpu_ids[slot], exit_code, r.group, r.name, wall, log_path)
                else:
                    logger.info("[GPU%d] done    (%s/%s) wall=%.0fs",
                                gpu_ids[slot], r.group, r.name, wall)
                manifest_f.write(
                    f"{r.group}\t{r.name}\t{gpu_ids[slot]}\t{proc.pid}\t{exit_code}\t{wall:.0f}\t{log_path}\n"
                )
                manifest_f.flush()
                workers[slot] = None

            # 2. Launch new work on free slots.
            for slot in range(args.n_gpus):
                if workers[slot] is None and queue:
                    launch(slot, queue.pop(0))

            # 3. Status heartbeat (every ~60s).
            if int(time.time() - t_session_start) % 60 == 0:
                running = sum(1 for w in workers if w is not None)
                logger.info(
                    "[heartbeat] remaining=%d  running=%d  completed=%d  failed=%d  elapsed=%.0fmin",
                    len(queue), running, completed, failed,
                    (time.time() - t_session_start) / 60.0,
                )

            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        logger.warning("Caught Ctrl-C, killing running children…")
        for w in workers:
            if w is None: continue
            r, proc, log_f, _, _ = w
            try: proc.terminate()
            except Exception: pass
        time.sleep(2)
        for w in workers:
            if w is None: continue
            r, proc, log_f, _, _ = w
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            log_f.close()

    finally:
        manifest_f.close()

    logger.info("Done.  completed=%d  failed=%d  manifest=%s",
                completed, failed, manifest_path)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
