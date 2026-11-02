#!/usr/bin/env python3
"""Reconstruct a per-cell manifest for the legacy D3IL sweep.

The run files hold summary rows only: no budgets, no identifiers, no record of
how a cell was launched. The SLURM logs do hold it -- every job writes its full
command line on line 2 -- so the manifest is rebuilt by parsing those commands
and joining them to the rows each job wrote through its ``--out`` path.

Fields the legacy pipeline never recorded anywhere are emitted as null rather
than guessed, so a reader can tell a recovered value from an absent one.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "_d3il"
LOGS = ROOT / "logs" / "cpu"
BACKBONES = Path("/shared/scratch/SCWF00111/d3il_backbones")

# Settings that change what a cell means. Recovered from the launch command.
SWEEP_FLAGS = ("task", "lam", "mode_groups", "ema", "divergence", "ref_smooth",
               "iters", "episodes_per_iter", "eval_episodes", "ref_episodes",
               "lr", "M", "n_rewarded", "coef_clip", "bmd_prior", "device")


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_logs() -> list[dict]:
    """One entry per SLURM job: its command, its output file, how it ended."""
    import run_d3il_headtohead as drv  # the real parser, so defaults are real

    parser = drv.build_parser()
    jobs = []
    for log in sorted(LOGS.glob("*.out")):
        head = log.open(errors="replace").readline(), None
        with log.open(errors="replace") as fh:
            lines = [next(fh, "") for _ in range(2)]
        m = re.match(r"cmd:\s*python\s+\S*run_d3il_headtohead\.py\s+(.*)", lines[1].strip())
        if not m:
            continue
        argv = shlex.split(m.group(1))
        try:
            ns = parser.parse_args(argv)
        except SystemExit:
            jobs.append(dict(log=log.name, parse_error=True, cmd=m.group(1)))
            continue
        tail = subprocess.run(["tail", "-c", "20000", str(log)],
                              capture_output=True, text=True).stdout
        job = dict(
            log=log.name,
            slurm_id=re.search(r"_(\d+)\.out$", log.name).group(1)
            if re.search(r"_(\d+)\.out$", log.name) else None,
            cmd=m.group(1),
            out=getattr(ns, "out", None),
            args={k: getattr(ns, k, None) for k in SWEEP_FLAGS},
            seeds=getattr(ns, "seeds", None),
            seed_offset=getattr(ns, "seed_offset", None),
            arms=list(getattr(ns, "arms", []) or []),
            skews=list(getattr(ns, "skews", []) or []),
            backbone_seeds=list(getattr(ns, "backbone_seeds", []) or []),
            resume=bool(getattr(ns, "resume", False)),
            traceback="Traceback (most recent call last)" in tail,
            oom="oom-kill" in tail.lower() or "out of memory" in tail.lower(),
            timeout="DUE TO TIME LIMIT" in tail or "CANCELLED AT" in tail,
        )
        jobs.append(job)
    return jobs


def planned_cells(job: dict) -> list[tuple]:
    """The cells a job's command asked for: skew x backbone x arm x seed."""
    if job.get("parse_error"):
        return []
    bbs = job["backbone_seeds"] or [None]
    seeds = [(job["seed_offset"] or 0) + i for i in range(job["seeds"] or 0)]
    return [(job["args"]["task"], sk, bb, arm, sd)
            for sk in job["skews"] for bb in bbs
            for arm in job["arms"] for sd in seeds]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs" / "_d3il_v2" / "manifest_legacy.jsonl"))
    a = ap.parse_args()

    import sys
    sys.path.insert(0, str(ROOT / "scripts"))

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()

    jobs = parse_logs()
    # rows actually written, indexed by output file
    rows_by_file: dict[str, list] = {}
    for f in sorted(RUNS.glob("*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if isinstance(d, list):
            rows_by_file[f.name] = d

    ck_hash: dict[tuple, str | None] = {}

    def ckpt(task, skew, bb):
        k = (task, skew, bb)
        if k not in ck_hash:
            if task == "pushing":
                p = BACKBONES / "pushing" / (f"pushing_skew{skew}_bb{bb}.pt" if bb is not None
                                             else f"pushing_skew{skew}.pt")
            else:
                p = BACKBONES / (f"avoiding_skew{skew}_bb{bb}.pt" if bb is not None
                                 else f"avoiding_skew{skew}.pt")
            ck_hash[k] = sha256(p)
        return ck_hash[k]

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as fh:
        for job in jobs:
            outfile = Path(job["out"]).name if job.get("out") else None
            written = rows_by_file.get(outfile, [])
            # index written rows by the identity a planned cell carries
            idx: dict[tuple, list] = {}
            for r in written:
                idx.setdefault((r.get("task"), r.get("skew"), r.get("bb_seed"),
                                r.get("arm"), r.get("seed")), []).append(r)
            for cell in planned_cells(job):
                task, skew, bb, arm, sd = cell
                hits = idx.get(cell, [])
                if not hits:
                    status = "attempted_no_row"
                elif any(h.get("failed") for h in hits):
                    status = "exception"
                else:
                    status = "completed"
                row = hits[0] if hits else {}
                cref = row.get("C_ref")
                rec = dict(
                    task=task, weighting=skew, backbone_seed=bb, arm=arm, rl_seed=sd,
                    lam=row.get("lam", job["args"]["lam"]),
                    divergence=job["args"]["divergence"],
                    mode_groups=job["args"]["mode_groups"],
                    bmd_prior=row.get("bmd_prior", job["args"]["bmd_prior"]),
                    optimiser="Adam", lr=job["args"]["lr"], euler_M=job["args"]["M"],
                    iters=job["args"]["iters"],
                    episodes_per_iter=job["args"]["episodes_per_iter"],
                    eval_episodes=job["args"]["eval_episodes"],
                    ref_episodes=job["args"]["ref_episodes"],
                    ema=job["args"]["ema"], ref_smooth=job["args"]["ref_smooth"],
                    coef_clip=job["args"]["coef_clip"],
                    n_rewarded=job["args"]["n_rewarded"],
                    checkpoint_sha256=ckpt(task, skew, bb),
                    reference_sha256=hashlib.sha256(
                        json.dumps([round(float(x), 9) for x in cref]).encode()
                    ).hexdigest() if cref else None,
                    reference_counts=None,       # never recorded by the legacy driver
                    valid_episodes=None,         # never recorded
                    code_commit=commit,          # commit at reconstruction, not at run
                    code_commit_is_run_time=False,
                    config_sha256=hashlib.sha256(job["cmd"].encode()).hexdigest(),
                    source_file=outfile, slurm_log=job["log"], slurm_id=job["slurm_id"],
                    launch_cmd=job["cmd"],
                    job_traceback=job["traceback"], job_timeout=job["timeout"],
                    job_oom=job["oom"], resumed_job=job["resume"],
                    status=status,
                    duplicate_rows=len(hits),
                    SR=row.get("SR"), rewarded_SR=row.get("rewarded_SR"),
                    tv_to_ref=row.get("tv_to_ref"), C_phi=row.get("C_phi"),
                )
                fh.write(json.dumps(rec) + "\n")
                n += 1
    print(f"{len(jobs)} jobs parsed, {n} planned cells written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
