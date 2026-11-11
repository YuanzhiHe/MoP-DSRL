#!/usr/bin/env python3
"""Forty cells through the whole pipeline, before the matrix is allowed to run.

Three times a cohort was launched, reported and withdrawn for something that a
small slice would have shown. This runs that slice: a reference, a handful of
cells on each task and each arm, a frozen evaluation, and then the checks that
the records the cohort will be read through are actually there and consistent.

It is deliberately small and deliberately end to end. A fast slice that skips
the scoring path tests nothing, because the failures that cost the most were in
measurement rather than in training.

The matrix depends on this job succeeding, so a non-zero exit here stops the
submission rather than being read later in a log.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("MOP_DSRL_ROOT",
                           Path(__file__).resolve().parents[2]))
RUNS = ROOT / "runs" / "_d3il_v2"
SRC = Path(__file__).resolve().parent

FAILED: list[str] = []

# Channels that are a sine or a cosine, and so bounded whatever happens.
ORIENT_CHANNELS = {"pushing": (6, 7, 10, 11), "avoiding": ()}


def check(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        FAILED.append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  {detail}" if detail else ""), flush=True)



def orientation_within_construction(task, rows):
    """Do the sine and cosine channels stay inside the bound they imply?

    A sine and a cosine lie in [-1, 1] whatever the policy does, so after the
    backbone's own normalisation each is bounded by (1 + |mean|) / sd. That
    number comes from the checkpoint, not from a threshold anyone chose, and a
    channel beyond it means the rollout and the clone disagree about what the
    channel is.
    """
    idx = ORIENT_CHANNELS.get(task, ())
    if not idx:
        return True, "no orientation channels on this task"
    import torch
    sys.path.insert(0, str(ROOT))
    from scripts.run_d3il_headtohead import backbone_path
    r0 = rows[0]
    blob = torch.load(backbone_path(task, r0["weighting"], r0["backbone_seed"]),
                      map_location="cpu", weights_only=False)
    mu = np.asarray(blob["obs_mean"], float).ravel()
    sd = np.asarray(blob["obs_std"], float).ravel()
    worst, worst_at = 0.0, None
    for r in rows:
        ch = r.get("obs_absmax_channels")
        if not ch:
            return False, "runs did not record the observation per channel"
        for i in idx:
            bound = (1.0 + abs(mu[i])) / sd[i]
            ratio = ch[i] / bound
            if ratio > worst:
                worst, worst_at = ratio, (i, ch[i], bound)
    i, got, bound = worst_at
    return worst <= 1.0 + 1e-6, (f"worst channel {i}: {got:.4g} against its "
                                 f"construction bound {bound:.4g}")


def workers(n=8) -> int:
    """Workers for a slice, bounded by the allocation and by the cells."""
    return min(n, max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", 8)) - 1))


def run(cmd) -> int:
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=ROOT).returncode


def slice_for(task, weighting, out, workers):
    """Two seeds of every arm on one frozen policy: eight cells."""
    return run([sys.executable, SRC / "run_d3il_v2.py",
                "--task", task, "--weightings", weighting,
                "--backbone-seeds", 0, "--seeds", 2, "--seed-offset", 900,
                "--lam", 1.0 if task == "avoiding" else 0.25,
                "--workers", workers, "--out", out])


def main() -> int:
    print("gates first; the slice is pointless if they are failing")
    if run([sys.executable, SRC / "gates.py"]) != 0:
        print("GATES FAILED, preflight stops here", flush=True)
        return 1

    outs = {}
    for task, w in (("avoiding", "0.00"), ("pushing", "0.50")):
        out = RUNS / f"preflight_{task}.jsonl"
        if out.exists():
            out.unlink()
        print(f"\nslice: {task} w{w}")
        if slice_for(task, w, out, workers()) != 0:
            check(f"{task} slice ran", False)
            return 1
        outs[task] = out
        check(f"{task} slice ran", True)

    print("\nfrozen evaluation on the same protocol")
    fz = RUNS / "preflight_frozen_avoiding.jsonl"
    if fz.exists():
        fz.unlink()
    if run([sys.executable, SRC / "frozen_baseline.py", "--task", "avoiding",
            "--weightings", "0.00", "--backbone-seeds", 0, "--seeds", 2,
            "--seed-offset", 900, "--workers", 2, "--out", fz]) != 0:
        check("frozen slice ran", False)
        return 1
    check("frozen slice ran", True)

    print("\nwhat the slice produced")
    for task, out in outs.items():
        rows = [json.loads(l) for l in out.open() if l.strip()]
        done = [r for r in rows if r.get("status") == "completed"]
        check(f"{task}: every cell completed", len(done) == len(rows) == 8,
              f"{len(done)} of {len(rows)}")
        check(f"{task}: all four arms present",
              len({r["arm"] for r in done}) == 4,
              ", ".join(sorted({r["arm"] for r in done})))
        check(f"{task}: every cell saved its policy",
              all(r.get("policy_sha256") for r in done))
        check(f"{task}: the environment protocol is recorded",
              all(r.get("env_protocol") for r in done),
              str({r.get("env_protocol") for r in done}))
        defined = [r for r in done if r.get("tv_to_ref") is not None]
        check(f"{task}: fidelity is defined somewhere",
              len(defined) > 0, f"{len(defined)} of {len(done)}")
        am = max(r["obs_absmax"] for r in done)
        # A pole in the representation is what this slice is here to exclude.
        # The orientation channels are a sine and a cosine, so each is bounded
        # by its own construction once normalised, and the bound is read off
        # the backbone rather than chosen. A channel past it is an encoding
        # fault. The position channels carry no such guarantee: the commanded
        # pose is an accumulator of the policy's own deltas, which both tasks
        # have always had and which the declared clip bounds for every arm
        # alike, so their size is recorded instead of gated.
        ok, detail = orientation_within_construction(task, done)
        check(f"{task}: the orientation channels obey their construction", ok,
              detail)
        # The pole the benchmark's tan(yaw) produced reached 1e12. Anything in
        # that regime is a representation fault rather than a hard-driven arm.
        check(f"{task}: no channel is in the pole regime", am < 1e6,
              f"largest {am:.3g}")
        frac = float(np.mean([r.get("obs_clipped_frac", 0.0) for r in done]))
        print(f"  [note] {task}: the bound engaged on {frac:.4f} of steps, "
              f"largest normalised observation {am:.3g}", flush=True)

    fzr = [json.loads(l) for l in fz.open() if l.strip()]
    fzd = [r for r in fzr if r.get("status") == "completed"]
    check("frozen: every cell completed", len(fzd) == len(fzr) == 2)
    check("frozen: measured on the same protocol as the arms",
          {r.get("env_protocol") for r in fzd}
          == {r.get("env_protocol") for r in
              [json.loads(l) for l in outs["avoiding"].open() if l.strip()]})

    pol = list((RUNS / "_policies").rglob("*.pt"))
    check("policies are on disk and loadable", len(pol) >= 16,
          f"{len(pol)} files")
    if pol:
        import torch
        b = torch.load(pol[0], map_location="cpu", weights_only=False)
        check("a saved policy carries its provenance",
              all(k in b for k in ("state_dict", "config_sha256",
                                   "env_protocol", "estimator_version")))

    print()
    if FAILED:
        print("failing: " + ", ".join(FAILED))
    print("PREFLIGHT: " + ("PASS" if not FAILED else "FAIL"))
    return 0 if not FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())
