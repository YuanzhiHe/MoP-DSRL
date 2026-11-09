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


def check(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        FAILED.append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  {detail}" if detail else ""), flush=True)


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
        if slice_for(task, w, out, 8) != 0:
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
        # The bound exists for a policy in trouble. On the new observation
        # nothing should be approaching it, and a slice that does says the
        # representation is still carrying a pole.
        check(f"{task}: the observation stays far from its bound",
              am < 1e3, f"largest {am:.3g}")
        frac = max(r.get("obs_clipped_frac", 0.0) for r in done)
        check(f"{task}: the bound engages rarely", frac < 0.01,
              f"largest clipped fraction {frac:.4f}")

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
