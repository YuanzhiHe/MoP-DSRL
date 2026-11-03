#!/usr/bin/env python3
"""Accounting and pre-registered analysis for the corrected cohort.

Reads the corrected run files and nothing else. Legacy rows carry a different
estimator version and are refused rather than quietly pooled.

What it produces, in the order docs/v2/PREREG_v2.md asks for:

  1. run accounting per task and arm, counting what was planned, what completed,
     what raised and what is missing;
  2. the primary family of policy-level paired contrasts on conditional
     fidelity, Holm adjusted, over every run;
  3. the same family on the success-floor cohort, labelled a sensitivity
     analysis, with its own n;
  4. success, rewarded success, return, completion and failure for every run,
     unfiltered;
  5. undefined fidelity counted, never imputed;
  6. the reference's own sampling error, resampled from the stored counts.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats as S  # noqa: E402

FLOOR = {"avoiding": 0.15, "pushing": 0.10}
ARMS = ("mop", "mi", "dsrl", "bmd")
ARM_NAME = {"mop": "reference matching", "mi": "geometric mode entropy",
            "dsrl": "no mode term", "bmd": "learned skills"}
# Which version is current is read from the rows, not fixed here, so that a
# version bump does not need an edit in two places. What is refused is a cohort
# that spans more than one, since two estimator versions are two experiments.


def policy_of(r):
    return (r["task"], r["weighting"], r["backbone_seed"])


def arm_of(r):
    return r["arm"]


def load(paths):
    """Latest attempt of each cell, plus every row for accounting."""
    rows = []
    for pat in paths:
        for f in sorted(Path().glob(str(pat))) or [Path(pat)]:
            if not f.exists():
                continue
            for line in f.open():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    versions = {r.get("estimator_version") for r in rows}
    if len(versions) > 1:
        raise SystemExit(f"refusing to pool estimator versions "
                         f"{sorted(map(str, versions))}: two estimator "
                         f"versions are two experiments")
    # Contrasts read closeness, so a positive difference always means the arm
    # sits nearer the reference and the win counter means what it says.
    for r in rows:
        for metric in ("tv", "hellinger", "js", "l2"):
            v = r.get(f"{metric}_to_ref")
            r[f"closeness_{metric}"] = None if v is None else -float(v)
        mb = r.get("max_bin_error")
        r["closeness_maxbin"] = None if mb is None else -float(mb)

    latest = {}
    for r in rows:
        k = (r["task"], r["weighting"], r["backbone_seed"], r["arm"],
             r["rl_seed"], r["config_sha256"], r["checkpoint_sha256"],
             r.get("reference_sha256"))
        if k not in latest or r.get("attempt", 1) >= latest[k].get("attempt", 1):
            latest[k] = r
    return rows, list(latest.values())


def accounting(all_rows, cells, planned):
    """What was planned against what exists, per task and arm."""
    out = {}
    for task in sorted({r["task"] for r in cells}):
        per = {}
        for arm in ARMS:
            sel = [r for r in cells if r["task"] == task and r["arm"] == arm]
            att = [r for r in all_rows if r["task"] == task and r["arm"] == arm]
            done = [r for r in sel if r.get("status") == "completed"]
            exc = [r for r in sel if r.get("status") == "exception"]
            und = [r for r in done if not r.get("fidelity_defined")]
            per[arm] = dict(
                planned=planned.get(task), attempts=len(att), cells=len(sel),
                completed=len(done), exception=len(exc),
                retried=len(att) - len(sel),
                missing=(planned[task] - len(done)) if planned.get(task) else None,
                fidelity_undefined=len(und),
                policies=len({policy_of(r) for r in done}),
                seconds_median=float(np.median([r["cell_seconds"] for r in sel]))
                if sel else None,
                peak_rss_mb_max=float(max((r["cell_peak_rss_mb"] for r in sel),
                                          default=0.0)))
            if exc:
                per[arm]["exception_kinds"] = dict(collections.Counter(
                    r.get("exception", "?").split(":")[0] for r in exc))
        out[task] = per
    return out


def descriptives(cells):
    """Success, rewarded success, return and completion, over every run."""
    out = {}
    for task in sorted({r["task"] for r in cells}):
        per = {}
        for arm in ARMS:
            sel = [r for r in cells
                   if r["task"] == task and r["arm"] == arm
                   and r.get("status") == "completed"]
            if not sel:
                continue
            def col(k):
                v = np.array([r[k] for r in sel], float)
                return dict(mean=float(v.mean()), sd=float(v.std(ddof=1))
                            if v.size > 1 else 0.0, n=int(v.size))
            per[arm] = dict(
                success=col("success_rate"),
                rewarded_success=col("rewarded_success_rate"),
                episode_return=col("episode_return"),
                completion=col("completion_rate"),
                below_floor=int(sum(1 for r in sel
                                    if r["success_rate"] < FLOOR[task])),
                collapsed=int(sum(1 for r in sel if r.get("collapsed"))),
                nonfinite=int(sum(r.get("n_nonfinite", 0) for r in sel)),
                nonfinite_grad=int(sum(r.get("n_nonfinite_grad", 0) for r in sel)),
                obs_absmax=float(max([r.get("obs_absmax", 0.0) for r in sel],
                                     default=0.0)),
                obs_clipped_frac=float(np.mean([r.get("obs_clipped_frac", 0.0)
                                                for r in sel])) if sel else 0.0,
                frac_inert=float(np.mean([r.get("frac_inert", 0.0)
                                          for r in sel])))
        out[task] = per
    return out


def contrasts(cells):
    out = {}
    for task in sorted({r["task"] for r in cells}):
        rows = [r for r in cells
                if r["task"] == task and r.get("status") == "completed"]
        scored = [r for r in rows if r.get("fidelity_defined")]
        kept = [r for r in scored if r["success_rate"] >= FLOOR[task]]
        for label, sel in (("primary, every run", scored),
                           ("sensitivity, success floor", kept)):
            fam = [S.paired(sel, "closeness_tv", other, "mop", policy_of,
                            arm_of, "greater",
                            name=f"mop closer than {other}, TV")
                   for other in ("mi", "dsrl", "bmd")]
            S.holm(fam)
            out[f"{task}/{label}"] = [c.__dict__ for c in fam]
        out[f"{task}/undefined fidelity"] = dict(
            runs=len(rows), scored=len(scored), undefined=len(rows) - len(scored),
            by_arm={a: sum(1 for r in rows
                           if r["arm"] == a and not r.get("fidelity_defined"))
                    for a in ARMS})
        res = {}
        for other in ("mi", "dsrl", "bmd"):
            res[other] = S.reference_resample(scored, policy_of, arm_of, "mop",
                                              other, metric_bins="C_phi",
                                              draws=200, seed=1)
        out[f"{task}/reference resampling"] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=[str(ROOT / "runs" / "_d3il_v2" / "main_*.jsonl")])
    ap.add_argument("--planned", type=json.loads,
                    default='{"avoiding": 100, "pushing": 100}',
                    help="completed cells expected per task per arm")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "runs" / "_d3il_v2" / "report_v2.json")
    args = ap.parse_args()
    if isinstance(args.planned, str):
        args.planned = json.loads(args.planned)

    all_rows, cells = load(args.runs)
    if not cells:
        raise SystemExit("no corrected rows found; nothing to report")
    report = dict(accounting=accounting(all_rows, cells, args.planned),
                  descriptives=descriptives(cells),
                  contrasts=contrasts(cells))

    for task, per in report["accounting"].items():
        print("=" * 78)
        print(f"{task}")
        for arm, a in per.items():
            print(f"  {arm:5s} planned {a['planned']}  completed {a['completed']}"
                  f"  exception {a['exception']}  missing {a['missing']}"
                  f"  undefined fidelity {a['fidelity_undefined']}"
                  f"  median {a['seconds_median'] and round(a['seconds_median'])}s")
    for key, val in report["contrasts"].items():
        if isinstance(val, list):
            print(f"\n--- {key} ---")
            for c in val:
                print("   ", S.Contrast(**c).line())

    args.out.write_text(json.dumps(report, indent=1, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
