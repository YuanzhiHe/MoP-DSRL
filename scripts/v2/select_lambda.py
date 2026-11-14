#!/usr/bin/env python3
"""Apply the pre-registered weight-selection rule to the calibration cohort.

Written before any calibration cell had run, so the rule cannot be shaped by
what the calibration shows. The rule is section 5 of docs/v2/PREREG_v2.md:

    the selection quantity is the fraction of an arm's calibration runs whose
    geometric success falls below the task's floor, pooled over the four
    weightings and all calibration seeds;

    if any candidate weight holds that fraction at or below 0.15, take the
    largest such candidate; otherwise take the candidate with the lowest
    fraction; ties go to the larger weight.

It reads success, never fidelity, and it reads only the calibration file. A run
that raised counts as a failure of that weight, since a weight that makes the
optimisation fall over is not a weight to carry into the main experiment.
"""
import argparse
import collections
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FLOOR = {"avoiding": 0.15, "pushing": 0.10}
TOLERATED = 0.15
MAIN_BACKBONES = (0, 1, 2, 3, 4)


def select(rows, task):
    """Chosen weight per arm, with the table the choice was made from."""
    versions = {r.get("estimator_version") for r in rows if r.get("task") == task}
    if len(versions) > 1:
        raise SystemExit(
            f"{task} calibration spans estimator versions {sorted(versions)}; "
            f"a weight cannot be selected across two update rules")
    # One backbone at one demonstration weighting is one checkpoint. Two means
    # the calibration files named here come from two cloning cohorts, which
    # doubles every cell count and pools runs the frozen policy of one cohort
    # could not have produced. The weight rule reads how often success falls
    # below the floor, so pooling a cohort that cannot reach the floor drives
    # the rule into its fallback branch whatever the current cohort shows.
    seen = collections.defaultdict(set)
    for r in rows:
        if r.get("task") != task:
            continue
        seen[(r["weighting"], r.get("backbone_seed"))].add(
            r.get("checkpoint_sha256"))
    mixed = {k: sorted(v) for k, v in seen.items() if len(v) > 1}
    if mixed:
        k, v = next(iter(sorted(mixed.items())))
        raise SystemExit(
            f"{task} calibration spans {len(mixed)} cells measured on more "
            f"than one checkpoint, the first being weighting {k[0]} backbone "
            f"{k[1]} on {[h[:12] for h in v]}; name the files of one cohort")
    by = collections.defaultdict(list)
    for r in rows:
        if r.get("task") != task or r["arm"] == "dsrl":
            continue
        if r.get("backbone_seed") in MAIN_BACKBONES:
            raise SystemExit(
                f"refusing to select on a main-cohort policy: "
                f"backbone_seed={r['backbone_seed']} in {r['arm']}")
        by[(r["arm"], float(r["lam"]))].append(r)

    out = {}
    for arm in sorted({a for a, _ in by}):
        table = []
        for lam in sorted({l for a, l in by if a == arm}):
            runs = by[(arm, lam)]
            bad = sum(1 for r in runs
                      if r.get("status") != "completed"
                      or r.get("success_rate", 0.0) < FLOOR[task])
            table.append(dict(lam=lam, n=len(runs), below_floor=bad,
                              fraction=bad / len(runs)))
        tolerable = [t for t in table if t["fraction"] <= TOLERATED]
        if tolerable:
            pick = max(tolerable, key=lambda t: t["lam"])
            why = f"largest weight holding the fraction at or below {TOLERATED}"
        else:
            best = min(t["fraction"] for t in table)
            pick = max([t for t in table if t["fraction"] == best],
                       key=lambda t: t["lam"])
            why = "lowest fraction, no candidate within tolerance"
        out[arm] = dict(lam=pick["lam"], reason=why, table=table)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", nargs="+",
                    default=[str(ROOT / "runs" / "_d3il_v2"
                                 / "calibration_*.jsonl")])
    ap.add_argument("--out", type=Path,
                    default=ROOT / "runs" / "_d3il_v2" / "lambda_choice.json")
    args = ap.parse_args()

    rows = []
    for pat in args.calibration:
        for f in sorted(Path(g) for g in glob.glob(str(pat))) or [Path(pat)]:
            if f.exists():
                rows += [json.loads(l) for l in f.open() if l.strip()]
    if not rows:
        raise SystemExit(f"no calibration rows under {args.calibration}")
    # A task's weights are selected when that task's calibration lands, which
    # need not be the same day. Keep what is already decided and add to it, so
    # that selecting the second task cannot silently redecide the first.
    report = json.loads(args.out.read_text()) if args.out.exists() else {}
    for task in ("avoiding", "pushing"):
        if not any(r.get("task") == task for r in rows):
            continue
        chosen = select(rows, task)
        if task in report and report[task] != chosen:
            raise SystemExit(
                f"{task} weights are already recorded in {args.out} and the "
                f"calibration now gives a different answer; resolve by hand")
        report[task] = chosen
        print(f"=== {task} (floor {FLOOR[task]}) ===")
        for arm, c in chosen.items():
            print(f"  {arm}: lam = {c['lam']}  ({c['reason']})")
            for t in c["table"]:
                print(f"      lam {t['lam']:.2f}  {t['below_floor']:3d}/"
                      f"{t['n']:3d} runs below the floor  "
                      f"fraction {t['fraction']:.3f}")
    args.out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
