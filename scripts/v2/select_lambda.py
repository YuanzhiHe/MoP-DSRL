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
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FLOOR = {"avoiding": 0.15, "pushing": 0.10}
TOLERATED = 0.15
MAIN_BACKBONES = (0, 1, 2, 3, 4)


def select(rows, task):
    """Chosen weight per arm, with the table the choice was made from."""
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
        for f in sorted(Path().glob(str(pat))) or [Path(pat)]:
            if f.exists():
                rows += [json.loads(l) for l in f.open() if l.strip()]
    if not rows:
        raise SystemExit(f"no calibration rows under {args.calibration}")
    report = {}
    for task in ("avoiding", "pushing"):
        chosen = select(rows, task)
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
