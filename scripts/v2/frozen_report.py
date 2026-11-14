#!/usr/bin/env python3
"""Where each frozen policy starts, against where the four objectives end.

Two quantities come out of this. The first is the frozen policy's own success,
rewarded success and completion, which places the columns of the main tables on
a scale. The second is the distance from the frozen policy's evaluation
histogram to the reference measured from the same policy: both samples come
from one distribution, so that distance is the sampling floor measured rather
than computed, and it is reported beside the computed one.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# Name prefix of the run files the analysis reads. The matrix was measured
# again after the environment rule changed, and both cohorts sit in the same
# directory; the prefix is what keeps the two apart, so a bare invocation
# reads the current one.
PREFIX = "v22_"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats as S  # noqa: E402

# The computed floor comes from sampling_floor.py, so the two scripts
# quote one number and cannot drift apart.
FLOOR_COMPUTED = {k: v["floor"] for k, v in json.loads(
    (ROOT / "runs" / "_d3il_v2" / "sampling_floor.json").read_text()).items()}

ARMS = ("dsrl", "mi", "bmd", "mop")

NAME = {"dsrl": "no mode term", "mi": "mode entropy",
        "bmd": "learned skills", "mop": "reference matching",
        "frozen": "frozen policy"}


def read(pattern):
    rows = []
    for f in sorted(glob.glob(str(ROOT / "runs" / "_d3il_v2" / pattern))):
        for line in open(f):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def skew(r):
    q = np.asarray(r["C_ref"], float)
    return float(0.5 * np.abs(q - 1.0 / q.size).sum())


def cell(rows, key):
    v = np.array([r[key] for r in rows if r.get(key) is not None], float)
    if not v.size:
        return None, None, 0
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)), int(v.size)


def main():
    # Two frozen-policy stages exist: the first measured every cell on the
    # environment its worker built first, which cost Avoiding about two tenths
    # of its completion rate, and the second builds a throwaway environment
    # ahead of the measured one. Naming the pattern keeps a report from
    # averaging the two together.
    pattern = sys.argv[1] if len(sys.argv) > 1 else "frozen2_*.jsonl"
    frozen = [r for r in read(pattern) if r.get("status") == "completed"]
    main_rows = [r for r in read(f"{PREFIX}main_*.jsonl") if r.get("status") == "completed"]
    if not frozen:
        raise SystemExit(f"no frozen-policy rows under {pattern}")
    versions = {r.get("estimator_version") for r in frozen + main_rows}
    if len(versions) > 1:
        raise SystemExit(f"refusing to pool estimator versions {sorted(versions)}")

    out = {}
    for task in sorted({r["task"] for r in frozen}):
        ws = sorted({r["weighting"] for r in frozen if r["task"] == task},
                    key=lambda w: np.mean([skew(r) for r in frozen
                                           if r["task"] == task
                                           and r["weighting"] == w]))
        per = {}
        print("=" * 78)
        print(f"{task}: frozen policy under its own latent prior")
        for w in ws:
            sel = [r for r in frozen if r["task"] == task and r["weighting"] == w]
            sk = float(np.mean([skew(r) for r in sel]))
            row = {k: cell(sel, k) for k in
                   ("success_rate", "rewarded_success_rate", "episode_return",
                    "completion_rate", "tv_to_ref")}
            per[w] = dict(skew=sk, **{k: dict(mean=v[0], se=v[1], n=v[2])
                                      for k, v in row.items()})
            print(f"  skew {sk:.2f}  succ {row['success_rate'][0]:.3f}"
                  f"  rew {row['rewarded_success_rate'][0]:.3f}"
                  f"  comp {row['completion_rate'][0]:.3f}"
                  f"  TV to own reference {row['tv_to_ref'][0]:.3f}"
                  f" (n={row['tv_to_ref'][2]})")
        tot = {k: cell([r for r in frozen if r["task"] == task], k)
               for k in ("success_rate", "rewarded_success_rate",
                         "completion_rate", "tv_to_ref")}
        out[task] = dict(by_level=per,
                         overall={k: dict(mean=v[0], se=v[1], n=v[2])
                                  for k, v in tot.items()},
                         floor_computed=FLOOR_COMPUTED[task])
        print(f"  all levels: succ {tot['success_rate'][0]:.3f}"
              f"  rew {tot['rewarded_success_rate'][0]:.3f}"
              f"  TV {tot['tv_to_ref'][0]:.3f}"
              f"  against a computed floor of {FLOOR_COMPUTED[task]:.3f}")

        # Paired against each objective, one point per frozen policy, on the
        # endpoints the frozen policy is a meaningful comparison for.
        pol = lambda r: (r["task"], r["weighting"], r["backbone_seed"])  # noqa: E731
        arm = lambda r: r["arm"]                                          # noqa: E731
        both = [r for r in frozen + main_rows if r["task"] == task]
        # These rows were measured after the main analysis, so the contrasts
        # below are exploratory and carry intervals without a p value: no
        # direction for them was declared before the runs they compare.
        for metric in ("success_rate", "rewarded_success_rate",
                       "completion_rate"):
            fam = [S.paired(both, metric, "frozen", a, pol, arm,
                            name=f"{NAME[a]} over the frozen policy")
                   for a in ARMS]
            print(f"  --- {metric}, each objective against the frozen policy ---")
            for c in fam:
                print(f"      {c.name:42s} {c.mean:+.3f} "
                      f"[{c.ci_low:+.3f}, {c.ci_high:+.3f}]  "
                      f"{c.wins}/{c.n} policies, df {c.df}")
            out[f"{task}/{metric}"] = [
                {k: v for k, v in c.__dict__.items()
                 if k not in ("p_one_sided", "p_adjusted", "t")} for c in fam]

    tag = pattern.split("_")[0]
    dest = ROOT / "runs" / "_d3il_v2" / f"{tag}_report.json"
    dest.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
