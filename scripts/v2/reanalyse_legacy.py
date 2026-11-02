#!/usr/bin/env python3
"""Reanalyse the legacy cohort under the pre-registered statistical scheme.

This repairs the analysis, not the experiment. The training defects stay in the
data: these rows were produced by the legacy update rule, their rewarded
success was computed on a coarsened label space, and their fidelity carries a
JS constant where the JS ablation is concerned. Nothing here changes that, and
no number produced here is a corrected experimental result.

What it does change: degrees of freedom follow the number of frozen policies
that actually contribute, the family of primary contrasts is adjusted together,
runs are no longer dropped by a success floor in the primary analysis, and the
shared reference's sampling error is carried separately.
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from math import lcm
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts" / "v2")]

import d3il_runs as D  # noqa: E402
import stats as S  # noqa: E402

FLOOR = {"avoiding": 0.15, "pushing": 0.10}


def policy_of(r):
    return (r.get("task"), r.get("skew"), r.get("bb_seed"))


def arm_of(r):
    return r["arm"]


def enrich(rows):
    """Add the endpoints the stored rows allow, and recover reference counts."""
    out = []
    for r in rows:
        r = dict(r)
        p, q = r.get("C_phi"), r.get("C_ref")
        if p and q:
            p = np.asarray(p, float); q = np.asarray(q, float)
            r["hellinger_to_ref"] = float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))
            r["max_bin_error"] = float(np.max(np.abs(p - q)))
            # closeness, so that "larger is better" holds for every contrast
            r["closeness_tv"] = -float(r["tv_to_ref"])
            r["closeness_hellinger"] = -r["hellinger_to_ref"]
            r["closeness_maxbin"] = -r["max_bin_error"]
            den = 1
            try:
                for x in q:
                    den = lcm(den, Fraction(float(x)).limit_denominator(2000).denominator)
                arr = q * den
                if den <= 2000 and np.max(np.abs(arr - np.round(arr))) < 1e-6:
                    r["reference_counts"] = [int(round(v)) for v in arr]
                    r["reference_valid_episodes"] = int(den)
            except (ValueError, ZeroDivisionError):
                pass
        out.append(r)
    return out


def cohort(task):
    if task == "avoiding":
        rows = D.load(["mb_skew*.json", "bf_skew*.json"], floor=0.0)
        rows = [r for r in rows if r.get("bmd_prior") in (None, "uniform")
                or r["arm"] != "bmd"]
    else:
        pm = [r for r in D.load("pm_skew*.json", floor=0.0) if r["arm"] != "bmd"]
        pu = D.load("pu_skew*.json", floor=0.0)
        rows = pm + pu
    rows = D.unify_reference(rows)
    return enrich([r for r in rows if r.get("task") == task or task == "avoiding"])


def main():
    report = {}
    for task in ("avoiding", "pushing"):
        rows = cohort(task)
        rows = [r for r in rows if r.get("task") in (task, None)]
        print("=" * 78)
        print(f"{task}: {len(rows)} runs, "
              f"{len({policy_of(r) for r in rows})} frozen policies")
        kept = [r for r in rows if r["SR"] >= FLOOR[task]]
        print(f"  success floor {FLOOR[task]} would drop "
              f"{len(rows) - len(kept)} of them")
        report[f"{task}/cohort"] = dict(
            runs=len(rows), policies=len({policy_of(r) for r in rows}),
            success_floor=FLOOR[task], runs_kept=len(kept),
            runs_dropped=len(rows) - len(kept),
            policies_kept=len({policy_of(r) for r in kept}),
            runs_per_arm={a: sum(1 for r in rows if arm_of(r) == a)
                          for a in ("mop", "mi", "dsrl", "bmd")})

        for label, sel in (("primary, every run", rows),
                           ("sensitivity, success floor", kept)):
            print(f"\n  --- {label} ---")
            fam = []
            for other in ("mi", "dsrl", "bmd"):
                fam.append(S.paired(sel, "closeness_tv", other, "mop",
                                    policy_of, arm_of, "greater",
                                    name=f"mop closer than {other}, TV"))
            S.holm(fam)
            for c in fam:
                print("   ", c.line())
                if c.policies_dropped:
                    print(f"        {c.policies_dropped} policies lacked one arm")
            report[f"{task}/{label}"] = [c.__dict__ for c in fam]

        print("\n  --- other distances, same cohort, every run ---")
        for metric, nm in (("closeness_hellinger", "Hellinger"),
                           ("closeness_maxbin", "largest bin error")):
            fam = [S.paired(rows, metric, o, "mop", policy_of, arm_of, "greater",
                            name=f"mop closer than {o}, {nm}")
                   for o in ("mi", "dsrl", "bmd")]
            S.holm(fam)
            report[f"{task}/other distances, every run/{nm}"] = \
                [c.__dict__ for c in fam]
            for c in fam:
                print("   ", c.line())

        print("\n  --- the shared reference's own sampling error ---")
        have = [r for r in rows if r.get("reference_counts")]
        print(f"    reference counts recovered for {len(have)}/{len(rows)} runs")
        resampled = dict(runs_with_counts=len(have), runs=len(rows),
                         policies_with_counts=len({policy_of(r) for r in have}))
        # Counts survive only where the stored probabilities rationalise, which
        # is not a random subset of the runs. The fixed-reference contrast is
        # recomputed on exactly the runs that enter the resampling so that the
        # two numbers are about the same cohort.
        for other in ("mi", "dsrl", "bmd"):
            res = S.reference_resample(have, policy_of, arm_of, "mop", other,
                                       draws=200, seed=1)
            # reference_resample returns TV(mop) - TV(other); report it in the
            # same direction as the paired contrasts, where positive is closer.
            for k in ("mean", "lo", "hi"):
                res[k] = -res[k]
            res["lo"], res["hi"] = res["hi"], res["lo"]
            fixed = S.paired(have, "closeness_tv", other, "mop", policy_of,
                             arm_of, "greater", name=f"subcohort vs {other}")
            res["fixed_reference_same_cohort"] = fixed.__dict__
            resampled[other] = res
            if res["draws"]:
                print(f"    vs {other:<5} fixed reference on this subcohort "
                      f"n={fixed.n:2d} mean={fixed.mean:+.4f}; "
                      f"resampled mean {res['mean']:+.4f} sd {res['sd']:.4f} "
                      f"95% [{res['lo']:+.4f}, {res['hi']:+.4f}]")

        report[f"{task}/reference resampling"] = resampled

        print("\n  --- what the manuscript's fixed critical value does ---")
        fixedcv = {}
        for other in ("mi", "dsrl", "bmd"):
            c = S.paired(kept, "closeness_tv", other, "mop", policy_of, arm_of)
            if c.n >= 2:
                half_fixed = 2.093 * c.se
                half_true = (c.ci_high - c.ci_low) / 2
                fixedcv[other] = dict(n=c.n, df=c.df, half_true=half_true,
                                      half_fixed=half_fixed,
                                      relative=half_fixed / half_true - 1)
                print(f"    vs {other:<5} n={c.n:2d} df={c.df:2d}: "
                      f"half-width {half_true:.4f} against {half_fixed:.4f} "
                      f"at the fixed t(19) value "
                      f"({100*(half_fixed/half_true - 1):+.1f}%)")

        report[f"{task}/fixed critical value"] = fixedcv

    out = ROOT / "runs" / "_d3il_v2" / "reanalysis_legacy.json"
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
