#!/usr/bin/env python3
"""The distance the endpoint shows for a policy that reproduces the reference.

Both histograms behind the endpoint are finite samples, so the distance between
them stays above zero even where the two distributions are identical. This
measures that offset at the counts the experiments actually produced: for each
frozen policy, one sample of the size a run's evaluation labels and one of the
size that policy's own reference labels, both drawn from the policy's measured
reference, scored by the same total variation the endpoint uses.

The evaluation count is the median labelled count over that task's runs, which
keeps one collapsed run from setting the offset for the task. The quartiles of
that count are reported beside it so the spread is visible.
"""
import os
import argparse
import glob
import json
from pathlib import Path

import numpy as np

# A job runs a frozen copy of scripts/ under runs/_d3il_v2/_src/<job>,
# so this file's own location says nothing about where the run files
# are. The environment names the root; the fallback is a local run.
ROOT = Path(os.environ.get("MOP_DSRL_ROOT")
            or Path(__file__).resolve().parents[2])
# Name prefix of the run files the analysis reads. The matrix was measured
# again after the environment rule changed, and both cohorts sit in the same
# directory; the prefix is what keeps the two apart, so a bare invocation
# reads the current one.
PREFIX = "v22_"

EVAL_EPISODES = {"avoiding": 500, "pushing": 300}


def read(pattern):
    rows = []
    for f in sorted(glob.glob(str(ROOT / "runs" / "_d3il_v2" / pattern))):
        for line in open(f):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def floor_for(rows, task, draws, rng):
    sel = [r for r in rows
           if r["task"] == task and r.get("status") == "completed"
           and r.get("fidelity_defined")]
    labelled = np.array([r["completion_rate"] * EVAL_EPISODES[task]
                         for r in sel], float)
    n_eval = int(round(float(np.median(labelled))))
    refs = {}
    for r in sel:
        refs[(r["weighting"], r["backbone_seed"])] = (
            np.asarray(r["C_ref"], float), int(r["reference_valid_episodes"]))
    d = []
    for q, n_ref in refs.values():
        for _ in range(draws):
            a = rng.multinomial(n_eval, q) / n_eval
            b = rng.multinomial(n_ref, q) / n_ref
            d.append(0.5 * np.abs(a - b).sum())
    d = np.asarray(d)
    return dict(task=task, policies=len(refs), draws=draws,
                labelled_median=n_eval,
                labelled_q25=int(round(float(np.percentile(labelled, 25)))),
                labelled_q75=int(round(float(np.percentile(labelled, 75)))),
                reference_labelled_median=int(np.median(
                    [n for _, n in refs.values()])),
                floor=float(d.mean()), floor_sd=float(d.std(ddof=1)))


def by_level(rows, task, draws, rng):
    """The same offset computed inside each demonstration weighting.

    The scoping statement in the manuscript is about one weighting, where the
    labelled counts differ from the task's median, so each level carries its
    own offset.
    """
    out = {}
    sel = [r for r in rows
           if r["task"] == task and r.get("status") == "completed"
           and r.get("fidelity_defined")]
    for w in sorted({r["weighting"] for r in sel}):
        lvl = [r for r in sel if r["weighting"] == w]
        labelled = np.array([r["completion_rate"] * EVAL_EPISODES[task]
                             for r in lvl], float)
        n_eval = max(1, int(round(float(np.median(labelled)))))
        refs = {}
        for r in lvl:
            refs[(r["weighting"], r["backbone_seed"])] = (
                np.asarray(r["C_ref"], float),
                int(r["reference_valid_episodes"]))
        d = []
        for q, n_ref in refs.values():
            for _ in range(draws):
                a = rng.multinomial(n_eval, q) / n_eval
                b = rng.multinomial(n_ref, q) / n_ref
                d.append(0.5 * np.abs(a - b).sum())
        skew = float(np.mean([0.5 * np.abs(q - 1.0 / q.size).sum()
                              for q, _ in refs.values()]))
        out[w] = dict(skew=skew, labelled_median=n_eval,
                      floor=float(np.mean(d)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "runs" / "_d3il_v2" / "sampling_floor.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = read(f"{PREFIX}main_*.jsonl")
    # An empty read wrote an empty file and reported success, and the table
    # that reads this file then had a floor of nothing to compare against.
    if not rows:
        raise SystemExit(f"no rows under {PREFIX}main_*.jsonl: the floor is "
                         f"measured from the runs it is a floor for")
    out = {}
    for task in sorted({r["task"] for r in rows}):
        f = floor_for(rows, task, args.draws, rng)
        f["by_level"] = by_level(rows, task, args.draws, rng)
        out[task] = f
        print(f"{task}: floor {f['floor']:.3f} (sd {f['floor_sd']:.3f}) from "
              f"{f['labelled_median']} labelled evaluation episodes "
              f"[{f['labelled_q25']}, {f['labelled_q75']}] against "
              f"{f['reference_labelled_median']} reference episodes, over "
              f"{f['policies']} policies and {f['draws']} draws each")
        for w, lv in sorted(f["by_level"].items(), key=lambda kv: kv[1]["skew"]):
            print(f"    skew {lv['skew']:.2f}: floor {lv['floor']:.3f} from "
                  f"{lv['labelled_median']} labelled episodes")
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
