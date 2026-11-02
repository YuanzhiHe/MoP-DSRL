#!/usr/bin/env python3
"""The three ablations, read from their own run files.

Divergence. At each task's highest demonstration weighting, total variation
against corrected Jensen-Shannon, Kullback-Leibler at an explicit numerical
guard, and Kullback-Leibler against a reference smoothed by an explicit weight.
Checkpoints, reference, budget and seeds are identical across the four, and the
total-variation rows are the main-matrix rows for exactly that configuration.
Every row reports the coefficient as it came out of the derivative and as it
reached the update, so a claim about why one divergence fails has the
diagnostics behind it rather than an argument from the size of a number.

Weight. On the two lowest Pushing weightings, the pre-fixed grid for reference
matching and for mode entropy, reporting failures and success for every
attempt rather than for the runs that pass a threshold.

Prior. The learned-skill arm with its prior fitted to the reference, against
the same arm under a uniform prior, on the same policies and seeds.
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
RUNS = ROOT / "runs" / "_d3il_v2"


def policy_of(r):
    return (r["task"], r["weighting"], r["backbone_seed"])


def load(patterns):
    rows = []
    for pat in patterns:
        for f in sorted(Path().glob(str(pat))):
            rows += [json.loads(l) for l in f.open() if l.strip()]
    for r in rows:
        v = r.get("tv_to_ref")
        r["closeness_tv"] = None if v is None else -float(v)
    return rows


def label(r):
    """What distinguishes this row inside an ablation."""
    if r["arm"] != "mop":
        return r["arm"]
    d = r["divergence"]
    if d == "kl":
        d += f" eps={r['kl_eps']:g}"
        if r.get("ref_smooth"):
            d += f" smoothing={r['ref_smooth']:g}"
    return d


def summarise(rows, task, key):
    """Every attempt, grouped by `key`. Nothing is filtered out."""
    out = {}
    for k in sorted({key(r) for r in rows}):
        sel = [r for r in rows if key(r) == k]
        done = [r for r in sel if r.get("status") == "completed"]
        scored = [r for r in done if r.get("fidelity_defined")]
        def m(field, src):
            v = [r[field] for r in src if r.get(field) is not None]
            return float(np.mean(v)) if v else None
        out[k] = dict(
            attempts=len(sel), completed=len(done), exception=len(sel) - len(done),
            fidelity_undefined=len(done) - len(scored),
            tv_to_ref=m("tv_to_ref", scored),
            success=m("success_rate", done),
            rewarded_success=m("rewarded_success_rate", done),
            episode_return=m("episode_return", done),
            completion=m("completion_rate", done),
            below_floor=sum(1 for r in done
                            if r["success_rate"] < FLOOR[task]) / max(len(done), 1),
            coef_absmax_raw=m("coef_absmax_raw", done),
            coef_absmax_used=m("coef_absmax", done),
            grad_norm_last=m("grad_norm_last", done),
            nonfinite=sum(r.get("n_nonfinite", 0) for r in done),
            nonfinite_grad=sum(r.get("n_nonfinite_grad", 0) for r in done),
            obs_clipped_frac=m("obs_clipped_frac", done),
            frac_inert=m("frac_inert", done),
            collapsed=sum(1 for r in done if r.get("collapsed")))
    return out


def table(title, d, cols):
    head = "| variant | " + " | ".join(c[0] for c in cols) + " |"
    rule = "|---" * (len(cols) + 1) + "|"
    lines = [f"### {title}", "", head, rule]
    for k, v in d.items():
        cells = []
        for _, field, fmt in cols:
            x = v.get(field)
            cells.append("undefined" if x is None else format(x, fmt))
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


COLS = [("runs", "completed", "d"), ("exceptions", "exception", "d"),
        ("undefined fidelity", "fidelity_undefined", "d"),
        ("TV to reference", "tv_to_ref", ".4f"),
        ("success", "success", ".3f"),
        ("rewarded success", "rewarded_success", ".3f"),
        ("return", "episode_return", ".3f"),
        ("completion", "completion", ".3f"),
        ("below floor", "below_floor", ".3f"),
        ("coefficient, raw", "coef_absmax_raw", ".3g"),
        ("coefficient, used", "coef_absmax_used", ".3g"),
        ("gradient norm", "grad_norm_last", ".3g"),
        ("non-finite", "nonfinite", "d"),
        ("non-finite gradients", "nonfinite_grad", "d"),
        ("steps at the obs bound", "obs_clipped_frac", ".4f"),
        ("inert batches", "frac_inert", ".3f")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, default=RUNS / "ablations.json")
    ap.add_argument("--out-doc", type=Path,
                    default=ROOT / "docs" / "v2" / "ABLATIONS.md")
    args = ap.parse_args()

    report, doc = {}, ["# Ablations, corrected implementation", ""]

    for task in ("avoiding", "pushing"):
        top = {"avoiding": "0.60", "pushing": "0.50"}[task]
        rows = [r for r in load([RUNS / f"ablation_divergence_{task}_*.jsonl",
                                 RUNS / f"main_{task}_{top}.jsonl"])
                if r["arm"] == "mop" and r["weighting"] == top]
        if rows:
            d = summarise(rows, task, label)
            report[f"{task}/divergence"] = d
            doc += table(f"{task.capitalize()}, divergence at weighting {top}",
                         d, COLS)

        fit = load([RUNS / f"ablation_fitted_prior_{task}.jsonl",
                    RUNS / f"main_{task}_{top}.jsonl"])
        fit = [r for r in fit if r["arm"] == "bmd" and r["weighting"] == top]
        if fit:
            d = summarise(fit, task, lambda r: f"prior {r['bmd_prior']}")
            report[f"{task}/skill prior"] = d
            doc += table(f"{task.capitalize()}, learned-skill prior at "
                         f"weighting {top}", d, COLS)
            # Does fitting the prior to the reference move the baseline at
            # all? Same policies, same seeds, so the comparison is paired.
            c = S.paired([r for r in fit if r.get("fidelity_defined")],
                         "closeness_tv", "uniform", "fitted", policy_of,
                         lambda r: r["bmd_prior"], "greater",
                         name=f"{task}: fitted prior closer than uniform")
            report[f"{task}/skill prior contrast"] = c.__dict__
            doc += [f"Fitted against uniform, paired over frozen policies: "
                    f"n = {c.n}, df = {c.df}, mean {c.mean:+.4f}, "
                    f"95% interval [{c.ci_low:+.4f}, {c.ci_high:+.4f}], "
                    f"p = {c.p_one_sided:.4f}.", ""]

    pw = load([RUNS / "ablation_weight_pushing_*.jsonl"])
    pw = [r for r in pw if r["weighting"] in ("0.00", "0.10")]
    if pw:
        d = summarise(pw, "pushing", lambda r: f"{r['arm']} lam={r['lam']:.2f}")
        report["pushing/weight"] = d
        doc += table("Pushing, weight grid on the two lowest weightings",
                     d, COLS)

    if not report:
        raise SystemExit("no ablation rows yet")
    args.out_json.write_text(json.dumps(report, indent=1, default=str))
    args.out_doc.parent.mkdir(parents=True, exist_ok=True)
    args.out_doc.write_text("\n".join(doc))
    print("\n".join(doc))
    print(f"wrote {args.out_json} and {args.out_doc}")


if __name__ == "__main__":
    main()
