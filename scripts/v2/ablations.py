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
import glob
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
        # Patterns here are absolute, which Path().glob refuses, so the
        # module-level glob does the expansion.
        for f in sorted(Path(g) for g in glob.glob(str(pat))):
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
        ("rewarded success and return", "rewarded_success", ".3f"),
        ("completion", "completion", ".3f"),
        ("below floor", "below_floor", ".3f"),
        ("coefficient, raw", "coef_absmax_raw", ".3g"),
        ("coefficient, used", "coef_absmax_used", ".3g"),
        ("gradient norm", "grad_norm_last", ".3g"),
        ("non-finite", "nonfinite", "d"),
        ("non-finite gradients", "nonfinite_grad", "d"),
        ("steps at the obs bound", "obs_clipped_frac", ".4f"),
        ("inert batches", "frac_inert", ".3f")]


def contrasts_against(rows, reference_key, key, task, metric, direction="greater"):
    """Paired contrasts of every variant against one, within the task.

    The variants share their checkpoints, their reference, their budget and
    their seeds, so the comparison is paired over frozen policies and the
    family is adjusted together. There are five policies behind each, which
    the reported degrees of freedom carry.
    """
    others = sorted({key(r) for r in rows} - {reference_key})
    fam = [S.paired(rows, metric, o, reference_key, policy_of, key, direction,
                    name=f"{reference_key} over {o}") for o in others]
    S.holm(fam)
    return fam


def contrast_lines(title, fam):
    lines = [title, "",
             "| contrast | n | df | mean | 95% interval | p | policies "
             "favouring |", "|---|---|---|---|---|---|---|"]
    for c in fam:
        if c.n < 2:
            continue
        pv = c.p_adjusted if c.p_adjusted is not None else c.p_one_sided
        note = "" if c.ci_low > 0 else " (interval contains zero)"
        lines.append(f"| {c.name}{note} | {c.n} | {c.df} | {c.mean:+.4f} | "
                     f"[{c.ci_low:+.4f}, {c.ci_high:+.4f}] | {pv:.4f} | "
                     f"{c.wins}/{c.n} |")
    lines.append("")
    return lines


def scale_note(d):
    """What a common weight actually buys each divergence.

    The weight was calibrated under total variation, and the five derivatives
    do not share a scale, so the same weight is not the same penalty strength.
    The spread is reported from the runs rather than argued from the formulas.
    """
    c = {k: v["coef_absmax_raw"] for k, v in d.items()
         if v.get("coef_absmax_raw")}
    if len(c) < 2:
        return []
    lo, hi = min(c, key=c.get), max(c, key=c.get)
    return [f"At the common weight the coefficient that reaches the update "
            f"differs by a factor of {c[hi] / c[lo]:.0f} across the variants, "
            f"from {c[lo]:.3g} for {lo} to {c[hi]:.3g} for {hi}. The weight "
            f"was calibrated under total variation, so this comparison holds "
            f"the weight fixed and lets the penalty strength vary with the "
            f"derivative's own scale. A difference between two variants is "
            f"therefore a difference between a divergence at its own scale "
            f"and another at its own scale.", ""]


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
            doc += scale_note(d)
            scored = [r for r in rows if r.get("fidelity_defined")]
            for metric, nm in (("closeness_tv", "fidelity"),
                               ("success_rate", "geometric success")):
                src = scored if metric == "closeness_tv" else rows
                fam = contrasts_against(src, "tv", label, task, metric)
                report[f"{task}/divergence contrasts/{nm}"] = \
                    [c.__dict__ for c in fam]
                doc += contrast_lines(
                    f"Total variation against each alternative, on {nm}, "
                    f"paired over the five frozen policies.", fam)

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
                    f"p = {S.fmt_p(c.p_one_sided)}.", ""]

    # The weight the main matrix already ran is not repeated: those cells
    # carry the same key, so the grid job resumed them and wrote nothing. The
    # rows live in the main files and are read from there, which is what makes
    # the grid a comparison across weights on one set of cells rather than two
    # cohorts that happen to share a configuration.
    pw = load([RUNS / "ablation_weight_pushing_*.jsonl",
               RUNS / "main_pushing_0.00.jsonl",
               RUNS / "main_pushing_0.10.jsonl"])
    pw = [r for r in pw if r["weighting"] in ("0.00", "0.10")
          and r["arm"] in ("mi", "mop")]
    if pw:
        d = summarise(pw, "pushing", lambda r: f"{r['arm']} lam={r['lam']:.2f}")
        report["pushing/weight"] = d
        doc += table("Pushing, weight grid on the two lowest weightings",
                     d, COLS)
        # The grid's point is how the separation between the two arms moves
        # with the weight, so it is read as a contrast at each weight rather
        # than as six independent cells.
        scored = [r for r in pw if r.get("fidelity_defined")]
        for metric, nm, src in (("closeness_tv", "fidelity", scored),
                                ("success_rate", "geometric success", pw)):
            fam = []
            for lam in sorted({r["lam"] for r in pw}):
                sel = [r for r in src if abs(r["lam"] - lam) < 1e-9]
                fam.append(S.paired(sel, metric, "mi", "mop", policy_of,
                                    lambda r: r["arm"], "greater",
                                    name=f"weight {lam:.2f}"))
            S.holm(fam)
            report[f"pushing/weight contrasts/{nm}"] = [c.__dict__ for c in fam]
            doc += contrast_lines(
                f"Reference matching against mode entropy at each weight, on "
                f"{nm}, paired over the frozen policies of the two lowest "
                f"demonstration weightings.", fam)
        doc += [f"The weight the main matrix runs is "
                f"{json.loads((RUNS / 'lambda_choice.json').read_text())['pushing']['mop']['lam']:.2f}, "
                f"chosen before any main cell existed by a rule that reads "
                f"only how often a weight drives geometric success below the "
                f"task floor. That rule never reads fidelity, and this grid "
                f"shows where the weight it chose sits relative to the "
                f"separation between the arms. The main analysis stays at the "
                f"chosen weight.", ""]

    if not report:
        raise SystemExit("no ablation rows yet")
    args.out_json.write_text(json.dumps(report, indent=1, default=str))
    args.out_doc.parent.mkdir(parents=True, exist_ok=True)
    args.out_doc.write_text("\n".join(doc))
    print("\n".join(doc))
    print(f"wrote {args.out_json} and {args.out_doc}")


if __name__ == "__main__":
    main()
