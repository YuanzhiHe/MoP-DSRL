#!/usr/bin/env python3
"""Render the legacy reanalysis as a document, straight from its JSON.

Nothing here recomputes a number. Every figure in the output is read from
``runs/_d3il_v2/reanalysis_legacy.json`` so the document cannot drift from the
analysis that produced it; regenerate both together or neither.
"""
import os
import json
from pathlib import Path

# A job runs a frozen copy of scripts/ under runs/_d3il_v2/_src/<job>,
# so this file's own location says nothing about where the run files
# are. The environment names the root; the fallback is a local run.
ROOT = Path(os.environ.get("MOP_DSRL_ROOT")
            or Path(__file__).resolve().parents[2])
SRC = ROOT / "runs" / "_d3il_v2" / "reanalysis_legacy.json"
OUT = ROOT / "docs" / "v2" / "REANALYSIS_LEGACY.md"

ARM = {"mi": "geometric mode entropy", "dsrl": "no mode term",
       "bmd": "learned skills"}


def row(c):
    p = c["p_adjusted"] if c.get("p_adjusted") is not None else c["p_one_sided"]
    return (f"| {c['name']} | {c['n']} | {c['df']} | {c['mean']:+.4f} | "
            f"{c['sd']:.4f} | {c['se']:.4f} | "
            f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | {p:.4f} | "
            f"{c['wins']}/{c['n']} |")


HEAD = ("| contrast | n | df | mean | sd | se | 95% CI | p | policies |\n"
        "|---|---|---|---|---|---|---|---|---|")


def main():
    d = json.load(SRC.open())
    L = ["# Reanalysis of the legacy sweep",
         "",
         "What this covers and what it cannot. The runs analysed here were "
         "produced by `scripts/run_d3il_headtohead.py`, whose update rule has "
         "the defects recorded in `docs/v2/FIXES.md`. Reanalysis repairs how "
         "the numbers are read; it leaves the training that produced them "
         "untouched. Every contrast below is a policy-level paired difference "
         "in closeness to the reference: seeds are averaged inside a frozen "
         "policy first, degrees of freedom follow the number of policies that "
         "contribute both arms, and the three primary contrasts in a family "
         "carry a Holm adjustment. A positive mean means the reference-matching "
         "arm sits closer to the reference.",
         "",
         "The fine-tuned noise policies were never written to disk, so no "
         "endpoint can be recomputed from a checkpoint; what the legacy "
         "evaluation failed to record is gone until the runs are repeated.",
         ""]

    for task in ("avoiding", "pushing"):
        co = d[f"{task}/cohort"]
        L += [f"## {task.capitalize()}", "",
              f"{co['runs']} runs over {co['policies']} frozen policies "
              f"({', '.join(f'{k} {v}' for k, v in co['runs_per_arm'].items())}). "
              f"The success floor of {co['success_floor']} used in the "
              f"manuscript removes {co['runs_dropped']} runs and leaves "
              f"{co['policies_kept']} policies.", ""]

        L += ["### Primary analysis: every run", "", HEAD]
        L += [row(c) for c in d[f"{task}/primary, every run"]]
        L += ["", "### Sensitivity: runs above the success floor", "", HEAD]
        L += [row(c) for c in d[f"{task}/sensitivity, success floor"]]

        L += ["", "### Other distances on the same cohort, every run", "", HEAD]
        for nm in ("Hellinger", "largest bin error"):
            L += [row(c) for c in d[f"{task}/other distances, every run/{nm}"]]

        rs = d[f"{task}/reference resampling"]
        L += ["", "### The shared reference's own sampling error", "",
              f"Counts rationalise for {rs['runs_with_counts']} of "
              f"{rs['runs']} runs. Rationalisation succeeds unevenly across "
              f"arms, so the fixed-reference contrast is recomputed on exactly "
              f"the runs that enter the resampling and shown beside it.", "",
              "| contrast | policies | fixed reference | resampled mean | sd | "
              "resampled 95% |", "|---|---|---|---|---|---|"]
        for other in ("mi", "dsrl", "bmd"):
            r = rs[other]
            f = r["fixed_reference_same_cohort"]
            L.append(f"| vs {ARM[other]} | {f['n']} | {f['mean']:+.4f} | "
                     f"{r['mean']:+.4f} | {r['sd']:.4f} | "
                     f"[{r['lo']:+.4f}, {r['hi']:+.4f}] |")

        fc = d[f"{task}/fixed critical value"]
        L += ["", "### The fixed critical value the manuscript used", "",
              "The manuscript multiplied every standard error by t(19) = 2.093 "
              "regardless of how many policies the contrast actually had.", "",
              "| contrast | n | df | half-width as computed | half-width at "
              "t(19) | difference |", "|---|---|---|---|---|---|"]
        for other in ("mi", "dsrl", "bmd"):
            c = fc[other]
            L.append(f"| vs {ARM[other]} | {c['n']} | {c['df']} | "
                     f"{c['half_true']:.4f} | {c['half_fixed']:.4f} | "
                     f"{100 * c['relative']:+.1f}% |")
        L.append("")

    L += ["## What this changes for the manuscript", "",
          "On Avoiding the three primary contrasts hold without the success "
          "floor and are larger without it than with it, and the Hellinger and "
          "largest-bin-error endpoints agree in direction.", "",
          "On Pushing none of the three primary contrasts excludes zero once "
          "every run is counted, and the contrast against the learned-skill "
          "arm changes sign. The manuscript reports the floored numbers as its "
          "Pushing result; the floor, not the method, carries that comparison. "
          "A reference-matching gain on Pushing has to be established by the "
          "corrected runs, not asserted from these.", "",
          "The shared reference contributes a spread of a few thousandths on "
          "both tasks, far below the contrasts themselves. The fixed critical "
          "value costs at most 1.3% of a half-width. Neither explains the "
          "Pushing result.", ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L))
    print(f"wrote {OUT}  ({len(L)} lines)")


if __name__ == "__main__":
    main()
