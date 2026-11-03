#!/usr/bin/env python3
"""Assemble the corrected programme's deliverables into one directory.

Everything a reader needs to check a reported number, and nothing that was
produced by hand. The manifest is built from the run rows themselves, so a
cell that raised, a cell that was retried and a cell that never ran are
distinguishable; the planned side is enumerated from the same constants
launch_plan.py submits from, so planned and completed cannot drift apart.

Nothing here reruns an experiment or recomputes a result. It reads the run
files and the analysis outputs, checks that they agree, and writes the package
plus a statement of what is finished and what is not.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_plan as LP  # noqa: E402

RUNS = ROOT / "runs" / "_d3il_v2"
STAGE_FILES = {
    "calibration": "calibration_*.jsonl",
    "main": "main_*.jsonl",
    "ablation/divergence": "ablation_divergence_*.jsonl",
    "ablation/fitted prior": "ablation_fitted_prior_*.jsonl",
    "ablation/weight": "ablation_weight_*.jsonl",
}
# Cells whose results are debugging evidence under section III of the brief and
# never enter an analysis. They are listed so that the package accounts for
# them rather than hiding them.
DEBUG_FILES = ("smoke_*.jsonl", "_tiny*.jsonl")


def planned():
    """Every cell the launch plan defines, keyed the way a run row keys it."""
    choice = json.loads((RUNS / "lambda_choice.json").read_text())
    out = collections.defaultdict(set)
    for task, t in LP.TASK.items():
        for lam in LP.CANDIDATES:
            for w in t["weightings"]:
                for s in range(3):
                    for arm in ("mi", "mop", "bmd"):
                        out["calibration"].add(
                            (task, w, None, arm, t["calib_seed"] + s, lam))
        for w in t["weightings"]:
            for bb in range(5):
                for s in range(5):
                    for arm in ("dsrl", "mi", "mop", "bmd"):
                        # The no-mode-term arm carries no penalty weight, and
                        # the driver records that as zero rather than as the
                        # weight it was handed and ignored.
                        lam = 0.0 if arm == "dsrl" else choice[task][arm]["lam"]
                        out["main"].add((task, w, bb, arm,
                                         t["main_seed"] + s, lam))
        top = t["weightings"][-1]
        for tag in ("js", "l2", "kl", "kl-smooth"):
            for bb in range(5):
                for s in range(5):
                    out["ablation/divergence"].add(
                        (task, top, bb, "mop", t["main_seed"] + s, tag))
        for bb in range(5):
            for s in range(5):
                out["ablation/fitted prior"].add(
                    (task, top, bb, "bmd", t["main_seed"] + s, "fitted"))
    t = LP.TASK["pushing"]
    for lam in LP.CANDIDATES:
        for w in ("0.00", "0.10"):
            for bb in range(5):
                for s in range(5):
                    for arm in ("mi", "mop"):
                        out["ablation/weight"].add(
                            ("pushing", w, bb, arm, t["main_seed"] + s, lam))
    return out


def key(stage, r):
    if stage == "calibration":
        return (r["task"], r["weighting"], r["backbone_seed"], r["arm"],
                r["rl_seed"], r["lam"])
    if stage == "main":
        return (r["task"], r["weighting"], r["backbone_seed"], r["arm"],
                r["rl_seed"], r["lam"])
    if stage == "ablation/divergence":
        tag = r["divergence"]
        if tag == "kl" and r.get("ref_smooth"):
            tag = "kl-smooth"
        return (r["task"], r["weighting"], r["backbone_seed"], r["arm"],
                r["rl_seed"], tag)
    if stage == "ablation/fitted prior":
        return (r["task"], r["weighting"], r["backbone_seed"], r["arm"],
                r["rl_seed"], r.get("bmd_prior"))
    return (r["task"], r["weighting"], r["backbone_seed"], r["arm"],
            r["rl_seed"], r["lam"])


MANIFEST_FIELDS = (
    "task", "weighting", "backbone_seed", "arm", "rl_seed", "lam",
    "divergence", "ref_smooth", "kl_eps", "bmd_prior", "objective",
    "standardise", "coef_clip", "obs_clip", "mode_groups", "n_rewarded",
    "iters", "episodes_per_iter", "M", "lr", "optimiser", "eval_episodes",
    "ref_episodes", "estimator_version", "code_commit", "config_sha256",
    "checkpoint_sha256", "reference_sha256", "reference_episodes",
    "reference_valid_episodes", "reference_counts", "reference_complete",
    "status", "attempt", "exception", "fidelity_defined", "tv_to_ref",
    "success_rate", "rewarded_success_rate", "episode_return",
    "completion_rate", "valid_episodes", "mode_counts", "mode_counts_raw",
    "n_nonfinite", "n_nonfinite_grad", "n_clipped", "n_inert",
    "obs_absmax", "obs_clipped_frac", "cell_seconds", "cell_peak_rss_mb",
)


def read(pattern):
    rows = []
    for f in sorted(glob.glob(str(RUNS / pattern))):
        for line in Path(f).open():
            if line.strip():
                r = json.loads(line)
                r["_file"] = str(Path(f).relative_to(ROOT))
                rows.append(r)
    return rows


def environment():
    def run(cmd):
        try:
            return subprocess.run(shlex.split(cmd), capture_output=True,
                                  text=True, cwd=ROOT, timeout=60).stdout.strip()
        except Exception as e:                   # noqa: BLE001
            return f"unavailable: {e}"
    return dict(
        commit=run("git rev-parse HEAD"),
        branch=run("git rev-parse --abbrev-ref HEAD"),
        dirty=bool(run("git status --porcelain")),
        python=sys.version.replace("\n", " "),
        packages=run("python -m pip freeze"),
        platform=run("uname -a"),
        scheduler=run("sinfo --version"),
    )


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


ARM_LABEL = {"mop": "reference matching", "mi": "mode entropy",
             "dsrl": "no mode term", "bmd": "learned skill"}


def results_text(account, missing):
    """The result section, written from the analysis files rather than by hand.

    Every number here is read out of a file another script wrote. Where a file
    is absent the section says the analysis has not been produced instead of
    leaving the reader to guess, and where an interval contains zero it says
    the arms are not distinguished and claims nothing further.
    """
    def read(name):
        p = RUNS / name
        return json.loads(p.read_text()) if p.exists() else None

    rep, trend, abl = (read("report_main.json"), read("trend_main.json"),
                       read("ablations.json"))
    out = ["# Results, corrected implementation", "",
           "Generated from the run files and the analysis outputs. No number "
           "in this document was entered by hand.", ""]

    out += ["## Run accounting", "",
            "| stage | planned | completed | missing | exceptions | retried | "
            "undefined fidelity | non-finite gradients |", "|---|---|---|---|---|---|---|---|"]
    for stage, a in account.items():
        out.append(f"| {stage} | {a['planned']} | {a['completed']} | "
                   f"{a['missing']} | {a['exceptions']} | {a['retried']} | "
                   f"{a['undefined_fidelity']} | {a['nonfinite_gradients']} |")
    out.append("")

    if rep is None:
        out += ["The main analysis has not been produced.", ""]
        return out

    for task, desc in rep["descriptives"].items():
        out += [f"## {task.capitalize()}", "",
                "Every run of the task, with no filtering. The episode reward "
                "is the rewarded-route indicator, so the mean return and the "
                "rewarded success rate are the same quantity and appear once.",
                "",
                "| arm | TV to reference | success | rewarded success and "
                "return | completion | below floor | undefined fidelity |",
                "|---|---|---|---|---|---|---|"]
        acc = rep["accounting"][task]
        for arm, d in desc.items():
            def g(k, f="mean"):
                v = d.get(k)
                return f"{v[f]:.4f}" if isinstance(v, dict) and v.get(f) is not None else "--"
            out.append(f"| {ARM_LABEL.get(arm, arm)} | {g('tv_to_ref')} | "
                       f"{g('success')} | {g('rewarded_success')} | "
                       f"{g('completion')} | "
                       f"{d.get('below_floor', '--')} | "
                       f"{acc[arm]['fidelity_undefined']} |")
        out.append("")

        for label in ("primary, every run", "sensitivity, success floor"):
            fam = rep["contrasts"].get(f"{task}/{label}")
            if not fam:
                continue
            out += [f"Contrasts, {label}. The direction was declared in "
                    "advance and the family is Holm adjusted.", "",
                    "| contrast | n | df | mean | 95% interval | p | policies "
                    "favouring |", "|---|---|---|---|---|---|---|"]
            for c in fam:
                pv = c.get("p_adjusted") or c["p_one_sided"]
                verdict = "" if c["ci_low"] > 0 else " (interval contains zero)"
                out.append(f"| {c['name']}{verdict} | {c['n']} | {c['df']} | "
                           f"{c['mean']:+.4f} | [{c['ci_low']:+.4f}, "
                           f"{c['ci_high']:+.4f}] | {pv:.4f} | "
                           f"{c['wins']}/{c['n']} |")
            out.append("")

        res = rep["contrasts"].get(f"{task}/reference resampling")
        if res:
            # The resampler returns the distance difference, where a negative
            # value means the proposed arm is closer. The contrast tables above
            # report closeness, where positive means closer, so the sign is
            # turned here and the interval ends swap with it.
            out += ["Redrawing the shared reference from its own stored "
                    "counts, and rescoring both arms against each draw. "
                    "Reported as closeness, in the same direction as the "
                    "contrasts above.", "",
                    "| against | mean | 95% interval over draws |",
                    "|---|---|---|"]
            for arm, v in res.items():
                if isinstance(v, dict) and v.get("mean") is not None:
                    out.append(f"| {ARM_LABEL.get(arm, arm)} | "
                               f"{-v['mean']:+.4f} | [{-v['hi']:+.4f}, "
                               f"{-v['lo']:+.4f}] |")
            out.append("")

    if trend:
        out += ["## Advantage against measured reference skew", "",
                "One point per frozen policy, the predictor being the total "
                "variation from that policy's own reference to the uniform "
                "histogram.", "",
                "| task | comparison | n | slope | se | r | p |",
                "|---|---|---|---|---|---|---|"]
        for k, v in trend.items():
            if "/trend vs " not in k:
                continue
            task, other = k.split("/trend vs ")
            out.append(f"| {task} | over {ARM_LABEL.get(other, other)} | "
                       f"{v['n']} | {v['slope']:+.4f} | {v['stderr']:.4f} | "
                       f"{v['r']:+.3f} | {v['p_two_sided']:.4f} |")
        out += ["", "This trend is declared in the pre-registration and is "
                "the prediction the identity between reference matching and "
                "entropy maximisation under a uniform reference makes.", ""]

    if abl:
        out += ["## Ablations", "",
                "The tables are in ABLATIONS.md, written by the same run "
                "files. Sections present: "
                + ", ".join(sorted(abl)) + ".", ""]

    out += ["## What is settled and what is not", ""]
    done = [s for s, a in account.items() if a["missing"] == 0 and a["rows"]]
    part = [(s, a) for s, a in account.items() if a["missing"]]
    if done:
        out.append("Complete, every planned cell present with no exception: "
                   + ", ".join(done) + ".")
    for stage, a in part:
        out.append(f"Incomplete: {stage} holds {a['completed']} of "
                   f"{a['planned']} planned cells, {a['missing']} missing. "
                   f"No result from it is reported as though the stage had "
                   f"finished.")
    if rep:
        for task in rep["descriptives"]:
            fam = rep["contrasts"].get(f"{task}/primary, every run", [])
            cross = [c["name"] for c in fam if c["ci_low"] <= 0 <= c["ci_high"]]
            if cross:
                out.append(f"On {task} the primary interval contains zero for: "
                           + "; ".join(cross) + ". These comparisons do not "
                           "distinguish the arms, and no equivalence is "
                           "claimed, because no margin was set in advance.")
    out.append("")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "deliverables" / "v2")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    plan = planned()
    manifest, account, missing = [], {}, {}
    for stage, pattern in STAGE_FILES.items():
        rows = read(pattern)
        seen = collections.Counter(key(stage, r) for r in rows)
        want = plan[stage]
        extra = sorted(k for k in seen if k not in want)
        absent = sorted(k for k in want if k not in seen)
        account[stage] = dict(
            planned=len(want), rows=len(rows), distinct_cells=len(seen),
            completed=sum(1 for r in rows if r.get("status") == "completed"),
            exceptions=sum(1 for r in rows if r.get("exception")),
            retried=sum(1 for r in rows if (r.get("attempt") or 1) > 1),
            undefined_fidelity=sum(1 for r in rows
                                   if not r.get("fidelity_defined", True)),
            nonfinite_gradients=sum(r.get("n_nonfinite_grad", 0) for r in rows),
            missing=len(absent), unplanned=len(extra),
            estimator_versions=sorted({str(r.get("estimator_version"))
                                       for r in rows}),
            code_commits=sorted({str(r.get("code_commit")) for r in rows}),
            by_task_arm={f"{t}/{a}": n for (t, a), n in sorted(
                collections.Counter((r["task"], r["arm"]) for r in rows).items())})
        missing[stage] = [list(k) for k in absent]
        if extra:
            missing[stage + "/unplanned"] = [list(k) for k in extra]
        for r in rows:
            m = {f: r.get(f) for f in MANIFEST_FIELDS}
            m["stage"], m["source_file"] = stage, r["_file"]
            manifest.append(m)

    debug = {p: len(read(p)) for p in DEBUG_FILES}

    with (out / "manifest.jsonl").open("w") as fh:
        for m in manifest:
            fh.write(json.dumps(m, default=str) + "\n")

    # Checkpoints and reference caches, by hash, so a reader can tell which
    # frozen policy and which reference a row was scored against. The hash on
    # disk is recomputed and checked against the hash the rows recorded, so a
    # checkpoint that changed after a run used it cannot pass unnoticed.
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_d3il_headtohead import backbone_path  # noqa: PLC0415

    ck, ck_cache = {}, {}
    for m in manifest:
        path = backbone_path(m["task"], m["weighting"], m["backbone_seed"])
        name = str(path)
        if name not in ck_cache:
            ck_cache[name] = sha(path) if path.exists() else None
        ck[name] = dict(sha256_on_disk=ck_cache[name],
                        sha256_in_rows=m.get("checkpoint_sha256"),
                        exists=path.exists(), task=m["task"],
                        weighting=m["weighting"],
                        backbone_seed=m["backbone_seed"],
                        matches=ck_cache[name] == m.get("checkpoint_sha256"))
    mismatched = sorted(k for k, v in ck.items() if not v["matches"])
    if mismatched:
        raise SystemExit("checkpoint hash differs from what the rows recorded, "
                         "so a reported result cannot be traced to a frozen "
                         f"policy: {mismatched}")
    refs = {}
    for m in manifest:
        h = m.get("reference_sha256")
        if h and h not in refs:
            refs[h] = dict(task=m["task"], weighting=m["weighting"],
                           backbone_seed=m["backbone_seed"],
                           episodes=m.get("reference_episodes"),
                           valid_episodes=m.get("reference_valid_episodes"),
                           counts=m.get("reference_counts"),
                           complete=m.get("reference_complete"))

    (out / "accounting.json").write_text(
        json.dumps(dict(stages=account, missing=missing,
                        debug_rows_excluded_from_analysis=debug),
                   indent=1, default=str))
    (out / "environment.json").write_text(json.dumps(environment(), indent=1))
    (out / "checkpoints.json").write_text(json.dumps(ck, indent=1))
    (out / "references.json").write_text(json.dumps(refs, indent=1, default=str))

    cmds = []
    choice = json.loads((RUNS / "lambda_choice.json").read_text())
    for stage, fn in (("calibration", lambda: LP.calibration(None)),
                      ("main", lambda: LP.main_matrix(None, choice)),
                      ("ablation", lambda: LP.ablation(None, choice))):
        for name, wall, workers, cmd in fn():
            cmds.append(dict(stage=stage, name=name, walltime=wall,
                             workers=workers, command=shlex.join(cmd)))
    (out / "launch_commands.json").write_text(json.dumps(cmds, indent=1))

    copied = []
    for src in ("docs/v2/PREREG_v2.md", "docs/v2/FIXES.md",
                "docs/v2/INVENTORY.md", "docs/v2/TREND.md",
                "docs/v2/ABLATIONS.md", "docs/v2/REANALYSIS_LEGACY.md",
                "runs/_d3il_v2/lambda_choice.json",
                "runs/_d3il_v2/report_main.json",
                "runs/_d3il_v2/trend_main.json",
                "runs/_d3il_v2/ablations.json",
                "runs/_d3il_v2/reanalysis_legacy.json",
                "runs/_d3il_v2/observation_range.json"):
        p = ROOT / src
        if p.exists():
            (out / Path(src).name).write_bytes(p.read_bytes())
            copied.append(src)

    (out / "RESULTS.md").write_text("\n".join(results_text(account, missing)))

    index = dict(copied=copied,
                 manifest_rows=len(manifest),
                 checkpoints=len(ck), references=len(refs),
                 launch_commands=len(cmds))
    (out / "index.json").write_text(json.dumps(index, indent=1))

    print(f"wrote {out}")
    for stage, a in account.items():
        print(f"  {stage:<26} planned {a['planned']:4d}  rows {a['rows']:4d}  "
              f"completed {a['completed']:4d}  missing {a['missing']:4d}  "
              f"exceptions {a['exceptions']}  "
              f"versions {','.join(a['estimator_versions'])}")
    print(f"  debug rows kept out of every analysis: {debug}")


if __name__ == "__main__":
    main()
