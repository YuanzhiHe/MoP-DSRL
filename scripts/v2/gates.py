#!/usr/bin/env python3
"""Checks that run before a cohort, each one a past failure made mechanical.

Every item here corresponds to something that reached a report and had to be
withdrawn. The point is that none of them is a judgement call: each either
passes or names what failed, and the launcher refuses to submit a matrix while
any of them is failing.

  environment   A process builds one environment. The parity probe showed the
                second one a process builds does not behave like the first, so
                the factory raises instead of returning it and every pool hands
                its tasks to processes that have built nothing.
  hash          The environment rule and the observation build enter the
                configuration hash and the reference cache key. Two protocols
                previously produced the same cell key, which let a resume
                satisfy a new cell with an old row.
  observation   The demonstrations and the environment present the same
                channels. A backbone cloned on one layout and rolled out on
                another sees something it was never trained on.
  estimators    The divergence gradients and the skill prior are checked
                against cases where a wrong one scores well: the prior's old
                objective rated weight 1.0 on a skill of zero completion as a
                perfect fit.
  coverage      Every run file on disk is read by some table or figure. The
                Pushing half of the divergence comparison sat unread while the
                Avoiding half carried a claim that it reverses.
  policies      Runs save their trained weights, so an error found on the
                evaluation side costs a rescore rather than the matrix.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import envfactory as EF  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  {detail}" if detail else ""), flush=True)


def gate_environment() -> None:
    print("environment")
    import run_d3il_v2 as D

    EF._reset_for_test()
    try:
        EF.build("avoiding", D.TASKS, 0, 1)
        first = True
    except Exception as exc:                                # noqa: BLE001
        check("first build succeeds", False, repr(exc))
        return
    check("first build succeeds", first, f"count {EF.built_count()}")

    try:
        EF.build("avoiding", D.TASKS, 0, 1)
        check("second build refused", False, "a second environment was returned")
    except EF.EnvironmentReused:
        check("second build refused", True, "EnvironmentReused raised")
    EF._reset_for_test()

    src = (Path(__file__).resolve().parent / "run_d3il_v2.py").read_text()
    frz = (Path(__file__).resolve().parent / "frozen_baseline.py").read_text()
    check("no stage builds an environment outside the factory",
          'TASKS[args.task]["env"](' not in src
          and 'TASKS[task]["env"](' not in src
          and 'TASKS[args.task]["env"](' not in frz)
    pools = src.count("ProcessPoolExecutor(") + frz.count("ProcessPoolExecutor(")
    per_task = (src.count("max_tasks_per_child=1")
                + frz.count("max_tasks_per_child=1"))
    check("every pool hands each task a fresh process",
          pools == per_task, f"{per_task} of {pools} pools")

    # A stage runs many processes at once and each one otherwise sizes its
    # thread pool to the whole allocation. Twenty concurrent clones asked for
    # 480 threads on 24 cores and finished none of twenty in six hours; with
    # one thread each they finished in five minutes.
    wrapper = (Path(__file__).resolve().parent / "cpu_job_v2.sh").read_text()
    pinned = [v for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                          "OPENBLAS_NUM_THREADS")
              if f"export {v}=1" in wrapper]
    check("the job wrapper pins the per-process thread count",
          len(pinned) == 3, ", ".join(pinned) or "none")
    import os as _os
    live = [v for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
            if _os.environ.get(v) == "1"]
    check("this job is running with threads pinned",
          len(live) == 2, ", ".join(f"{v}={_os.environ.get(v)}"
                                    for v in ("OMP_NUM_THREADS",
                                              "MKL_NUM_THREADS")))


def gate_hash() -> None:
    print("hash")
    import run_d3il_v2 as D

    ap = D.build_parser()
    args = ap.parse_args(["--task", "avoiding", "--weightings", "0.00",
                          "--out", "/dev/null"])
    h1 = D.config_hash(args, "mop", 1.0)
    saved = EF.ENV_PROTOCOL
    EF.ENV_PROTOCOL = saved + "-other"
    h2 = D.config_hash(args, "mop", 1.0)
    EF.ENV_PROTOCOL = saved
    check("environment protocol changes the config hash", h1 != h2)

    saved_obs = D.TASKS["avoiding"]["obs_version"]
    D.TASKS["avoiding"]["obs_version"] = saved_obs + "-other"
    h3 = D.config_hash(args, "mop", 1.0)
    D.TASKS["avoiding"]["obs_version"] = saved_obs
    check("observation version changes the config hash", h1 != h3)

    src = (Path(__file__).resolve().parent / "run_d3il_v2.py").read_text()
    key = src[src.index("    key = dict("):src.index("    kh = hashlib")]
    check("reference cache key carries both",
          "env_protocol" in key and "obs_version" in key)
    check("estimator version moved past the old cohort",
          D.ESTIMATOR_VERSION != "v2.1", D.ESTIMATOR_VERSION)


def gate_observation() -> None:
    print("observation")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import train_d3il_backbone as T
    import run_d3il_v2 as D

    files = sorted(T.TASK_DATA["pushing"].glob("*.pkl"))[:3]
    obs, act, _ = T.load_pairs(files, history=1, task="pushing")
    check("demonstration frame has the sine-cosine layout",
          obs.shape[1] == 12, f"{obs.shape[1]} channels")
    for lo in (6, 10):
        n = np.hypot(obs[:, lo], obs[:, lo + 1])
        check(f"demonstration channels {lo},{lo + 1} lie on the unit circle",
              bool(np.allclose(n, 1.0, atol=1e-5)),
              f"max |norm-1| {float(np.abs(n - 1).max()):.2e}")

    EF._reset_for_test()
    env = EF.build("pushing", D.TASKS, 0, 1)
    o = np.asarray(env.reset(), dtype=np.float64)
    EF._reset_for_test()
    check("environment presents the same channel count",
          o.size == obs.shape[1], f"{o.size} against {obs.shape[1]}")
    for lo in (6, 10):
        n = float(np.hypot(o[lo], o[lo + 1]))
        check(f"environment channels {lo},{lo + 1} lie on the unit circle",
              abs(n - 1.0) < 1e-5, f"norm {n:.6f}")
    check("no tangent is left in either path",
          "np.tan(" not in T._frame_pushing.__code__.co_consts.__str__()
          and "np.tan" not in
          (Path(__file__).resolve().parents[1] / "train_d3il_backbone.py"
           ).read_text().split("def _frame_pushing")[1].split("FRAME_FN")[0])


def gate_estimators() -> None:
    print("estimators")
    import conditional_prior as CP
    import estimators as EST

    A = np.array([[0.40, 0.10, 0.00, 0.00],
                  [0.05, 0.35, 0.05, 0.00],
                  [0.00, 0.05, 0.30, 0.10],
                  [0.00, 0.00, 0.00, 0.00]])
    q = np.array([0.30, 0.25, 0.25, 0.20])
    r = CP.fit_conditional_prior(A, q)
    check("a skill that never finishes carries no weight",
          r["prior"][3] == 0.0 and r["inestimable"] == [3])
    check("the fitted prior deploys a policy that finishes",
          r["predicted_completion"] > 0.1,
          f"completion {r['predicted_completion']:.3f}")

    Bk = np.array([[0.7, 0.2, 0.1, 0.0], [0.1, 0.6, 0.2, 0.1],
                   [0.0, 0.1, 0.5, 0.4], [0.25, 0.25, 0.25, 0.25]])
    av = np.array([0.6, 0.3, 0.45, 0.2])
    alpha = np.array([0.4, 0.1, 0.35, 0.15])
    r2 = CP.fit_conditional_prior(Bk * av[:, None], alpha @ Bk)
    err = float(np.abs(np.asarray(r2["predicted_conditional"]) - alpha @ Bk).max())
    check("a reachable target is reached", err < 1e-6, f"max error {err:.1e}")

    # Each gradient against a finite difference of its own value.
    rng = np.random.default_rng(0)
    for name in ("tv", "js", "l2", "kl"):
        p0 = rng.dirichlet(np.ones(5))
        qq = rng.dirichlet(np.ones(5))
        g = EST.DIVERGENCES[name](p0, qq, EST.EPS)
        f = EST.VALUES[name]
        num = np.zeros_like(p0)
        h = 1e-6
        for k in range(len(p0)):
            e = np.zeros_like(p0); e[k] = h
            num[k] = (f(p0 + e, qq, EST.EPS)
                      - f(p0 - e, qq, EST.EPS)) / (2 * h)
        # The gradients are partials of the value, free of the simplex
        # constraint, so they agree up to a constant shift along it.
        d = (g - num) - (g - num).mean()
        check(f"{name} gradient matches its value",
              float(np.abs(d).max()) < 1e-4,
              f"max departure {float(np.abs(d).max()):.2e}")


def gate_coverage() -> None:
    print("coverage")
    root = Path(__file__).resolve().parents[2]
    runs = root / "runs" / "_d3il_v2"
    stages = json.loads((root / "scripts" / "v2" / "stage_map.json").read_text()) \
        if (root / "scripts" / "v2" / "stage_map.json").exists() else None
    if stages is None:
        check("stage map present", False,
              "scripts/v2/stage_map.json lists which files each table reads")
        return
    claimed = {f for k, pats in stages.items() if k != "_comment"
               for f in pats}
    on_disk = {p.name for p in runs.glob("*.jsonl")
               if not p.name.startswith("_")}
    unread = sorted(on_disk - claimed)
    check("every run file on disk is read by something",
          not unread, ", ".join(unread) if unread else "")

    # The map names files; the stages write them. Nothing made the two agree,
    # and a stage that writes a name the map does not carry produces a file
    # the analysis never reads -- which is the failure this map exists to
    # prevent, arriving at the end of the cohort instead of before it.
    import os
    os.environ.setdefault("SLURM_CPUS_PER_TASK", "8")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_stage as RS
    planned = set()
    for fn in RS.STAGES.values():
        for cmd in fn():
            if "--out" in cmd:
                planned.add(Path(cmd[cmd.index("--out") + 1]).name)
    reported = {f for k, v in stages.items()
                if k != "_comment"
                and not k.startswith(("superseded", "validation"))
                for f in v}
    missing = sorted(f for f in planned - reported if f.endswith(".jsonl"))
    stale = sorted(f for f in reported - planned)
    check("every file a stage writes is claimed by the map", not missing,
          ", ".join(missing))
    check("every file the map claims is written by a stage", not stale,
          ", ".join(stale))


def gate_policies() -> None:
    print("policies")
    src = (Path(__file__).resolve().parent / "run_d3il_v2.py").read_text()
    check("runs save their trained policy",
          "def save_policy(" in src and 'base["policy_sha256"]' in src)
    check("both arms hand their policy back",
          src.count('row["_policy"] = policy') == 2,
          f'{src.count(chr(34) + "_policy" + chr(34))} references')


def main() -> int:
    only = sys.argv[1:] or ["environment", "hash", "observation",
                            "estimators", "coverage", "policies"]
    fns = dict(environment=gate_environment, hash=gate_hash,
               observation=gate_observation, estimators=gate_estimators,
               coverage=gate_coverage, policies=gate_policies)
    for name in only:
        try:
            fns[name]()
        except Exception as exc:                            # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(f"{name} gate ran", False, repr(exc))
        print()
    bad = [n for n, ok, _ in RESULTS if not ok]
    print(f"{len(RESULTS) - len(bad)} of {len(RESULTS)} checks passed")
    if bad:
        print("failing: " + ", ".join(bad))
    # The launcher reads this line.
    print("GATES: " + ("PASS" if not bad else "FAIL"))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
