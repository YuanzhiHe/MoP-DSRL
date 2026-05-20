#!/usr/bin/env python3
"""Diagnose LIBERO task ↔ language ↔ bddl alignment.

Prints, for every task in every LIBERO suite:
  - task.name
  - task.language (if set, else "(none)")
  - task.bddl_file (filename only)
  - task.problem_folder
  - the objects mentioned in the bddl file (first 10)

Use this to verify that the instruction we pass to pi_0.5 matches the actual
scene that LIBERO loads.  If task[0].name says "open_the_middle_drawer" but
the bddl file mentions cream_cheese / wine_bottle / plate, we have a task
indexing bug — that explains 0% success.

Usage:
    python scripts/diag_libero_tasks.py
"""

from __future__ import annotations

import os
import sys
import pathlib

# IPv4 patch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _ipv4_patch  # noqa: F401,E402


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    from libero.libero import benchmark, get_libero_path

    bddl_root = get_libero_path("bddl_files")
    suites = ["libero_goal", "libero_spatial", "libero_object", "libero_10"]

    for suite in suites:
        try:
            b = benchmark.get_benchmark(suite)()
        except Exception as e:
            print(f"[{suite}] benchmark unavailable: {e}")
            continue
        print(f"\n============ {suite}  (n_tasks={b.n_tasks}) ============")
        for i in range(b.n_tasks):
            t = b.get_task(i)
            lang = getattr(t, "language", None) or "(none)"
            bddl_path = os.path.join(bddl_root, t.problem_folder, t.bddl_file)
            objs_str = ""
            try:
                with open(bddl_path) as f:
                    content = f.read()
                import re
                # bddl object declarations look like "(:objects xxx_1 yyy_1 - ...)"
                m = re.search(r":objects\s+(.+?)\s*\)", content, re.S)
                if m:
                    objs = m.group(1).split()
                    objs = [o for o in objs if not o.startswith("-")]
                    objs_str = ", ".join(objs[:10])
            except Exception as e:
                objs_str = f"(bddl read failed: {e})"
            print(f"  [{i:2d}] name = {t.name}")
            print(f"       language = {lang!r}")
            print(f"       bddl = {t.problem_folder}/{t.bddl_file}")
            print(f"       objects(first10) = {objs_str}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
