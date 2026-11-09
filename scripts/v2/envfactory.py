#!/usr/bin/env python3
"""The one place an environment is built, and the invariant it enforces.

Every stage of the sweep rolled a policy out, and each built its environment
its own way: one per reference chunk, one per fine-tuning cell, and for a
while one that was built and dropped before the one a frozen evaluation
measured on. The parity probe replayed a fixed latent sequence through
environments that differed in nothing but how many the process had already
built, and found that only the first one a process builds behaves:

    avoiding w0.00 bb0   env 1   completion 0.667   observations reach 2.5
                         env 2   completion 0.867   five episodes in sixty
                                                    reach 9.0e6 and run to the
                                                    step cap
    avoiding w0.30 bb0   env 1   completion 0.700
                         env 2   completion 1.000

Environments two and three are identical to each other, so this is a split
between the first and the rest rather than a drift. A single-worker run put the
point beyond doubt: a configuration that asked to measure on a first
environment, but ran seventh in a process that had already built six, behaved
exactly like a second one.

The invariant is therefore that a process builds one environment over its whole
life. ``build`` raises on the second call rather than returning something whose
dynamics cannot be compared with the first, and the drivers obtain their
workers with ``max_tasks_per_child=1`` so that every cell and every chunk
arrives in a process that has built nothing.

``ENV_PROTOCOL`` names this rule and enters the configuration hash, so a record
measured under it can never be served to a configuration that was not, and a
resume cannot satisfy a new cell with an old row.
"""
from __future__ import annotations

import os

# v1: whatever the stage happened to do, which is what every v2.1 record was
#     measured under and is not reproducible across stages.
# v2: one environment per process, enforced here.
ENV_PROTOCOL = "env-v2-one-per-process"

_BUILT: list[str] = []


class EnvironmentReused(RuntimeError):
    """A second environment was requested in a process that already has one."""


def built_count() -> int:
    """How many environments this process has built."""
    return len(_BUILT)


def build(task: str, tasks_table: dict, seed: int, history: int):
    """Build this process's environment. Valid once per process.

    ``tasks_table`` is the driver's ``TASKS`` mapping, passed in so this module
    stays free of the driver and can be imported by a probe on its own.
    """
    if _BUILT:
        raise EnvironmentReused(
            f"process {os.getpid()} already built an environment for "
            f"{_BUILT[0]!r}; a second one does not behave like the first, so "
            f"the work that needs it belongs in a fresh process "
            f"(pools take max_tasks_per_child=1)")
    _BUILT.append(task)
    return tasks_table[task]["env"](seed=seed, history=history)


def _reset_for_test() -> None:
    """Clear the counter. For tests that build several in one process."""
    _BUILT.clear()
