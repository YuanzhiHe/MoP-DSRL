#!/usr/bin/env python3
"""The IK guard must end one episode, not the sweep.

The simulator's inverse-kinematics SVD fails occasionally on Pushing, and three
jobs died from it after a day of work each because the failure surfaced inside
reference_profile, which runs outside the per-cell guard. This injects the
failure deterministically and checks that an episode ends, the counter records
it, and the next episode runs normally.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import envs_d3il


class Exploding:
    """Stands in for the D3IL env, raising LinAlgError on the nth step."""

    def __init__(self, fail_on):
        self.fail_on, self.n = fail_on, 0

    def reset(self):
        self.n = 0
        return np.zeros(4)

    def robot_state(self):
        return np.array([0.5, 0.0, 0.1])

    def step(self, pose):
        self.n += 1
        if self.n == self.fail_on:
            raise np.linalg.LinAlgError("SVD did not converge")
        return np.zeros(4), None, False, {"mode": -1, "success": False}


ok = True
for cls_name, cls in (("D3ILPushingEnv", envs_d3il.D3ILPushingEnv),
                      ("D3ILAvoidingEnv", envs_d3il.D3ILAvoidingEnv)):
    env = cls.__new__(cls)
    env._env = Exploding(fail_on=3)
    env.action_dim, env.max_steps, env.rewarded_modes = 2, 50, None
    env.history = 1
    cls.ik_failures = 0

    env.reset()
    steps, done = 0, False
    while not done and steps < 10:
        _, r, done, info = env.step(np.zeros(2))
        steps += 1
    hit = info.get("ik_failure", False)
    print(f"{cls_name}: ended after {steps} steps, ik_failure={hit}, "
          f"counter={cls.ik_failures}, reward={r}")
    good = steps == 3 and hit and cls.ik_failures == 1 and r == 0.0
    # the wrapper must still be usable afterwards
    env._env.fail_on = 99
    env.reset()
    _, _, done2, info2 = env.step(np.zeros(2))
    good &= not info2.get("ik_failure", False)
    print(f"{cls_name}: reusable after the failure = {not info2.get('ik_failure', False)}")
    ok &= good

print("\nVERDICT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
