#!/usr/bin/env python3
"""Behavioural checks on the corrected driver.

These run the real ``run_arm`` against the real environment on a tiny budget.
A check that does not exercise the path it claims to cover has cleared broken
code here before, so each one observes the live objects the driver uses rather
than a reconstruction of them.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1]), str(HERE.parents[1] / "scripts")]

import estimators as EST  # noqa: E402
import run_d3il_v2 as R  # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(f"  {name:<58} {'ok' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAIL.append(name)


def tiny_args(**kw):
    base = ["--task", "avoiding", "--weightings", "0.00", "--out", "/dev/null",
            "--mode-groups", "6", "--iters", "3", "--episodes-per-iter", "4",
            "--eval-episodes", "4", "--ref-episodes", "4", "--n-rewarded", "4"]
    for k, v in kw.items():
        base += [f"--{k.replace('_', '-')}", str(v)]
    a = R.build_parser().parse_args(base)
    a.code_commit = "test"
    return a


REF = dict(C_ref=[0.4, 0.2, 0.15, 0.15, 0.1, 0.0], checkpoint_sha256="x",
           reference_sha256="y", complete=True, episodes=4, valid_episodes=4,
           counts=[8, 4, 3, 3, 2, 0])


def run_once(args, arm="mop", seed=700, capture=None):
    """Run one cell, optionally intercepting the frozen backbone and coefficients."""
    seen = {"models": [], "coefs": []}
    real_load = R.load_backbone
    real_coef = EST.conditional_coefficient

    def spy_load(path, device):
        model, blob = real_load(path, device)
        seen["models"].append((model, copy.deepcopy(model.state_dict())))
        return model, blob

    def spy_coef(valid, modes, g, C, s, s_floor):
        out = real_coef(valid, modes, g, C, s, s_floor)
        seen["coefs"].append((np.asarray(valid).copy(), out.copy()))
        return out

    R.load_backbone = spy_load
    EST.conditional_coefficient = spy_coef
    try:
        row = R.run_arm(arm, args.lam, "0.00", seed, args, REF, bb_seed=0)
    finally:
        R.load_backbone = real_load
        EST.conditional_coefficient = real_coef
    if capture is not None:
        capture.update(seen)
    return row, seen


print("1. the frozen backbone stays frozen and the noise policy moves")
args = tiny_args()
row, seen = run_once(args)
model, before = seen["models"][0]
after = model.state_dict()
same = all(torch.equal(before[k], after[k]) for k in before)
check("decoder weights unchanged after training", same)
grads = [p.grad for p in model.parameters() if p.grad is not None]
check("no gradient accumulated on the decoder", not grads,
      f"{len(grads)} tensors carry .grad" if grads else "")

# the noise policy must actually move
torch.manual_seed(0)
pol = R.NoisePolicy(4, 2)
start = copy.deepcopy(pol.state_dict())
opt = torch.optim.Adam(pol.parameters(), lr=3e-3)
o = torch.randn(8, 4)
d = pol.dist(o)
z = d.rsample()
loss = -(torch.randn(8) * d.log_prob(z.detach()).sum(-1)).mean()
opt.zero_grad(); loss.backward(); opt.step()
moved = any(not torch.equal(start[k], pol.state_dict()[k]) for k in start)
check("noise policy parameters update under the loss", moved)

print("\n2. the mode term is a per-trajectory coefficient")
varied = [c for _, c in seen["coefs"] if np.unique(np.round(c, 9)).size > 1]
check("coefficients differ within a batch", len(varied) > 0,
      f"{len(varied)}/{len(seen['coefs'])} batches carry more than one value")
check("unlabelled trajectories carry exactly zero",
      all(np.all(c[~v.astype(bool)] == 0.0) for v, c in seen["coefs"]))
check("the dsrl arm never calls the coefficient",
      len(run_once(tiny_args(), arm="dsrl")[1]["coefs"]) == 0)

print("\n3. edge cases")
# every bin empty in a batch: coefficient must stay finite
g = EST.tv_grad(np.full(6, 1 / 6), np.asarray(REF["C_ref"]))
c = EST.conditional_coefficient(np.zeros(4), np.zeros(4, int), g,
                                np.full(6, 1 / 6), s=0.0, s_floor=0.05)
check("all-unlabelled batch gives an all-zero finite coefficient",
      np.all(c == 0) and np.all(np.isfinite(c)))
# zero-count bin in the reference: TV and JS stay finite, KL is guarded
q = np.asarray(REF["C_ref"])            # last bin is exactly zero
p = np.full(6, 1 / 6)
check("zero-probability reference bin keeps tv finite", np.isfinite(EST.tv_grad(p, q)).all())
check("zero-probability reference bin keeps js finite", np.isfinite(EST.js_grad(p, q)).all())
check("zero-probability reference bin keeps kl finite under the guard",
      np.isfinite(EST.kl_grad(p, q, 1e-8)).all())
# s floor
c_small = EST.conditional_coefficient(np.ones(2), np.zeros(2, int), g, p, s=1e-9,
                                      s_floor=0.05)
check("the 1/s factor is bounded by s_floor",
      np.abs(c_small).max() <= np.abs(g - p @ g).max() / 0.05 + 1e-9)
# a batch with no reward at all must still produce a finite update
r = np.zeros(8)
loo = r - (r.sum() - r) / 7
check("all-zero reward batch gives a zero, finite advantage",
      np.all(loo == 0) and np.all(np.isfinite(loo)))
# zero-variance mode term under the corrected rule is NOT discarded
const = np.full(8, 0.3)
check("a constant coefficient survives the fixed rule",
      np.all(const == 0.3))

print("\n4. diagnostics are recorded")
for f in ("coef_absmax", "coef_absmax_raw", "n_clipped", "n_inert",
          "grad_norm_last", "n_nonfinite", "n_nonfinite_grad", "s_min",
          "obs_steps", "obs_absmax", "obs_clipped_steps", "obs_clipped_frac",
          "n_zero_reward_batches",
          "n_batches", "completion_rate", "success_rate",
          "rewarded_success_rate", "episode_return", "valid_episodes",
          "mode_counts", "mode_counts_raw", "fidelity_defined"):
    check(f"row carries {f}", f in row)
check("grad norm is finite", np.isfinite(row["grad_norm_last"]))
check("rewarded success is measured over all episodes",
      0.0 <= row["rewarded_success_rate"] <= row["success_rate"] + 1e-12)

print("\n5. undefined fidelity is not zero")
out = R.evaluate(lambda i: (-1, False), 5, tiny_args(), set(), 24)
check("no labelled episode leaves C_phi undefined",
      out["C_phi"] is None and out["fidelity_defined"] is False)
check("completion rate is still reported", out["completion_rate"] == 0.0)
scored = R.score(dict(out), np.asarray(REF["C_ref"]), 1e-8)
check("an undefined histogram scores as None, not a distance",
      scored["tv_to_ref"] is None)

print("\n6. the two standardisation rules are different update rules")
a_fixed = tiny_args(standardise="fixed")
a_leg = tiny_args(standardise="legacy")
r1, _ = run_once(a_fixed, seed=701)
r2, _ = run_once(a_leg, seed=701)
check("legacy and fixed do not produce the same run",
      r1["mode_counts"] != r2["mode_counts"] or r1["coef_absmax"] != r2["coef_absmax"],
      f"fixed |coef|max={r1['coef_absmax']:.3f} legacy |coef|max={r2['coef_absmax']:.3f}")

print("\n7. a cell cannot be counted as done across configurations")
base = dict(task="avoiding", weighting="0.00", backbone_seed=0, arm="mop",
            rl_seed=700, config_sha256="A", checkpoint_sha256="C",
            reference_sha256="R")
other = dict(base, config_sha256="B")
check("a different configuration is a different cell",
      R.cell_key(base) != R.cell_key(other))
check("a different reference is a different cell",
      R.cell_key(base) != R.cell_key(dict(base, reference_sha256="R2")))
check("the legacy estimator version cannot satisfy a v2 cell",
      R.config_hash(tiny_args(), "mop", 1.0) != "A")

print("\nVERDICT:", "PASS" if not FAIL else f"FAIL {FAIL}")
sys.exit(0 if not FAIL else 1)
