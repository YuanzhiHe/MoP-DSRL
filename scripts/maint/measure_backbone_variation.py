#!/usr/bin/env python3
"""How much does C_ref vary between backbones trained at the SAME skew level?

The review's pseudoreplication objection: with one backbone per level, every run
seed shares that checkpoint's idiosyncrasies, so a trend across levels has as
many independent units as it has levels. The objection only bites if backbones
trained on identical demonstration weights actually differ. This measures that
directly -- if between-backbone spread at one level is comparable to the spread
between levels, the original single-backbone design could not have identified a
skew effect at all.
"""
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.run_d3il_headtohead import reference_profile  # noqa: E402

K = 6
args = types.SimpleNamespace(workers=8, ref_episodes=900, M=10,
                             device="cpu", mode_groups=K, task="avoiding")

print(f"C_ref measured per backbone, {args.ref_episodes} rollouts each, {K} route groups\n")
print(f"{'demo skew':>10}  {'bb':>3}  {'KL(C_ref||u)':>13}  {'H':>7}  {'TV(C_ref,u)':>12}")
summary = {}
for skew in ("0.00", "0.15", "0.30", "0.60"):
    kls, tvs = [], []
    for bb in range(5):
        p = reference_profile(skew, args, bb)
        nz = p > 0
        kl = float(np.sum(p[nz] * np.log(p[nz] * K)))
        H = float(-(p[nz] * np.log(p[nz])).sum())
        tv = float(0.5 * np.abs(p - 1.0 / K).sum())
        kls.append(kl); tvs.append(tv)
        print(f"{skew:>10}  {bb:>3}  {kl:>13.3f}  {H:>7.3f}  {tv:>12.3f}")
    summary[skew] = (np.mean(kls), np.std(kls, ddof=1), np.mean(tvs))
    print(f"{'':>10}  {'mean':>3}  {np.mean(kls):>13.3f}  {'':>7}  {np.mean(tvs):>12.3f}"
          f"   (sd across backbones = {np.std(kls, ddof=1):.3f})")
    print()

means = np.array([v[0] for v in summary.values()])
sds = np.array([v[1] for v in summary.values()])
print(f"between-level spread of KL(C_ref||u):  sd = {means.std(ddof=1):.3f}")
print(f"within-level spread (mean over levels): sd = {sds.mean():.3f}")
ratio = sds.mean() / means.std(ddof=1) if means.std(ddof=1) > 0 else float("inf")
print(f"ratio within/between = {ratio:.2f}")
print()
if ratio > 0.5:
    print("Between-backbone variation is a large fraction of between-level variation,")
    print("so a single backbone per level cannot identify the skew effect: the")
    print("review's objection is substantive and the multi-backbone design is required.")
else:
    print("Between-backbone variation is small against between-level variation, so the")
    print("single-backbone design was less biased than feared -- still report with")
    print("backbone-level standard errors.")
