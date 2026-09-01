#!/usr/bin/env python3
"""Weight D3IL demonstrations to realise a controlled behavioural-mode skew.

Why. The toy experiment established a necessary condition for this project's
contribution to be measurable at all: KL(C_phi || C_ref) and an entropy/MI
objective are the SAME objective when the reference is uniform, because
KL(p || uniform) = ln K - H(p). They were numerically identical to three
decimals in that setting. The two only separate when the pretrained policy's own
mode distribution is skewed -- MoP then preserves that profile while MI drives
the histogram toward uniform, i.e. away from the backbone.

D3IL is deliberately balanced ("we collected data such that we have an
approximately equal number of demonstrations for each behavior descriptor"), and
Avoiding is exactly uniform by construction: 96 demos = 24 modes x 4. Used
as-is it would reproduce the LIBERO outcome for a different reason -- a null
that was structurally guaranteed.

So the skew becomes the independent variable. This script emits per-demo
SAMPLING WEIGHTS realising a target KL(demo || uniform), so the head-to-head runs
across a skew sweep rather than at a single arbitrary point. The curve is the
argument: the arms should coincide at KL=0 and separate as the backbone's profile
departs from uniform.

Weights, not subsets. Avoiding has 4 demos per mode, so choosing a subset cannot
express a skewed profile -- a 90x ratio needs 90 demos in one mode against 1 in
another when only 5 are available. Measured, subsetting undershot badly (target
0.30 -> 0.131, target 0.60 -> 0.089) while shrinking the set to 30 demos.
Weighting hits every target exactly and keeps all 96 demonstrations.

    python scripts/make_skewed_demos.py --data-root <...>/avoiding \\
        --target-kl 0 0.05 0.15 0.3 0.6 --out-dir <...>/avoiding/skew
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def geometric_profile(k: int, target_kl: float, tol: float = 1e-4) -> np.ndarray:
    """Mode probabilities p_i ∝ r^i whose KL to uniform matches ``target_kl``.

    A geometric profile is used because it is monotone in a single parameter, so
    the skew is describable by one number in the paper and the sweep is ordered.
    """
    if target_kl <= 0:
        return np.full(k, 1.0 / k)

    def kl_of(r: float) -> float:
        p = r ** np.arange(k, dtype=float)
        p /= p.sum()
        return float(np.sum(p * np.log(p * k)))

    lo, hi = 1e-6, 1.0 - 1e-9          # r -> 1 is uniform, r -> 0 is a point mass
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if kl_of(mid) > target_kl:
            lo = mid
        else:
            hi = mid
        if abs(kl_of(mid) - target_kl) < tol:
            break
    p = (0.5 * (lo + hi)) ** np.arange(k, dtype=float)
    return p / p.sum()


def load_mode_map(data_root: Path) -> dict[int, list[str]]:
    """mode id -> demonstration files, from D3IL's own all_modes manifest."""
    for name in ("all_modes.pkl", "mode_files.pkl"):
        f = data_root / name
        if f.exists():
            with open(f, "rb") as fh:
                raw = pickle.load(fh)
            return {int(k): list(v) for k, v in raw.items()}
    raise FileNotFoundError(
        f"no all_modes.pkl under {data_root}; the D3IL archive must be extracted first"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--target-kl", nargs="+", type=float,
                    default=[0.0, 0.05, 0.15, 0.3, 0.6])
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    mode_map = load_mode_map(args.data_root)
    K = len(mode_map)
    counts = np.array([len(v) for _, v in sorted(mode_map.items())])
    print(f"modes={K}  demos={counts.sum()}  per-mode min/median/max="
          f"{counts.min()}/{int(np.median(counts))}/{counts.max()}")
    emp = counts / counts.sum()
    print(f"native KL(demo || uniform) = {np.sum(emp * np.log(emp * K + 1e-12)):.4f} nats")

    out_dir = args.out_dir or args.data_root
    out_dir.mkdir(parents=True, exist_ok=True)

    for kl in args.target_kl:
        p = geometric_profile(K, kl)
        # Emit per-file SAMPLING WEIGHTS rather than a subset. Avoiding has only
        # 96 demos over 24 modes -- 4 per mode -- so selecting a subset cannot
        # realise a skewed profile: the integer granularity is far too coarse
        # (a 90x ratio needs 90 demos in one mode and 1 in another, when the
        # maximum available is 5). Subsetting was measured to undershoot badly
        # (target 0.30 -> 0.131, target 0.60 -> 0.089, and shrinking the set to
        # 30 demos in the process). Weighting hits the target exactly and keeps
        # every demonstration, which is also how one would normally train on an
        # imbalanced distribution.
        weights: dict[str, float] = {}
        for i, (_, flist) in enumerate(sorted(mode_map.items())):
            if not flist:
                continue
            w = p[i] / len(flist)          # split the mode's mass over its demos
            for fn in flist:
                weights[fn] = w
        tot = sum(weights.values())
        weights = {k: v / tot for k, v in weights.items()}
        # verify: the induced mode distribution
        got = np.array([sum(weights[f] for f in mode_map[m]) if mode_map.get(m) else 0.0
                        for m in range(K)])
        nz = got > 0
        got_kl = float(np.sum(got[nz] * np.log(got[nz] * K)))
        out = out_dir / f"demo_weights_skew{kl:.2f}.pkl"
        with open(out, "wb") as fh:
            pickle.dump({"weights": weights, "mode_profile": got.tolist(),
                         "target_kl": kl, "achieved_kl": got_kl}, fh)
        print(f"  target_kl={kl:.2f}  achieved={got_kl:.4f}  files={len(weights)}  "
              f"mode p max/min={got[nz].max() / got[nz].min():.1f}x  -> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
