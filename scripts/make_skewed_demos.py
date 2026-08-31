#!/usr/bin/env python3
"""Build demonstration subsets with a controlled behavioural-mode skew.

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

So the skew becomes the independent variable. This script filters the
demonstration manifest to realise a target KL(demo || uniform), which lets the
head-to-head be run across a skew sweep rather than at a single arbitrary point.
The curve is the argument: the arms should coincide at KL=0 and separate as the
backbone's profile departs from uniform. D3IL itself already subsets manifests
this way for its data-size ablation (train_files_{0.1,0.25,0.5,0.75}_.pkl).

    python scripts/make_skewed_demos.py --data-root <d3il>/environments/dataset/data/avoiding \\
        --target-kl 0 0.05 0.15 0.3 0.6 --out-dir <dir>
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
        # Largest subset whose per-mode counts match p and stay within supply.
        scale = min(counts[i] / p[i] for i in range(K) if p[i] > 0)
        take = np.maximum(1, np.floor(p * scale).astype(int))
        take = np.minimum(take, counts)
        files: list[str] = []
        for i, (_, flist) in enumerate(sorted(mode_map.items())):
            pick = rng.choice(len(flist), size=take[i], replace=False)
            files.extend(flist[j] for j in pick)
        got = take / take.sum()
        got_kl = float(np.sum(got * np.log(got * K + 1e-12)))
        out = out_dir / f"train_files_skew{kl:.2f}.pkl"
        with open(out, "wb") as fh:
            pickle.dump(files, fh)
        print(f"  target_kl={kl:.2f}  achieved={got_kl:.4f}  demos={len(files)}  "
              f"max/min mode count={take.max()}/{take.min()}  -> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
