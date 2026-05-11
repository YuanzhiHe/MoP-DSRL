#!/usr/bin/env python3
"""Build a K-means action-cluster vocabulary from a demonstration set.

Loads a demonstration set (LIBERO HDF5 layout, RoboCasa, or a flat .npy of
flattened action chunks), fits the §4.1 K-means vocabulary, and writes the
result to a pickle file for reuse during training and eval.

Usage:
    python scripts/build_vocab.py \\
        --demos /path/to/libero_goal \\
        --H 50 --d_a 7 --K 16 --seed 42 \\
        --out vocabs/libero_goal_K16.pkl
"""

from __future__ import annotations

# Force IPv4 for HF / requests; see scripts/_ipv4_patch.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _ipv4_patch  # noqa: F401,E402

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np

from mop_dsrl.components.vocab import build_kmeans_vocab

logger = logging.getLogger(__name__)


def _load_chunks_from_npy(path: Path, H: int) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 2:  # already flattened [N, H*d_a]
        return arr.astype(np.float64)
    if arr.ndim == 3 and arr.shape[1] == H:  # [N, H, d_a]
        return arr.reshape(arr.shape[0], -1).astype(np.float64)
    raise ValueError(f"Unexpected .npy shape {arr.shape} for H={H}")


def _load_chunks_from_hdf5_dir(dir_path: Path, H: int, d_a: int) -> np.ndarray:
    """Load every *_demo.hdf5 in a LIBERO-style suite directory and tile actions.

    LIBERO HDF5 layout: data/demo_<i>/actions [T, d_a] per demonstration.
    We slide a window of length H over each demo to produce action chunks.
    """
    try:
        import h5py
    except ImportError as e:
        raise RuntimeError(
            "h5py is required to load LIBERO demos. Run `pip install h5py`."
        ) from e

    chunks = []
    files = sorted(dir_path.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No .hdf5 files in {dir_path}")
    logger.info("Found %d demo files in %s", len(files), dir_path)

    for f in files:
        with h5py.File(f, "r") as hf:
            if "data" not in hf:
                logger.warning("Skip %s: no 'data' group", f)
                continue
            for demo_name in hf["data"].keys():
                acts = np.asarray(hf["data"][demo_name]["actions"], dtype=np.float64)
                if acts.shape[1] != d_a:
                    raise ValueError(
                        f"Action dim mismatch in {f}/{demo_name}: got {acts.shape[1]}, expected d_a={d_a}"
                    )
                T = acts.shape[0]
                if T < H:
                    pad = np.tile(acts[-1:], (H - T, 1))
                    chunks.append(np.concatenate([acts, pad], axis=0).reshape(-1))
                    continue
                # Stride 1 sliding window for maximum coverage.
                for s in range(T - H + 1):
                    chunks.append(acts[s : s + H].reshape(-1))
    return np.stack(chunks, axis=0)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    p = argparse.ArgumentParser()
    p.add_argument("--demos", required=True, help="Demo directory or .npy file.")
    p.add_argument("--H", type=int, required=True)
    p.add_argument("--d_a", type=int, required=True)
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--n_init", type=int, default=10)
    p.add_argument("--max_iter", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_chunks", type=int, default=200_000,
                   help="Cap on chunks fed to K-means (subsampled uniformly).")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    src = Path(args.demos)
    if src.is_file() and src.suffix == ".npy":
        chunks = _load_chunks_from_npy(src, args.H)
    elif src.is_dir():
        chunks = _load_chunks_from_hdf5_dir(src, args.H, args.d_a)
    else:
        raise FileNotFoundError(f"--demos must be a .npy file or HDF5 dir; got {src}")

    logger.info("Loaded %d action chunks of shape [%d * %d = %d]",
                chunks.shape[0], args.H, args.d_a, args.H * args.d_a)

    if chunks.shape[0] > args.max_chunks:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(chunks.shape[0], size=args.max_chunks, replace=False)
        chunks = chunks[idx]
        logger.info("Subsampled to %d chunks (seed=%d)", chunks.shape[0], args.seed)

    vocab = build_kmeans_vocab(
        chunks,
        K=args.K,
        n_init=args.n_init,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(vocab, f)
    logger.info("Wrote vocab K=%d to %s", args.K, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
