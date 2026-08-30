"""K-means action-cluster vocabulary builder (method_spec §4.1).

Public functions:
- robust_z_normalize(actions, median, mad) -> normalized actions
- fit_robust_z_stats(actions) -> (median, mad)
- build_kmeans_vocab(actions, K, n_init, max_iter, seed) -> KMeansVocab
- KMeansVocab.assign(actions) -> integer cluster ids in {0, ..., K-1}

Determinism contract (Task 16, mandatory test (c)):
For a fixed (actions, K, n_init, max_iter, seed) tuple, two calls to
build_kmeans_vocab return identical centroids and identical cluster assignments
on the original training set. The test in tests/test_unit_contracts.py verifies
this with seed=42 (the method_spec default) on a synthetic 3-cluster mixture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


_MAD_SCALE = 1.4826  # consistent estimator of sigma for Gaussian data
_EPS = 1e-8
# Degenerate dimensions (near-constant across demos, e.g. a near-binary gripper
# channel) have MAD ~ 0. Flooring them at an absolute _EPS made 1/mad ~ 1e8, so
# those coordinates dominated the normalized space by ~16 orders of magnitude:
# centroid separations reached ~1e17 and softmax(-d^2/T) saturated to an exact
# one-hot, i.e. the mode regularizer had an identically-zero gradient. Floor
# relative to the typical informative dimension instead, so a constant channel
# contributes O(1) rather than O(1e8).
_MAD_REL_FLOOR = 1e-2  # fraction of the median informative MAD
# A dimension with no demonstration variance (e.g. a gripper channel that is
# constant across every demo) carries no information about which *behavioural*
# mode a chunk belongs to. Rather than flooring its MAD -- which amplifies it and
# lets it dominate both K-means and the assignment distances -- give it an
# enormous scale so the normalized coordinate is ~0 and the dimension drops out
# of every distance computation. This keeps the torch consumers in sac.py /
# train.py unchanged: they only ever divide by `mad`.
_DEGENERATE_MAD = 1e12


def fit_robust_z_stats(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension median and scaled-MAD over the demonstration set.

    Dimensions with (near-)zero demonstration variance are assigned
    ``_DEGENERATE_MAD`` so they are excluded from the normalized space; the rest
    are floored RELATIVE to the median informative MAD so no single low-variance
    but still informative channel can dominate.

    Args:
        actions: float array of shape [N, H * d_a] (already flattened).

    Returns:
        (median, mad) each of shape [H * d_a].
    """
    if actions.ndim != 2:
        raise ValueError(f"actions must be 2D, got shape {actions.shape}")
    median = np.median(actions, axis=0)
    mad = np.median(np.abs(actions - median), axis=0) * _MAD_SCALE
    degenerate = mad <= _EPS
    informative = mad[~degenerate]
    scale = float(np.median(informative)) if informative.size else 1.0
    mad = np.maximum(mad, _MAD_REL_FLOOR * scale)
    mad[degenerate] = _DEGENERATE_MAD
    return median.astype(np.float64), mad.astype(np.float64)


def degenerate_dims(mad: np.ndarray) -> np.ndarray:
    """Boolean mask of dimensions excluded from the vocabulary space."""
    return np.asarray(mad) >= _DEGENERATE_MAD * 0.999


def robust_z_normalize(actions: np.ndarray, median: np.ndarray, mad: np.ndarray) -> np.ndarray:
    """Apply per-dimension robust-z normalization with frozen stats."""
    return (actions - median) / mad


@dataclass
class KMeansVocab:
    """Frozen K-means vocabulary; serialize via numpy.save / pickle."""

    K: int
    median: np.ndarray  # [d]
    mad: np.ndarray  # [d]
    centroids: np.ndarray  # [K, d]
    seed: int

    def assign(self, actions: np.ndarray) -> np.ndarray:
        """Return integer cluster ids in {0, ..., K-1} for each row.

        Args:
            actions: float array of shape [..., H * d_a]; the leading shape is
                preserved in the output.
        """
        flat = actions.reshape(-1, actions.shape[-1])
        normalized = robust_z_normalize(flat, self.median, self.mad)
        # squared Euclidean distance to every centroid
        # ||x - mu||^2 = ||x||^2 - 2 x.mu + ||mu||^2 (last term constant per centroid)
        dists = ((normalized[:, None, :] - self.centroids[None, :, :]) ** 2).sum(-1)
        ids = np.argmin(dists, axis=1).astype(np.int64)
        return ids.reshape(actions.shape[:-1])


def build_kmeans_vocab(
    actions: np.ndarray,
    K: int = 16,
    n_init: int = 10,
    max_iter: int = 300,
    seed: int = 42,
) -> KMeansVocab:
    """Fit the §4.1 vocabulary on the demonstration set.

    Args:
        actions: float array of shape [N, H * d_a]; each row is a flattened
            action chunk.
        K, n_init, max_iter, seed: K-means hyperparameters per method_spec §4.1.

    Returns:
        A frozen KMeansVocab object.
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if actions.shape[0] < K:
        raise ValueError(
            f"need at least K={K} demonstrations, got {actions.shape[0]}"
        )
    median, mad = fit_robust_z_stats(actions)
    normalized = robust_z_normalize(actions, median, mad)
    km = KMeans(
        n_clusters=K,
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
        algorithm="lloyd",
    )
    km.fit(normalized)
    return KMeansVocab(
        K=K,
        median=median,
        mad=mad,
        centroids=km.cluster_centers_.astype(np.float64),
        seed=seed,
    )
