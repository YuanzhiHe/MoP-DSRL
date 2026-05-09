"""Reference cluster distribution C_ref(o) (method_spec §4.2).

C_ref is computed once before any RL run by sampling z ~ N(0, I) and rolling the
*frozen* denoiser pi_theta to produce action chunks, then assigning each chunk
through the K-means vocabulary kappa.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .vocab import KMeansVocab


def compute_reference_histogram(
    sample_actions_at: Callable[[int, int], np.ndarray],
    n_obs: int,
    n_ref: int,
    vocab: KMeansVocab,
) -> np.ndarray:
    """Compute C_ref of shape [n_obs, K] by sampling z ~ N(0, I) per o.

    Args:
        sample_actions_at: callable (obs_index, n_samples) -> array
            [n_samples, H * d_a]. Caller is responsible for fixing the
            observation set and for sampling z ~ N(0, I) when constructing
            the action chunks (this function never sees z).
        n_obs: number of held-out observations.
        n_ref: number of action samples per observation.
        vocab: the §4.1 frozen K-means vocabulary.

    Returns:
        C_ref: float array of shape [n_obs, K], rows summing to 1.
    """
    K = vocab.K
    C_ref = np.zeros((n_obs, K), dtype=np.float64)
    for i in range(n_obs):
        actions = sample_actions_at(i, n_ref)  # [n_ref, H * d_a]
        ids = vocab.assign(actions)
        for k in range(K):
            C_ref[i, k] = float((ids == k).sum())
    C_ref /= float(n_ref)
    return C_ref
