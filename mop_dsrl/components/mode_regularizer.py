"""Mode-coverage regularizer R_mode (method_spec §4.3).

R_mode(phi) = (1/B) sum_b KL( C_phi(o_b) || C_ref(o_b) )

C_phi(o_b) is the on-policy histogram over the §4.1 vocabulary, computed from
n_on action samples per observation; C_ref(o_b) is the precomputed reference
distribution. Both histograms are Laplace-smoothed with alpha = 1 / (K * n_on)
so the KL is finite (method_spec §4.3).

The function is differentiable wrt the *cluster-assignment one-hots* via a
straight-through path: cluster ids are nondifferentiable, but the gradient
flows through `mu_phi` via the reparameterization trick already used to draw
z, and the regularizer attaches to the Monte-Carlo estimate of the
categorical histogram. We therefore expose two entry points:

- `kl_histogram_to_reference(C_phi, C_ref, eps)` — pure-tensor KL, used when
  the caller has already produced the histograms (e.g., as a soft estimate).
- `mode_kl_from_cluster_ids(ids, C_ref_batch, K, n_on, B)` — the standard
  hard-assignment estimator from §4.3; non-differentiable wrt phi (used as a
  measurement signal, e.g., for UT-2 and as the dual-ascent target on lambda).

For the actor-loss contribution under reparameterization (method_spec §5
"Actor objective with mode-coverage regularizer"), we use a soft-assignment
variant where cluster ids are replaced by softmax(-distances / temperature).
The temperature is fixed at 1 by default; the soft assignment recovers the
hard assignment as temperature -> 0 and is differentiable everywhere.
"""

from __future__ import annotations

import torch


def laplace_smooth(hist: torch.Tensor, alpha: float) -> torch.Tensor:
    """Add Laplace smoothing alpha to each bin and renormalize.

    Args:
        hist: [..., K] tensor with rows summing to 1 (or to n_on / n_ref pre-norm).
        alpha: smoothing constant per method_spec §4.3.

    Returns:
        smoothed: [..., K] tensor with rows summing to 1.
    """
    K = hist.shape[-1]
    smoothed = hist + alpha
    return smoothed / smoothed.sum(dim=-1, keepdim=True)


def kl_histogram_to_reference(
    C_phi: torch.Tensor,
    C_ref: torch.Tensor,
    laplace_alpha: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """KL( C_phi || C_ref ) per row, averaged over the batch.

    Both inputs must be [B, K] with rows summing to 1 *before* smoothing.
    """
    if C_phi.shape != C_ref.shape:
        raise ValueError(
            f"C_phi {tuple(C_phi.shape)} vs C_ref {tuple(C_ref.shape)}"
        )
    p = laplace_smooth(C_phi, laplace_alpha)
    q = laplace_smooth(C_ref, laplace_alpha)
    return torch.sum(p * (torch.log(p + eps) - torch.log(q + eps)), dim=-1).mean()


def soft_assign_distances(
    flat_actions: torch.Tensor,
    centroids: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Soft cluster-assignment via softmax(-distances / T).

    Used to produce a differentiable surrogate of the cluster histogram for
    the actor-loss path. The hard-assignment K-means rule is recovered as
    temperature -> 0+.

    Args:
        flat_actions: [N, d] (already robust-z normalized).
        centroids: [K, d].
        temperature: positive scalar.

    Returns:
        weights: [N, K] with rows summing to 1.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    sq = ((flat_actions[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    return torch.softmax(-sq / temperature, dim=-1)


def mode_kl_from_assignments(
    soft_assignments: torch.Tensor,
    C_ref_batch: torch.Tensor,
    n_on: int,
    laplace_alpha: float,
) -> torch.Tensor:
    """Aggregate soft assignments into a per-observation histogram and return KL.

    Args:
        soft_assignments: [B * n_on, K] soft cluster assignments for the
            B observations x n_on samples per observation, ordered so that
            consecutive groups of n_on rows belong to the same observation.
        C_ref_batch: [B, K] precomputed reference histogram for the batch.
        n_on: number of samples per observation.
        laplace_alpha: §4.3 smoothing constant.

    Returns:
        scalar tensor R_mode(phi) for this batch.
    """
    if soft_assignments.shape[0] % n_on != 0:
        raise ValueError(
            f"soft_assignments rows ({soft_assignments.shape[0]}) must be a "
            f"multiple of n_on ({n_on})"
        )
    B = soft_assignments.shape[0] // n_on
    K = soft_assignments.shape[1]
    if C_ref_batch.shape != (B, K):
        raise ValueError(
            f"C_ref_batch {tuple(C_ref_batch.shape)} vs expected ({B}, {K})"
        )
    # average n_on soft assignments per observation
    C_phi = soft_assignments.view(B, n_on, K).mean(dim=1)
    return kl_histogram_to_reference(C_phi, C_ref_batch, laplace_alpha)
