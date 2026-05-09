"""MoP-DSRL: Mode-Preserving Noise-Space Reinforcement Learning.

Reference: Ideation/ideas/method_spec.md (Sections 2-12).

Public surface:
- Config (config.py)
- FrozenDenoiser, ToyDenoiser (denoiser.py)
- OuterGaussianPolicy (components.policy)
- TwinQEnsemble (components.critic)
- ReplayBuffer (components.replay)
- build_kmeans_vocab, assign_clusters (components.vocab)
- compute_reference_histogram (components.reference_dist)
- batched_kl_to_reference (components.mode_regularizer)
- l2_clip_with_logprob_correction (components.action_clip)
- SACTrainer (sac)
"""

from .config import Config
from .denoiser import FrozenDenoiser, ToyDenoiser
from .sac import SACTrainer

__all__ = ["Config", "FrozenDenoiser", "ToyDenoiser", "SACTrainer"]
