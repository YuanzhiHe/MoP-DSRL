"""Configuration dataclass for MoP-DSRL.

Defaults follow method_spec.md §8 (LIBERO + pi_0 column). Toggle flags follow §11.
Only fields used by the SAC core, the regularizer, and the unit tests are populated;
backbone-specific fields (LIBERO version, image cache, etc.) live in their own
configs and are not the responsibility of this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # --- backbone / chunk geometry (method_spec §8) ---
    H: int = 50  # action chunk horizon (50 for pi_0, 16 for DP-T)
    d_a: int = 14  # per-step action dimension (14 for ALOHA bimanual, 7 for LIBERO single-arm)

    # --- vocabulary (method_spec §4.1) ---
    K: int = 16
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300
    kmeans_seed: int = 42

    # --- reference distribution (method_spec §4.2) ---
    n_ref: int = 50
    n_obs: int = 1000

    # --- on-policy histogram (method_spec §4.3) ---
    n_on: int = 8

    # --- regularizer weight (method_spec §4 / §8) ---
    lambda_: float = 0.1
    epsilon: float = 0.1  # nats; only used when tune_lambda is True
    alpha_lambda: float = 1e-3

    # --- SAC core (method_spec §5 / §8) ---
    replay_capacity: int = int(1e6)
    batch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    tau_polyak: float = 5e-3
    gamma: float = 0.99
    warmup_env_steps: int = int(1e4)
    total_env_steps: int = int(5e5)
    eval_interval: int = int(2.5e4)

    # --- action-space bounding (method_spec §5) ---
    # r_z = 4 * sqrt(H * d_a); computed in property below.
    clip_radius_sigma_factor: float = 4.0

    # --- log-std bounds for the outer policy (method_spec §3) ---
    log_std_min: float = -6.907755  # log(1e-3)
    log_std_max: float = 0.0  # log(1.0)

    # --- toggle flags (method_spec §11) ---
    backbone_family: str = "diffusion"  # {"diffusion", "flow"}
    value_guidance: bool = False
    tune_lambda: bool = False
    pref_reward: bool = False
    hierarchical: bool = False
    weight_level: bool = False  # counter-hypothesis (X1)
    imitation_only: bool = False  # counter-hypothesis (X2)

    # --- value-guidance arm (method_spec §6, only used when value_guidance=True) ---
    beta_max: float = 0.5
    critic_tikhonov_weight: float = 1e-4

    # --- engineering ---
    seed: int = 0
    device: str = "cpu"  # "cpu" for unit tests; "cuda" for benchmark runs.

    # --- target entropy override (defaults to -dim(z) per method_spec §5) ---
    target_entropy: float = field(default=float("nan"))

    @property
    def z_dim(self) -> int:
        return self.H * self.d_a

    @property
    def clip_radius(self) -> float:
        # method_spec §5 "Action-space bounding": r_z = 4 * sqrt(H * d_a).
        return float(self.clip_radius_sigma_factor) * (self.z_dim ** 0.5)

    @property
    def effective_target_entropy(self) -> float:
        # method_spec §5 "Temperature": H_target = -dim(z) by default.
        if self.target_entropy != self.target_entropy:  # NaN sentinel = unset
            return -float(self.z_dim)
        return float(self.target_entropy)

    def laplace_alpha(self) -> float:
        # method_spec §4.3: alpha = 1 / (K * n_on).
        return 1.0 / (self.K * self.n_on)

    def shape(self) -> Tuple[int, int]:
        return (self.H, self.d_a)
