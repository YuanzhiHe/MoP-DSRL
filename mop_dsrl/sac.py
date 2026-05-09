"""SAC outer-loop trainer for MoP-DSRL (method_spec §5).

This trainer is intentionally minimal and agnostic to:
- the backbone family (diffusion vs flow), which lives behind the
  FrozenDenoiser interface.
- the optional value-guidance arm, which is a property of the denoiser
  (a `GuidedDenoiser` subclass that wraps the same interface and consumes
  this trainer's Q ensemble), not of the SAC update rule.

The §4.3 mode-coverage regularizer is added to the actor loss only when the
config's `lambda_` is strictly positive *and* a vocabulary plus C_ref are
provided. With `lambda_ = 0` the actor loss is, by construction, identical
(up to floating-point) to a vanilla DSRL-style SAC actor loss; this is the
contract tested by UT-3 (mandatory test (a) of Task 16).

The trainer does not own the environment loop. The caller drives interaction
and calls `update(batch, vocab=None, C_ref_batch=None)` per gradient step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .components.action_clip import l2_clip_with_logprob_correction
from .components.critic import TwinQEnsemble, polyak_update
from .components.mode_regularizer import (
    mode_kl_from_assignments,
    soft_assign_distances,
)
from .components.policy import OuterGaussianPolicy
from .components.vocab import KMeansVocab
from .config import Config
from .denoiser import FrozenDenoiser


@dataclass
class UpdateInfo:
    critic_loss: float
    actor_loss_q: float  # the J_Q + J_H part (i.e., DSRL part)
    actor_loss_total: float
    r_mode: float  # 0.0 if regularizer not applied
    alpha: float
    lambda_: float


class SACTrainer:
    def __init__(
        self,
        config: Config,
        denoiser: FrozenDenoiser,
        obs_dim: int,
        device: str | None = None,
    ):
        self.cfg = config
        self.denoiser = denoiser
        self.device = torch.device(device or config.device)
        self.obs_dim = obs_dim
        z_dim = config.z_dim

        # outer policy mu_phi
        self.policy = OuterGaussianPolicy(
            z_dim=z_dim,
            obs_dim=obs_dim,
            log_std_min=config.log_std_min,
            log_std_max=config.log_std_max,
        ).to(self.device)

        # twin Q ensemble + targets
        self.critic = TwinQEnsemble(obs_dim, z_dim).to(self.device)
        self.critic_target = TwinQEnsemble(obs_dim, z_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # auto-tuned alpha (entropy coefficient)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.target_entropy = config.effective_target_entropy

        # optimizers
        self.actor_opt = torch.optim.Adam(self.policy.parameters(), lr=config.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)

        # mutable lambda for dual ascent
        self._lambda = float(config.lambda_)

        # ensure denoiser is frozen
        for p in self.denoiser.parameters():
            p.requires_grad_(False)

    # --- helpers ---

    def _sample_z(self, o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z, log_prob = self.policy.sample(o)
        z, log_prob = l2_clip_with_logprob_correction(z, log_prob, self.cfg.clip_radius)
        return z, log_prob

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def lambda_(self) -> float:
        return self._lambda

    # --- core updates ---

    def critic_loss(self, batch: dict) -> torch.Tensor:
        o = torch.as_tensor(batch["o"], dtype=torch.float32, device=self.device)
        z = torch.as_tensor(batch["z"], dtype=torch.float32, device=self.device)
        r = torch.as_tensor(batch["r"], dtype=torch.float32, device=self.device)
        o_next = torch.as_tensor(batch["o_next"], dtype=torch.float32, device=self.device)
        d = torch.as_tensor(batch["d"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            z_next, log_pi_next = self._sample_z(o_next)
            q1_t, q2_t = self.critic_target(o_next, z_next)
            q_min = torch.min(q1_t, q2_t)
            target = r + self.cfg.gamma * (1.0 - d) * (q_min - self.alpha.detach() * log_pi_next)

        q1, q2 = self.critic(o, z)
        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        return loss

    def actor_loss(
        self,
        batch: dict,
        vocab: KMeansVocab | None = None,
        C_ref_batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (actor_loss, j_q, j_h, r_mode_scalar).

        j_q + j_h is the DSRL-equivalent part. r_mode_scalar is the un-weighted
        mode-coverage value (zero when not applied) used for logging and for
        the dual-ascent update on lambda.
        """
        o = torch.as_tensor(batch["o"], dtype=torch.float32, device=self.device)

        z, log_pi = self._sample_z(o)
        q1, q2 = self.critic(o, z)
        q_min = torch.min(q1, q2)

        j_q = q_min.mean()
        j_h = -(self.alpha.detach() * log_pi).mean()
        loss = -(j_q + j_h)

        r_mode = torch.zeros((), device=self.device)
        if self.cfg.lambda_ > 0.0 and vocab is not None and C_ref_batch is not None:
            r_mode = self._mode_regularizer(o, vocab, C_ref_batch)
            loss = loss + self._lambda * r_mode

        return loss, j_q.detach(), j_h.detach(), r_mode.detach()

    def _mode_regularizer(
        self,
        o: torch.Tensor,
        vocab: KMeansVocab,
        C_ref_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the soft-assignment R_mode for the actor-loss path.

        Implements method_spec §4.3 with the soft-assignment surrogate so that
        gradients flow through `mu_phi` via the reparameterization trick. The
        denoiser forward pass is wrapped in `torch.no_grad()` because
        pi_theta is frozen and §4.3 only requires the categorical histogram,
        not per-step likelihoods.
        """
        n_on = self.cfg.n_on
        B = o.shape[0]

        # repeat each observation n_on times
        o_rep = o.repeat_interleave(n_on, dim=0)  # [B*n_on, obs_dim]
        # draw n_on independent z per observation (reparameterized => gradient flows)
        z_rep, _ = self._sample_z(o_rep)

        # frozen denoiser forward
        a_rep = self.denoiser.denoise(o_rep, z_rep)  # [B*n_on, z_dim]

        # robust-z normalization with frozen vocab stats
        median = torch.as_tensor(vocab.median, dtype=a_rep.dtype, device=self.device)
        mad = torch.as_tensor(vocab.mad, dtype=a_rep.dtype, device=self.device)
        a_norm = (a_rep - median) / mad

        centroids = torch.as_tensor(vocab.centroids, dtype=a_rep.dtype, device=self.device)
        soft = soft_assign_distances(a_norm, centroids, temperature=1.0)

        return mode_kl_from_assignments(soft, C_ref_batch, n_on, self.cfg.laplace_alpha())

    def alpha_loss(self, batch: dict) -> torch.Tensor:
        o = torch.as_tensor(batch["o"], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, log_pi = self._sample_z(o)
        return -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

    def update(
        self,
        batch: dict,
        vocab: KMeansVocab | None = None,
        C_ref_batch: torch.Tensor | None = None,
    ) -> UpdateInfo:
        # critic
        c_loss = self.critic_loss(batch)
        self.critic_opt.zero_grad(set_to_none=True)
        c_loss.backward()
        self.critic_opt.step()

        # actor (with optional R_mode)
        a_loss, j_q, j_h, r_mode = self.actor_loss(batch, vocab, C_ref_batch)
        self.actor_opt.zero_grad(set_to_none=True)
        a_loss.backward()
        self.actor_opt.step()

        # alpha
        al_loss = self.alpha_loss(batch)
        self.alpha_opt.zero_grad(set_to_none=True)
        al_loss.backward()
        self.alpha_opt.step()

        # dual ascent on lambda (method_spec §4.5)
        if self.cfg.tune_lambda:
            self._lambda = max(
                0.0,
                self._lambda + self.cfg.alpha_lambda * (float(r_mode.item()) - self.cfg.epsilon),
            )

        # Polyak target update (method_spec §5)
        polyak_update(self.critic_target, self.critic, self.cfg.tau_polyak)

        return UpdateInfo(
            critic_loss=float(c_loss.item()),
            actor_loss_q=float(-(j_q + j_h).item()),
            actor_loss_total=float(a_loss.item()),
            r_mode=float(r_mode.item()),
            alpha=float(self.alpha.item()),
            lambda_=float(self._lambda),
        )
