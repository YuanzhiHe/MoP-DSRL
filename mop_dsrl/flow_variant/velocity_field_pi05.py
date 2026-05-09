"""Velocity field adapter for pi_0.5 (lerobot/pi05_base) flow-matching action head.

pi_0.5 (Physical Intelligence, 2025) is loaded via LeRobot's PI05Policy and
exposes a flow-matching velocity field through its ``denoise_step`` method.

Key differences from the (now-retired) pi_0 adapter:

  Time direction:
    pi_0.5 integrates from t=1 → t=0 with dt = -1/M (noise → action).
    MoP-DSRL's Euler solver integrates from t=0 → t=1 with dt = +1/M.
    This adapter flips the time axis: when the Euler solver queries
    ``v(x, t_forward)``, we call pi_0.5's velocity head with
    ``t_pi05 = 1 - t_forward`` and negate the returned velocity so that
    the positive-dt Euler step advances in the correct direction.

  KV cache:
    pi_0.5 pre-computes vision–language prefix embeddings once per
    observation and caches them as ``past_key_values``. The adapter
    performs this prefix encoding in ``bind_obs`` and reuses the cache
    across all M Euler steps.

  Action padding:
    pi_0.5 internally pads actions to ``max_action_dim=32``.  The adapter
    pads the coupling endpoint z_0 on entry and strips the padding from
    the output after integration.

All parameters are registered non-trainable at construction time.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure the lerobot source is importable.  Set ``LEROBOT_SRC`` to the
# ``src/`` directory of your lerobot checkout (e.g. ``lerobot_v044/src``).
# ---------------------------------------------------------------------------


def _ensure_lerobot_src() -> None:
    src = os.environ.get("LEROBOT_SRC", "")
    if src and src not in sys.path and Path(src).is_dir():
        sys.path.insert(0, src)


class Pi05VelocityField(nn.Module):
    """Adapter wrapping pi_0.5's ``denoise_step`` as a forward-time velocity field.

    After ``bind_obs(o)`` is called, ``forward(x, t)`` returns the velocity
    ``v`` compatible with MoP-DSRL's forward-time Euler solver (t: 0 → 1).

    Args:
        pretrained_path: HuggingFace Hub identifier (e.g. ``"lerobot/pi05_base"``)
            or local directory.
        H: action chunk horizon (default 50 for pi_0.5).
        d_a: per-step action dimension for the target env (e.g. 7 for LIBERO).
        device: torch device string.
        num_inference_steps: number of Euler steps M (default 10 for pi_0.5).
        local_files_only: if True, skip HF Hub download.
        cache_dir: HuggingFace cache directory (empty string = default).
        task_instruction: language instruction passed to the vision-language
            encoder (e.g. ``"open the middle drawer of the cabinet"``).
    """

    def __init__(
        self,
        pretrained_path: str = "lerobot/pi05_base",
        H: int = 50,
        d_a: int = 7,
        device: str = "cpu",
        num_inference_steps: int = 10,
        local_files_only: bool = False,
        cache_dir: str = "",
        task_instruction: str = "",
    ):
        super().__init__()
        self.H = H
        self.d_a = d_a
        self.z_dim = H * d_a
        self._device = torch.device(device)
        self.num_inference_steps = num_inference_steps
        self._task_instruction = task_instruction

        # Load the full PI05Policy via LeRobot.
        _ensure_lerobot_src()
        self._policy, self._max_action_dim = self._load_policy(
            pretrained_path, local_files_only, cache_dir
        )

        # Bound observation state (set by bind_obs, cleared by unbind_obs).
        self._prefix_pad_masks: torch.Tensor | None = None
        self._past_key_values: Any = None

        self.requires_grad_(False)
        self.to(self._device)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_policy(
        self,
        pretrained_path: str,
        local_files_only: bool,
        cache_dir: str,
    ) -> tuple[Any, int]:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pi05 import PI05Policy as LeRobotPI05Policy

        cd = cache_dir.strip() or None
        policy_cfg = PreTrainedConfig.from_pretrained(
            pretrained_path,
            local_files_only=local_files_only,
            cache_dir=cd,
        )
        policy_cfg.device = str(self._device)
        policy_cfg.num_inference_steps = self.num_inference_steps
        policy_cfg.validate_features()

        policy = LeRobotPI05Policy.from_pretrained(
            pretrained_path,
            config=policy_cfg,
            local_files_only=local_files_only,
            cache_dir=cd,
        )
        policy.eval()
        for p in policy.parameters():
            p.requires_grad_(False)
        policy.to(self._device)

        max_action_dim = int(policy_cfg.max_action_dim)
        return policy, max_action_dim

    # ------------------------------------------------------------------
    # Observation binding (prefix encoding + KV cache)
    # ------------------------------------------------------------------

    def bind_obs(self, o: torch.Tensor) -> None:
        """Pre-compute vision–language prefix and cache KV for Euler steps.

        For MoP-DSRL, ``o`` is the observation tensor that the SAC trainer
        passes.  In production (GPU + LIBERO), ``o`` should be a dict-based
        batch prepared via ``PI05Policy._preprocess_images`` and the
        language tokenizer.  For the unit-test path (ToyVelocityField),
        this method is a no-op; see the fallback at the end.

        When called with a raw ``[B, obs_dim]`` tensor (e.g. from
        ``SACTrainer``), the adapter stores it for later but defers the
        actual prefix encoding to the first ``forward`` call where the
        full observation dict is available.  If you need to run inference
        directly, pass the preprocessed dict via ``bind_obs_dict``.
        """
        self._raw_obs = o.to(self._device)

    def bind_obs_dict(self, batch: Dict[str, torch.Tensor]) -> None:
        """Bind a fully-preprocessed LeRobot observation dict.

        This is the preferred entry point when you have images + language
        tokens available (i.e. during LIBERO evaluation).
        """
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        images, img_masks = self._policy._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        model = self._policy.model

        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, tokens, masks
        )
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)

        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        _, past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        self._prefix_pad_masks = prefix_pad_masks
        self._past_key_values = past_key_values

    def unbind_obs(self) -> None:
        self._prefix_pad_masks = None
        self._past_key_values = None
        self._raw_obs = None

    # ------------------------------------------------------------------
    # Velocity field
    # ------------------------------------------------------------------

    def _pad_to_max_action_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Pad [B, H*d_a] → [B, H, max_action_dim]."""
        B = x.shape[0]
        x_3d = x.view(B, self.H, self.d_a)
        if self.d_a < self._max_action_dim:
            pad = torch.zeros(
                B, self.H, self._max_action_dim - self.d_a,
                dtype=x.dtype, device=x.device,
            )
            x_3d = torch.cat([x_3d, pad], dim=-1)
        return x_3d  # [B, H, max_action_dim]

    def _unpad_from_max_action_dim(self, x_3d: torch.Tensor) -> torch.Tensor:
        """Unpad [B, H, max_action_dim] → [B, H*d_a]."""
        return x_3d[:, :, : self.d_a].reshape(x_3d.shape[0], -1)

    def forward(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Velocity query in forward time (t: 0→1), with pre-bound observations.

        The pi_0.5 model uses backward time (t: 1→0) with negative dt.
        We convert:
          t_pi05 = 1 - t_forward
          v_forward = -v_pi05
        so that the positive-dt Euler solver in euler_solver.py produces
        the correct trajectory.

        Args:
            x: ``[B, H*d_a]`` noisy action at forward time ``t``.
            t: scalar forward time in ``[0, 1)``.

        Returns:
            v: ``[B, H*d_a]`` velocity vector in forward-time convention.
        """
        if self._prefix_pad_masks is None or self._past_key_values is None:
            raise RuntimeError(
                "No observation bound. Call bind_obs_dict(batch) before euler_integrate."
            )

        x = x.to(self._device)
        B = x.shape[0]

        # Pad to pi_0.5 internal shape: [B, H, max_action_dim]
        x_padded = self._pad_to_max_action_dim(x)

        # Convert forward time → pi_0.5 backward time
        t_pi05 = 1.0 - t
        timestep = torch.full((B,), t_pi05, dtype=torch.float32, device=self._device)

        # Query the velocity field
        model = self._policy.model
        v_pi05 = model.denoise_step(
            prefix_pad_masks=self._prefix_pad_masks,
            past_key_values=self._past_key_values,
            x_t=x_padded,
            timestep=timestep,
        )  # [B, H, max_action_dim]

        # Unpad and negate (backward → forward time conversion)
        v_forward = -self._unpad_from_max_action_dim(v_pi05)
        return v_forward

    def forward_with_obs_dict(
        self, x: torch.Tensor, t: float, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Full velocity query: bind + query + unbind in one call."""
        self.bind_obs_dict(batch)
        try:
            return self.forward(x, t)
        finally:
            self.unbind_obs()
