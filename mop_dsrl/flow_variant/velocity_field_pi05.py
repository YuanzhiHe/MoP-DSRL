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
        self._pretrained_path = pretrained_path

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
    # LIBERO → pi_0.5 batch adapter
    # ------------------------------------------------------------------

    def _resolve_tokenizer(self):
        """Best-effort lookup of the policy's language tokenizer.

        Order of attempts:
          1. Direct attribute on the policy (covers most lerobot versions).
          2. Nested processor / tokenizer fields inside the model tree.
          3. ``AutoTokenizer.from_pretrained`` against lerobot's public pi05
             repos (already cached locally), which avoids the gated PaliGemma
             repo that requires HF auth.
        Cached so repeated calls during training don't re-load.
        """
        if getattr(self, "_cached_tokenizer", None) is not None:
            return self._cached_tokenizer

        from transformers import AutoTokenizer

        # 1. Direct attribute lookup
        for attr in ("language_tokenizer", "tokenizer", "_language_tokenizer",
                     "text_tokenizer", "_tokenizer"):
            tok = getattr(self._policy, attr, None)
            if tok is not None:
                self._cached_tokenizer = tok
                return tok

        # 2. Nested paths
        nested_paths = [
            ("processor", "tokenizer"),
            ("model", "tokenizer"),
            ("model", "language_tokenizer"),
            ("model", "paligemma_with_expert", "paligemma", "tokenizer"),
            ("model", "paligemma_with_expert", "paligemma", "language_tokenizer"),
        ]
        for path in nested_paths:
            obj = self._policy
            for part in path:
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                self._cached_tokenizer = obj
                return obj

        # 3. lerobot's preprocessor builder may construct a tokenizer for us.
        try:
            from lerobot.policies.pi05 import make_pi05_pre_post_processors
            cfg = getattr(self._policy, "config", None)
            pre, _ = make_pi05_pre_post_processors(config=cfg, dataset_stats=None)
            for attr in ("tokenizer", "language_tokenizer", "_tokenizer"):
                tok = getattr(pre, attr, None)
                if tok is not None:
                    self._cached_tokenizer = tok
                    return tok
            # Sometimes the tokenizer is on a sub-step of the preprocessor.
            for step in getattr(pre, "steps", []) or []:
                for attr in ("tokenizer", "_tokenizer"):
                    tok = getattr(step, attr, None)
                    if tok is not None:
                        self._cached_tokenizer = tok
                        return tok
        except Exception:
            pass

        # 4. Public mirrors of the PaliGemma / Gemma tokenizer (same vocab,
        # no gating).  Try a few well-known ones.
        for repo in (
            "google/paligemma-3b-mix-224",   # works if user did huggingface-cli login
            "philschmid/gemma-tokenizer-chatml",
            "unsloth/gemma-2b",
            "google/gemma-2b",
        ):
            try:
                tok = AutoTokenizer.from_pretrained(repo)
                self._cached_tokenizer = tok
                return tok
            except Exception:
                continue

        # 5. lerobot pi05 repos as last resort (usually have no tokenizer files,
        # but try anyway in case a newer release bundles them).
        for repo in ("lerobot/pi05_libero", "lerobot/pi05_libero_base",
                     "lerobot/pi05_base"):
            try:
                tok = AutoTokenizer.from_pretrained(repo)
                self._cached_tokenizer = tok
                return tok
            except Exception:
                continue

        raise RuntimeError(
            "Could not locate pi_0.5's tokenizer.  Run `huggingface-cli login` "
            "and accept the google/paligemma-3b-mix-224 gated repo, OR "
            "`pip install --upgrade transformers` to a version that bundles a "
            "public Gemma tokenizer."
        )

    def _resolve_max_token_length(self) -> int:
        cfg = getattr(self._policy, "config", None)
        for key in ("tokenizer_max_length", "max_decoder_prompt_length",
                    "max_language_tokens", "max_prompt_length"):
            v = getattr(cfg, key, None) if cfg is not None else None
            if isinstance(v, int) and v > 0:
                return v
        return 48  # pi_0.5 default

    def _get_preprocessor(self):
        """Lazily build & cache lerobot's official preprocessor (normalizer +
        tokenizer + device transfer) for this policy."""
        if getattr(self, "_lerobot_preproc", None) is not None:
            return self._lerobot_preproc
        from lerobot.policies.factory import make_pre_post_processors
        cfg = self._policy.config
        # `pretrained_path` may have been a HF id like "lerobot/pi05_libero_finetuned"
        # — we don't have it as an instance attribute anymore.  Pull it from the
        # config's _name_or_path if available, else fall back to a sensible default.
        pretrained_path = (
            getattr(cfg, "_name_or_path", None)
            or getattr(self, "_pretrained_path", None)
            or "lerobot/pi05_libero_finetuned"
        )
        try:
            pre, post = make_pre_post_processors(cfg, pretrained_path=pretrained_path)
        except Exception:
            pre, post = make_pre_post_processors(cfg, pretrained_path=None)
        self._lerobot_preproc = pre
        self._lerobot_postproc = post
        return pre

    def _get_postprocessor(self):
        if getattr(self, "_lerobot_postproc", None) is None:
            self._get_preprocessor()
        return self._lerobot_postproc

    def prepare_libero_batch(
        self, raw_obs: dict, instruction: str, image_size: int = 224
    ) -> Dict[str, torch.Tensor]:
        """Convert a raw LIBERO obs dict + language instruction into a fully-
        prepared batch suitable for ``bind_obs_dict``.

        Layout for pi05_libero_finetuned (verified via verify_finetuned.py):
          observation.images.image          : agentview, [1, 3, 224, 224] float ∈ [0,1]
          observation.images.image2         : robot0_eye_in_hand, same shape
          observation.images.empty_camera_0 : zeros placeholder, same shape
          observation.state                 : [1, 8] = eef_pos(3) + eef_quat(4) + gripper(1)
          task                              : list[str] of instructions

        We then run this batch through lerobot's official preprocessor which:
          1. RenameObservationsProcessorStep   (id passthrough)
          2. AddBatchDimensionProcessorStep   (id since we're already batched)
          3. NormalizerProcessorStep          (state/action MEAN_STD norm)
          4. Pi05PrepareStateTokenizerProcessorStep
          5. TokenizerProcessorStep           (tokenizes language)
          6. DeviceProcessorStep              (moves to GPU)
        and returns the fully-prepared batch.
        """
        import numpy as np
        import torch
        import torch.nn.functional as F

        device = self._device
        batch: Dict[str, torch.Tensor] = {}

        def _img(arr):
            t = torch.from_numpy(arr.copy()).float() / 255.0
            if t.ndim == 3 and t.shape[-1] == 3:
                t = t.permute(2, 0, 1)
            if t.shape[-1] != image_size or t.shape[-2] != image_size:
                t = F.interpolate(
                    t.unsqueeze(0), size=(image_size, image_size),
                    mode="bilinear", align_corners=False,
                ).squeeze(0)
            return t.unsqueeze(0).to(device)

        if "agentview_image" in raw_obs:
            batch["observation.images.image"] = _img(raw_obs["agentview_image"])
        if "robot0_eye_in_hand_image" in raw_obs:
            batch["observation.images.image2"] = _img(raw_obs["robot0_eye_in_hand_image"])
        # 3rd-camera placeholder (training distribution had 3 cams; LIBERO has 2).
        batch["observation.images.empty_camera_0"] = torch.zeros(
            1, 3, image_size, image_size, device=device, dtype=torch.float32
        )

        # State: 8-d = eef_pos(3) + eef_quat(4) + gripper_qpos[:1] (single coord)
        state_parts = []
        for k in ("robot0_eef_pos", "robot0_eef_quat"):
            if k in raw_obs:
                state_parts.append(torch.from_numpy(raw_obs[k].copy()).float().flatten())
        gp = raw_obs.get("robot0_gripper_qpos", None)
        if isinstance(gp, np.ndarray) and gp.size > 0:
            state_parts.append(torch.from_numpy(gp[:1].copy()).float())
        if state_parts:
            batch["observation.state"] = torch.cat(state_parts, dim=0).unsqueeze(0).to(device)
        else:
            batch["observation.state"] = torch.zeros(1, 8, device=device, dtype=torch.float32)

        batch["task"] = [instruction]

        # Now route through lerobot's official preprocessor: tokenize + normalize.
        try:
            pre = self._get_preprocessor()
            batch = pre(batch)
        except Exception as e:
            # Fall back to a tiny zero-tokens batch so the rest of the model
            # at least loads.  Production runs hit the success path above.
            import logging as _log
            _log.warning("lerobot preproc failed: %s; falling back to raw batch", e)
        return batch

    def _OLD_prepare_libero_batch_DEPRECATED(self, raw_obs, instruction, image_size=224):
        """Old hand-rolled batch builder (kept inert for reference).

        This was bypassing lerobot's NormalizerProcessorStep which meant state
        and action were never normalized → BC=0%.  Use prepare_libero_batch
        above instead, which delegates to lerobot's official preproc.
        """
        import numpy as np
        import torch
        import torch.nn.functional as F
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS,
        )
        device = self._device
        batch: Dict[str, torch.Tensor] = {}
        for src_key, dst_key in [
            ("agentview_image",          "observation.images.image"),
            ("robot0_eye_in_hand_image", "observation.images.wrist_image"),
        ]:
            if src_key in raw_obs:
                arr = raw_obs[src_key]
                if not isinstance(arr, np.ndarray):
                    continue
                img = torch.from_numpy(arr.copy()).float() / 255.0
                if img.ndim == 3 and img.shape[-1] == 3:
                    img = img.permute(2, 0, 1)
                if img.shape[-1] != image_size or img.shape[-2] != image_size:
                    img = F.interpolate(
                        img.unsqueeze(0), size=(image_size, image_size),
                        mode="bilinear", align_corners=False,
                    ).squeeze(0)
                batch[dst_key] = img.unsqueeze(0).to(device)
        state_parts = []
        for k in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
            if k in raw_obs:
                v = raw_obs[k]
                if isinstance(v, np.ndarray):
                    state_parts.append(torch.from_numpy(v.copy()).float().flatten())
        if state_parts:
            state = torch.cat(state_parts, dim=0).unsqueeze(0).to(device)
        else:
            state = torch.zeros(1, 8, device=device, dtype=torch.float32)
        batch["observation.state"] = state
        max_len = self._resolve_max_token_length()
        warned = getattr(self, "_tokenizer_fallback_warned", False)
        try:
            tok = self._resolve_tokenizer()
            tok_out = tok(
                instruction,
                return_tensors="pt",
                padding="max_length",
                max_length=max_len,
                truncation=True,
            )
            batch[OBS_LANGUAGE_TOKENS] = tok_out["input_ids"].long().to(device)
            batch[OBS_LANGUAGE_ATTENTION_MASK] = tok_out["attention_mask"].bool().to(device)
        except Exception as e:
            if not warned:
                import logging as _log
                _log.warning(
                    "Tokenizer fallback engaged (%s).  Using zero tokens — pi_0.5 "
                    "output will not reflect the language goal.",
                    e,
                )
                self._tokenizer_fallback_warned = True
            zeros = torch.zeros(1, max_len, dtype=torch.long, device=device)
            mask = torch.zeros(1, max_len, dtype=torch.bool, device=device)
            mask[0, 0] = True
            batch[OBS_LANGUAGE_TOKENS] = zeros
            batch[OBS_LANGUAGE_ATTENTION_MASK] = mask

        return batch

    @staticmethod
    def repeat_batch(batch: Dict[str, torch.Tensor], n: int) -> Dict[str, torch.Tensor]:
        """Repeat every tensor in ``batch`` n times along dim 0 (broadcasts B=1)."""
        out: Dict[str, torch.Tensor] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if v.shape[0] == 1:
                    out[k] = v.expand(n, *v.shape[1:]).contiguous()
                else:
                    out[k] = v.repeat(n, *([1] * (v.ndim - 1)))
            else:
                out[k] = v
        return out

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
