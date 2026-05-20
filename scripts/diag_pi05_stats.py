#!/usr/bin/env python3
"""Inspect pi_0.5's preprocessor normalizer stats.

We suspect BC=0% because `make_pre_post_processors` loaded an empty / default
stats dict, so NormalizerProcessorStep is a no-op.  This script dumps the
actual normalize / unnormalize stats loaded into preproc/postproc so we can
confirm.

If stats are missing → fix by reading policy_preprocessor.json directly from
the HF cache and feeding it as dataset_stats=...

Usage:  python scripts/diag_pi05_stats.py
"""

from __future__ import annotations
import os, sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _ipv4_patch  # noqa: F401,E402
os.environ.setdefault("MUJOCO_GL", "egl")

import torch


def main():
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained("lerobot/pi05_libero")
    cfg.device = "cuda"

    print("== Step 1: locate the cached policy_preprocessor.json on disk ==")
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id="lerobot/pi05_libero")
    print(f"  snapshot at: {path}")
    print(f"  files: {sorted(os.listdir(path))}")
    for fn in ("policy_preprocessor.json", "policy_postprocessor.json"):
        p = os.path.join(path, fn)
        if os.path.exists(p):
            print(f"\n  === {fn} (size={os.path.getsize(p)} B) ===")
            data = json.loads(open(p).read())
            print(json.dumps(data, indent=2)[:4000])
        else:
            print(f"  {fn} MISSING")

    print("\n== Step 2: build preproc via make_pre_post_processors ==")
    preproc, postproc = make_pre_post_processors(cfg, pretrained_path="lerobot/pi05_libero")
    print("  preproc steps:")
    for i, s in enumerate(preproc.steps):
        cname = type(s).__name__
        print(f"    [{i}] {cname}")
        # Try to dump stats / config of NormalizerProcessorStep
        for attr in ("stats", "_stats", "norm_stats", "_norm_stats",
                     "mean", "std", "min", "max", "dataset_stats",
                     "features", "_features", "norm_map"):
            if hasattr(s, attr):
                v = getattr(s, attr)
                if isinstance(v, dict):
                    print(f"        .{attr} keys = {list(v.keys())}")
                    for k, vv in v.items():
                        if isinstance(vv, dict):
                            for k2, vv2 in vv.items():
                                if isinstance(vv2, torch.Tensor):
                                    print(f"          [{k}].{k2}: shape={tuple(vv2.shape)} mean={vv2.float().mean().item():.4f} range=[{vv2.float().min().item():.4f},{vv2.float().max().item():.4f}]")
                                else:
                                    print(f"          [{k}].{k2}: {vv2}")
                elif isinstance(v, torch.Tensor):
                    print(f"        .{attr}: tensor shape={tuple(v.shape)}")
                else:
                    print(f"        .{attr}: {type(v).__name__} = {str(v)[:200]}")

    print("\n  postproc steps:")
    for i, s in enumerate(postproc.steps):
        cname = type(s).__name__
        print(f"    [{i}] {cname}")
        for attr in ("stats", "_stats", "norm_stats", "_norm_stats",
                     "mean", "std", "min", "max", "dataset_stats", "norm_map"):
            if hasattr(s, attr):
                v = getattr(s, attr)
                if isinstance(v, dict):
                    print(f"        .{attr} keys = {list(v.keys())}")
                    for k, vv in v.items():
                        if isinstance(vv, dict):
                            for k2, vv2 in vv.items():
                                if isinstance(vv2, torch.Tensor):
                                    print(f"          [{k}].{k2}: shape={tuple(vv2.shape)} mean={vv2.float().mean().item():.4f} range=[{vv2.float().min().item():.4f},{vv2.float().max().item():.4f}]")
                                else:
                                    print(f"          [{k}].{k2}: {vv2}")
                elif isinstance(v, torch.Tensor):
                    print(f"        .{attr}: tensor shape={tuple(v.shape)}")
                else:
                    print(f"        .{attr}: {type(v).__name__} = {str(v)[:200]}")

    print("\n== Step 3: verify normalizer actually changes state ==")
    batch = {
        "observation.images.image": torch.rand(1, 3, 224, 224, device="cuda"),
        "observation.images.wrist_image": torch.rand(1, 3, 224, 224, device="cuda"),
        "observation.state": torch.tensor([[0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]], device="cuda"),
        "task": ["open the middle drawer of the cabinet"],
    }
    print(f"  state BEFORE preproc: {batch['observation.state'].cpu().numpy()}")
    print(f"  image[0,0,0,0] BEFORE: {batch['observation.images.image'][0,0,0,0].item():.4f}")
    out = preproc(batch)
    print(f"  state AFTER preproc:  {out['observation.state'].cpu().numpy()}")
    print(f"  image[0,0,0,0] AFTER: {out['observation.images.image'][0,0,0,0].item():.4f}")

    print()
    print("INTERPRETATION:")
    print("  • If preproc state OUT == state IN, normalizer has no stats → BC will fail")
    print("  • If image[0,0,0,0] changed (e.g. 0.5→0.0 via 2x-1), image norm is active")
    print("  • If postproc has an Unnormalizer with non-trivial stats, action de-norm OK")


if __name__ == "__main__":
    raise SystemExit(main())
