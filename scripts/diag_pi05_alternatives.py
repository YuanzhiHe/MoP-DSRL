#!/usr/bin/env python3
"""Try alternative pi_0.5 LIBERO checkpoints / sources to find one with
non-empty normalizer stats.

The `lerobot/pi05_libero` HF repo's policy_preprocessor.json has empty
`features: {}`, so the normalizer is identity — that's why BC=0%.
We try every plausible alternative repo and report which (if any) ship
populated stats.

Usage:  python scripts/diag_pi05_alternatives.py
"""

from __future__ import annotations
import os, sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _ipv4_patch  # noqa: F401,E402

CANDIDATES = [
    "lerobot/pi05_libero",
    "lerobot/pi05_libero_finetuned",
    "lerobot/pi05_libero_base",
    "lerobot/pi05_libero_v044",
    "lerobot/pi05_libero_finetuned_v044",
    "lerobot/pi05_base",
    "physical-intelligence/pi0_libero",
    "openpi/pi0_libero",
]


def check_repo(repo: str) -> dict:
    from huggingface_hub import snapshot_download, list_repo_files
    info = {"repo": repo, "exists": False, "preproc_json": None,
            "has_features": False, "has_action_stats": False,
            "has_state_stats": False, "files": []}
    try:
        files = list_repo_files(repo)
        info["exists"] = True
        info["files"] = files
    except Exception as e:
        info["error"] = str(e)[:200]
        return info

    if "policy_preprocessor.json" not in files:
        info["error"] = "no policy_preprocessor.json"
        return info

    try:
        path = snapshot_download(repo_id=repo, allow_patterns=["policy_preprocessor.json", "policy_postprocessor.json"])
    except Exception as e:
        info["error"] = f"download failed: {str(e)[:200]}"
        return info

    pp = os.path.join(path, "policy_preprocessor.json")
    if not os.path.exists(pp):
        return info
    data = json.loads(open(pp).read())
    info["preproc_json"] = pp
    for step in data.get("steps", []):
        if step.get("registry_name") == "normalizer_processor":
            feats = step.get("config", {}).get("features", {})
            info["has_features"] = bool(feats)
            info["features_count"] = len(feats)
            info["feature_keys"] = list(feats.keys())[:10]
            for fkey, fval in feats.items():
                if "action" in fkey.lower():
                    info["has_action_stats"] = True
                if "state" in fkey.lower():
                    info["has_state_stats"] = True
    return info


def main():
    print("Probing candidate pi_0.5 LIBERO repos for non-empty normalizer stats…\n")
    best = None
    for repo in CANDIDATES:
        print(f"=== {repo} ===")
        info = check_repo(repo)
        if not info.get("exists"):
            print(f"  ✗ {info.get('error', 'not found')}")
            print()
            continue
        if info.get("error"):
            print(f"  ✗ {info['error']}")
            print()
            continue
        feats = info.get("features_count", 0)
        print(f"  files ({len(info['files'])}): {', '.join(info['files'][:8])}")
        print(f"  has_features: {info['has_features']} (count={feats})")
        print(f"  feature_keys: {info['feature_keys']}")
        print(f"  has_action_stats: {info['has_action_stats']}")
        print(f"  has_state_stats: {info['has_state_stats']}")
        if info["has_action_stats"] and info["has_state_stats"]:
            print(f"  ✓ THIS REPO HAS POPULATED STATS")
            if best is None:
                best = repo
        print()

    print(f"\nBest candidate: {best!r}")
    if best is None:
        print("\nNo HF repo has populated stats.  Fall back to computing stats from")
        print("LIBERO demo HDF5s (see scripts/build_norm_stats.py — TBD).")


if __name__ == "__main__":
    raise SystemExit(main())
