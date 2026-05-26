#!/usr/bin/env python3
"""Find out exactly how lerobot's eval_policy calls env_pre + ObservationProcessorStep.
Look at the source code, then build a minimal correct call to env_pre.
"""
from __future__ import annotations
import os, sys, pathlib, inspect
_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))
import _ipv4_patch  # noqa: F401
os.environ.setdefault("MUJOCO_GL", "egl")


def main():
    print("== Inspect lerobot eval_policy source ==")
    from lerobot.scripts import lerobot_eval
    src = inspect.getsource(lerobot_eval)
    # Show every line that mentions env_pre / env_preproc / preprocessor / observation
    for i, line in enumerate(src.split("\n"), 1):
        if any(k in line for k in ("env_pre", "env_preprocessor", "preprocessor", "EnvTransition")):
            print(f"  {i:4d}: {line}")

    print("\n== ObservationProcessorStep.__call__ source ==")
    from lerobot.processor.pipeline import ObservationProcessorStep
    print(inspect.getsource(ObservationProcessorStep))

    print("\n== EnvTransition (or whatever transition type lerobot uses) ==")
    try:
        from lerobot.processor.pipeline import EnvTransition
        print(f"  type: {type(EnvTransition).__name__}")
        if hasattr(EnvTransition, "__annotations__"):
            print(f"  fields: {list(EnvTransition.__annotations__.keys())}")
    except ImportError as e:
        print(f"  no EnvTransition: {e}")
        # Try alternative names
        from lerobot.processor import pipeline
        for name in dir(pipeline):
            if "transition" in name.lower() or "envT" in name:
                obj = getattr(pipeline, name)
                if inspect.isclass(obj):
                    print(f"  candidate: {name}: {obj}")
                    if hasattr(obj, "__annotations__"):
                        print(f"    fields: {list(obj.__annotations__.keys())}")

    print("\n== LiberoProcessorStep source ==")
    try:
        from lerobot.envs.libero import LiberoProcessorStep
    except Exception:
        # find it
        import pkgutil, lerobot.envs
        for m in pkgutil.walk_packages(lerobot.envs.__path__, prefix='lerobot.envs.'):
            try:
                mod = __import__(m.name, fromlist=['_'])
                if hasattr(mod, "LiberoProcessorStep"):
                    LiberoProcessorStep = mod.LiberoProcessorStep
                    print(f"  found in {m.name}")
                    break
            except Exception:
                continue
    try:
        print(inspect.getsource(LiberoProcessorStep))
    except Exception as e:
        print(f"  couldn't get source: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
