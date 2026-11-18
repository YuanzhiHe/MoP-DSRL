#!/usr/bin/env python3
"""Can this node take a picture of the simulator?

The paper shows the tasks only as a block diagram. A frame from the
simulator's own cage camera would show what a mode is. The camera needs a GL
context, which a batch node has only through EGL or OSMesa, so this probe
reports which of them works before anything is built around it.
"""
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("MOP_DSRL_ROOT") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT))


def main() -> int:
    backend = os.environ.get("MUJOCO_GL", "unset")
    print(f"MUJOCO_GL={backend}", flush=True)
    import numpy as np
    from scripts.envs_d3il import D3ILAvoidingEnv
    env = D3ILAvoidingEnv(seed=0, history=1)
    env.reset()
    print("env built and reset", flush=True)
    img = env._env.bp_cam.get_image(width=640, height=640, depth=False)
    arr = np.asarray(img)
    print("image", arr.shape, arr.dtype, "min", arr.min(), "max", arr.max(), flush=True)
    out = ROOT / "runs" / "_render" / f"probe_{backend}.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arr)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
