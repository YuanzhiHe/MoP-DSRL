#!/usr/bin/env python3
"""Step-by-step parity between the environment paths the stages use.

Three stages roll a frozen policy out and each builds its environment
differently: ``_ref_chunk`` builds one per chunk, ``run_arm`` and ``run_bmd``
build one per cell, and ``frozen_baseline`` builds one it drops before the one
it measures on. A completion rate measured under one of them was compared
against a histogram measured under another, so the first question is whether
the environment path changes a trajectory at all.

Every configuration here replays ONE pre-generated latent sequence, so the
sampler contributes nothing and the only variable left is how the environment
was constructed. Under a deterministic reset the trajectories are then
comparable step by step: if two configurations agree, the environment path is
not what moved the completion rate, and if they part, the probe reports the
step index where they first did.

The native paths run alongside with their own samplers, which is what separates
a sampling difference from an environment-path difference: a native path that
matches its replayed twin in aggregate leaves the sampler out of the account.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_d3il_v2 as D  # noqa: E402

# A job runs from a frozen copy of the source, so the repository root has to
# come from the driver rather than from this file's location.
ROOT = D.ROOT
OUT = ROOT / "runs" / "_d3il_v2"


# How many environments this process has built, over its whole life. A pool
# reuses its workers, so a configuration that asks to measure on its first
# environment gets one only while its worker is fresh; every later task in the
# same worker measures on a later environment whatever it asked for. The number
# is reported with each result so the ordinal a configuration received can be
# read apart from the ordinal it requested.
_BUILT = 0


def _build(task, seed, history):
    """Construct one environment and count it against this process."""
    global _BUILT
    _BUILT += 1
    return D.TASKS[task]["env"](seed=seed, history=history)


def _digest(parts) -> str:
    """Order-sensitive digest of a step sequence, for comparing whole paths."""
    h = hashlib.blake2b(digest_size=12)
    for p in parts:
        h.update(np.ascontiguousarray(p, dtype=np.float64).tobytes())
    return h.hexdigest()


def _env_identity(env) -> dict:
    """Whatever identifies this environment's simulator instance."""
    inner = getattr(env, "_env", None)
    scene = getattr(inner, "scene", None)
    ident = dict(pid=os.getpid(),
                 wrapper_id=id(env),
                 inner_id=id(inner) if inner is not None else None,
                 scene_id=id(scene) if scene is not None else None)
    for name in ("sim", "model", "data", "viewer"):
        obj = getattr(scene, name, None)
        ident[f"scene_{name}_id"] = id(obj) if obj is not None else None
    return ident


def _reset_state(env) -> list:
    """The state a reset left behind, so two resets can be compared directly."""
    inner = getattr(env, "_env", None)
    out = []
    for name in ("robot_state", "get_observation"):
        fn = getattr(inner, name, None)
        if callable(fn):
            try:
                out += np.asarray(fn(), dtype=np.float64).reshape(-1).tolist()
            except Exception:                              # noqa: BLE001
                out += [float("nan")]
    return [round(float(v), 10) for v in out[:16]]


def replay_episodes(env, model, blob, z_seq, clip, n_episodes, keep_steps=3):
    """Run episodes on ``env``, taking each step's latent from ``z_seq``.

    The arithmetic is the one both ``rollout`` and ``_ref_chunk`` perform; only
    the latent comes from the caller, so two calls differing in nothing but the
    environment they were handed are comparable step by step.
    """
    out = []
    k = 0
    for _ in range(n_episodes):
        obs = env.reset()
        rs = _reset_state(env)
        steps, done, head = [], False, []
        while not done:
            o_n = D.norm_obs(obs, blob, clip)
            o = torch.tensor(o_n, dtype=torch.float32)[None]
            z = torch.tensor(z_seq[k % len(z_seq)][None], dtype=torch.float32)
            k += 1
            a = D.decode(model, blob, o, z, 10)[0]
            obs, r, done, info = env.step(a)
            steps.append((o_n, a, float(r), float(done)))
            if len(head) < keep_steps:
                head.append(dict(obs=[round(float(v), 8) for v in np.ravel(o_n)],
                                 act=[round(float(v), 8) for v in np.ravel(a)]))
        out.append(dict(
            mode=int(info["mode_id"]), success=bool(info["success"]),
            n_steps=len(steps), reset_state=rs, head=head,
            path_digest=_digest([p for s in steps for p in (s[0], s[1])]),
            obs_absmax=float(max(np.abs(s[0]).max() for s in steps)),
        ))
    return out


def _cell(payload):
    """One configuration, in its own process."""
    D._single_threaded()
    cfg = payload
    task, weighting, bb = cfg["task"], cfg["weighting"], cfg["bb"]
    ck = D.backbone_path(task, weighting, bb)
    model, blob = D.load_backbone(ck, "cpu")
    hist = blob.get("history", 1)
    clip = cfg["clip"]
    rng = np.random.default_rng(cfg["z_seed"])
    z_seq = rng.standard_normal((cfg["z_len"], blob["act_dim"]))

    t0 = time.time()
    # Build and drop ``ordinal - 1`` environments, then measure on the next one.
    built = []
    for _ in range(cfg["ordinal"] - 1):
        built.append(_env_identity(_build(task, cfg["env_seed"], hist)))
    env = _build(task, cfg["env_seed"], hist)
    ident = _env_identity(env)
    ident["ordinal_requested"] = cfg["ordinal"]
    ident["ordinal_in_process"] = _BUILT
    ident["fresh_worker"] = (_BUILT == cfg["ordinal"])

    if cfg["mode"] == "replay":
        eps = replay_episodes(env, model, blob, z_seq, clip, cfg["episodes"])
    elif cfg["mode"] == "native_ref":
        # The reference sampler: numpy, one draw per step, no policy.
        eps = []
        r2 = np.random.default_rng(cfg["env_seed"])
        for _ in range(cfg["episodes"]):
            obs, done = env.reset(), False
            n = 0
            while not done:
                o = torch.tensor(D.norm_obs(obs, blob, clip),
                                 dtype=torch.float32)[None]
                z = torch.tensor(r2.standard_normal((1, blob["act_dim"])),
                                 dtype=torch.float32)
                obs, _, done, info = env.step(D.decode(model, blob, o, z, 10)[0])
                n += 1
            eps.append(dict(mode=int(info["mode_id"]),
                            success=bool(info["success"]), n_steps=n))
    else:                                                   # native_eval
        # The evaluation sampler: an untrained NoisePolicy, which is N(0, I).
        torch.manual_seed(cfg["env_seed"])
        policy = D.NoisePolicy(blob["obs_dim"], blob["act_dim"])
        eps = []
        for _ in range(cfg["episodes"]):
            with torch.no_grad():
                _, _, m, s = D.rollout(env, policy, model, blob, set(), "cpu",
                                       10, clip, None)
            eps.append(dict(mode=int(m), success=bool(s), n_steps=None))

    done_n = sum(1 for e in eps if e["mode"] >= 0)
    return dict(name=cfg["name"], cfg={k: v for k, v in cfg.items()
                                      if k != "name"},
                identity=ident, discarded=built,
                completion=done_n / max(len(eps), 1),
                success=sum(e["success"] for e in eps) / max(len(eps), 1),
                episodes=eps, seconds=round(time.time() - t0, 1))


def plan(task, weighting, bb, episodes, z_len, z_seed):
    """The configurations compared, each identical but for one stated thing."""
    base = dict(task=task, weighting=weighting, bb=bb, episodes=episodes,
                z_len=z_len, z_seed=z_seed, env_seed=700, clip=50.0,
                mode="replay")
    cfgs = []
    # Construction ordinal, with the latent sequence held fixed: this is the
    # environment path on its own.
    for o in (1, 2, 3):
        cfgs.append({**base, "ordinal": o, "name": f"replay_env{o}"})
    # The same two ordinals with the bound lifted, which is the other candidate.
    for o in (1, 2):
        cfgs.append({**base, "ordinal": o, "clip": 1e12,
                     "name": f"replay_env{o}_noclip"})
    # The native samplers, for the sampling half of the account.
    for o in (1, 2):
        cfgs.append({**base, "ordinal": o, "mode": "native_ref",
                     "name": f"native_ref_env{o}"})
        cfgs.append({**base, "ordinal": o, "mode": "native_eval",
                     "name": f"native_eval_env{o}"})
    return cfgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="avoiding")
    ap.add_argument("--weighting", default="0.00")
    ap.add_argument("--bb", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--z-len", type=int, default=4096)
    ap.add_argument("--z-seed", type=int, default=20261109)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfgs = plan(a.task, a.weighting, a.bb, a.episodes, a.z_len, a.z_seed)
    print(f"{len(cfgs)} configurations, {a.workers} workers, "
          f"{a.task} w{a.weighting} bb{a.bb}", flush=True)

    # Always dispatch through a pool, which is how every driver runs its cells.
    # With one worker the pool reuses a single process, so the configurations
    # queue behind one another and only the first measures on a fresh one; that
    # is the comparison --workers asks for, and the ordinals reported below say
    # which configuration got what.
    res = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=D._MP) as ex:
        for r in ex.map(_cell, cfgs):
            res.append(r)
            i = r["identity"]
            print(f"  {r['name']:<24} completion {r['completion']:.3f}  "
                  f"pid {i['pid']}  env {i['ordinal_in_process']} of this "
                  f"process (asked for {i['ordinal_requested']})", flush=True)

    out = Path(a.out) if a.out else (
        OUT / f"parity_{a.task}_{a.weighting}_bb{a.bb}_w{a.workers}.json")
    out.write_text(json.dumps(res, indent=1))

    print("\n=== replayed paths, one latent sequence, "
          "episode-by-episode ===", flush=True)
    rep = {r["name"]: r for r in res if r["cfg"]["mode"] == "replay"}
    ref = rep.get("replay_env1")
    for name, r in rep.items():
        if ref is None:
            break
        same = [i for i, (x, y) in enumerate(zip(ref["episodes"],
                                                 r["episodes"]))
                if x["path_digest"] == y["path_digest"]]
        first_diff = next((i for i, (x, y) in enumerate(
            zip(ref["episodes"], r["episodes"]))
            if x["path_digest"] != y["path_digest"]), None)
        print(f"  {name:<24} env {r['identity']['ordinal_in_process']:>2}  "
              f"completion {r['completion']:.3f}  "
              f"identical episodes {len(same)}/{len(r['episodes'])}  "
              f"first difference at episode {first_diff}  "
              f"obs_absmax {max(e['obs_absmax'] for e in r['episodes']):.3g}",
              flush=True)

    print("\n=== native samplers ===", flush=True)
    for r in res:
        if r["cfg"]["mode"] != "replay":
            print(f"  {r['name']:<24} env "
                  f"{r['identity']['ordinal_in_process']:>2}  completion "
                  f"{r['completion']:.3f}  success {r['success']:.3f}",
                  flush=True)

    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
