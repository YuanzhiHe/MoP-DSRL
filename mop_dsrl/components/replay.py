"""FIFO replay buffer for chunk-level transitions (method_spec §5).

A transition is (o, z, r_outer, o', d). We store r_outer *without* the
regularization term so retraining with a different lambda does not require
re-collection (method_spec §5 "Replay buffer").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    o: np.ndarray
    z: np.ndarray
    r: float
    o_next: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, z_dim: int, seed: int = 0):
        self.capacity = int(capacity)
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.o_buf = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.z_buf = np.zeros((self.capacity, z_dim), dtype=np.float32)
        self.r_buf = np.zeros((self.capacity,), dtype=np.float32)
        self.o_next_buf = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.d_buf = np.zeros((self.capacity,), dtype=np.float32)
        self.size = 0
        self.ptr = 0
        self._rng = np.random.default_rng(seed)

    def push(self, o: np.ndarray, z: np.ndarray, r: float, o_next: np.ndarray, done: bool) -> None:
        i = self.ptr
        self.o_buf[i] = o
        self.z_buf[i] = z
        self.r_buf[i] = r
        self.o_next_buf[i] = o_next
        self.d_buf[i] = float(done)
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        if self.size == 0:
            raise RuntimeError("replay buffer is empty")
        idx = self._rng.integers(0, self.size, size=batch_size)
        return {
            "o": self.o_buf[idx],
            "z": self.z_buf[idx],
            "r": self.r_buf[idx],
            "o_next": self.o_next_buf[idx],
            "d": self.d_buf[idx],
        }

    def __len__(self) -> int:
        return self.size
