from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ReplayBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    weights: np.ndarray
    indices: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_states": self.next_states,
            "dones": self.dones,
            "weights": self.weights,
            "indices": self.indices,
        }


class UniformReplayBuffer:
    def __init__(
        self,
        capacity: int,
        frame_stack: int = 4,
        frame_shape: tuple[int, int] = (84, 84),
    ):
        self.capacity = int(capacity)
        self.frame_stack = int(frame_stack)
        self.frame_shape = frame_shape

        self.frames = np.zeros((self.capacity, *frame_shape), dtype=np.uint8)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.bool_)

        self.num_added = 0

    def __len__(self) -> int:
        return min(self.num_added, self.capacity)

    @property
    def size(self) -> int:
        return len(self)

    def add(self, frame: np.ndarray, action: int, reward: float, done: bool) -> None:
        idx = self.num_added % self.capacity
        self.frames[idx] = frame
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.num_added += 1

    def can_sample(self, batch_size: int) -> bool:
        if self.size < self.frame_stack + 1:
            return False
        lower_bound = max(0, self.num_added - self.size)
        min_abs = max(lower_bound + self.frame_stack - 1, self.frame_stack - 1)
        max_abs = self.num_added - 2
        return max_abs >= min_abs and batch_size > 0

    def sample(self, batch_size: int) -> dict[str, Any]:
        if not self.can_sample(batch_size):
            raise RuntimeError("Not enough data in replay buffer")

        indices = np.array([self._sample_abs_index() for _ in range(batch_size)], dtype=np.int64)
        states = np.stack([self._encode_stack(i) for i in indices], axis=0)
        next_states = np.stack([self._encode_stack(i + 1) for i in indices], axis=0)

        actions = np.array([self._get_action(i) for i in indices], dtype=np.int64)
        rewards = np.array([self._get_reward(i) for i in indices], dtype=np.float32)
        dones = np.array([self._get_done(i) for i in indices], dtype=np.bool_)

        batch = ReplayBatch(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            weights=np.ones(batch_size, dtype=np.float32),
            indices=np.array([i % self.capacity for i in indices], dtype=np.int64),
        )
        return batch.as_dict()

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        _ = indices, priorities
        return

    def _sample_abs_index(self) -> int:
        lower_bound = max(0, self.num_added - self.size)
        min_abs = max(lower_bound + self.frame_stack - 1, self.frame_stack - 1)
        max_abs = self.num_added - 2
        return int(np.random.randint(min_abs, max_abs + 1))

    def _encode_stack(self, abs_index: int) -> np.ndarray:
        stack = np.zeros((self.frame_stack, *self.frame_shape), dtype=np.uint8)
        lower_bound = max(0, self.num_added - self.size)

        start = max(lower_bound, abs_index - self.frame_stack + 1)
        for j in range(start, abs_index):
            if self._get_done(j):
                start = j + 1

        num = abs_index - start + 1
        for offset, j in enumerate(range(start, abs_index + 1)):
            stack[self.frame_stack - num + offset] = self._get_frame(j)
        return stack

    def _get_frame(self, abs_index: int) -> np.ndarray:
        return self.frames[abs_index % self.capacity]

    def _get_action(self, abs_index: int) -> int:
        return int(self.actions[abs_index % self.capacity])

    def _get_reward(self, abs_index: int) -> float:
        return float(self.rewards[abs_index % self.capacity])

    def _get_done(self, abs_index: int) -> bool:
        return bool(self.dones[abs_index % self.capacity])
