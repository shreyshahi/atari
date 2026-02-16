from __future__ import annotations

from typing import Any

import numpy as np

from .sum_tree import SumTree
from .uniform import ReplayBatch, UniformReplayBuffer


class PrioritizedReplayBuffer(UniformReplayBuffer):
    def __init__(
        self,
        capacity: int,
        frame_stack: int = 4,
        frame_shape: tuple[int, int] = (84, 84),
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_decay_frames: int = 50_000_000,
        priority_eps: float = 1e-6,
    ):
        super().__init__(capacity=capacity, frame_stack=frame_stack, frame_shape=frame_shape)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_decay_frames = max(1, beta_decay_frames)
        self.priority_eps = priority_eps

        self.sum_tree = SumTree(capacity)
        self.phys_to_abs = np.full(capacity, -1, dtype=np.int64)
        self.max_priority = 1.0

    def add(self, frame: np.ndarray, action: int, reward: float, done: bool) -> None:
        abs_index = self.num_added
        phys = abs_index % self.capacity
        super().add(frame=frame, action=action, reward=reward, done=done)
        self.phys_to_abs[phys] = abs_index
        self.sum_tree.update(phys, self.max_priority)

    def sample(self, batch_size: int) -> dict[str, Any]:
        if not self.can_sample(batch_size):
            raise RuntimeError("Not enough data in replay buffer")

        beta = self._beta_by_frame(self.num_added)
        total = self.sum_tree.total
        if total <= 0:
            return super().sample(batch_size)

        segment = total / batch_size
        abs_indices: list[int] = []
        phys_indices: list[int] = []
        priorities: list[float] = []

        while len(abs_indices) < batch_size:
            i = len(abs_indices)
            low = segment * i
            high = segment * (i + 1)
            mass = np.random.uniform(low, high)
            phys, priority = self.sum_tree.get(mass)
            abs_index = int(self.phys_to_abs[phys])

            if not self._is_valid_abs_index(abs_index):
                # fallback to uniform valid index if sampled slot is stale/newest
                abs_index = self._sample_abs_index()
                phys = abs_index % self.capacity
                priority = max(self.sum_tree.tree[phys + self.capacity], self.priority_eps)

            abs_indices.append(abs_index)
            phys_indices.append(phys)
            priorities.append(float(priority))

        probs = np.array(priorities, dtype=np.float64) / total
        probs = np.clip(probs, self.priority_eps, 1.0)
        n = max(1, self.size)
        weights = (n * probs) ** (-beta)
        weights /= weights.max()

        states = np.stack([self._encode_stack(i) for i in abs_indices], axis=0)
        next_states = np.stack([self._encode_stack(i + 1) for i in abs_indices], axis=0)
        actions = np.array([self._get_action(i) for i in abs_indices], dtype=np.int64)
        rewards = np.array([self._get_reward(i) for i in abs_indices], dtype=np.float32)
        dones = np.array([self._get_done(i) for i in abs_indices], dtype=np.bool_)

        batch = ReplayBatch(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            weights=weights.astype(np.float32),
            indices=np.array(phys_indices, dtype=np.int64),
        )
        return batch.as_dict()

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        priorities = np.asarray(priorities, dtype=np.float64)
        indices = np.asarray(indices, dtype=np.int64)
        adjusted = np.power(np.abs(priorities) + self.priority_eps, self.alpha)

        for idx, p in zip(indices, adjusted):
            self.sum_tree.update(int(idx), float(p))
            self.max_priority = max(self.max_priority, float(p))

    def _beta_by_frame(self, frame: int) -> float:
        frac = min(1.0, frame / self.beta_decay_frames)
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def _is_valid_abs_index(self, abs_index: int) -> bool:
        if abs_index < 0:
            return False
        lower_bound = max(0, self.num_added - self.size)
        min_abs = max(lower_bound + self.frame_stack - 1, self.frame_stack - 1)
        max_abs = self.num_added - 2
        return min_abs <= abs_index <= max_abs
