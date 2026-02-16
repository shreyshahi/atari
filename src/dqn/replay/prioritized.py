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

    def sample(self, batch_size: int, env_frames: int | None = None) -> dict[str, Any]:
        if not self.can_sample(batch_size):
            raise RuntimeError("Not enough data in replay buffer")

        frame_count = self.num_added if env_frames is None else int(env_frames)
        beta = self._beta_by_frame(frame_count)
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
            abs_index = -1
            phys = -1
            priority = self.priority_eps

            for _ in range(10):
                mass = np.random.uniform(low, high)
                phys_cand, priority_cand = self.sum_tree.get(mass)
                abs_cand = int(self.phys_to_abs[phys_cand])
                if self._is_valid_abs_index(abs_cand):
                    abs_index = abs_cand
                    phys = phys_cand
                    priority = priority_cand
                    break

            if phys < 0:
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

        valid_leaf_probs = self._valid_leaf_probs(total)
        if valid_leaf_probs.size > 0:
            min_prob = float(valid_leaf_probs.min())
            max_weight = (n * min_prob) ** (-beta)
            weights /= max_weight
        else:
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

    def _valid_leaf_probs(self, total: float) -> np.ndarray:
        if total <= 0:
            return np.array([], dtype=np.float64)
        valid_phys = [
            phys for phys, abs_idx in enumerate(self.phys_to_abs) if self._is_valid_abs_index(int(abs_idx))
        ]
        if not valid_phys:
            return np.array([], dtype=np.float64)
        leaves = self.sum_tree.tree[self.capacity + np.array(valid_phys, dtype=np.int64)]
        leaves = np.clip(leaves / total, self.priority_eps, 1.0)
        return leaves.astype(np.float64)

    def diagnostics(self, sample_indices: np.ndarray, sample_weights: np.ndarray) -> dict[str, float]:
        sample_indices = np.asarray(sample_indices, dtype=np.int64)
        sample_weights = np.asarray(sample_weights, dtype=np.float32)

        valid_phys = [
            phys for phys, abs_idx in enumerate(self.phys_to_abs) if self._is_valid_abs_index(int(abs_idx))
        ]
        if valid_phys:
            priorities = self.sum_tree.tree[self.capacity + np.array(valid_phys, dtype=np.int64)]
        else:
            priorities = np.array([0.0], dtype=np.float64)

        sampled_abs = self.phys_to_abs[sample_indices]
        sample_ages = self.num_added - sampled_abs

        return {
            "replay_priority_mean": float(np.mean(priorities)),
            "replay_priority_std": float(np.std(priorities)),
            "replay_priority_max": float(np.max(priorities)),
            "replay_sample_age_mean": float(np.mean(sample_ages)),
            "replay_sample_age_std": float(np.std(sample_ages)),
            "replay_sample_age_max": float(np.max(sample_ages)),
            "replay_is_weight_mean": float(np.mean(sample_weights)),
            "replay_is_weight_std": float(np.std(sample_weights)),
            "replay_is_weight_max": float(np.max(sample_weights)),
        }

    def _is_valid_abs_index(self, abs_index: int) -> bool:
        if abs_index < 0:
            return False
        lower_bound = max(0, self.num_added - self.size)
        min_abs = max(lower_bound + self.frame_stack - 1, self.frame_stack - 1)
        max_abs = self.num_added - 2
        return min_abs <= abs_index <= max_abs
