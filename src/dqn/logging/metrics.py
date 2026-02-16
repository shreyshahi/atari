from __future__ import annotations

from collections import deque
from statistics import mean
from typing import Any


class MetricsAggregator:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.episode_returns = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.losses = deque(maxlen=window_size)
        self.mean_qs = deque(maxlen=window_size)
        self.max_qs = deque(maxlen=window_size)
        self.grad_norms = deque(maxlen=window_size)

    def add_episode(self, episode_return: float, episode_length: int) -> None:
        self.episode_returns.append(float(episode_return))
        self.episode_lengths.append(int(episode_length))

    def add_update(self, loss: float, mean_q: float, max_q: float, grad_norm: float) -> None:
        self.losses.append(float(loss))
        self.mean_qs.append(float(mean_q))
        self.max_qs.append(float(max_q))
        self.grad_norms.append(float(grad_norm))

    def summarize(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.episode_returns:
            out.update(
                {
                    "episode_return_mean": mean(self.episode_returns),
                    "episode_return_last": self.episode_returns[-1],
                    "episode_length_mean": mean(self.episode_lengths),
                    "episode_length_last": self.episode_lengths[-1],
                }
            )
        if self.losses:
            out.update(
                {
                    "loss_mean": mean(self.losses),
                    "loss_last": self.losses[-1],
                    "mean_q_mean": mean(self.mean_qs),
                    "mean_q_last": self.mean_qs[-1],
                    "max_q_mean": mean(self.max_qs),
                    "max_q_last": self.max_qs[-1],
                    "grad_norm_mean": mean(self.grad_norms),
                    "grad_norm_last": self.grad_norms[-1],
                }
            )
        return out
