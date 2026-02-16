from __future__ import annotations

from collections import deque
from statistics import mean
from typing import Any


class MetricsAggregator:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.episode_returns = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)

    def add_episode(self, episode_return: float, episode_length: int) -> None:
        self.episode_returns.append(float(episode_return))
        self.episode_lengths.append(int(episode_length))

    def summarize(self) -> dict[str, Any]:
        if not self.episode_returns:
            return {}
        return {
            "episode_return_mean": mean(self.episode_returns),
            "episode_return_last": self.episode_returns[-1],
            "episode_length_mean": mean(self.episode_lengths),
            "episode_length_last": self.episode_lengths[-1],
        }
