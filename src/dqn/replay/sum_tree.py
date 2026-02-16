from __future__ import annotations

import numpy as np


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def update(self, index: int, priority: float) -> None:
        idx = index + self.capacity
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        idx //= 2
        while idx >= 1:
            self.tree[idx] += change
            idx //= 2

    def get(self, mass: float) -> tuple[int, float]:
        idx = 1
        while idx < self.capacity:
            left = idx * 2
            if mass <= self.tree[left]:
                idx = left
            else:
                mass -= self.tree[left]
                idx = left + 1
        leaf = idx - self.capacity
        return leaf, float(self.tree[idx])
