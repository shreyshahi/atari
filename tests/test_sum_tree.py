from __future__ import annotations

import numpy as np

from dqn.replay.sum_tree import SumTree


def test_sum_tree_update_and_sample():
    tree = SumTree(capacity=8)
    for i in range(8):
        tree.update(i, i + 1)

    assert abs(tree.total - 36.0) < 1e-6

    idx, p = tree.get(1.0)
    assert 0 <= idx < 8
    assert p > 0


def test_sum_tree_sampling_is_proportional_over_many_samples():
    tree = SumTree(capacity=4)
    priorities = [1.0, 2.0, 3.0, 4.0]
    for i, p in enumerate(priorities):
        tree.update(i, p)

    counts = np.zeros(4, dtype=np.int64)
    for _ in range(10_000):
        mass = np.random.uniform(0.0, tree.total)
        idx, _ = tree.get(mass)
        counts[idx] += 1

    # Highest priority sampled most often; ordering should follow priorities.
    assert counts[3] > counts[2] > counts[1] > counts[0]
