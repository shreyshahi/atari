from __future__ import annotations

from dqn.replay.sum_tree import SumTree


def test_sum_tree_update_and_sample():
    tree = SumTree(capacity=8)
    for i in range(8):
        tree.update(i, i + 1)

    assert abs(tree.total - 36.0) < 1e-6

    idx, p = tree.get(1.0)
    assert 0 <= idx < 8
    assert p > 0
