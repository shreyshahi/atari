from __future__ import annotations

import numpy as np

from dqn.replay.prioritized import PrioritizedReplayBuffer
from dqn.replay.uniform import UniformReplayBuffer


def test_uniform_replay_shapes():
    rb = UniformReplayBuffer(capacity=64, frame_stack=4)
    for i in range(20):
        frame = np.full((84, 84), i, dtype=np.uint8)
        rb.add(frame=frame, action=i % 4, reward=float(i), done=(i % 7 == 0))

    batch = rb.sample(8)
    assert batch["states"].shape == (8, 4, 84, 84)
    assert batch["next_states"].shape == (8, 4, 84, 84)
    assert batch["actions"].shape == (8,)
    assert batch["weights"].shape == (8,)


def test_prioritized_replay_weights_and_indices():
    rb = PrioritizedReplayBuffer(capacity=64, frame_stack=4)
    for i in range(25):
        frame = np.full((84, 84), i, dtype=np.uint8)
        rb.add(frame=frame, action=i % 3, reward=float(i), done=(i % 9 == 0))

    batch = rb.sample(8)
    assert batch["weights"].max() <= 1.0 + 1e-6
    assert batch["weights"].min() > 0.0
    assert batch["indices"].shape == (8,)

    rb.update_priorities(batch["indices"], np.abs(np.random.randn(8)))
