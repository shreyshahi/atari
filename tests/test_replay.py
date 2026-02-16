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
    assert batch["states"].dtype == np.uint8
    assert batch["rewards"].dtype == np.float32


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


def test_uniform_replay_circular_wrap():
    rb = UniformReplayBuffer(capacity=16, frame_stack=4)
    for i in range(50):
        frame = np.full((84, 84), i % 255, dtype=np.uint8)
        rb.add(frame=frame, action=i % 4, reward=float(i), done=False)
    assert rb.size == 16
    batch = rb.sample(4)
    assert batch["states"].shape == (4, 4, 84, 84)


def test_episode_boundary_zero_fills_previous_frames():
    rb = UniformReplayBuffer(capacity=32, frame_stack=4)
    for i in range(10):
        done = i == 4
        frame = np.full((84, 84), i + 1, dtype=np.uint8)
        rb.add(frame=frame, action=0, reward=0.0, done=done)

    # Index right after terminal should not include pre-terminal frames.
    stack = rb._encode_stack(5)
    assert np.all(stack[0] == 0)
