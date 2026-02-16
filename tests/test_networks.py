from __future__ import annotations

import torch

from dqn.networks import DuelingDQN, NatureDQN


def test_nature_dqn_output_shape():
    model = NatureDQN(n_actions=6)
    x = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
    y = model(x)
    assert y.shape == (2, 6)


def test_dueling_dqn_output_shape():
    model = DuelingDQN(n_actions=4)
    x = torch.randint(0, 256, (3, 4, 84, 84), dtype=torch.uint8)
    y = model(x)
    assert y.shape == (3, 4)
