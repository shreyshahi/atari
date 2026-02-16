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


def test_nature_dqn_parameter_count_roughly_expected():
    model = NatureDQN(n_actions=6)
    params = sum(p.numel() for p in model.parameters())
    assert 1_650_000 <= params <= 1_750_000


def test_gradients_flow_to_all_trainable_layers():
    model = NatureDQN(n_actions=4)
    x = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
    out = model(x)
    loss = out.mean()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
