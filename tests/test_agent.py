from __future__ import annotations

import numpy as np
import torch
from torch import nn

from dqn.agent import DQNAgent


class StubNet(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, x):
        return self.values.unsqueeze(0).repeat(x.shape[0], 1)


def test_agent_update_runs(cfg):
    agent = DQNAgent(cfg, n_actions=4, device=torch.device("cpu"))
    batch = {
        "states": np.random.randint(0, 256, size=(4, 4, 84, 84), dtype=np.uint8),
        "actions": np.array([0, 1, 2, 3], dtype=np.int64),
        "rewards": np.random.randn(4).astype(np.float32),
        "next_states": np.random.randint(0, 256, size=(4, 4, 84, 84), dtype=np.uint8),
        "dones": np.array([False, False, True, False], dtype=np.bool_),
        "weights": np.ones(4, dtype=np.float32),
        "indices": np.arange(4, dtype=np.int64),
    }
    info = agent.update(batch)
    assert np.isfinite(info["loss"])
    assert np.isfinite(info["mean_q"])


def test_double_target_differs_when_argmax_differs(cfg):
    agent = DQNAgent(cfg, n_actions=2, device=torch.device("cpu"))
    next_states = torch.randint(0, 256, (3, 4, 84, 84), dtype=torch.uint8)
    rewards = torch.zeros(3, dtype=torch.float32)
    dones = torch.zeros(3, dtype=torch.bool)

    agent.online_net = StubNet([1.0, 0.0])
    agent.target_net = StubNet([0.0, 2.0])

    agent.use_double = False
    vanilla = agent.compute_target_q(next_states, rewards, dones)

    agent.use_double = True
    double = agent.compute_target_q(next_states, rewards, dones)

    assert not torch.allclose(vanilla, double)


def test_select_action_epsilon_zero_is_greedy(cfg):
    agent = DQNAgent(cfg, n_actions=2, device=torch.device("cpu"))
    agent.online_net = StubNet([0.1, 0.9])
    state = np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)
    actions = [agent.select_action(state, epsilon=0.0) for _ in range(50)]
    assert all(a == 1 for a in actions)


def test_select_action_epsilon_one_is_random(cfg):
    agent = DQNAgent(cfg, n_actions=3, device=torch.device("cpu"))
    agent.online_net = StubNet([0.1, 0.9, 0.2])
    state = np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)
    actions = [agent.select_action(state, epsilon=1.0) for _ in range(200)]
    assert len(set(actions)) > 1
