from __future__ import annotations

import torch
from torch import nn

from .encoder import ConvEncoder


class DuelingDQN(nn.Module):
    def __init__(self, n_actions: int, frame_stack: int = 4):
        super().__init__()
        self.encoder = ConvEncoder(in_channels=frame_stack)
        in_features = self.encoder.out_features

        self.value = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions),
        )
        self._init_linear()

    def _init_linear(self) -> None:
        value_hidden = self.value[0]
        value_out = self.value[2]
        adv_hidden = self.advantage[0]
        adv_out = self.advantage[2]

        nn.init.kaiming_uniform_(value_hidden.weight, nonlinearity="relu")
        nn.init.zeros_(value_hidden.bias)
        nn.init.kaiming_uniform_(adv_hidden.weight, nonlinearity="relu")
        nn.init.zeros_(adv_hidden.bias)

        nn.init.kaiming_uniform_(value_out.weight, nonlinearity="linear")
        nn.init.zeros_(value_out.bias)
        nn.init.kaiming_uniform_(adv_out.weight, nonlinearity="linear")
        nn.init.zeros_(adv_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        value = self.value(z)
        advantage = self.advantage(z)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
