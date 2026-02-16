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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        value = self.value(z)
        advantage = self.advantage(z)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
