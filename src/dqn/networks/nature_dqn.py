from __future__ import annotations

import torch
from torch import nn

from .encoder import ConvEncoder


class NatureDQN(nn.Module):
    def __init__(self, n_actions: int, frame_stack: int = 4):
        super().__init__()
        self.encoder = ConvEncoder(in_channels=frame_stack)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions),
        )
        self._init_linear()

    def _init_linear(self) -> None:
        hidden = self.head[0]
        output = self.head[2]
        nn.init.kaiming_uniform_(hidden.weight, nonlinearity="relu")
        nn.init.zeros_(hidden.bias)
        nn.init.kaiming_uniform_(output.weight, nonlinearity="linear")
        nn.init.zeros_(output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))
