from __future__ import annotations

from typing import Iterable

import torch
from torch.optim.optimizer import Optimizer


class DeepMindRMSprop(Optimizer):
    """RMSProp variant used by DQN paper (running mean + variance style)."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 2.5e-4,
        decay: float = 0.95,
        momentum: float = 0.95,
        eps: float = 0.01,
    ):
        defaults = dict(lr=lr, decay=decay, momentum=momentum, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            decay = group["decay"]
            momentum = group["momentum"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("DeepMindRMSprop does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["square_avg"] = torch.zeros_like(p)
                    state["grad_avg"] = torch.zeros_like(p)
                    state["momentum_buffer"] = torch.zeros_like(p)

                square_avg = state["square_avg"]
                grad_avg = state["grad_avg"]
                momentum_buffer = state["momentum_buffer"]

                square_avg.mul_(decay).addcmul_(grad, grad, value=1.0 - decay)
                grad_avg.mul_(decay).add_(grad, alpha=1.0 - decay)
                denom = (square_avg - grad_avg.pow(2) + eps).sqrt_()

                momentum_buffer.mul_(momentum).addcdiv_(grad, denom, value=-lr)
                p.add_(momentum_buffer)

        return loss
