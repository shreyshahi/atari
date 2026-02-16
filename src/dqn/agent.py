from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from torch.optim import Adam, RMSprop

from dqn.networks import DuelingDQN, NatureDQN
from dqn.utils.optimizers import DeepMindRMSprop


def _build_network(cfg: DictConfig, n_actions: int) -> nn.Module:
    if cfg.agent.network == "dueling":
        return DuelingDQN(n_actions=n_actions, frame_stack=int(cfg.agent.frame_stack))
    return NatureDQN(n_actions=n_actions, frame_stack=int(cfg.agent.frame_stack))


class DQNAgent:
    def __init__(self, cfg: DictConfig, n_actions: int, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.n_actions = n_actions

        self.online_net = _build_network(cfg, n_actions).to(device)
        self.target_net = _build_network(cfg, n_actions).to(device)
        self.sync_target_network()
        self.target_net.eval()

        self.gamma = float(cfg.agent.gamma)
        self.use_double = bool(cfg.agent.double)
        self.max_grad_norm = cfg.agent.max_grad_norm

        self.optimizer = self._build_optimizer(cfg)

        self.online_net.train()

        if bool(cfg.agent.use_c51):
            raise NotImplementedError("C51 is roadmap/future in this implementation")

    def _build_optimizer(self, cfg: DictConfig):
        opt_name = str(cfg.preset.optimizer).lower()
        if opt_name == "adam":
            return Adam(
                self.online_net.parameters(),
                lr=float(cfg.preset.lr),
                eps=float(cfg.preset.adam_eps),
            )
        if opt_name == "deepmind_rmsprop":
            return DeepMindRMSprop(
                self.online_net.parameters(),
                lr=float(cfg.preset.lr),
                decay=float(cfg.preset.rmsprop_decay),
                momentum=float(cfg.preset.rmsprop_momentum),
                eps=float(cfg.preset.rmsprop_eps),
            )
        return RMSprop(
            self.online_net.parameters(),
            lr=float(cfg.preset.lr),
            alpha=float(cfg.preset.rmsprop_decay),
            momentum=float(cfg.preset.rmsprop_momentum),
            eps=float(cfg.preset.rmsprop_eps),
            centered=True,
        )

    @torch.no_grad()
    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if np.random.random() < epsilon:
            return int(np.random.randint(0, self.n_actions))

        state_t = torch.from_numpy(state[None, ...]).to(self.device)
        q_values = self.online_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def compute_target_q(
        self,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            if self.use_double:
                next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_states).max(dim=1).values
            return rewards + self.gamma * next_q * (1.0 - dones.float())

    def update(self, batch: dict[str, Any]) -> dict[str, Any]:
        states = torch.from_numpy(batch["states"]).to(self.device)
        actions = torch.from_numpy(batch["actions"]).to(self.device)
        rewards = torch.from_numpy(batch["rewards"]).to(self.device)
        next_states = torch.from_numpy(batch["next_states"]).to(self.device)
        dones = torch.from_numpy(batch["dones"]).to(self.device)
        weights = torch.from_numpy(batch["weights"]).to(self.device)

        q_values = self.online_net(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        target_q = self.compute_target_q(next_states, rewards, dones)
        td_errors = target_q - q_selected

        per_sample_loss = F.smooth_l1_loss(q_selected, target_q, reduction="none")
        loss = (per_sample_loss * weights).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_norm = torch.tensor(0.0, device=self.device)
        if self.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.online_net.parameters(), float(self.max_grad_norm)
            )
        else:
            norms = []
            for p in self.online_net.parameters():
                if p.grad is not None:
                    norms.append(p.grad.detach().norm(2))
            if norms:
                grad_norm = torch.norm(torch.stack(norms), 2)

        self.optimizer.step()

        with torch.no_grad():
            mean_q = float(q_selected.mean().item())
            max_q = float(q_selected.max().item())

        return {
            "loss": float(loss.item()),
            "mean_q": mean_q,
            "max_q": max_q,
            "grad_norm": float(grad_norm.item()),
            "td_errors": td_errors.detach().cpu().numpy(),
        }

    def sync_target_network(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    def state_dict(self) -> dict[str, Any]:
        return {
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.online_net.load_state_dict(state["online_net"])
        self.target_net.load_state_dict(state["target_net"])
        self.optimizer.load_state_dict(state["optimizer"])
