from __future__ import annotations

from omegaconf import DictConfig

from .prioritized import PrioritizedReplayBuffer
from .uniform import UniformReplayBuffer


def build_replay_buffer(cfg: DictConfig):
    frame_stack = int(cfg.agent.frame_stack)
    capacity = int(cfg.replay.capacity)

    if bool(cfg.replay.prioritized):
        return PrioritizedReplayBuffer(
            capacity=capacity,
            frame_stack=frame_stack,
            alpha=float(cfg.replay.alpha),
            beta_start=float(cfg.replay.beta_start),
            beta_end=float(cfg.replay.beta_end),
            beta_decay_frames=int(cfg.replay.beta_decay_frames),
            priority_eps=float(cfg.replay.priority_eps),
        )
    return UniformReplayBuffer(capacity=capacity, frame_stack=frame_stack)
