from __future__ import annotations

from typing import Any

import gymnasium as gym
from omegaconf import DictConfig

from .wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    FrameStack,
    MaxAndSkipEnv,
    NoopResetEnv,
    WarpFrame,
)


def _validate_protocol(cfg: DictConfig) -> None:
    required = [
        "repeat_action_probability",
        "frameskip",
        "action_repeat",
        "full_action_space",
        "noop_max",
        "terminal_on_life_loss_train",
        "terminal_on_life_loss_eval",
        "eval_max_episode_frames",
    ]
    missing = [k for k in required if k not in cfg.env_protocol]
    if missing:
        raise ValueError(f"Missing env protocol fields: {missing}")


def _make_base_env(cfg: DictConfig, render_mode: str | None = None):
    kwargs: dict[str, Any] = {
        "frameskip": int(cfg.env_protocol.frameskip),
        "repeat_action_probability": float(cfg.env_protocol.repeat_action_probability),
        "full_action_space": bool(cfg.env_protocol.full_action_space),
    }
    if render_mode is not None:
        kwargs["render_mode"] = render_mode

    try:
        return gym.make(cfg.env.game_id, **kwargs)
    except TypeError:
        # Compatibility fallback for minor ALE/Gym API differences.
        kwargs.pop("full_action_space", None)
        return gym.make(cfg.env.game_id, **kwargs)


def make_atari_env(cfg: DictConfig, eval_mode: bool = False, render_mode: str | None = None):
    _validate_protocol(cfg)

    env = _make_base_env(cfg, render_mode=render_mode)
    env = NoopResetEnv(env, noop_max=int(cfg.env_protocol.noop_max))
    env = MaxAndSkipEnv(env, skip=int(cfg.env_protocol.action_repeat))

    if not eval_mode and bool(cfg.env_protocol.terminal_on_life_loss_train):
        env = EpisodicLifeEnv(env)

    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    env = WarpFrame(env)

    if not eval_mode and bool(cfg.preset.reward_clip):
        env = ClipRewardEnv(env)

    env = FrameStack(env, num_stack=int(cfg.agent.frame_stack))
    return env
