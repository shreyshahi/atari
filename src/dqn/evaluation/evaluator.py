from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from omegaconf import DictConfig

from dqn.envs.factory import make_atari_env


@dataclass
class EvalResult:
    mean_return: float
    std_return: float
    min_return: float
    max_return: float
    episode_returns: list[float]


class Evaluator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def evaluate(
        self,
        agent,
        episodes: int,
        epsilon: float,
        record_video: bool = False,
        video_path: Path | None = None,
    ) -> EvalResult:
        env = make_atari_env(self.cfg, eval_mode=True, render_mode="rgb_array" if record_video else None)
        returns: list[float] = []

        writer = None
        if record_video and video_path is not None:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(video_path, fps=30)

        max_env_frames = int(self.cfg.env_protocol.eval_max_episode_frames)
        max_steps = max(1, max_env_frames // 4)

        for ep in range(episodes):
            state, _ = env.reset()
            done = False
            ep_return = 0.0
            steps = 0

            while not done and steps < max_steps:
                if writer is not None and ep == 0:
                    frame = env.render()
                    if frame is not None:
                        writer.append_data(frame)

                action = agent.select_action(state, epsilon)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_return += reward
                state = next_state
                steps += 1

            returns.append(ep_return)

        if writer is not None:
            writer.close()

        env.close()
        arr = np.array(returns, dtype=np.float32)
        return EvalResult(
            mean_return=float(arr.mean()),
            std_return=float(arr.std()),
            min_return=float(arr.min()),
            max_return=float(arr.max()),
            episode_returns=returns,
        )
