from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from omegaconf import OmegaConf


class DummyAtariEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=np.uint8,
        )
        self.action_space = gym.spaces.Discrete(4)
        self._step = 0
        self._lives = 5
        self.last_obs = np.zeros((210, 160, 3), dtype=np.uint8)

    def get_action_meanings(self):
        return ["NOOP", "FIRE", "RIGHT", "LEFT"]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        self._lives = 5
        self.last_obs = self.np_random.integers(0, 256, size=(210, 160, 3), dtype=np.uint8)
        return self.last_obs, {"lives": self._lives}

    def step(self, action):
        self._step += 1
        if self._step % 5 == 0 and self._lives > 0:
            self._lives -= 1

        self.last_obs = self.np_random.integers(0, 256, size=(210, 160, 3), dtype=np.uint8)
        reward = float((action % 3) - 1)
        terminated = self._step >= 20
        truncated = False
        info = {"lives": self._lives}
        return self.last_obs, reward, terminated, truncated, info

    def render(self):
        return self.last_obs


@pytest.fixture
def dummy_env():
    return DummyAtariEnv()


@pytest.fixture
def cfg(tmp_path):
    return OmegaConf.create(
        {
            "seed": 1,
            "output_dir": str(tmp_path / "outputs"),
            "run_name": "test_run",
            "wandb": {
                "enabled": False,
                "project": "test",
                "entity": None,
                "run_name": None,
                "tags": [],
            },
            "env": {"name": "pong", "game_id": "ALE/Pong-v5"},
            "env_protocol": {
                "name": "paper_v4",
                "repeat_action_probability": 0.0,
                "frameskip": 1,
                "action_repeat": 4,
                "full_action_space": False,
                "noop_max": 30,
                "terminal_on_life_loss_train": True,
                "terminal_on_life_loss_eval": False,
                "eval_max_episode_frames": 18000,
            },
            "preset": {
                "name": "paper",
                "optimizer": "adam",
                "lr": 1e-4,
                "rmsprop_decay": 0.95,
                "rmsprop_momentum": 0.95,
                "rmsprop_eps": 0.01,
                "adam_eps": 1e-4,
                "epsilon_start": 1.0,
                "epsilon_end": 0.1,
                "epsilon_decay_frames": 1000,
                "reward_clip": True,
            },
            "agent": {
                "name": "dqn",
                "network": "nature",
                "double": False,
                "gamma": 0.99,
                "frame_stack": 4,
                "update_frequency": 4,
                "target_update_freq": 100,
                "replay_start_size": 8,
                "batch_size": 4,
                "max_grad_norm": None,
                "n_step": 1,
                "use_c51": False,
                "num_atoms": 51,
                "v_min": -10.0,
                "v_max": 10.0,
            },
            "replay": {
                "capacity": 100,
                "prioritized": False,
                "alpha": 0.6,
                "beta_start": 0.4,
                "beta_end": 1.0,
                "beta_decay_frames": 1000,
                "priority_eps": 1e-6,
            },
            "training": {
                "total_env_frames": 1000,
                "warmup_random_steps": 10,
                "log_every_steps": 10,
                "replay_diag_every_steps": 10,
                "light_eval_every_frames": 100,
                "full_eval_every_frames": 200,
                "checkpoint_every_frames": 200,
                "save_video_every_frames": 200,
            },
            "eval": {
                "light_episodes": 2,
                "full_episodes": 2,
                "epsilon": 0.05,
                "vector_envs": 1,
                "record_video": False,
            },
            "hardware": {"device": "cpu", "pin_memory": False, "non_blocking": False},
        }
    )
