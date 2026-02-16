from __future__ import annotations

import numpy as np
import gymnasium as gym

from dqn.training.scheduler import EpsilonScheduler
from dqn.training.trainer import Trainer


def test_epsilon_scheduler():
    sched = EpsilonScheduler(start=1.0, end=0.1, decay_frames=100)
    assert abs(sched.value(0) - 1.0) < 1e-9
    assert sched.value(100) == 0.1
    assert sched.value(200) == 0.1


def test_trainer_initialization(cfg):
    trainer = Trainer(cfg)
    assert trainer.run_dir.exists()


class DummyStackEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(4)
        self._step = 0
        self.last_frame = np.zeros((84, 84, 3), dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        obs = self.np_random.integers(0, 256, size=(4, 84, 84), dtype=np.uint8)
        return obs, {"lives": 5}

    def step(self, action):
        self._step += 1
        obs = self.np_random.integers(0, 256, size=(4, 84, 84), dtype=np.uint8)
        self.last_frame = self.np_random.integers(0, 256, size=(84, 84, 3), dtype=np.uint8)
        reward = float((action % 3) - 1)
        done = self._step >= 10
        return obs, reward, done, False, {"lives": max(0, 5 - self._step // 2)}

    def render(self):
        return self.last_frame


def test_trainer_smoke_train(cfg, monkeypatch):
    cfg.training.total_env_frames = 80
    cfg.training.warmup_random_steps = 8
    cfg.training.light_eval_every_frames = 40
    cfg.training.full_eval_every_frames = 80
    cfg.eval.light_episodes = 1
    cfg.eval.full_episodes = 1
    cfg.eval.record_video = False

    monkeypatch.setattr("dqn.training.trainer.make_atari_env", lambda *_args, **_kwargs: DummyStackEnv())
    monkeypatch.setattr("dqn.evaluation.evaluator.make_atari_env", lambda *_args, **_kwargs: DummyStackEnv())

    trainer = Trainer(cfg)
    summary = trainer.train()
    assert "final_eval_mean_return" in summary
