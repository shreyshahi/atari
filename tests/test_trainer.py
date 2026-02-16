from __future__ import annotations

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
