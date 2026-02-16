from __future__ import annotations

from pathlib import Path

import torch

from dqn.training.checkpoint import CheckpointManager


def test_checkpoint_save_and_load(tmp_path):
    run_dir = tmp_path / "run"
    manager = CheckpointManager(run_dir)

    agent_state = {
        "online_net": {"w": torch.tensor([1.0])},
        "target_net": {"w": torch.tensor([2.0])},
        "optimizer": {},
    }
    rng_state = {"python_random": (3, (1, 2, 3), None), "numpy_random": None, "torch_cpu": torch.tensor([1])}
    metrics = {"eval_mean_return": 1.23}
    config = {"a": 1}

    ckpt_dir = manager.save(
        step=10,
        agent_state=agent_state,
        rng_state=rng_state,
        metrics=metrics,
        config=config,
        video_path=None,
    )

    assert (ckpt_dir / "agent.pt").exists()
    loaded = manager.load_agent_state(Path(ckpt_dir))
    assert "online_net" in loaded
