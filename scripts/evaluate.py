from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from dqn.agent import DQNAgent
from dqn.envs.factory import make_atari_env
from dqn.evaluation.evaluator import Evaluator
from dqn.utils.device import resolve_device
from dqn.utils.seeding import seed_everything


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    checkpoint_path = cfg.get("checkpoint_path")
    if checkpoint_path is None:
        raise ValueError("Provide checkpoint path: +checkpoint_path=outputs/.../checkpoints/step_x")

    seed_everything(int(cfg.seed))
    device = resolve_device(str(cfg.hardware.device))

    env = make_atari_env(cfg, eval_mode=True)
    agent = DQNAgent(cfg, n_actions=int(env.action_space.n), device=device)
    env.close()

    try:
        ckpt = torch.load(Path(checkpoint_path) / "agent.pt", map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(Path(checkpoint_path) / "agent.pt", map_location=device)
    agent.load_state_dict(ckpt)

    evaluator = Evaluator(cfg)
    result = evaluator.evaluate(
        agent,
        episodes=int(cfg.eval.full_episodes),
        epsilon=float(cfg.eval.epsilon),
        record_video=False,
    )

    payload = {
        "mean_return": result.mean_return,
        "std_return": result.std_return,
        "min_return": result.min_return,
        "max_return": result.max_return,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
