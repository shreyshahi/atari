from __future__ import annotations

import hydra
from omegaconf import DictConfig

from dqn.training.trainer import Trainer


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    trainer = Trainer(cfg)
    summary = trainer.train()
    print(summary)


if __name__ == "__main__":
    main()
