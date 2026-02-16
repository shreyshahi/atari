from __future__ import annotations

from typing import Any


class WandbLogger:
    def __init__(self, enabled: bool, config: dict[str, Any] | None = None):
        self.enabled = enabled
        self._wandb = None
        if enabled:
            import wandb

            self._wandb = wandb
            self._wandb.init(**(config or {}))

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled or self._wandb is None:
            return
        self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.finish()
