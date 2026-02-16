from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def _symlink_force(target: Path, link_name: Path) -> None:
    if link_name.exists() or link_name.is_symlink():
        link_name.unlink()
    link_name.symlink_to(target.name)


class CheckpointManager:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.checkpoints_dir = run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.checkpoints_dir / "catalog.json"
        self._catalog: dict[str, Any] = {"items": []}
        if self.catalog_path.exists():
            self._catalog = json.loads(self.catalog_path.read_text())

    def save(
        self,
        step: int,
        agent_state: dict[str, Any],
        rng_state: dict[str, Any],
        metrics: dict[str, Any],
        config: dict[str, Any],
        video_path: Path | None = None,
    ) -> Path:
        ckpt_dir = self.checkpoints_dir / f"step_{step:09d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.save(agent_state, ckpt_dir / "agent.pt")
        torch.save(rng_state, ckpt_dir / "rng_states.pt")

        with (ckpt_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        with (ckpt_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        if video_path is not None and video_path.exists():
            os.replace(video_path, ckpt_dir / "video.mp4")

        self._catalog["items"].append(
            {
                "step": step,
                "path": str(ckpt_dir),
                "eval_score": metrics.get("eval_mean_return"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.catalog_path.write_text(json.dumps(self._catalog, indent=2), encoding="utf-8")

        _symlink_force(ckpt_dir, self.checkpoints_dir / "latest")
        self._update_best_link()
        return ckpt_dir

    def _update_best_link(self) -> None:
        if not self._catalog["items"]:
            return
        best = max(self._catalog["items"], key=lambda item: item.get("eval_score", float("-inf")))
        target = Path(best["path"])
        _symlink_force(target, self.checkpoints_dir / "best")

    def load_agent_state(self, checkpoint_path: Path) -> dict[str, Any]:
        try:
            return torch.load(checkpoint_path / "agent.pt", map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(checkpoint_path / "agent.pt", map_location="cpu")
