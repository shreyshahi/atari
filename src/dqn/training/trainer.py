from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

from dqn.agent import DQNAgent
from dqn.envs.factory import make_atari_env
from dqn.evaluation.evaluator import Evaluator
from dqn.logging.metrics import MetricsAggregator
from dqn.logging.wandb_logger import WandbLogger
from dqn.replay.factory import build_replay_buffer
from dqn.training.checkpoint import CheckpointManager
from dqn.training.scheduler import EpsilonScheduler
from dqn.utils.device import resolve_device
from dqn.utils.seeding import capture_rng_states, seed_everything


class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = resolve_device(str(cfg.hardware.device))

        seed_everything(int(cfg.seed))

        self.run_dir = self._create_run_dir(cfg)
        (self.run_dir / ".hydra").mkdir(parents=True, exist_ok=True)
        (self.run_dir / ".hydra" / "config.yaml").write_text(
            OmegaConf.to_yaml(cfg), encoding="utf-8"
        )

        self.checkpoint_manager = CheckpointManager(self.run_dir)
        self.metrics = MetricsAggregator(window_size=100)

        wandb_config = None
        if bool(cfg.wandb.enabled):
            wandb_config = {
                "project": cfg.wandb.project,
                "entity": cfg.wandb.entity,
                "name": cfg.wandb.run_name,
                "tags": list(cfg.wandb.tags),
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
        self.wandb = WandbLogger(enabled=bool(cfg.wandb.enabled), config=wandb_config)

        self.train_csv = self.run_dir / "train_log.csv"
        self.eval_csv = self.run_dir / "eval_log.csv"

    def _create_run_dir(self, cfg: DictConfig) -> Path:
        run_name = cfg.run_name
        if run_name is None:
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{cfg.env.name}_{cfg.agent.name}_{cfg.preset.name}_{ts}"
        run_dir = Path(cfg.output_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _append_csv(self, path: Path, row: dict[str, Any]) -> None:
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _prefill_replay(self, env, replay, steps: int) -> np.ndarray:
        state, _ = env.reset()
        for _ in trange(steps, desc="Prefill", leave=False):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay.add(frame=state[-1], action=action, reward=reward, done=done)
            state = next_state
            if done:
                state, _ = env.reset()
        return state

    def _check_failure_gates(self, update_info: dict[str, Any]) -> None:
        if not np.isfinite(update_info["loss"]) or not np.isfinite(update_info["mean_q"]):
            raise RuntimeError("Failure gate triggered: non-finite loss/Q")

    def train(self) -> dict[str, Any]:
        env = make_atari_env(self.cfg, eval_mode=False)
        n_actions = int(env.action_space.n)

        agent = DQNAgent(self.cfg, n_actions=n_actions, device=self.device)
        replay = build_replay_buffer(self.cfg)
        evaluator = Evaluator(self.cfg)

        scheduler = EpsilonScheduler(
            start=float(self.cfg.preset.epsilon_start),
            end=float(self.cfg.preset.epsilon_end),
            decay_frames=int(self.cfg.preset.epsilon_decay_frames),
        )

        warmup = int(self.cfg.training.warmup_random_steps)
        state = self._prefill_replay(env, replay, warmup)

        total_env_frames = int(self.cfg.training.total_env_frames)
        frame_skip = 4
        total_steps = total_env_frames // frame_skip

        update_count = 0
        env_steps = warmup
        env_frames = env_steps * frame_skip
        episode_return = 0.0
        episode_length = 0
        episode_num = 0

        next_log = env_steps + int(self.cfg.training.log_every_steps)
        next_light_eval = int(self.cfg.training.light_eval_every_frames)
        next_full_eval = int(self.cfg.training.full_eval_every_frames)

        pbar = trange(env_steps, total_steps, desc="Train", leave=False)
        for _ in pbar:
            epsilon = scheduler.value(env_frames)
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay.add(frame=state[-1], action=action, reward=reward, done=done)

            state = next_state
            episode_return += reward
            episode_length += 1
            env_steps += 1
            env_frames = env_steps * frame_skip

            if done:
                self.metrics.add_episode(episode_return, episode_length)
                self._append_csv(
                    self.train_csv,
                    {
                        "step": env_steps,
                        "env_frames": env_frames,
                        "episode_number": episode_num,
                        "episode_return": episode_return,
                        "episode_length": episode_length,
                    },
                )
                episode_num += 1
                episode_return = 0.0
                episode_length = 0
                state, _ = env.reset()

            if (
                env_steps >= int(self.cfg.agent.replay_start_size)
                and env_steps % int(self.cfg.agent.update_frequency) == 0
                and replay.can_sample(int(self.cfg.agent.batch_size))
            ):
                batch = replay.sample(int(self.cfg.agent.batch_size))
                update_info = agent.update(batch)
                self._check_failure_gates(update_info)
                update_count += 1

                if bool(self.cfg.replay.prioritized):
                    replay.update_priorities(
                        indices=batch["indices"], priorities=np.abs(update_info["td_errors"])
                    )

                if update_count % int(self.cfg.agent.target_update_freq) == 0:
                    agent.sync_target_network()

                pbar.set_postfix(loss=f"{update_info['loss']:.4f}", eps=f"{epsilon:.3f}")

            if env_steps >= next_log:
                log_row = {
                    "step": env_steps,
                    "env_frames": env_frames,
                    "epsilon": epsilon,
                }
                log_row.update(self.metrics.summarize())
                self.wandb.log(log_row, step=env_steps)
                next_log += int(self.cfg.training.log_every_steps)

            if env_frames >= next_light_eval:
                result = evaluator.evaluate(
                    agent,
                    episodes=int(self.cfg.eval.light_episodes),
                    epsilon=float(self.cfg.eval.epsilon),
                    record_video=False,
                )
                eval_row = {
                    "step": env_steps,
                    "env_frames": env_frames,
                    "eval_type": "light",
                    "eval_mean_return": result.mean_return,
                    "eval_std_return": result.std_return,
                    "eval_min_return": result.min_return,
                    "eval_max_return": result.max_return,
                }
                self._append_csv(self.eval_csv, eval_row)
                self.wandb.log(eval_row, step=env_steps)
                next_light_eval += int(self.cfg.training.light_eval_every_frames)

            if env_frames >= next_full_eval:
                video_tmp = self.run_dir / "tmp_video.mp4"
                result = evaluator.evaluate(
                    agent,
                    episodes=int(self.cfg.eval.full_episodes),
                    epsilon=float(self.cfg.eval.epsilon),
                    record_video=bool(self.cfg.eval.record_video),
                    video_path=video_tmp,
                )
                eval_row = {
                    "step": env_steps,
                    "env_frames": env_frames,
                    "eval_type": "full",
                    "eval_mean_return": result.mean_return,
                    "eval_std_return": result.std_return,
                    "eval_min_return": result.min_return,
                    "eval_max_return": result.max_return,
                }
                self._append_csv(self.eval_csv, eval_row)
                self.wandb.log(eval_row, step=env_steps)

                cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
                assert isinstance(cfg_dict, dict)
                self.checkpoint_manager.save(
                    step=env_steps,
                    agent_state=agent.state_dict(),
                    rng_state=capture_rng_states(),
                    metrics=eval_row,
                    config=cfg_dict,
                    video_path=video_tmp if video_tmp.exists() else None,
                )

                next_full_eval += int(self.cfg.training.full_eval_every_frames)

            if env_steps >= total_steps:
                break

        final_eval = evaluator.evaluate(
            agent,
            episodes=int(self.cfg.eval.full_episodes),
            epsilon=float(self.cfg.eval.epsilon),
            record_video=bool(self.cfg.eval.record_video),
            video_path=self.run_dir / "tmp_video_final.mp4",
        )

        final_row = {
            "step": env_steps,
            "env_frames": env_steps * frame_skip,
            "eval_type": "final",
            "eval_mean_return": final_eval.mean_return,
            "eval_std_return": final_eval.std_return,
            "eval_min_return": final_eval.min_return,
            "eval_max_return": final_eval.max_return,
        }
        self._append_csv(self.eval_csv, final_row)

        cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
        assert isinstance(cfg_dict, dict)
        self.checkpoint_manager.save(
            step=env_steps,
            agent_state=agent.state_dict(),
            rng_state=capture_rng_states(),
            metrics=final_row,
            config=cfg_dict,
            video_path=self.run_dir / "tmp_video_final.mp4",
        )

        summary = {
            "run_dir": str(self.run_dir),
            "final_eval_mean_return": final_eval.mean_return,
            "final_eval_std_return": final_eval.std_return,
            "steps": env_steps,
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.wandb.finish()
        env.close()
        return summary
