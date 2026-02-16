# Atari DQN User Manual

This project trains Atari agents with a configurable DQN pipeline:

- `Track A` (paper-style): vanilla DQN + DeepMind-style RMSProp + no sticky actions.
- `Track B` (modern): Double DQN + Dueling + Prioritized Replay + sticky actions.

This README explains how to use the system, not how internal code is written.

## 1. Prerequisites

- Python 3.10+
- `uv` installed
- NVIDIA GPU optional but recommended for training

## 2. Install and Validate

From repo root:

```bash
uv sync --extra dev --extra dashboard
uv run pytest -q
```

If PyTorch wheel resolution fails for your CUDA stack:

```bash
uv add torch --index-url https://download.pytorch.org/whl/cu124
uv sync --extra dev --extra dashboard
```

## 3. Run Training

### Paper-style baseline (Track A)

```bash
uv run python scripts/train.py \
  agent=dqn preset=paper replay=uniform env=pong env_protocol=paper_v4
```

### Modern baseline (Track B)

```bash
uv run python scripts/train.py \
  agent=modern preset=modern replay=prioritized env=pong env_protocol=modern_v5_sticky
```

### Quick smoke run

```bash
uv run python scripts/train.py \
  training.total_env_frames=200000 \
  training.warmup_random_steps=10000 \
  eval.full_episodes=5 \
  wandb.enabled=false
```

## 4. Evaluate a Checkpoint

Run full evaluation for an existing checkpoint:

```bash
uv run python scripts/evaluate.py \
  +checkpoint_path=outputs/<run_name>/checkpoints/step_<N>
```

Tip: Use the same config family (`agent/preset/env/env_protocol`) that was used during training.

## 5. Record Gameplay Video

```bash
uv run python scripts/record_video.py \
  +checkpoint_path=outputs/<run_name>/checkpoints/step_<N> \
  +output_video=demo.mp4
```

## 6. Open the Dashboard

```bash
uv run streamlit run dashboard/app.py
```

Pages:

- `Training Curves`: plots episode metrics from `train_log.csv`.
- `Video Player`: plays `video.mp4` from saved checkpoints.
- `Game Comparison`: compares best full-eval return across runs.
- `Ablation Comparator`: overlays eval progress for selected runs.
- `Config Diff Viewer`: side-by-side run config view.

## 7. Understand Output Files

Runs are saved under `outputs/<run_name>/`.

Important files:

- `train_log.csv`: episode-level training progress.
- `eval_log.csv`: periodic eval results (`light`, `full`, `final`).
- `.hydra/config.yaml`: exact resolved config for reproducibility.
- `checkpoints/catalog.json`: index of checkpoints and scores.
- `checkpoints/step_<N>/agent.pt`: model + optimizer state.
- `checkpoints/step_<N>/metrics.json`: eval metrics for that checkpoint.
- `checkpoints/step_<N>/video.mp4`: recorded gameplay for that checkpoint.
- `summary.json`: end-of-run summary.

## 8. Interpret Results

Use these rules when reading outcomes:

- Prioritize `full` and `final` eval scores over train episode returns.
- Compare runs only when `env_protocol` is the same.
- Track `env_frames` for sample-efficiency comparisons.
- Use multi-seed medians/IQR for any final claim.
- Treat single-run thresholds (for example Pong milestones) as heuristics, not hard truth.

## 9. Common Issues

- `PyPI/DNS` errors during `uv sync`: network issue on current machine/session.
- `CUDA init warning` during tests: often harmless in CPU-only test execution.
- No videos in dashboard: ensure full eval/checkpoint completed and `eval.record_video=true`.
- Wrong eval behavior: check that `env_protocol` matches your intended track.

## 10. Useful Short Commands

```bash
make test
make train
make evaluate
make video
make dashboard
```

These map to the same `uv run ...` commands above.
