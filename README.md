# Atari DQN

Reimplementation of DeepMind's DQN (Nature 2015) with modern improvements. Two tracks:

- **Track A** — Paper-faithful: vanilla DQN, DeepMind RMSProp, uniform replay, no sticky actions
- **Track B** — Modern: Double DQN + Dueling + Prioritized Replay, Adam, sticky actions

---

## 1. Setup

```bash
uv sync --extra dev --extra dashboard
```

If torch doesn't pick up your CUDA version automatically:

```bash
# For CUDA 12.4
uv add torch --index-url https://download.pytorch.org/whl/cu124
uv sync --extra dev --extra dashboard
```

Verify GPU detection:

```bash
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

Atari ROMs are auto-installed by `ale-py`. No manual ROM download needed.

Run tests to confirm everything works:

```bash
make test
```

---

## 2. Config System

This project uses [Hydra](https://hydra.cc/) for configuration. Every run is composed from config groups:

| Group | Options | What it controls |
|---|---|---|
| `agent` | `dqn`, `modern` | Network type, Double DQN toggle, grad clipping |
| `preset` | `paper`, `modern` | Optimizer, learning rate, epsilon schedule |
| `replay` | `uniform`, `prioritized` | Replay buffer type, PER hyperparameters |
| `env` | `pong`, `breakout`, `space_invaders`, `seaquest`, `qbert` | Which Atari game |
| `env_protocol` | `paper_v4`, `modern_v5_sticky` | Sticky actions, eval episode length |
| `training` | `default` | Total frames, warmup, eval cadence |
| `eval` | `default` | Eval episodes, epsilon |
| `hardware` | `default` | Device, pin_memory |

### Defaults

With no arguments, `configs/config.yaml` defaults to Track A on Pong:

```yaml
defaults:
  - agent: dqn
  - env: pong
  - preset: paper
  - env_protocol: paper_v4
  - replay: uniform
  - training: default
  - eval: default
  - hardware: default

seed: 42
wandb:
  enabled: false
```

So `make train` with no overrides runs vanilla DQN on Pong with seed 42 and wandb off.

### Two ways to change config

**Edit the YAML files** for permanent changes. For example, to always use a different learning rate, edit `configs/preset/paper.yaml` and change `lr: 0.00025` to your value. Every run that uses `preset=paper` will pick up the change.

**Inline overrides** for one-off experiments. Pass them on the command line:

```bash
make train seed=123 preset.lr=0.001
```

Inline overrides take highest priority — they override whatever is in the YAML files, without modifying them.

### Valid track combinations

**Track A** (paper-faithful):
```
agent=dqn preset=paper replay=uniform env_protocol=paper_v4
```

**Track B** (modern):
```
agent=modern preset=modern replay=prioritized env_protocol=modern_v5_sticky
```

Do not mix across tracks (e.g., `agent=dqn preset=modern`) unless you are deliberately running an ablation and understand what you're changing.

---

## 3. Commands

The `Makefile` provides targets for all common operations. Hydra overrides pass through directly:

| Target | What it runs |
|---|---|
| `make train` | `uv run python scripts/train.py` |
| `make evaluate` | `uv run python scripts/evaluate.py` |
| `make video` | `uv run python scripts/record_video.py` |
| `make dashboard` | `uv run streamlit run dashboard/app.py` |
| `make test` | `uv run pytest -q` |
| `make lint` | `uv run ruff check .` |
| `make fmt` | `uv run ruff format .` |

### Training

**Track A (default — just run it):**

```bash
make train
```

**Track B:**

```bash
make train agent=modern preset=modern replay=prioritized env_protocol=modern_v5_sticky
```

**Different game:**

```bash
make train env=breakout
```

Replace `pong` with any of: `breakout`, `space_invaders`, `seaquest`, `qbert`.

**Override a hyperparameter:**

```bash
make train preset.lr=0.001
make train agent.batch_size=64
```

**Custom run name:**

```bash
make train run_name=my_experiment
```

**Smoke test (quick sanity check):**

```bash
make train \
  training.total_env_frames=200000 \
  training.warmup_random_steps=10000 \
  eval.full_episodes=5 \
  wandb.enabled=false
```

Runs ~200K frames (~50K agent steps) in a few minutes. Useful to verify nothing crashes before launching a full run.

**Multi-seed runs:**

```bash
for s in 1 2 3; do
  make train \
    agent=modern preset=modern replay=prioritized env=pong env_protocol=modern_v5_sticky \
    seed=$s run_name=pong_modern_seed${s}
done
```

Report median and IQR across seeds for any performance claim.

### Evaluate a checkpoint

```bash
make evaluate \
  agent=modern preset=modern env=pong env_protocol=modern_v5_sticky replay=prioritized \
  +checkpoint_path=outputs/pong_modern_seed1/checkpoints/step_12500000
```

You **must** pass the same config groups (`agent`, `preset`, `env`, `env_protocol`, `replay`) that were used during training. The checkpoint only stores model weights — the config determines the network architecture and environment setup.

This runs 30 episodes at epsilon=0.05 and prints mean/std/min/max return as JSON.

To use the `best` or `latest` checkpoint symlink:

```bash
+checkpoint_path=outputs/pong_modern_seed1/checkpoints/best
+checkpoint_path=outputs/pong_modern_seed1/checkpoints/latest
```

### Record gameplay video

```bash
make video \
  agent=modern preset=modern env=pong env_protocol=modern_v5_sticky replay=prioritized \
  +checkpoint_path=outputs/pong_modern_seed1/checkpoints/best \
  +output_video=pong_best.mp4
```

Records one episode of gameplay at epsilon=0.05. Output is an MP4 at 30 fps. Same config requirement as evaluation.

### Weights & Biases

```bash
wandb login   # one-time, paste your API key

make train \
  wandb.enabled=true \
  wandb.project=atari-dqn \
  wandb.entity=your-username \
  wandb.run_name=pong_trackA_seed42 \
  wandb.tags='[track_a,pong,seed42]'
```

Wandb is off by default (`wandb.enabled=false`). All metrics are also written to local CSV files (`train_log.csv`, `eval_log.csv`) in the run directory regardless of wandb settings.

---

## 4. What to Expect During Training

- **Prefill phase**: ~50K random transitions are collected before training starts. Takes ~10-15 seconds. You'll see a "Prefill" progress bar.
- **Training loop**: A "Train" progress bar shows agent steps. The postfix displays current `loss` and `eps` (epsilon). On an RTX 4090, expect ~70-90 minutes for a full 50M-frame run.
- **Eval checkpoints**: Every 2M frames, the trainer pauses to run 30 eval episodes and save a checkpoint with a gameplay video. Every 500K frames, a lightweight 10-episode eval runs (no video, no checkpoint).
- **Pong heuristics**: For Pong specifically, you should see score > 0 within ~500K frames and score > 18 within ~3M frames. These are rough debugging heuristics, not hard pass/fail criteria.

### Training timeline (50M frames, RTX 4090)

| Milestone | ~Time | What happens |
|---|---|---|
| 0 | 0 min | Prefill starts |
| 50K steps | ~15 sec | Prefill done, training begins |
| 500K frames | ~5 min | First lightweight eval |
| 2M frames | ~15 min | First full eval + checkpoint + video |
| 50M frames | ~70-90 min | Training complete, final eval + checkpoint |

---

## 5. Output Directory Structure

Each run creates `outputs/<run_name>/`:

```
outputs/pong_modern_seed1/
  .hydra/config.yaml          # Exact resolved config (all groups merged)
  train_log.csv               # Per-episode: step, env_frames, episode_return, episode_length
  update_log.csv              # Every 1K steps: loss/Q/grad_norm + rolling stats
  eval_log.csv                # Eval results: step, type (light/full/final), mean/std/min/max return
  summary.json                # End-of-run summary
  checkpoints/
    catalog.json              # Index of all checkpoints with eval scores
    step_002000000/
      agent.pt                # Model weights + optimizer state
      rng_states.pt           # RNG states for exact reproducibility
      metrics.json            # Eval scores at this checkpoint
      video.mp4               # One episode of gameplay (~2MB)
    step_004000000/
      ...
    best -> step_010000000/   # Symlink to highest eval score
    latest -> step_012500000/ # Symlink to most recent
```

### Key files

| File | When written | Contents |
|---|---|---|
| `train_log.csv` | Every episode end | `step`, `env_frames`, `episode_number`, `episode_return`, `episode_length` |
| `update_log.csv` | Every 1K steps | `step`, `env_frames`, `epsilon`, `loss`, `mean_q`, `max_q`, `grad_norm`, rolling means |
| `eval_log.csv` | Every eval (500K and 2M frame intervals) | `step`, `env_frames`, `eval_type`, `eval_mean_return`, `eval_std_return`, `eval_min_return`, `eval_max_return` |
| `.hydra/config.yaml` | Run start | Full merged config — use this to see exactly what parameters a run used |
| `catalog.json` | Every full eval | List of `{step, path, eval_score, timestamp}` for all checkpoints |
| `agent.pt` | Every full eval | `online_net`, `target_net`, `optimizer` state dicts |

The replay buffer is NOT checkpointed (it's ~7GB). On resume, it must be re-filled from scratch (~12 seconds).

---

## 6. Ablations

To measure the incremental contribution of each improvement, run these configs on the same game and seed:

```bash
GAME=pong
SEED=42

# 1. Vanilla DQN (baseline)
make train agent=dqn preset=paper replay=uniform env=$GAME env_protocol=paper_v4 seed=$SEED run_name=${GAME}_vanilla_s${SEED}

# 2. + Double DQN only
make train agent.double=true preset=paper replay=uniform env=$GAME env_protocol=paper_v4 seed=$SEED run_name=${GAME}_double_s${SEED}

# 3. + Double DQN + PER
make train agent.double=true preset=paper replay=prioritized env=$GAME env_protocol=paper_v4 seed=$SEED run_name=${GAME}_double_per_s${SEED}

# 4. Full modern (Double + Dueling + PER)
make train agent=modern preset=modern replay=prioritized env=$GAME env_protocol=modern_v5_sticky seed=$SEED run_name=${GAME}_modern_s${SEED}
```

Note: ablation #4 uses a different `env_protocol` (sticky actions) and `preset` (Adam), so it's not a pure single-variable ablation. For a pure comparison, keep `preset=paper env_protocol=paper_v4` and only toggle `agent.network=dueling`:

```bash
# 3b. + Double DQN + PER + Dueling (paper preset, no sticky actions)
make train agent.double=true agent.network=dueling preset=paper replay=prioritized env=$GAME env_protocol=paper_v4 seed=$SEED run_name=${GAME}_double_per_dueling_s${SEED}
```

---

## 7. Adding a New Game

Create a config file at `configs/env/<game>.yaml`:

```yaml
name: <game>
game_id: ALE/<GameName>-v5
```

The `game_id` must match an ALE environment name. Find available games:

```bash
uv run python -c "import ale_py; print([e for e in ale_py.roms.get_all_rom_ids()])"
```

Then train:

```bash
make train env=<game>
```

---

## 8. Dashboard

```bash
make dashboard
```

The dashboard reads from `outputs/` to find all completed runs. It has 5 pages:

| Page | What it shows |
|---|---|
| **Training Curves** | Episode return and episode length over training steps. Select a run directory, choose metric, adjust smoothing slider. Data comes from `train_log.csv`. |
| **Video Player** | Select a run, scrub through checkpoints with a slider, play the `video.mp4` for each checkpoint. Shows eval score alongside the video. |
| **Game Comparison** | Select multiple runs, see overlaid eval curves and a score table. Useful for comparing across games or agent variants. |
| **Ablation Comparator** | Select runs from an ablation series (vanilla → +Double → +PER → +Dueling). Shows incremental improvement of each addition. |
| **Config Diff Viewer** | Select two runs, see their configs side-by-side. Useful for understanding why two runs produced different results. |

---

## 9. Full Config Reference

### `agent/dqn.yaml` (Track A default)

| Key | Default | Description |
|---|---|---|
| `network` | `nature` | Network architecture: `nature` or `dueling` |
| `double` | `false` | Enable Double DQN target computation |
| `gamma` | `0.99` | Discount factor |
| `frame_stack` | `4` | Number of stacked frames as input |
| `update_frequency` | `4` | Gradient update every N agent steps |
| `target_update_freq` | `10000` | Sync target network every N gradient steps |
| `replay_start_size` | `50000` | Min buffer size before training starts |
| `batch_size` | `32` | Training batch size |
| `max_grad_norm` | `null` | Gradient clipping (null = disabled) |

### `agent/modern.yaml` (Track B default)

Same keys as above, with: `network=dueling`, `double=true`, `max_grad_norm=10.0`.

### `preset/paper.yaml` (Track A)

| Key | Default | Description |
|---|---|---|
| `optimizer` | `deepmind_rmsprop` | Optimizer: `deepmind_rmsprop`, `adam`, or `torch_rmsprop` |
| `lr` | `0.00025` | Learning rate |
| `rmsprop_decay` | `0.95` | RMSProp decay (alpha) |
| `rmsprop_momentum` | `0.95` | RMSProp momentum |
| `rmsprop_eps` | `0.01` | RMSProp epsilon |
| `epsilon_start` | `1.0` | Initial exploration epsilon |
| `epsilon_end` | `0.1` | Final exploration epsilon |
| `epsilon_decay_frames` | `1000000` | Frames over which epsilon anneals |
| `reward_clip` | `true` | Clip rewards to {-1, 0, +1} |

### `preset/modern.yaml` (Track B)

Same structure. Key differences: `optimizer=adam`, `lr=6.25e-5`, `adam_eps=1.5e-4`, `epsilon_end=0.01`.

### `env_protocol/paper_v4.yaml`

| Key | Default | Description |
|---|---|---|
| `repeat_action_probability` | `0.0` | Sticky action probability (0 = deterministic) |
| `frameskip` | `1` | ALE-level frameskip (action repeat is handled by MaxAndSkipEnv wrapper at 4x) |
| `full_action_space` | `false` | Use minimal action set |
| `noop_max` | `30` | Max random no-ops on reset |
| `terminal_on_life_loss_train` | `true` | Treat life loss as episode end during training |
| `terminal_on_life_loss_eval` | `false` | Do NOT treat life loss as episode end during eval |
| `eval_max_episode_frames` | `18000` | Max emulator frames per eval episode (18K = 5 min at 60 Hz) |

### `env_protocol/modern_v5_sticky.yaml`

Same structure. Key differences: `repeat_action_probability=0.25`, `eval_max_episode_frames=108000` (30 min).

### `training/default.yaml`

| Key | Default | Description |
|---|---|---|
| `total_env_frames` | `50000000` | Total emulator frames (50M = 12.5M agent steps) |
| `warmup_random_steps` | `50000` | Random transitions before training begins |
| `log_every_steps` | `1000` | Log metrics every N agent steps |
| `light_eval_every_frames` | `500000` | Lightweight eval (10 episodes) every N frames |
| `full_eval_every_frames` | `2000000` | Full eval (30 episodes + video + checkpoint) every N frames |

### `eval/default.yaml`

| Key | Default | Description |
|---|---|---|
| `light_episodes` | `10` | Episodes per lightweight eval |
| `full_episodes` | `30` | Episodes per full eval |
| `epsilon` | `0.05` | Exploration during eval |
| `record_video` | `true` | Record video during full eval |

### `replay/prioritized.yaml`

| Key | Default | Description |
|---|---|---|
| `capacity` | `1000000` | Replay buffer size (1M transitions) |
| `prioritized` | `true` | Enable prioritized experience replay |
| `alpha` | `0.6` | PER priority exponent |
| `beta_start` | `0.4` | Initial IS weight correction |
| `beta_end` | `1.0` | Final IS weight correction |
| `beta_decay_frames` | `50000000` | Frames over which beta anneals |
| `priority_eps` | `0.000001` | Small constant added to TD error for priority |

### Top-level (`config.yaml`)

| Key | Default | Description |
|---|---|---|
| `seed` | `42` | Random seed for reproducibility |
| `output_dir` | `outputs` | Base directory for run outputs |
| `run_name` | `null` | Custom run name (auto-generated if null) |
| `wandb.enabled` | `false` | Enable Weights & Biases logging |
| `wandb.project` | `atari-dqn` | Wandb project name |
| `wandb.entity` | `null` | Wandb username/team |

---

## 10. Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: ale_py` | ROMs not installed | `uv sync` should handle this. If not: `uv run ale-import-roms --import-from-pkg` |
| Training seems stuck (no loss decrease) | Normal for first ~100K frames | Wait. DQN takes time to start learning. Check again after 500K frames. |
| `RuntimeError: Failure gate triggered: non-finite loss/Q` | NaN in training | Check learning rate and epsilon config. This usually means a hyperparameter mismatch. |
| OOM on GPU | Replay buffer is on CPU, model is small (~1.7M params). GPU OOM is unlikely. | If it happens: reduce `agent.batch_size` |
| Videos are black/empty | `render_mode` not set | Ensure `eval.record_video=true`. The evaluator handles render_mode internally. |
| Eval scores are clipped to {-1,0,+1} | Using training env for eval | Evaluation uses `eval_mode=True` which disables reward clipping. If you see clipped scores, check your eval script. |
| `catalog.json` shows no checkpoints | Training hasn't reached 2M frames yet | Full eval + checkpoint happens every `full_eval_every_frames` (default 2M). For faster checkpoints: `training.full_eval_every_frames=500000` |
| Runs not appearing in dashboard | Wrong `outputs/` path or run incomplete | Dashboard scans `outputs/`. Ensure runs have `eval_log.csv` and `checkpoints/`. |

---

## Two-Track Approach

**Track A — Paper-Faithful Reproduction**
- DeepMind-style RMSProp (custom implementation; PyTorch's RMSprop differs from the paper's variant)
- Explicit optimizer fields: `lr=2.5e-4`, `decay=0.95`, `momentum=0.95`, `eps=0.01`
- ε: 1.0 → 0.1 over 1M frames
- Uniform replay, standard DQN target
- Reward clipping, no sticky actions (`repeat_action_probability=0.0`, paper-style v4 protocol)
- Purpose: validate our infrastructure against known results

**Track B — Modern Agent**
- Adam (lr=6.25e-5, eps=1.5e-4), Huber loss, gradient clipping
- Double DQN + Dueling Architecture + Prioritized Replay
- ε: 1.0 → 0.01
- Future: n-step returns, C51 distributional RL
- Purpose: best performance using the three most impactful post-DQN improvements (Double, Dueling, PER)

Rule: never mix Track A and B settings in one claim. Each run's track is logged in config.

---

## Algorithm Roadmap (Incremental)

1. **DQN baseline** — target network, uniform replay, reward clipping (Track A)
2. **+ Double DQN** — decouple action selection from evaluation
3. **+ Prioritized Replay** — sample proportional to |TD error|, IS weight correction
4. **+ Dueling architecture** — separate V(s) and A(s,a) streams
5. **+ n-step returns (n=3)** — faster credit assignment (future phase)
6. **+ C51 distributional RL** — model full return distribution (future phase)
7. **+ Noisy Nets** — learned exploration, replace ε-greedy (optional future)

Deliverables: `baseline_dqn` (step 1) and `modern_dqn` (steps 2-4), with a clean path to Rainbow-lite (steps 5-6).

---

## Project Structure

```
atari/
├── paper/                              # Original Nature paper (exists)
├── pyproject.toml                      # uv project, all deps including PyTorch via uv
├── Makefile                            # train, eval, dashboard, test, lint targets
│
├── configs/                            # Hydra YAML configs
│   ├── config.yaml                     # Top-level defaults
│   ├── agent/
│   │   ├── dqn.yaml                   # Vanilla DQN (Track A default)
│   │   └── modern.yaml                # Double+Dueling+PER (Track B default)
│   ├── env/
│   │   ├── pong.yaml
│   │   ├── breakout.yaml
│   │   ├── space_invaders.yaml
│   │   ├── seaquest.yaml
│   │   └── qbert.yaml
│   ├── preset/
│   │   ├── paper.yaml                 # Track A: DeepMind RMSProp, ε→0.1, paper settings
│   │   └── modern.yaml               # Track B: Adam, ε→0.01, Rainbow-tuned
│   ├── env_protocol/
│   │   ├── paper_v4.yaml             # No sticky actions; paper-style Atari protocol
│   │   └── modern_v5_sticky.yaml     # Sticky actions (0.25) + modern ALE protocol
│   ├── replay/
│   │   ├── uniform.yaml
│   │   └── prioritized.yaml
│   ├── training/
│   │   └── default.yaml
│   ├── eval/
│   │   └── default.yaml
│   └── hardware/
│       └── default.yaml
│
├── src/dqn/
│   ├── __init__.py
│   ├── networks/
│   │   ├── __init__.py
│   │   ├── encoder.py                 # ConvEncoder (shared 3-layer CNN)
│   │   ├── nature_dqn.py             # Standard FC head → Q(s,a)
│   │   └── dueling.py                # V(s) + A(s,a) split head
│   ├── agent.py                       # Single DQNAgent with feature toggles
│   ├── replay/
│   │   ├── __init__.py
│   │   ├── uniform.py                # UniformReplayBuffer (uint8 circular)
│   │   ├── sum_tree.py               # SumTree for O(log n) PER
│   │   ├── prioritized.py            # PrioritizedReplayBuffer + IS weights
│   │   └── factory.py                # build_replay_buffer(cfg)
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── wrappers.py               # All 7 Atari wrappers
│   │   └── factory.py                # make_atari_env(cfg), make_eval_env(cfg)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                # Main training loop
│   │   ├── scheduler.py              # Epsilon linear annealing
│   │   └── checkpoint.py             # Save/load/catalog + failure gates
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py              # Eval protocol + video recording
│   ├── logging/
│   │   ├── __init__.py
│   │   ├── metrics.py                # Windowed stats + replay diagnostics
│   │   └── wandb_logger.py           # W&B integration
│   └── utils/
│       ├── __init__.py
│       ├── seeding.py                # Deterministic seeding (torch, np, gym, RNG state)
│       ├── device.py
│       └── optimizers.py             # DeepMindRMSprop (custom, differs from torch.optim.RMSprop)
│
├── scripts/
│   ├── train.py                       # Hydra entry point
│   ├── evaluate.py                    # Eval from checkpoint
│   └── record_video.py               # Record gameplay from checkpoint
│
├── dashboard/
│   ├── app.py                         # Streamlit main app
│   ├── pages/
│   │   ├── training_curves.py        # Loss, return, Q-value, epsilon
│   │   ├── video_player.py           # Checkpoint scrubber + gameplay
│   │   ├── comparison.py             # Cross-game, cross-agent scores
│   │   ├── ablations.py              # Incremental improvement comparison
│   │   └── config_diff.py            # Side-by-side config diff between runs
│   └── utils.py
│
└── tests/
    ├── conftest.py
    ├── test_networks.py
    ├── test_replay.py
    ├── test_sum_tree.py
    ├── test_wrappers.py
    ├── test_agent.py
    ├── test_trainer.py
    └── test_checkpoint.py
```
