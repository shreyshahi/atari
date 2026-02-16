# Atari DQN Reimplementation — Merged Plan

## Context

Reimplementing the DeepMind DQN paper ("Human-level control through deep reinforcement learning", Mnih et al., Nature 2015) from scratch with modern improvements. The paper introduced Deep Q-Networks that learn 49 Atari games from raw pixels. We rebuild this with a modular, feature-toggled architecture that supports both a paper-faithful baseline (Track A) and a modernized agent (Track B), with a clear incremental path toward Rainbow.

**Hardware**: RTX 4090 (24GB), 64GB RAM, Xeon E5-2620 v4 (16 threads), CUDA 12.9
**Estimated training time**: ~70-90 min per game (50M frames)

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

Key structural change vs our original plan: **single `agent.py` with feature toggles** instead of separate agent classes. Algorithm features (double, dueling, PER) are config flags, not class forks.

---

## Core Design: Toggle-Based Agent

Instead of an inheritance hierarchy (DQNAgent → DoubleDQNAgent), we use a single `DQNAgent` class with composable features controlled by config:

```python
class DQNAgent:
    """
    Unified DQN agent. Features controlled by cfg flags:
      - cfg.agent.double: bool    → Double DQN target computation
      - cfg.agent.network: str    → "nature" | "dueling"
      - cfg.agent.replay: str     → "uniform" | "prioritized"

    This avoids combinatorial class explosion when adding n-step, C51, etc.
    """
    def __init__(self, cfg, n_actions: int, device: torch.device):
        # Build network based on config
        NetworkClass = DuelingDQN if cfg.agent.network == "dueling" else NatureDQN
        self.online_net = NetworkClass(n_actions).to(device)
        self.target_net = NetworkClass(n_actions).to(device)
        self.sync_target_network()
        self.target_net.eval()

        # Build optimizer based on preset
        if cfg.preset.optimizer == "adam":
            self.optimizer = Adam(self.online_net.parameters(),
                                  lr=cfg.preset.lr, eps=cfg.preset.adam_eps)
        elif cfg.preset.optimizer == "deepmind_rmsprop":
            # Custom DeepMind-style RMSProp (differs from torch.optim.RMSprop)
            self.optimizer = DeepMindRMSprop(
                self.online_net.parameters(),
                lr=cfg.preset.lr, decay=cfg.preset.rmsprop_decay,
                momentum=cfg.preset.rmsprop_momentum, eps=cfg.preset.rmsprop_eps)
        else:  # torch_rmsprop (debug/ablation only, NOT paper-faithful)
            self.optimizer = RMSprop(self.online_net.parameters(),
                                     lr=cfg.preset.lr, alpha=cfg.preset.rmsprop_decay,
                                     momentum=cfg.preset.rmsprop_momentum,
                                     eps=cfg.preset.rmsprop_eps)

        self.use_double = cfg.agent.double
        self.gamma = cfg.agent.gamma
        self.grad_clip_norm = cfg.agent.grad_clip_norm  # None for Track A

    def compute_target_q(self, next_states, rewards, dones):
        if self.use_double:
            # Online net selects, target net evaluates
            best_actions = self.online_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
        else:
            # Vanilla: target net both selects and evaluates
            next_q = self.target_net(next_states).max(dim=1).values
        return rewards + self.gamma * next_q * (~dones).float()

    def update(self, batch: dict) -> dict:
        """Huber loss, optional gradient clipping, optional PER IS weights."""
        # ... gather Q for taken actions
        # ... compute targets
        # ... weighted Huber loss (weights=1 for uniform, IS weights for PER)
        # ... backward + optional clip_grad_norm + step
        # Returns: loss, mean_q, max_q, td_errors, grad_norm
```

This makes adding n-step or C51 later a matter of adding another flag, not a new class.

---

## Network Architectures

### `src/dqn/networks/encoder.py` — Shared Conv Encoder
```
Input: (batch, 4, 84, 84) uint8 → normalized to [0,1] in forward()
Conv2d(4, 32, 8, stride=4) + ReLU
Conv2d(32, 64, 4, stride=2) + ReLU
Conv2d(64, 64, 3, stride=1) + ReLU
Flatten → 7×7×64 = 3136 features
```
Kaiming initialization. Normalization happens here (uint8/255.0) so the replay buffer stays in uint8.

### `src/dqn/networks/nature_dqn.py` — Standard Head
```
encoder(3136) → Linear(3136, 512) + ReLU → Linear(512, n_actions)
```
~1.69M parameters total. Used for DQN and Double DQN.

### `src/dqn/networks/dueling.py` — Dueling Head
```
encoder(3136) → Value:     Linear(3136, 512) + ReLU → Linear(512, 1)
              → Advantage: Linear(3136, 512) + ReLU → Linear(512, n_actions)
Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
```
Mean-centering ensures identifiability.

---

## Replay Buffers

### Uniform Replay (`src/dqn/replay/uniform.py`)
- Pre-allocated circular numpy arrays: `frames[capacity+4, 84, 84]` uint8, `actions[capacity]` int8, `rewards[capacity]` float32, `dones[capacity]` bool
- Frame stacks assembled at sample time: index `[i-3:i+1]`, handling episode boundaries (zero-fill across resets)
- **Memory**: 1M transitions ≈ 7.07GB (vs 28GB if storing 4×84×84 per transition)
- Returns dict: `states (B,4,84,84) uint8`, `actions (B,) int64`, `rewards (B,) float32`, `next_states (B,4,84,84) uint8`, `dones (B,) bool`

### Prioritized Replay (`src/dqn/replay/prioritized.py`)
- Same frame storage as uniform
- SumTree (`src/dqn/replay/sum_tree.py`): binary tree, float64 for precision, O(log n) sample + update
- Priority: `p_i = (|δ_i| + ε)^α` where α=0.6, ε=1e-6
- IS weights: `w_i = (N · P(i))^(-β) / max(w)`, β annealed 0.4→1.0 over training
- Stratified sampling: divide `[0, total_priority]` into `batch_size` segments
- New transitions get max priority (always sampled at least once)
- Returns same dict as uniform plus: `weights (B,) float32`, `indices (B,) int` for priority update

---

## Environment Wrappers (`src/dqn/envs/wrappers.py`)

Applied in this order:

1. **NoopResetEnv**: 1-30 random no-ops on reset (diverse starting states)
2. **MaxAndSkipEnv**: Repeat action 4×, return max of last 2 frames (handles sprite flickering)
3. **EpisodicLifeEnv**: done=True on life loss *(training only)*
4. **FireResetEnv**: Auto-press FIRE after reset (for Breakout, Space Invaders)
5. **WarpFrame**: Grayscale + resize to 84×84 via cv2
6. **ClipRewardEnv**: sign(reward) ∈ {-1, 0, +1} *(training only)*
7. **FrameStack**: Stack last 4 frames, returns uint8

Training env: all 7 wrappers.
Eval env: NoopReset + MaxAndSkip + FireReset + WarpFrame + FrameStack (no EpisodicLife, no ClipReward → raw scores).

## Environment Protocol Locking (Reproducibility-Critical)

Each run selects `env_protocol=paper_v4` or `env_protocol=modern_v5_sticky`. The resolved protocol is logged with every checkpoint.

Required protocol fields in config:
- `game_id` (exact env id string, e.g. `ALE/Pong-v5`)
- `frameskip`, `repeat_action_probability` (0.0 for Track A, 0.25 for Track B)
- `full_action_space` (false — minimal action set)
- `terminal_on_life_loss_train`, `terminal_on_life_loss_eval`

Rules:
- Track A uses `repeat_action_probability=0.0` (deterministic, paper-style).
- Track B uses `repeat_action_probability=0.25` (sticky actions, modern protocol).
- Results are never compared across protocols unless differences are explicitly called out.

---

## Training Pipeline

```
1. INIT
   Hydra composes config (agent + env + preset + replay + training + eval + hardware)
   Seed everything (torch, numpy, random, gym) + save RNG states
   Create training env → build agent → build replay buffer → init wandb

2. PRE-FILL REPLAY (50K random transitions, ~12 seconds)
   Random actions, store transitions, no gradient updates.

3. TRAINING LOOP (12.5M agent steps = 50M frames)
   For each step:
   a) ε-greedy action selection (ε annealed per schedule)
   b) env.step() → store (frame, action, reward, done)
   c) Every 4 steps: sample batch → agent.update()
      - For PER: stratified sample with IS weights, update priorities after
   d) Every 10K gradient steps: sync target network
   e) Every 500K frames: lightweight eval (10 episodes, fast feedback)
   f) Every 2M frames: full eval (30 episodes + video recording + checkpoint)
   g) Every 1K steps: log metrics to wandb

4. FAILURE GATES (checked continuously)
   - NaN/inf in loss or Q-values → halt training, log alert
   - Persistent NaN/inf gradients (>100 consecutive updates) → halt training
   - Single large grad norm spike (>1000) → warning + log, NOT immediate halt
   - Eval score drops >50% from best for 3 consecutive evals → warning + auto-tag run for inspection

5. FINAL: full eval + final checkpoint + wandb.finish()
```

---

## Evaluation Protocol

### Two-Tier Cadence
- **Lightweight** (every 500K frames): 10 episodes, ε=0.05, log mean return. Fast feedback.
- **Full** (every 2M frames): 30 episodes, ε=0.05, raw scores (no clipping). Record 1-episode video. Save checkpoint.
  - Track A: up to 18,000 emulator frames per episode (= 4,500 agent steps with frame_skip=4, = 5 min at 60 Hz, matching the Nature paper's evaluation protocol)
  - Track B: up to 108,000 emulator frames per episode (= 27,000 agent steps with frame_skip=4, = 30 min at 60 Hz, modern convention for stronger agents)

### Statistical Reporting
- Dev runs: 1 seed for rapid iteration
- Candidate runs: 2 seeds
- Final claims: 3+ seeds, report **median and IQR**, not just mean
- Human-normalized score: `100 × (agent - random) / (human - random)`

---

## Metrics to Log

### Training (every 1K steps to wandb)
- `loss`, `mean_q`, `max_q`, `epsilon`, `grad_norm`
- Throughput: `env_fps`, `learner_updates_per_sec`
- `env_frames` and `learner_updates` as separate x-axes

### Replay Diagnostics (every 10K steps)
- Priority distribution stats (mean, std, max) for PER
- Sample age distribution (how old are sampled transitions)
- IS weight stats (mean, std, max)

### Evaluation
- `eval_mean_return`, `eval_std_return`, `eval_min`, `eval_max`
- Episode length distribution
- Video as `wandb.Video`

### Per-Episode (on episode end)
- `episode_return`, `episode_length`, `episode_number`

---

## Checkpointing (`src/dqn/training/checkpoint.py`)

```
outputs/{run_name}/
  .hydra/                      # Hydra auto-saved config
  checkpoints/
    catalog.json               # Index: step → {eval_score, path, timestamp}
    step_000000000/
      agent.pt                 # online_net, target_net, optimizer, step, epsilon
      rng_states.pt            # torch, numpy, python random, CUDA RNG states
      metrics.json             # eval scores at this checkpoint
      video.mp4                # 1 episode gameplay recording (~2MB)
    step_002000000/
      ...
    best -> step_010000000/    # symlink
    latest -> step_012500000/  # symlink
  train_log.csv                # Local backup of training metrics
  eval_log.csv                 # Local backup of eval metrics
```

- Replay buffer NOT saved (7GB; re-fill takes 12 sec)
- RNG states ARE saved (for exact reproducibility on resume)
- ~12MB per checkpoint × ~25 checkpoints = ~300MB per run

---

## Hyperparameters

### Track A — Paper Preset (`configs/preset/paper.yaml`)
| Parameter | Value | Source |
|---|---|---|
| optimizer | deepmind_rmsprop | Paper-faithful (custom impl) |
| lr | 2.5e-4 | Paper |
| rmsprop_decay | 0.95 | Paper |
| rmsprop_momentum | 0.95 | Paper |
| rmsprop_eps | 0.01 | Paper |
| epsilon_start → end | 1.0 → 0.1 over 1M frames | Paper |
| grad_clip_norm | null (disabled) | Paper |
| loss | huber (equivalent to paper's error clipping) | Paper |

Note: `torch_rmsprop` is available for debug/ablation but must not be labeled as paper-faithful reproduction.

**DeepMind RMSProp update rule** (Graves-style, as used in the Nature DQN paper — differs from PyTorch's `torch.optim.RMSprop`):
```
g  = α * g  + (1 - α) * grad          # running mean of gradient
n  = α * n  + (1 - α) * grad²         # running mean of squared gradient
Δ  = momentum * Δ - lr * grad / √(n - g² + ε)   # parameter update
θ  = θ + Δ
```
Two key differences from PyTorch's `torch.optim.RMSprop`:
1. **Epsilon placement**: DeepMind puts ε *inside* the sqrt: `√(n - g² + ε)`. PyTorch puts ε *outside*: `√(n - g²) + ε`. With ε=0.01, the minimum denominator is `√0.01 = 0.1` (DeepMind) vs `0.01` (PyTorch) — a **10× difference** in gradient scaling. This is the primary reason we need a custom optimizer.
2. **Gradient mean tracking**: DeepMind always tracks `g` (the running gradient mean) for the `n - g²` variance term. PyTorch only does this with `centered=True` (default is `False`). While `centered=True` solves this, it does NOT fix the epsilon placement, so we still need a custom implementation.

### Track B — Modern Preset (`configs/preset/modern.yaml`)
| Parameter | Value | Source |
|---|---|---|
| optimizer | Adam | Rainbow |
| lr | 6.25e-5 | Rainbow |
| adam_eps | 1.5e-4 | Rainbow |
| epsilon_start → end | 1.0 → 0.01 over 250K agent steps | Modern best practice |
| grad_clip_norm | 10.0 | Modern best practice |
| loss | huber | Same |

### Shared (both tracks)
| Parameter | Value |
|---|---|
| batch_size | 32 |
| replay_capacity | 1,000,000 |
| frame_stack | 4 |
| frame_skip | 4 |
| target_update_freq | 10,000 gradient steps |
| gamma | 0.99 |
| replay_start_size | 50,000 |
| update_frequency | every 4 agent steps |
| noop_max | 30 |
| PER alpha | 0.6 |
| PER beta | 0.4 → 1.0 |
| eval_epsilon | 0.05 |

---

## Dependencies (`pyproject.toml`)

```toml
[project]
name = "atari-dqn"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2.0",
    "numpy>=1.26.0,<2.0",
    "gymnasium[atari]>=1.0.0",
    "ale-py>=0.10.0",
    "shimmy[atari]>=2.0.0",
    "opencv-python-headless>=4.9.0",
    "hydra-core>=1.3.0",
    "omegaconf>=2.3.0",
    "wandb>=0.17.0",
    "imageio[ffmpeg]>=2.34.0",
    "tqdm>=4.66.0",
    "rich>=13.0.0",
]

[project.optional-dependencies]
dashboard = ["streamlit>=1.40.0", "plotly>=5.24.0", "pandas>=2.2.0"]
dev = ["pytest>=8.0.0", "pytest-cov>=5.0.0", "ruff>=0.5.0"]
```

PyTorch installed via uv (not system torch):
```bash
uv add torch --index-url https://download.pytorch.org/whl/cu124
```

Reproducibility rule: commit `uv.lock` to git. Log exact runtime package versions (torch, gymnasium, ale-py) to `metrics.json` at every checkpoint.

---

## Dashboard (Streamlit, 5 pages)

1. **Training Curves**: episode return, loss, mean Q, epsilon, grad norm over steps. Smoothing slider. Data from wandb API or local CSV.

2. **Video Player**: checkpoint slider to scrub through training. Embedded MP4 player. Side-by-side: early vs late training. Eval score displayed per checkpoint.

3. **Game Comparison**: overlaid training curves across games. Score table with human-normalized %. Bar chart comparing agent variants.

4. **Ablation Comparator**: baseline → +Double → +PER → +Dueling progression. Shows incremental gain of each improvement. Confidence bands across seeds.

5. **Config Diff Viewer**: select two runs, see side-by-side YAML diff highlighting what changed. Explains why results differ.

Launch: `uv run streamlit run dashboard/app.py`

---

## Implementation Phases

### Phase 1: Project Setup + Environment
- `uv init`, install all deps (torch via uv with CUDA index)
- Create full directory structure
- Implement 7 Atari wrappers in `src/dqn/envs/wrappers.py`
- Implement `src/dqn/envs/factory.py` with env protocol validation
- Implement `configs/env_protocol/{paper_v4,modern_v5_sticky}.yaml`
- Implement `src/dqn/utils/seeding.py`
- **Verify**: obs shape (4, 84, 84), dtype uint8, frame stacking correct across resets, max of last 2 frames in MaxAndSkip, protocol fields logged and match selected track

### Phase 2: Networks
- Implement `ConvEncoder` in `src/dqn/networks/encoder.py`
- Implement `NatureDQN` in `src/dqn/networks/nature_dqn.py`
- Implement `DuelingDQN` in `src/dqn/networks/dueling.py`
- **Verify**: input (1,4,84,84) → output (1, n_actions), ~1.69M params (NatureDQN), 3136 conv features, gradients flow to all layers, Dueling satisfies Q = V + A - mean(A)

### Phase 3: Replay Buffers
- Implement `UniformReplayBuffer` in `src/dqn/replay/uniform.py`
- Implement `SumTree` in `src/dqn/replay/sum_tree.py`
- Implement `PrioritizedReplayBuffer` in `src/dqn/replay/prioritized.py`
- **Verify**: correct shapes/dtypes, circular wrap at capacity, episode boundary handling (no cross-reset stacking), sum_tree.total == sum of leaves, PER sampling proportional to priority (statistical test over 10K samples), IS weights max == 1.0, new transitions get max priority

### Phase 4: Agent + Training Infrastructure
- Implement toggle-based `DQNAgent` in `src/dqn/agent.py`
- Implement `EpsilonScheduler` in `src/dqn/training/scheduler.py`
- Implement `CheckpointManager` with failure gates in `src/dqn/training/checkpoint.py`
- Implement `MetricsAggregator` with replay diagnostics in `src/dqn/logging/metrics.py`
- Implement `WandbLogger` in `src/dqn/logging/wandb_logger.py`
- Implement `Trainer` in `src/dqn/training/trainer.py`
- Implement `Evaluator` + video recording in `src/dqn/evaluation/evaluator.py`
- Write all Hydra YAML configs
- Implement `scripts/train.py`, `scripts/evaluate.py`, `scripts/record_video.py`
- **Verify**: smoke test 100K frames (no crash, no NaN), checkpoints created, epsilon anneals correctly, select_action with ε=0 returns argmax, ε=1 is random, Double DQN target differs from vanilla, state_dict save/load round-trip, eval produces valid MP4

### Phase 5: Baseline Training (Track A)
- Train vanilla DQN on Pong: `uv run python scripts/train.py agent=dqn preset=paper env=pong replay=uniform env_protocol=paper_v4`
- **Verify**: expect Pong score >0 within ~500K frames as a learning signal heuristic, and >18 within ~3M frames as a rough benchmark (not a hard gate — investigate if missed, but don't auto-fail)

### Phase 6: Modern Training (Track B)
- Train modern agent on Pong: `uv run python scripts/train.py agent=modern preset=modern env=pong replay=prioritized env_protocol=modern_v5_sticky`
- Train on Breakout, Space Invaders
- **Verify**: expect faster convergence than Track A as a sanity check; Breakout ~300+ within 10M frames as a rough target (investigate if significantly below); Q-values should trend reasonably (use Pong Q ∈ [-20, 30] as a rough heuristic, not a hard bound)

### Phase 7: Multi-Seed Comparison
- Run Track A vs Track B on Pong + Breakout with 3 seeds each
- Incremental ablations: DQN → +Double → +PER → +Dueling
- Produce comparison tables with median + IQR

### Phase 8: Dashboard
- Implement all 5 Streamlit pages
- **Verify**: training curves render, videos play, ablation comparator works, config diff highlights changes

---

## Correctness Checks

- Conv output: 84×84 → 7×7×64 = 3136 after 3 conv layers
- Q-value sanity is game/protocol dependent; use trend-based alerts (sustained divergence), not fixed universal bounds. Rough guide: Pong Q ∈ [-20, 30], but treat as heuristic
- Loss scale is algorithm-dependent; alert on NaN/inf or abrupt multi-sigma jumps, not fixed thresholds
- Frame normalization: once in encoder.forward() (uint8/255), NOT also in wrappers (double-normalize = silent failure)
- Done mask: `target = r + γ * next_q * (1 - done)` — forgetting mask = common bug
- Action gather: Q for action TAKEN, not argmax
- Buffer dtype: uint8 not float32 (4× memory)
- Target net: eval mode, no requires_grad
- PER IS weights: multiply per-sample loss BEFORE reduction, not after
- EpisodicLife: training only, NOT eval
- Pong milestone checks (score > 0 by 500K frames) are heuristics for debugging, not binary failure criteria
- Failure gates: NaN/inf → halt; persistent NaN grads (>100 consecutive) → halt; single grad spikes → warning; 3 consecutive eval drops → warning + inspection tag
