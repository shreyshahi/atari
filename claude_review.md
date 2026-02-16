# Code Review: DQN Atari Implementation

## Verdict

The architecture follows the plan well — toggle-based agent, correct network shapes, proper encoder normalization, correct Dueling mean-centering, correct DeepMind RMSProp epsilon placement. But there are several bugs that will silently corrupt training results if not fixed.

---

## CRITICAL — Will produce wrong results silently

### 1. PER beta never reaches 1.0
**`src/dqn/replay/prioritized.py:45`**
```python
beta = self._beta_by_frame(self.num_added)
```
`self.num_added` counts agent steps (~12.5M at end of training). `beta_decay_frames=50_000_000` is in emulator frames. Beta anneals to `0.4 + (12.5M/50M) * 0.6 = 0.55` instead of `1.0`. IS weight correction never fully activates — the agent trains with biased updates throughout.

**Fix**: Either pass `env_frames` into `sample()`, or set `beta_decay_frames` to `12_500_000` (agent steps).

### 2. Training loss / Q-values / grad norm never logged anywhere
**`src/dqn/logging/metrics.py`** and **`src/dqn/training/trainer.py:173-180`**

`MetricsAggregator` only tracks episode returns/lengths. The `update_info` dict (containing `loss`, `mean_q`, `max_q`, `grad_norm`) is computed by `agent.update()` but:
- Never aggregated into windowed stats
- Never written to CSV (only per-episode data goes to `train_log.csv`)
- Never logged to wandb (the `log_row` only includes `epsilon` + episode stats)

You will have **zero visibility** into training health — no loss curves, no Q-value trends, no gradient diagnostics. The plan explicitly requires all of these every 1K steps.

### 3. Three of four failure gates missing
**`src/dqn/training/trainer.py:85-87`**

Only checks `isfinite(loss)` and `isfinite(mean_q)`. Missing:
- Persistent NaN/inf gradients (>100 consecutive) → halt
- Grad norm spike (>1000) → warning
- 3 consecutive eval drops >50% from best → warning + inspection tag

If gradients go NaN but loss stays finite (possible with clipping), training degrades silently.

### 4. Environment never seeded — reproducibility broken
**`src/dqn/utils/seeding.py`** and **`src/dqn/training/trainer.py:74,151`**

`seed_everything()` seeds torch/numpy/random, but `env.reset()` is called without `seed=` in both the prefill loop (line 74) and the training loop (line 151). The ALE environment gets OS-random initial state. Same config + same seed = different trajectories. Exact reproducibility is impossible.

**Fix**: Pass `seed=cfg.seed` to the first `env.reset()` call.

### 5. MaxAndSkipEnv obs_buffer cross-contamination
**`src/dqn/envs/wrappers.py:31,39-50`**

`obs_buffer` is allocated once and never cleared between steps. If the episode terminates during frame 0 or 1 of the skip (before reaching `i >= skip-2`), the buffer still holds stale data from the previous step. The max-pooled frame then combines current and previous-step observations — cross-episode frame contamination.

This affects terminal/near-terminal states in every game, subtly corrupting the training signal.

---

## HIGH — Significant plan deviations or correctness risks

### 6. Epsilon starts at ~0.82, not 1.0
**`src/dqn/training/trainer.py:111-112,123`**
```python
env_steps = warmup           # 50000
env_frames = env_steps * 4   # 200000
epsilon = scheduler.value(env_frames)  # value(200000)
```
The warmup frames are counted toward the epsilon schedule. At training start, epsilon = `1.0 + (200K/1M) * (0.1-1.0) = 0.82` (Track A). The fully-exploratory phase (ε near 1.0) is skipped. This may reduce early exploration quality.

### 7. `frame_skip = 4` hardcoded in two places
**`trainer.py:107`** and **`evaluator.py:44`**

```python
frame_skip = 4                              # trainer.py
max_steps = max(1, max_env_frames // 4)    # evaluator.py
```

The env protocol configs have `frameskip: 1` (ALE native), and `MaxAndSkipEnv` handles the 4x skip. This works today but the constant should be derived from config, not a magic number. If `MaxAndSkipEnv.skip` ever changes, frame counters, eval caps, and epsilon annealing all break silently.

### 8. Wandb video logging absent
**`src/dqn/logging/wandb_logger.py`**

The plan requires `wandb.Video` uploads. The logger has only `log()` and `finish()` — no video method. Videos are saved to disk in checkpoints but never uploaded to wandb.

### 9. Wandb step uses agent steps, not emulator frames
**`trainer.py:180`**: `self.wandb.log(log_row, step=env_steps)`

The plan specifies `env_frames` as the x-axis. Using agent steps makes wandb charts incomparable with the literature (which plots against emulator frames).

### 10. `target_net` parameters still have `requires_grad=True`
**`agent.py:31`**

`target_net.eval()` only affects BatchNorm/Dropout behavior. Parameters still allocate gradient buffers. The plan says "no requires_grad". Should add `for p in self.target_net.parameters(): p.requires_grad_(False)`.

### 11. Replay diagnostics completely missing
**Plan requirement**: "Every 10K steps: priority distribution stats, sample age distribution, IS weight stats."

No code anywhere computes or logs these. `MetricsAggregator` has no methods for it, trainer never calls it, logger has no support.

### 12. `EvalResult` has no episode lengths
**`src/dqn/evaluation/evaluator.py:14-20`**

The plan requires "Episode length distribution" from eval. `EvalResult` only stores `episode_returns`, not episode lengths. This data is never collected or logged.

---

## MEDIUM — Correctness nits or minor plan deviations

### 13. Kaiming-ReLU init on output layers
**`nature_dqn.py:20-24`**, **`dueling.py:27-31`**

`_init_linear()` applies `kaiming_uniform_(nonlinearity="relu")` to ALL Linear layers including the final output layer (which has no ReLU after it). The gain of sqrt(2) inflates initial Q-value variance. Should skip the last layer or use `nonlinearity="linear"` for it.

### 14. PER IS weight normalization by batch max, not global min-probability
**`prioritized.py:77`**: `weights /= weights.max()`

Normalizes by the max weight in the current batch rather than the theoretical max weight (from the minimum-priority item in the entire buffer). This makes weight scale inconsistent across batches. The PER paper normalizes by `max_j(w_j)` over all stored transitions.

### 15. PER fallback breaks stratification
**`prioritized.py:63-67`**

When a sampled index is invalid, the fallback uses `_sample_abs_index()` (uniform random across the entire buffer) instead of retrying within the same stratified segment. The fallback's priority is also re-read from the tree, making the IS weight inconsistent for that sample.

### 16. Per-episode metrics never logged to wandb
**`trainer.py:136-151`**

Episode end metrics (`episode_return`, `episode_length`, `episode_number`) go to CSV but not to wandb. Only 1K-step aggregated stats go to wandb.

### 17. `catalog.json` missing `timestamp` field
**`checkpoint.py:51-57`**

Plan specifies entries should have `{step, path, eval_score, timestamp}`. No timestamp is recorded.

### 18. `torch.load` without `weights_only=True`
**`checkpoint.py:72`**, **`evaluate.py:30`**, **`record_video.py:30`**

PyTorch 2.x security warning. Low risk for internal checkpoints but noisy.

### 19. Config: `preset/paper.yaml` has spurious `adam_eps`
**`configs/preset/paper.yaml:7`**

`adam_eps: 0.00015` in the paper preset which uses `deepmind_rmsprop`. Unused but confusing.

### 20. Config: `replay/uniform.yaml` has PER fields
**`configs/replay/uniform.yaml`**

Contains `alpha`, `beta_start`, `beta_end`, etc. that are irrelevant to uniform replay. Confusing.

### 21. Tests: Major gaps vs plan
- `test_networks.py`: Missing param count test (~1.69M), missing gradient flow test
- `test_replay.py`: Missing circular wrap test, episode boundary test, dtype assertion
- `test_sum_tree.py`: Missing proportional sampling statistical test (plan says "10K samples")
- `test_agent.py`: Missing `select_action` ε=0/ε=1 tests
- `test_trainer.py`: Smoke test only instantiates `Trainer`, never calls `train()`

### 22. Dashboard: `config_diff.py` has no actual diff
Displays configs side-by-side with no highlighting of differences. Acknowledged as incomplete in the code itself.

---

## Confirmed Correct

These are commonly-buggy areas that the implementation gets right:

| Item | Verdict |
|------|---------|
| DeepMind RMSProp: ε inside sqrt (`√(n-g²+ε)`) | **Correct** |
| Double DQN: online selects, target evaluates | **Correct** |
| Done mask: `γ * next_q * (1 - done)` | **Correct** |
| IS weights: per-sample loss BEFORE `.mean()` | **Correct** |
| uint8 normalization ONLY in encoder, not in wrappers | **Correct** |
| Dueling: mean-centering, not max-centering | **Correct** |
| Action gather: taken action, not argmax | **Correct** |
| Conv output: 7x7x64 = 3136 | **Correct** |
| EpisodicLife: training only, not eval | **Correct** |
| Wrapper ordering matches plan | **Correct** |
| Target sync every 10K gradient steps | **Correct** |

---

## Fix Priority

If fixing these, this is the recommended order:

1. **#1** (PER beta) — silent corruption of all Track B results
2. **#2** (metrics never logged) — flying blind during training
3. **#4** (env not seeded) — reproducibility broken
4. **#5** (MaxAndSkip stale buffer) — corrupts terminal transitions
5. **#3** (failure gates) — safety net missing
6. **#6** (epsilon starts at 0.82) — early exploration reduced
7. **#7** (hardcoded frame_skip) — maintenance hazard
8. **#13** (Kaiming init on output layer) — noisy early training
