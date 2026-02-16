from __future__ import annotations

import numpy as np

from dqn.envs.wrappers import ClipRewardEnv, FrameStack, MaxAndSkipEnv, NoopResetEnv, WarpFrame


def test_warp_and_stack(dummy_env):
    env = NoopResetEnv(dummy_env, noop_max=5)
    env = MaxAndSkipEnv(env, skip=4)
    env = WarpFrame(env)
    env = FrameStack(env, num_stack=4)

    obs, _ = env.reset()
    assert obs.shape == (4, 84, 84)
    assert obs.dtype == np.uint8

    obs, _, _, _, _ = env.step(0)
    assert obs.shape == (4, 84, 84)


def test_clip_reward(dummy_env):
    env = ClipRewardEnv(dummy_env)
    _, _ = env.reset()
    _, reward, _, _, _ = env.step(2)
    assert reward in {-1.0, 0.0, 1.0}
