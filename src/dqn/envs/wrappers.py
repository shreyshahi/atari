from __future__ import annotations

from collections import deque
from typing import Deque

import cv2
import gymnasium as gym
import numpy as np


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = int(self.unwrapped.np_random.integers(1, self.noop_max + 1))
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self.skip = skip
        self.obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for i in range(self.skip):
            obs, reward, term, trunc, info = self.env.step(action)
            if i >= self.skip - 2:
                self.obs_buffer[i - (self.skip - 2)] = obs
            total_reward += reward
            terminated = terminated or term
            truncated = truncated or trunc
            if terminated or truncated:
                break

        max_frame = self.obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated

        lives = info.get("lives", 0)
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = info.get("lives", 0)
        return obs, info


class FireResetEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        action_meanings = env.unwrapped.get_action_meanings()
        if "FIRE" not in action_meanings:
            raise ValueError("Environment does not use FIRE action")
        self.fire_action = action_meanings.index("FIRE")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(self.fire_action)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(height, width),
            dtype=np.uint8,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)


class ClipRewardEnv(gym.RewardWrapper):
    def reward(self, reward):
        return float(np.sign(reward))


class FrameStack(gym.Wrapper):
    def __init__(self, env: gym.Env, num_stack: int = 4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames: Deque[np.ndarray] = deque(maxlen=num_stack)

        low = np.repeat(env.observation_space.low[None, ...], num_stack, axis=0)
        high = np.repeat(env.observation_space.high[None, ...], num_stack, axis=0)
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.stack(self.frames, axis=0)
