"""Dict replay buffers for encoder v5 (spec Pack 2 S1, S3).

Two opt-in modes over SB3's ``DictReplayBuffer``; both default off, so the code path the
accepted runs used is unchanged:

- ``n_steps > 1`` (S3) replaces each transition's reward and bootstrap discount with the
  n-step return at sampling time, in the Bellman target only. The stored transitions are the
  same; only ``sample()`` differs. An episode end (done or timeout) truncates the sum.
- ``optimize_memory_usage=True`` (S1) stores only ``observations`` and reads ``next_obs`` from
  the following ring slot, halving the buffer's RAM. The slot written immediately after a
  transition holds its next observation (the SB3 trick), so the ring is self-consistent; the
  one slot that has not been written yet is excluded from sampling.

n-step math mirrors ``stable_baselines3.common.buffers.NStepReplayBuffer`` (2.9.0); the
per-mode behaviour is pinned by ``tests/test_replay.py``.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples


class FlyDictReplayBuffer(DictReplayBuffer):
    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device="auto",
        n_envs=1,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
        n_steps=1,
        gamma=0.99,
    ):
        # SB3's DictReplayBuffer rejects optimize_memory_usage; allocate the plain layout and
        # then drop the duplicate array for the memory-optimised variant.
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=False,
            handle_timeout_termination=handle_timeout_termination,
        )
        self.n_steps = int(n_steps)
        self.gamma = float(gamma)
        self.optimize_memory_usage = bool(optimize_memory_usage)
        if self.optimize_memory_usage:
            # The single ring then holds obs[i] at i and next_obs[i] at i+1.
            self.next_observations = self.observations

    def add(self, obs, next_obs, action, reward, done, infos):
        for key in self.observations:
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = np.array(obs[key])
        if self.optimize_memory_usage:
            for key in self.observations:
                if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                    next_obs[key] = next_obs[key].reshape(
                        (self.n_envs,) + self.obs_shape[key]
                    )
                # Write next_obs into the slot the next transition will occupy.
                self.observations[key][(self.pos + 1) % self.buffer_size] = np.array(
                    next_obs[key]
                )
        else:
            for key in self.next_observations:
                if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                    next_obs[key] = next_obs[key].reshape(
                        (self.n_envs,) + self.obs_shape[key]
                    )
                self.next_observations[key][self.pos] = np.array(next_obs[key])

        action = action.reshape((self.n_envs, self.action_dim))
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)
        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array(
                [info.get("TimeLimit.truncated", False) for info in infos]
            )
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size, env=None):
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # The slot at self.pos holds an old observation whose successor is unknown, so it is
        # never sampled (this is what SB3's flat memory-optimised buffer does).
        if self.full:
            batch_inds = (
                np.random.randint(1, self.buffer_size, size=batch_size) + self.pos
            ) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds, env=None):
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        obs = {
            key: values[batch_inds, env_indices]
            for key, values in self.observations.items()
        }
        actions = self.actions[batch_inds, env_indices]
        dones = self.dones[batch_inds, env_indices] * (
            1 - self.timeouts[batch_inds, env_indices]
        )

        if self.n_steps <= 1:
            if self.optimize_memory_usage:
                next_obs = {
                    key: values[(batch_inds + 1) % self.buffer_size, env_indices]
                    for key, values in self.observations.items()
                }
                rewards = self._normalize_reward(
                    self.rewards[batch_inds, env_indices].reshape(-1, 1), env
                )
                return self._samples(obs, actions, next_obs, dones, rewards, None, env)
            return super()._get_samples(batch_inds, env=env)

        # n-step: accumulate the next n rewards, stopping at the first done or timeout.
        last_valid = self.pos - 1
        saved_timeout = self.timeouts[last_valid].copy()
        self.timeouts[last_valid] = np.logical_or(
            saved_timeout, np.logical_not(self.dones[last_valid])
        )
        try:
            steps = np.arange(self.n_steps).reshape(1, -1)
            indices = (batch_inds[:, None] + steps) % self.buffer_size
            rewards_seq = self._normalize_reward(
                self.rewards[indices, env_indices[:, None]], env
            )
            ended = np.logical_or(
                self.dones[indices, env_indices[:, None]],
                self.timeouts[indices, env_indices[:, None]],
            )
            has_end = ended.any(axis=1)
            done_idx = np.where(has_end, ended.argmax(axis=1), self.n_steps - 1)
            mask = np.arange(self.n_steps).reshape(1, -1) <= done_idx[:, None]
            discounts = self.gamma ** mask.sum(axis=1, keepdims=True).astype(np.float32)
            decay = self.gamma ** np.arange(self.n_steps, dtype=np.float32)
            n_step_rewards = (rewards_seq * decay * mask).sum(axis=1, keepdims=True)

            last_indices = (batch_inds + done_idx) % self.buffer_size
            if self.optimize_memory_usage:
                next_obs = {
                    key: values[(last_indices + 1) % self.buffer_size, env_indices]
                    for key, values in self.observations.items()
                }
            else:
                next_obs = {
                    key: values[last_indices, env_indices]
                    for key, values in self.next_observations.items()
                }
            final_dones = self.dones[last_indices, env_indices][:, None].astype(
                np.float32
            ) * (
                1.0
                - self.timeouts[last_indices, env_indices][:, None].astype(np.float32)
            )
        finally:
            self.timeouts[last_valid] = saved_timeout
        return self._samples(
            obs, actions, next_obs, final_dones[:, 0], n_step_rewards, discounts, env
        )

    def _samples(self, obs, actions, next_obs, dones, rewards, discounts, env):
        return DictReplayBufferSamples(
            observations={
                key: self.to_torch(value)
                for key, value in self._normalize_obs(obs, env).items()
            },
            actions=self.to_torch(actions),
            next_observations={
                key: self.to_torch(value)
                for key, value in self._normalize_obs(next_obs, env).items()
            },
            dones=self.to_torch(dones).reshape(-1, 1),
            rewards=self.to_torch(np.asarray(rewards).reshape(-1, 1)),
            discounts=None if discounts is None else self.to_torch(discounts),
        )
