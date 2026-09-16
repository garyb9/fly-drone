import numpy as np
import pytest
import torch
from fly_drone.replay import FlyDictReplayBuffer
from fly_drone.sac import SpacesOnlyEnv, build_sac
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.logger import Logger


def make_buffer(cls=FlyDictReplayBuffer, **kwargs):
    env = SpacesOnlyEnv("decoder", n_features=4)
    buf = cls(64, env.observation_space, env.action_space, device="cpu", **kwargs)
    return env, buf


def add(buf, index, reward=0.0, done=False, timeout=False):
    obs = {
        key: np.full(space.shape, float(index), dtype=space.dtype)
        for key, space in buf.observation_space.spaces.items()
    }
    next_obs = {
        key: np.full(space.shape, float(index) + 1.0, dtype=space.dtype)
        for key, space in buf.observation_space.spaces.items()
    }
    info = {"TimeLimit.truncated": timeout} if timeout else {}
    buf.add(
        obs,
        next_obs,
        np.zeros(4, dtype=np.float32),
        np.array([reward], dtype=np.float32),
        np.array([done]),
        [info],
    )


def test_the_default_mode_reproduces_a_plain_dict_buffer():
    env = SpacesOnlyEnv("decoder", n_features=4)
    ours = FlyDictReplayBuffer(
        64, env.observation_space, env.action_space, device="cpu"
    )
    plain = DictReplayBuffer(64, env.observation_space, env.action_space, device="cpu")
    for i in range(10):
        add(ours, i, reward=float(i))
        add(plain, i, reward=float(i))

    inds = np.array([0, 3, 7, 9])
    a, b = ours._get_samples(inds), plain._get_samples(inds)
    for key in a.observations:
        assert torch.equal(a.observations[key], b.observations[key])
        assert torch.equal(a.next_observations[key], b.next_observations[key])
    assert torch.equal(a.rewards, b.rewards)
    assert torch.equal(a.dones, b.dones)
    assert a.discounts is None  # n_steps=1 leaves the bootstrap to SAC's gamma


def test_by_index_reads_the_following_slot_and_drops_the_duplicate_array():
    env = SpacesOnlyEnv("decoder", n_features=4)
    buf = FlyDictReplayBuffer(
        64,
        env.observation_space,
        env.action_space,
        device="cpu",
        optimize_memory_usage=True,
    )
    assert buf.next_observations is buf.observations  # one ring, not two
    for i in range(10):
        add(buf, i)
    for i in range(10):
        sample = buf._get_samples(np.array([i]))
        for value in sample.next_observations.values():
            assert torch.allclose(value, torch.full_like(value, float(i) + 1.0))

    plain = DictReplayBuffer(64, env.observation_space, env.action_space, device="cpu")
    plain_bytes = sum(v.nbytes for v in plain.observations.values())
    assert sum(v.nbytes for v in plain.next_observations.values()) == plain_bytes
    assert sum(v.nbytes for v in buf.observations.values()) == plain_bytes


def test_optimized_sampling_never_touches_the_unwritten_slot():
    env = SpacesOnlyEnv("decoder", n_features=4)
    buf = FlyDictReplayBuffer(
        8,
        env.observation_space,
        env.action_space,
        device="cpu",
        optimize_memory_usage=True,
    )
    for i in range(3):  # not full
        add(buf, i)
    for _ in range(20):
        sample = buf.sample(4)
        assert sample.rewards.shape == (4, 1)
    for i in range(3, 12):  # wraps the ring
        add(buf, i)
    assert buf.full
    for _ in range(20):
        assert buf.sample(4).rewards.shape == (4, 1)


def test_n_step_accumulates_rewards_and_stops_at_an_episode_end():
    _, buf = make_buffer(n_steps=3)
    for i, reward in enumerate([1.0, 2.0, 3.0, 4.0]):
        add(buf, i, reward=reward)

    sample = buf._get_samples(np.array([0]))
    assert sample.rewards[0, 0].item() == pytest.approx(1 + 0.99 * 2 + 0.99**2 * 3)
    assert sample.discounts[0, 0].item() == pytest.approx(0.99**3)

    _, ended = make_buffer(n_steps=3)
    for i, reward in enumerate([1.0, 2.0, 3.0, 4.0]):
        add(ended, i, reward=reward, done=(i == 1))
    sample = ended._get_samples(np.array([0]))
    assert sample.rewards[0, 0].item() == pytest.approx(
        1 + 0.99 * 2
    )  # stops after the done
    assert sample.discounts[0, 0].item() == pytest.approx(0.99**2)


def test_n_step_stops_at_a_timeout_too():
    _, buf = make_buffer(n_steps=3)
    for i, reward in enumerate([1.0, 2.0, 3.0]):
        add(buf, i, reward=reward, timeout=(i == 1))
    sample = buf._get_samples(np.array([0]))
    assert sample.rewards[0, 0].item() == pytest.approx(1 + 0.99 * 2)
    assert sample.dones[0, 0].item() == 0.0  # a timeout is not a terminal state


def test_build_sac_uses_the_fly_buffer_and_trains_through_it():
    env = SpacesOnlyEnv("decoder", n_features=4)
    model = build_sac(
        "decoder",
        env,
        buffer_size=64,
        device="cpu",
        actor_warmup=0,
        n_step=2,
        optimize_memory=True,
    )
    assert isinstance(model.replay_buffer, FlyDictReplayBuffer)
    assert model.replay_buffer.n_steps == 2
    model.set_logger(Logger(folder=None, output_formats=[]))
    spaces = env.observation_space.spaces
    rng = np.random.default_rng(0)
    for _ in range(32):
        obs = {k: rng.uniform(0, 1, s.shape).astype(s.dtype) for k, s in spaces.items()}
        nxt = {k: rng.uniform(0, 1, s.shape).astype(s.dtype) for k, s in spaces.items()}
        model.replay_buffer.add(
            obs,
            nxt,
            rng.uniform(-1, 1, 4).astype(np.float32),
            rng.normal(size=1).astype(np.float32),
            np.zeros(1, dtype=bool),
            [{}],
        )
    model.train(gradient_steps=2, batch_size=8)  # the n-step path signs off here
