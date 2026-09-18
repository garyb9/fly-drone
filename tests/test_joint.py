import numpy as np
import pytest
import torch
from fly_drone.encoder import eyes_space
from fly_drone.joint_policy import (
    JointSACPolicy,
    joint_action_dim,
    split_joint_action,
)
from fly_drone.spatial_encoder import flat_dim
from gymnasium import spaces


def _policy(n_dn=16):
    space = spaces.Dict(
        {
            "eyes": eyes_space(),
            "dn": spaces.Box(0, 1, (n_dn,), np.float32),
            "geometry": spaces.Box(-np.inf, np.inf, (8,), np.float32),
        }
    )
    return JointSACPolicy(
        space,
        spaces.Box(-1, 1, (joint_action_dim(),), np.float32),
        lambda _: 3e-4,
        net_arch={"pi": [], "qf": [8]},
    )


def _obs(n_dn=16, n=2):
    return {
        "eyes": torch.randint(0, 255, (n, 6, 48, 64), dtype=torch.float32),
        "dn": torch.rand(n, n_dn),
        "geometry": torch.rand(n, 8),
    }


def test_joint_action_dim_and_split():
    assert joint_action_dim() == flat_dim() + 4
    action = np.arange(joint_action_dim(), dtype=np.float32)
    currents, velocity = split_joint_action(action)
    assert currents.shape == (flat_dim(),)
    assert velocity.tolist() == [
        flat_dim(),
        flat_dim() + 1,
        flat_dim() + 2,
        flat_dim() + 3,
    ]


def test_joint_actor_heads_are_independent():
    policy = _policy()
    obs = _obs()
    with torch.no_grad():
        mean, log_std, _ = policy.actor.get_action_dist_params(obs)
    assert mean.shape == (2, joint_action_dim())
    assert log_std.shape == (2, joint_action_dim())

    other_dn = dict(obs)
    other_dn["dn"] = torch.rand_like(obs["dn"])
    other_eyes = dict(obs)
    other_eyes["eyes"] = torch.randint(0, 255, obs["eyes"].shape, dtype=torch.float32)
    with torch.no_grad():
        mean_dn, _, _ = policy.actor.get_action_dist_params(other_dn)
        mean_eyes, _, _ = policy.actor.get_action_dist_params(other_eyes)
    # dn moves only the 4 velocity means; eyes only the 816 current logits.
    torch.testing.assert_close(mean[:, : flat_dim()], mean_dn[:, : flat_dim()])
    assert not torch.allclose(mean[:, flat_dim() :], mean_dn[:, flat_dim() :])
    torch.testing.assert_close(mean[:, flat_dim() :], mean_eyes[:, flat_dim() :])
    assert not torch.allclose(mean[:, : flat_dim()], mean_eyes[:, : flat_dim()])


def test_joint_spaces_and_build():
    from fly_drone.joint_policy import JointSACPolicy
    from fly_drone.sac import SpacesOnlyEnv, build_sac, learner_spaces

    obs_space, action_space = learner_spaces("joint", n_features=16)
    assert set(obs_space.spaces) == {"eyes", "dn", "geometry"}
    assert action_space.shape == (joint_action_dim(),)
    model = build_sac(
        "joint", SpacesOnlyEnv("joint", n_features=16), buffer_size=1, device="cpu"
    )
    assert isinstance(model.policy, JointSACPolicy)
    assert model.action_space.shape == (joint_action_dim(),)


def test_joint_env_step():
    from fly_drone.sac import JointRoamEnv

    env = JointRoamEnv(level=2)
    try:
        obs, _ = env.reset(seed=1)
        assert set(obs) == {"eyes", "dn", "geometry"}
        obs, _, _, _, info = env.step(env.action_space.sample())
        assert obs["eyes"].shape == (6, 48, 64)
        assert obs["dn"].shape == (2022,)
        assert "metabolic_cost" in info
    finally:
        env.close()


def test_joint_export_round_trip(tmp_path):
    import json

    from fly_drone.sac import SpacesOnlyEnv, build_sac, export_checkpoint
    from fly_drone.spatial_encoder import SpatialEncoder

    model = build_sac(
        "joint", SpacesOnlyEnv("joint", n_features=2022), buffer_size=1, device="cpu"
    )
    model.save(tmp_path / "m")
    report = export_checkpoint(str(tmp_path / "m.zip"), "joint", tmp_path / "out")
    assert report["encoder_version"].startswith("learned-v6:")
    assert (
        SpatialEncoder.load(tmp_path / "out" / "encoder.pt").version
        == report["encoder_version"]
    )
    decoder = json.loads((tmp_path / "out" / "decoder.json").read_text())
    assert decoder["encoder_version"] == report["encoder_version"]
    assert decoder["output"] == "tanh"


def _joint_model(actor_warmup, predictive_weight=0.0, n_dn=16):
    from fly_drone.sac import SpacesOnlyEnv, build_sac
    from stable_baselines3.common.logger import Logger

    env = SpacesOnlyEnv("joint", n_features=n_dn)
    model = build_sac(
        "joint",
        env,
        buffer_size=64,
        device="cpu",
        actor_warmup=actor_warmup,
        predictive_weight=predictive_weight,
    )
    model.set_logger(Logger(folder=None, output_formats=[]))
    rng = np.random.default_rng(11)
    for _ in range(32):
        obs = {
            k: rng.uniform(0, 1, (1, *s.shape)).astype(s.dtype)
            for k, s in env.observation_space.items()
        }
        nxt = {
            k: rng.uniform(0, 1, (1, *s.shape)).astype(s.dtype)
            for k, s in env.observation_space.items()
        }
        model.replay_buffer.add(
            obs,
            nxt,
            rng.uniform(-1, 1, (1, joint_action_dim())).astype(np.float32),
            rng.normal(size=1).astype(np.float32),
            np.zeros(1, dtype=bool),
            [{}],
        )
    return model


def test_per_head_log_prob_sums_to_the_total():
    policy = _policy()
    obs = _obs()
    with torch.no_grad():
        action, lp_enc, lp_vel = policy.actor.action_log_prob_heads(obs)
        # action_log_prob_heads leaves the distribution set from the same obs, so this is the
        # same per-action squashed log-density SB3 sums, for the same sampled action.
        total = policy.actor.action_dist.log_prob(action)
    assert lp_enc.shape == (2, 1) and lp_vel.shape == (2, 1)
    torch.testing.assert_close((lp_enc + lp_vel).squeeze(1), total)


def test_joint_sac_uses_per_head_alpha():
    import math

    from fly_drone.sac import (
        ENT_COEF_INIT,
        TARGET_ENTROPY_PER_DIM,
        JointSAC,
        ent_coef_init,
    )

    model = _joint_model(actor_warmup=0)
    assert isinstance(model, JointSAC)
    assert model.log_ent_coef.shape == (2,)
    assert model.target_entropy_heads == (
        flat_dim() * TARGET_ENTROPY_PER_DIM,
        4 * TARGET_ENTROPY_PER_DIM,
    )
    assert model.log_ent_coef[0].item() == pytest.approx(
        math.log(ent_coef_init(flat_dim())), abs=1e-6
    )
    assert model.log_ent_coef[1].item() == pytest.approx(
        math.log(ENT_COEF_INIT), abs=1e-6
    )


def test_joint_per_head_alpha_trains_and_warmup_freezes_it():
    from fly_drone.sac import JointSAC

    model = _joint_model(actor_warmup=100)
    assert isinstance(model, JointSAC)
    model.num_timesteps = 60
    frozen = model.log_ent_coef.detach().clone()
    model.train(gradient_steps=2, batch_size=16)
    assert torch.equal(model.log_ent_coef.detach(), frozen)

    model.num_timesteps = 100
    before = model.log_ent_coef.detach().clone()
    model.train(gradient_steps=4, batch_size=16)
    assert not torch.equal(model.log_ent_coef.detach(), before)


def test_joint_training_runs_the_predictive_auxiliary():
    model = _joint_model(actor_warmup=0, predictive_weight=1.0)
    model.train(gradient_steps=1, batch_size=4)
    assert model._n_updates == 1
    assert "train/predictive_loss" in model.logger.name_to_value
    assert "train/ent_coef_enc" in model.logger.name_to_value
    assert "train/ent_coef_vel" in model.logger.name_to_value


def test_attach_predictor_uses_the_encoder_head_dim():
    from fly_drone.sac import _attach_predictor

    model = _joint_model(actor_warmup=0)
    _attach_predictor(model, 1.0)
    first = next(m for m in model.predictor.modules() if isinstance(m, torch.nn.Linear))
    assert first.in_features == 16 + flat_dim()
