import numpy as np
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
