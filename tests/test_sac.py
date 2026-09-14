import json

import mujoco
import numpy as np
import pytest
import torch
from fly_drone.brain import ENCODER_VERSION, BrainRuntime
from fly_drone.env import ConnectomeEnv
from fly_drone.sac import (
    GEOMETRY,
    AsymmetricSACPolicy,
    SacRoamEnv,
    visible_geometry,
)
from gymnasium import spaces


def zero_actor(path, brain):
    n = len(brain.feature_ids)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "encoder_version": ENCODER_VERSION,
                "dataset_hash": brain.dataset_hash,
                "feature_ids": brain.feature_ids,
                "mean": [0.0] * n,
                "scale": [1.0] * n,
                "layers": [{"weights": [[0.0] * n] * 4, "bias": [0.0] * 4}],
                "action_limits": [0.7, 0.5, 0.3, 0.8],
            }
        )
    )
    return path


def test_visible_geometry_reports_only_what_the_eyes_can_see():
    env = ConnectomeEnv(task="free_roam", level=0, respawn=False)
    try:
        env.reset(seed=31)
        env.plant.teleport([0.0, 0.0, 1.0], 0.0)
        env.plant.set_objects(target=[3.0, 0.5, 1.0])
        g = visible_geometry(env)
        assert g[4] == 1.0 and g[5] == pytest.approx(3.0, abs=0.05)
        assert g[6] == pytest.approx(0.5, abs=0.05)
        assert not g[:4].any()  # no threat flying
        env.plant.set_objects(target=[-3.0, 0.0, 1.0])
        assert not visible_geometry(env)[4:].any()
        assert env.launch_threat()
        assert visible_geometry(env)[0] == 1.0
        env.plant.set_ghost(True)
        assert not visible_geometry(env)[:4].any()
    finally:
        env.close()


def test_encoder_env_spaces_and_metabolic_cost(tmp_path):
    decoder = zero_actor(tmp_path / "decoder.json", BrainRuntime())
    env = SacRoamEnv("encoder", decoder=decoder, level=3)
    try:
        obs, _ = env.reset(seed=32)
        plant = env.env.plant
        bands = [
            g
            for g in range(plant.model.ngeom)
            if (
                mujoco.mj_id2name(plant.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            ).endswith("_band")
        ]
        greys = plant.model.geom_rgba[bands, :3]
        assert (
            len(bands) == 4
            and np.all(greys == greys[0, 0])
            and 0.0 <= greys[0, 0] <= 0.15
        )
        assert np.all(plant.model.geom_rgba[bands, 3] == 1.0)
        assert obs["eyes"].shape == (6, 48, 64) and obs["eyes"].dtype == np.uint8
        assert obs["dn"].shape == (2022,) and obs["geometry"].shape == (GEOMETRY,)
        assert env.action_space.shape == (8,)
        _, _, _, _, info = env.step(-np.ones(8))
        assert info["metabolic_cost"] == pytest.approx(0.0)
        _, _, _, _, info = env.step(np.ones(8))
        assert info["metabolic_cost"] == pytest.approx(0.01 * 2 + 0.002 * 2)
        np.testing.assert_array_equal(env.env.brain.cues, np.full(8, 2.0))
    finally:
        env.env.close()


def test_learner_arguments_are_validated(tmp_path):
    with pytest.raises(ValueError):
        SacRoamEnv("planner")
    with pytest.raises(ValueError):
        SacRoamEnv("encoder")
    with pytest.raises(ValueError):
        SacRoamEnv("decoder")


@pytest.mark.parametrize("actor_key", ["eyes", "dn"])
def test_actor_reads_only_its_key_and_critic_reads_all(actor_key):
    space = spaces.Dict(
        {
            "eyes": spaces.Box(0, 255, (6, 48, 64), np.uint8),
            "dn": spaces.Box(0, 1, (5,), np.float32),
            "geometry": spaces.Box(-np.inf, np.inf, (GEOMETRY,), np.float32),
        }
    )
    policy = AsymmetricSACPolicy(
        space,
        spaces.Box(-1, 1, (4,), np.float32),
        lambda _: 3e-4,
        actor_key=actor_key,
        net_arch={"pi": [8], "qf": [8]},
    )
    torch.manual_seed(0)
    a = {
        "eyes": torch.rand(2, 6, 48, 64),
        "dn": torch.rand(2, 5),
        "geometry": torch.rand(2, 8),
    }
    b = dict(a)
    for key in a:
        if key != actor_key:
            b[key] = torch.rand_like(a[key])
    action = torch.zeros(2, 4)
    with torch.no_grad():
        torch.testing.assert_close(
            policy.actor(a, deterministic=True), policy.actor(b, deterministic=True)
        )
        assert not torch.allclose(
            policy.critic(a, action)[0], policy.critic(b, action)[0]
        )
