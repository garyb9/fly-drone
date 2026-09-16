import json

import numpy as np
import pytest
import torch
from fly_drone.brain import ENCODER_VERSION, BrainRuntime
from fly_drone.sac import (
    LAMBDA_LIGHT,
    LAMBDA_LOOM,
    SacRoamEnv,
    SpacesOnlyEnv,
    build_sac,
)
from fly_drone.spatial_encoder import flat_dim


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


def test_v6_encoder_env_spaces_and_metabolic_cost(tmp_path):
    decoder = zero_actor(tmp_path / "decoder.json", BrainRuntime())
    env = SacRoamEnv("encoder", decoder=decoder, level=3, spatial=True)
    try:
        obs, _ = env.reset(seed=42)
        assert obs["eyes"].shape == (6, 48, 64)
        assert obs["dn"].shape == (2022,)
        assert env.action_space.shape == (flat_dim(),)
        _, _, _, _, info = env.step(-np.ones(flat_dim(), np.float32))
        assert info["metabolic_cost"] == pytest.approx(0.0)
        _, _, _, _, info = env.step(np.ones(flat_dim(), np.float32))
        assert info["metabolic_cost"] == pytest.approx(
            LAMBDA_LOOM * 2.0 + LAMBDA_LIGHT * 2.0
        )
        np.testing.assert_array_equal(env.env.brain.cues, np.full(flat_dim(), 2.0))
    finally:
        env.env.close()


def test_spatial_bypass_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="bypass"):
        SacRoamEnv("bypass", encoder=None, spatial=True)


def test_v6_policy_builds_with_the_spatial_actor():
    model = build_sac(
        "encoder",
        SpacesOnlyEnv("encoder", spatial=True),
        buffer_size=1,
        device="cpu",
        spatial=True,
    )
    from fly_drone.spatial_policy import SpatialActor, SpatialSACPolicy

    assert isinstance(model.policy, SpatialSACPolicy)
    assert isinstance(model.actor, SpatialActor)
    assert isinstance(model.actor.mu, torch.nn.Identity)
    assert model.action_space.shape == (flat_dim(),)
