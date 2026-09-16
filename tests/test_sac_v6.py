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


def test_v6_export_checkpoint_round_trips(tmp_path):
    from fly_drone.sac import export_checkpoint
    from fly_drone.spatial_encoder import SpatialEncoder

    model = build_sac(
        "encoder",
        SpacesOnlyEnv("encoder", spatial=True),
        buffer_size=1,
        device="cpu",
        spatial=True,
    )
    model.save(tmp_path / "encoder")
    report = export_checkpoint(
        str(tmp_path / "encoder.zip"), "encoder", tmp_path / "e.pt"
    )
    assert report["encoder_version"].startswith("learned-v6:")
    assert SpatialEncoder.load(tmp_path / "e.pt").version == report["encoder_version"]


def test_repin_decoder_pins_the_v6_version(tmp_path):
    from fly_drone.sac import repin_decoder
    from fly_drone.spatial_encoder import SpatialEncoder

    enc = SpatialEncoder.fresh(seed=0)
    enc.save(tmp_path / "e.pt")
    n = len(BrainRuntime().feature_ids)
    src = tmp_path / "dec.json"
    src.write_text(
        json.dumps(
            {
                "version": 1,
                "encoder_version": ENCODER_VERSION,
                "dataset_hash": "h",
                "feature_ids": [0] * n,
                "mean": [0.0] * n,
                "scale": [1.0] * n,
                "layers": [{"weights": [[0.0] * n] * 4, "bias": [0.0] * 4}],
                "action_limits": [0.7, 0.5, 0.3, 0.8],
                "output": "tanh",
            }
        )
    )
    out = repin_decoder(src, tmp_path / "e.pt", tmp_path / "pinned.json")
    assert out["encoder_version"] == enc.version


def test_encoder_scores_uses_the_v6_groups():
    from fly_drone.roam_eval import encoder_scores
    from fly_drone.spatial_encoder import group_slices

    currents = np.zeros((4, flat_dim()), np.float32)
    loom = group_slices()["loom"]
    currents[0, loom] = 2.0
    currents[1, loom] = 2.0
    scores = encoder_scores(currents, [1, 1, 0, -1], [False, False, True, False])
    assert scores["E1"]["loom_auc"] == pytest.approx(1.0)
    assert scores["E1"]["passed"] is True


def test_infer_spatial_detects_v6_encoders(tmp_path):
    from fly_drone.cli import _infer_spatial
    from fly_drone.spatial_encoder import SpatialEncoder

    path = tmp_path / "e.pt"
    SpatialEncoder.fresh(seed=0).save(path)
    assert _infer_spatial("decoder", str(path), None) is True
    assert _infer_spatial("encoder", None, str(path)) is True
    assert _infer_spatial("encoder", None, "runs/x.zip") is False
    assert _infer_spatial("encoder", None, None) is False
