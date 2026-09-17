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


def test_anchor_and_saturation_penalties():
    from fly_drone.sac import anchor_penalty, saturation_penalty

    assert anchor_penalty([1.0, 1.0], [1.0, 1.0]) == pytest.approx(0.0)
    assert anchor_penalty([0.0, 0.0], [1.0, 1.0]) == pytest.approx(1.0)
    assert saturation_penalty([1.0, 1.0]) == pytest.approx(0.0)
    assert saturation_penalty([2.0, 0.0]) == pytest.approx(0.04)


def test_encoder_env_applies_anchor_and_saturation_cost(tmp_path):
    from fly_drone.sac import SacRoamEnv
    from fly_drone.spatial_encoder import SpatialEncoder

    decoder = zero_actor(tmp_path / "d.json", BrainRuntime())
    anchor = SpatialEncoder.fresh(seed=1)
    anchor.save(tmp_path / "a.pt")
    env = SacRoamEnv(
        "encoder", decoder=decoder, level=3, spatial=True, anchor=str(tmp_path / "a.pt")
    )
    try:
        env.reset(seed=7)
        _, _, _, _, info = env.step(np.ones(flat_dim(), np.float32))
        assert info["anchor_cost"] > 0
        assert info["saturation_cost"] > 0
        assert info["metabolic_cost"] >= info["anchor_cost"] + info["saturation_cost"]
    finally:
        env.env.close()


def test_anchor_rejected_for_non_encoder_learner(tmp_path):
    from fly_drone.sac import SacRoamEnv
    from fly_drone.spatial_encoder import SpatialEncoder

    SpatialEncoder.fresh(seed=0).save(tmp_path / "e.pt")
    SpatialEncoder.fresh(seed=1).save(tmp_path / "a.pt")
    with pytest.raises(ValueError, match="anchor"):
        SacRoamEnv(
            "decoder",
            encoder=str(tmp_path / "e.pt"),
            spatial=True,
            anchor=str(tmp_path / "a.pt"),
        )


def test_ent_coef_scales_with_action_dimension():
    from fly_drone.sac import ENT_COEF_INIT, ENT_COEF_REF_DIM, ent_coef_init

    assert ent_coef_init(4) == ENT_COEF_INIT
    assert ent_coef_init(8) == ENT_COEF_INIT
    assert ent_coef_init(flat_dim()) == pytest.approx(
        ENT_COEF_INIT * ENT_COEF_REF_DIM / flat_dim()
    )


def test_spatial_decoder_uses_the_velocity_policy_not_the_spatial_actor():
    from fly_drone.sac import AsymmetricSACPolicy, build_sac

    model = build_sac(
        "decoder",
        SpacesOnlyEnv("decoder", spatial=True),
        buffer_size=1,
        device="cpu",
        spatial=True,
    )
    assert isinstance(model.policy, AsymmetricSACPolicy)
    assert model.action_space.shape == (4,)


def test_spatial_encoder_uses_the_scaled_entropy_alpha():
    from fly_drone.sac import build_sac, ent_coef_init

    model = build_sac(
        "encoder",
        SpacesOnlyEnv("encoder", spatial=True),
        buffer_size=1,
        device="cpu",
        spatial=True,
    )
    assert model.ent_coef == f"auto_{ent_coef_init(flat_dim())}"


def test_encoder_predictive_head_is_wired_and_trains():
    from fly_drone.sac import SpacesOnlyEnv, build_sac
    from stable_baselines3.common.logger import configure

    env = SpacesOnlyEnv("encoder", spatial=True)
    model = build_sac(
        "encoder",
        env,
        buffer_size=64,
        device="cpu",
        spatial=True,
        predictive_weight=1.0,
    )
    assert model.predictor is not None
    # The predictor has its own optimizer; the actor optimizer stays SB3-loadable (one group).
    assert model.aux_optimizer is not None
    assert len(model.actor.optimizer.param_groups) == 1
    space = env.observation_space
    for _ in range(8):
        obs = {k: space[k].sample() for k in space.spaces}
        nxt = {k: space[k].sample() for k in space.spaces}
        model.replay_buffer.add(obs, nxt, model.action_space.sample(), 1.0, False, [{}])
    model.set_logger(configure(None, []))
    model.num_timesteps = model.actor_warmup  # past the critic-only warm-up
    model.train(1, batch_size=4)


def test_predictive_model_saves_and_resumes(tmp_path):
    from fly_drone.sac import SpacesOnlyEnv, WarmupSAC, _attach_predictor, build_sac
    from stable_baselines3.common.logger import configure

    env = SpacesOnlyEnv("encoder", spatial=True)
    model = build_sac(
        "encoder",
        env,
        buffer_size=64,
        device="cpu",
        spatial=True,
        predictive_weight=1.0,
    )
    model.set_logger(configure(None, []))
    model.save(tmp_path / "model")
    # The actor optimizer is untouched, so SB3's load must succeed; the predictor is re-attached.
    loaded = WarmupSAC.load(
        tmp_path / "model.zip", env=env, device="cpu", buffer_size=64, actor_warmup=0
    )
    _attach_predictor(loaded, 1.0)
    assert loaded.predictor is not None
    assert loaded.aux_optimizer is not None
