import hashlib
import json

import mujoco
import numpy as np
import pytest
import torch
from fly_drone import distill
from fly_drone.arena import ArenaSpec
from fly_drone.brain import DATA, ENCODER_VERSION, BrainRuntime
from fly_drone.encoder import LearnedEncoder
from fly_drone.env import ConnectomeEnv
from fly_drone.sac import (
    GEOMETRY,
    AsymmetricSACPolicy,
    SacRoamEnv,
    SpacesOnlyEnv,
    build_sac,
    export_decoder,
    set_dn_stats,
    train_round,
    validate,
    visible_geometry,
    warm_start_decoder,
)
from fly_drone.teacher import DRIVES
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


def test_export_decoder_matches_rust_with_tanh_output_and_pins_the_encoder(tmp_path):
    LearnedEncoder.fresh(seed=5).save(tmp_path / "e.pt")
    brain = BrainRuntime(encoder=tmp_path / "e.pt")
    model = build_sac("decoder", SpacesOnlyEnv("decoder"), buffer_size=1, device="cpu")
    rng = np.random.default_rng(0)
    set_dn_stats(model, rng.uniform(0, 0.1, 2022), rng.uniform(0.5, 1.5, 2022))
    error = export_decoder(model, brain, tmp_path / "d.json", ArenaSpec().limits)
    assert error <= 1e-4
    data = json.loads((tmp_path / "d.json").read_text())
    assert data["output"] == "tanh" and data["encoder_version"] == brain.encoder_version
    assert [len(layer["bias"]) for layer in data["layers"]] == [64, 64, 4]
    with pytest.raises(ValueError, match="encoder"):
        BrainRuntime().load_policy(tmp_path / "d.json")


def test_warm_start_decoder_learns_labels_and_shares_normalisation(tmp_path):
    rng = np.random.default_rng(1)
    n = 400
    x = rng.uniform(0, 1, (n, 2022)).astype(np.float32)
    path = tmp_path / "dagger.npz"
    np.savez(
        path,
        x=x.astype(np.float16),
        y=(x[:, :4] * 1.6 - 0.8).astype(np.float32),
        drive=rng.integers(0, len(DRIVES), n).astype(np.int8),
        flight=np.repeat(np.arange(20), n // 20).astype(np.int32),
        dataset_hash=hashlib.sha256((DATA / "graph.bin").read_bytes()).hexdigest(),
        encoder_version=ENCODER_VERSION,
        student="None",
        beta=1.0,
    )
    model = build_sac("decoder", SpacesOnlyEnv("decoder"), buffer_size=1, device="cpu")
    digest = hashlib.sha256((DATA / "graph.bin").read_bytes()).hexdigest()
    report = warm_start_decoder(model, [path], digest, steps=300)
    assert report["held_out_flights"] == 2 and set(report["drives"]) == set(DRIVES)
    assert report["loss_last"] < report["loss_first"]
    actor_norm = model.actor.features_extractor
    assert float(actor_norm.scale.min()) >= 0.003
    assert torch.equal(model.critic.features_extractor.dn_norm.mean, actor_norm.mean)
    assert torch.equal(
        model.critic_target.features_extractor.dn_norm.scale, actor_norm.scale
    )


def test_rounds_for_every_learner_resume_export_and_validate(tmp_path):
    decoder0 = zero_actor(tmp_path / "decoder0.json", BrainRuntime())
    LearnedEncoder.fresh(seed=7).save(tmp_path / "clone.pt")
    small = {"workers": 1, "buffer_size": 200, "device": "cpu", "learning_starts": 20}

    enc = train_round(
        "encoder",
        tmp_path / "enc1",
        40,
        decoder=decoder0,
        init=tmp_path / "clone.pt",
        **small,
    )
    assert (
        enc["encoder_version"]
        == LearnedEncoder.load(tmp_path / "enc1" / "encoder.pt").version
    )
    assert enc["frames"] >= 40

    again = train_round(
        "encoder",
        tmp_path / "enc2",
        30,
        decoder=decoder0,
        init=tmp_path / "enc1" / "encoder.zip",
        **small,
    )
    assert again["encoder_version"] != enc["encoder_version"]

    encoder_pt = tmp_path / "enc1" / "encoder.pt"
    dec = train_round("decoder", tmp_path / "dec1", 40, encoder=encoder_pt, **small)
    assert dec["export_max_error"] <= 1e-4
    BrainRuntime(encoder=encoder_pt).load_policy(tmp_path / "dec1" / "decoder.json")

    train_round("bypass", tmp_path / "byp", 40, encoder=encoder_pt, **small)
    job = (
        f"bypass:{tmp_path / 'byp' / 'bypass.zip'}",
        [3],
        0.4,
        3,
        "none",
        str(encoder_pt),
    )
    assert len(distill._screen_job(job)[2]) == 1

    summary = validate(
        tmp_path / "dec1" / "decoder.json",
        encoder_pt,
        tmp_path / "val.json",
        seeds=1,
        seconds=0.4,
        workers=1,
    )
    assert {"near_dodge_rate", "beacons_per_min", "E1", "E2"} <= set(summary)
    assert json.loads((tmp_path / "val.json").read_text())["encoder"] == str(
        encoder_pt.resolve()
    )


def test_train_round_rejects_a_missing_frozen_partner_before_spawning(tmp_path):
    with pytest.raises(ValueError, match="frozen decoder"):
        train_round("encoder", tmp_path / "e", 10, encoder="x.pt")
    for learner in ("decoder", "bypass"):
        with pytest.raises(ValueError, match="frozen encoder"):
            train_round(learner, tmp_path / learner, 10, decoder="x.json")
    assert not any(tmp_path.iterdir())
