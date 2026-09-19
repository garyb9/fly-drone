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
    ACTOR_WARMUP_FRAMES,
    ENT_COEF_INIT,
    GEOMETRY,
    LOG_STD_MAX,
    LOG_STD_MIN,
    TARGET_ENTROPY_PER_DIM,
    WARM_START_LOG_STD,
    AsymmetricSACPolicy,
    ClampedActor,
    SacRoamEnv,
    SpacesOnlyEnv,
    WarmupSAC,
    build_sac,
    export_checkpoint,
    export_decoder,
    repin_decoder,
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
        assert info["loom_cost"] == pytest.approx(0.0)
        assert info["light_cost"] == pytest.approx(0.0)
        _, _, _, _, info = env.step(np.ones(8))
        assert info["loom_cost"] == pytest.approx(0.01 * 2)
        assert info["light_cost"] == pytest.approx(0.002 * 2)
        assert info["metabolic_cost"] == pytest.approx(
            info["loom_cost"] + info["light_cost"]
        )
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
    report = warm_start_decoder(model, [path], digest, ENCODER_VERSION, steps=300)
    assert report["held_out_flights"] == 2 and set(report["drives"]) == set(DRIVES)
    assert report["loss_last"] < report["loss_first"]
    for split in ("train", "held_out"):
        axes = {f"{split}_{a}_mse" for a in ("vx", "vy", "vz", "yaw")}
        assert axes <= set(report["drives"]["threat"])
    actor_norm = model.actor.features_extractor
    assert float(actor_norm.scale.min()) >= 0.003
    assert torch.equal(model.critic.features_extractor.dn_norm.mean, actor_norm.mean)
    assert torch.equal(
        model.critic_target.features_extractor.dn_norm.scale, actor_norm.scale
    )


def test_warm_start_decoder_fits_saturated_labels_through_the_pre_tanh_mean(tmp_path):
    rng = np.random.default_rng(1)
    n = 400
    x = rng.uniform(0, 1, (n, 2022)).astype(np.float32)
    y = np.clip((x[:, :4] * 2 - 1) * 4, -1, 1).astype(np.float32)
    saturated = np.abs(y) >= 0.97
    assert saturated.mean() > 0.5
    digest = hashlib.sha256((DATA / "graph.bin").read_bytes()).hexdigest()
    path = tmp_path / "saturated.npz"
    np.savez(
        path,
        x=x.astype(np.float16),
        y=y,
        drive=rng.integers(0, len(DRIVES), n).astype(np.int8),
        flight=np.repeat(np.arange(20), n // 20).astype(np.int32),
        dataset_hash=digest,
        encoder_version=ENCODER_VERSION,
        student="None",
        beta=1.0,
    )
    model = build_sac("decoder", SpacesOnlyEnv("decoder"), buffer_size=1, device="cpu")
    # 50 steps: the old tanh-space loss left 0.57 here (vanishing gradient near +-1).
    warm_start_decoder(model, [path], digest, ENCODER_VERSION, steps=50)
    with torch.no_grad():
        action = model.actor({"dn": torch.as_tensor(x)}, deterministic=True).numpy()
    err = (action - np.clip(y, -0.97, 0.97)) ** 2
    assert err[saturated].mean() < 0.3


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
    assert enc["actor_warmup"] == ACTOR_WARMUP_FRAMES
    saved = json.loads((tmp_path / "enc1" / "round.json").read_text())
    assert saved["actor_warmup"] == ACTOR_WARMUP_FRAMES

    # A resumed round takes its warm-up from the argument, not the .zip.
    again = train_round(
        "encoder",
        tmp_path / "enc2",
        30,
        decoder=decoder0,
        init=tmp_path / "enc1" / "encoder.zip",
        actor_warmup=0,
        **small,
    )
    assert again["encoder_version"] != enc["encoder_version"]
    assert again["actor_warmup"] == 0

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


def test_export_checkpoint_writes_encoder_and_parity_checked_decoder(tmp_path):
    enc_model = build_sac(
        "encoder", SpacesOnlyEnv("encoder"), buffer_size=1, device="cpu"
    )
    enc_model.save(tmp_path / "enc_ckpt")
    result = export_checkpoint(
        tmp_path / "enc_ckpt.zip", "encoder", tmp_path / "encoder.pt"
    )
    assert result["learner"] == "encoder"
    assert (
        result["encoder_version"]
        == LearnedEncoder.load(tmp_path / "encoder.pt").version
    )

    LearnedEncoder.fresh(seed=9).save(tmp_path / "e2.pt")
    dec_model = build_sac(
        "decoder", SpacesOnlyEnv("decoder"), buffer_size=1, device="cpu"
    )
    rng = np.random.default_rng(3)
    set_dn_stats(dec_model, rng.uniform(0, 0.1, 2022), rng.uniform(0.5, 1.5, 2022))
    dec_model.save(tmp_path / "dec_ckpt")
    result = export_checkpoint(
        tmp_path / "dec_ckpt.zip",
        "decoder",
        tmp_path / "decoder.json",
        encoder=tmp_path / "e2.pt",
    )
    assert result["export_max_error"] <= 1e-4
    BrainRuntime(encoder=tmp_path / "e2.pt").load_policy(tmp_path / "decoder.json")

    with pytest.raises(ValueError, match="learner"):
        export_checkpoint(tmp_path / "dec_ckpt.zip", "bypass", tmp_path / "x.json")
    with pytest.raises(ValueError, match="encoder"):
        export_checkpoint(tmp_path / "dec_ckpt.zip", "decoder", tmp_path / "x.json")


def test_repin_decoder_copies_weights_verbatim_and_updates_pinning(tmp_path):
    LearnedEncoder.fresh(seed=1).save(tmp_path / "old.pt")
    LearnedEncoder.fresh(seed=2).save(tmp_path / "new.pt")
    old_brain = BrainRuntime(encoder=tmp_path / "old.pt")
    model = build_sac("decoder", SpacesOnlyEnv("decoder"), buffer_size=1, device="cpu")
    rng = np.random.default_rng(4)
    set_dn_stats(model, rng.uniform(0, 0.1, 2022), rng.uniform(0.5, 1.5, 2022))
    export_decoder(model, old_brain, tmp_path / "decoder.json", ArenaSpec().limits)
    source = json.loads((tmp_path / "decoder.json").read_text())

    new_encoder_version = LearnedEncoder.load(tmp_path / "new.pt").version
    result = repin_decoder(
        tmp_path / "decoder.json", tmp_path / "new.pt", tmp_path / "repinned.json"
    )
    assert result["encoder_version"] == new_encoder_version
    assert result["repinned_from"] == old_brain.encoder_version

    repinned = json.loads((tmp_path / "repinned.json").read_text())
    assert repinned["layers"] == source["layers"]
    assert repinned["mean"] == source["mean"] and repinned["scale"] == source["scale"]
    assert repinned["output"] == source["output"]
    assert repinned["encoder_version"] == new_encoder_version
    assert repinned["repinned_from"] == old_brain.encoder_version

    # Loads strictly (check_encoder=True) into a runtime built on the new encoder.
    BrainRuntime(encoder=tmp_path / "new.pt").load_policy(tmp_path / "repinned.json")

    bad = dict(source)
    bad["output"] = "linear"
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="tanh"):
        repin_decoder(tmp_path / "bad.json", tmp_path / "new.pt", tmp_path / "out.json")

    missing = dict(source)
    del missing["layers"]
    (tmp_path / "missing.json").write_text(json.dumps(missing))
    with pytest.raises(ValueError, match="missing fields"):
        repin_decoder(
            tmp_path / "missing.json", tmp_path / "new.pt", tmp_path / "out.json"
        )


def test_resumed_round_uses_the_given_seed_not_the_saved_one(tmp_path, monkeypatch):
    """SAC.load restores the saved seed unless told otherwise; a resumed round must
    still draw its first training episode from the `--seed` the caller passed in."""
    decoder0 = zero_actor(tmp_path / "decoder0.json", BrainRuntime())
    LearnedEncoder.fresh(seed=7).save(tmp_path / "clone.pt")
    small = {"workers": 1, "buffer_size": 50, "device": "cpu", "learning_starts": 5}
    train_round(
        "encoder",
        tmp_path / "base",
        10,
        decoder=decoder0,
        init=tmp_path / "clone.pt",
        **small,
    )

    seen = []
    original_reset = SacRoamEnv.reset

    def recording_reset(self, seed=None, options=None):
        seen.append(seed)
        return original_reset(self, seed=seed, options=options)

    monkeypatch.setattr(SacRoamEnv, "reset", recording_reset)

    seen.clear()
    train_round(
        "encoder",
        tmp_path / "resumeA",
        10,
        decoder=decoder0,
        init=tmp_path / "base" / "encoder.zip",
        seed=101,
        **small,
    )
    first_a = seen[1]  # index 0 is the factory's construction reset

    seen.clear()
    train_round(
        "encoder",
        tmp_path / "resumeB",
        10,
        decoder=decoder0,
        init=tmp_path / "base" / "encoder.zip",
        seed=202,
        **small,
    )
    first_b = seen[1]

    assert first_a == 101 and first_b == 202


def test_train_round_resumes_from_a_snapshot(tmp_path, monkeypatch):
    """A crash snapshot continues the round from its saved timestep instead of restarting
    the warm-up: the resumed call runs the remainder of the round's target."""
    from fly_drone import sac

    decoder0 = zero_actor(tmp_path / "decoder0.json", BrainRuntime())
    LearnedEncoder.fresh(seed=7).save(tmp_path / "clone.pt")
    small = {
        "workers": 1,
        "buffer_size": 50,
        "device": "cpu",
        "learning_starts": 5,
        "resume_every": 10,
    }
    out = tmp_path / "round"
    # Simulate a crash right after the last snapshot by keeping the resume directory.
    monkeypatch.setattr(sac.shutil, "rmtree", lambda path: None)
    first = train_round(
        "encoder", out, 20, decoder=decoder0, init=tmp_path / "clone.pt", **small
    )
    assert (out / "resume" / "model.zip").exists()
    assert (out / "resume" / "buffer.pkl").exists()
    assert first["resumed_from"] is None

    second = train_round(
        "encoder",
        out,
        40,
        decoder=decoder0,
        init=tmp_path / "clone.pt",
        resume=True,
        **small,
    )
    assert second["resumed_from"] == first["frames"]  # continued, not restarted
    assert second["frames"] == 40  # `frames` is the round total, not extra steps


def test_keep_resume_leaves_the_snapshot_for_a_second_stage(tmp_path):
    decoder0 = zero_actor(tmp_path / "decoder0.json", BrainRuntime())
    LearnedEncoder.fresh(seed=7).save(tmp_path / "clone.pt")
    out = tmp_path / "round"
    report = train_round(
        "encoder",
        out,
        20,
        decoder=decoder0,
        init=tmp_path / "clone.pt",
        workers=1,
        buffer_size=50,
        device="cpu",
        learning_starts=5,
        resume_every=10,
        keep_resume=True,
    )
    assert (out / "resume" / "model.zip").exists()
    assert (out / "resume" / "buffer.pkl").exists()
    assert report["snapshot"] == str(out / "resume")


def small_warmup_model(actor_warmup):
    from stable_baselines3.common.logger import Logger

    env = SpacesOnlyEnv("decoder", n_features=16)
    model = build_sac(
        "decoder", env, buffer_size=64, device="cpu", actor_warmup=actor_warmup
    )
    model.set_logger(Logger(folder=None, output_formats=[]))
    rng = np.random.default_rng(11)
    space = env.observation_space
    for _ in range(32):
        obs = {
            k: rng.uniform(0, 1, (1, *s.shape)).astype(s.dtype)
            for k, s in space.items()
        }
        nxt = {
            k: rng.uniform(0, 1, (1, *s.shape)).astype(s.dtype)
            for k, s in space.items()
        }
        model.replay_buffer.add(
            obs,
            nxt,
            rng.uniform(-1, 1, (1, 4)).astype(np.float32),
            rng.normal(size=1).astype(np.float32),
            np.zeros(1, dtype=bool),
            [{}],
        )
    return model


def snapshot(module):
    return [p.detach().clone() for p in module.parameters()]


def unchanged(module, before):
    return all(
        torch.equal(p, q) for p, q in zip(module.parameters(), before, strict=True)
    )


def test_warmup_trains_only_the_critic_then_normal_sac():
    assert ACTOR_WARMUP_FRAMES == 50_000
    model = small_warmup_model(actor_warmup=100)
    model.num_timesteps = 60
    actor, critic = snapshot(model.actor), snapshot(model.critic)
    target = snapshot(model.critic_target)
    alpha = model.log_ent_coef.detach().clone()
    model.train(gradient_steps=4, batch_size=16)
    assert unchanged(model.actor, actor)
    assert torch.equal(model.log_ent_coef.detach(), alpha)
    assert not model.actor.optimizer.state  # Adam moments untouched too
    assert not unchanged(model.critic, critic)
    assert not unchanged(model.critic_target, target)
    assert "step" not in vars(model.actor.optimizer)  # patch removed

    model.num_timesteps = 100  # warm-up over
    model.train(gradient_steps=4, batch_size=16)
    assert not unchanged(model.actor, actor)
    assert not torch.equal(model.log_ent_coef.detach(), alpha)


def test_warmup_skips_the_actor_forward_pass():
    model = small_warmup_model(actor_warmup=100)
    model.num_timesteps = 60
    calls = []
    original = model.actor.action_log_prob

    def counting(obs):
        calls.append(1)
        return original(obs)

    model.actor.action_log_prob = counting
    model.train(gradient_steps=4, batch_size=16)
    # Once per step, for the target's next action; the actor-loss call is gone.
    assert len(calls) == 4
    assert "train/critic_loss" in model.logger.name_to_value
    assert "train/actor_loss" not in model.logger.name_to_value

    model.num_timesteps = 100  # warm-up over: SAC's full loop returns
    before = len(calls)
    model.train(gradient_steps=4, batch_size=16)
    assert len(calls) - before == 8  # next-action target + the actor loss
    assert "train/actor_loss" in model.logger.name_to_value


def test_zero_warmup_updates_the_actor_immediately():
    model = small_warmup_model(actor_warmup=0)
    model.num_timesteps = 0
    actor = snapshot(model.actor)
    model.train(gradient_steps=4, batch_size=16)
    assert not unchanged(model.actor, actor)


def test_warmup_survives_save_and_load(tmp_path):
    model = small_warmup_model(actor_warmup=100)
    model.num_timesteps = 70
    model.save(tmp_path / "m")
    loaded = WarmupSAC.load(tmp_path / "m.zip", device="cpu", buffer_size=1)
    assert loaded.actor_warmup == 100 and loaded.num_timesteps == 70
    assert loaded.actor_frozen
    override = WarmupSAC.load(
        tmp_path / "m.zip", device="cpu", buffer_size=1, actor_warmup=0
    )
    assert override.actor_warmup == 0 and not override.actor_frozen

    from stable_baselines3 import SAC

    plain = SAC("MlpPolicy", "Pendulum-v1", buffer_size=1, device="cpu")
    plain.save(tmp_path / "plain")
    # A .zip saved by plain SAC takes the warm-up from the load argument.
    restored = WarmupSAC.load(tmp_path / "plain.zip", device="cpu", actor_warmup=7)
    assert restored.actor_warmup == 7 and restored.actor_frozen


def test_decoder_in_loss_bc_anchor_is_logged_and_pulls_the_actor(tmp_path):
    import json

    from fly_drone.sac import _attach_decoder_anchor

    n = 16
    payload = {
        "version": 1,
        "encoder_version": "learned-v6:" + "0" * 16,
        "dataset_hash": "h",
        "mean": [0.0] * n,
        "scale": [1.0] * n,
        "layers": [{"weights": [[0.0] * n] * 4, "bias": [0.0] * 4}],
        "action_limits": [1.0, 1.0, 1.0, 1.0],
    }
    path = tmp_path / "anchor.json"
    path.write_text(json.dumps(payload))
    model = small_warmup_model(actor_warmup=0)
    _attach_decoder_anchor(model, path)
    assert model.decoder_anchor is not None and model.bc_weight > 0
    model.num_timesteps = 0
    model.train(gradient_steps=4, batch_size=16)
    assert "train/bc_loss" in model.logger.name_to_value
    assert "train/actor_loss" in model.logger.name_to_value


def test_attach_decoder_anchor_rejects_a_mismatched_encoder(tmp_path):
    import json

    from fly_drone.sac import _attach_decoder_anchor

    version = LearnedEncoder.fresh(seed=6).save(tmp_path / "e.pt")
    n = 16
    payload = {
        "version": 1,
        "encoder_version": "learned-v5:" + "0" * 16,
        "dataset_hash": "h",
        "mean": [0.0] * n,
        "scale": [1.0] * n,
        "layers": [{"weights": [[0.0] * n] * 4, "bias": [0.0] * 4}],
        "action_limits": [1.0, 1.0, 1.0, 1.0],
    }
    path = tmp_path / "anchor.json"
    path.write_text(json.dumps(payload))
    model = small_warmup_model(actor_warmup=0)
    assert version != payload["encoder_version"]
    with pytest.raises(ValueError, match="decoder anchor"):
        _attach_decoder_anchor(model, path, encoder=tmp_path / "e.pt")


def test_train_round_rejects_a_missing_frozen_partner_before_spawning(tmp_path):
    with pytest.raises(ValueError, match="frozen decoder"):
        train_round("encoder", tmp_path / "e", 10, encoder="x.pt")
    for learner in ("decoder", "bypass"):
        with pytest.raises(ValueError, match="frozen encoder"):
            train_round(learner, tmp_path / learner, 10, decoder="x.json")
    assert not any(tmp_path.iterdir())


def test_fresh_round_starts_at_the_small_alpha_and_warm_start_target_entropy():
    assert LOG_STD_MIN < WARM_START_LOG_STD < LOG_STD_MAX
    model = build_sac("decoder", SpacesOnlyEnv("decoder"), buffer_size=1, device="cpu")
    assert isinstance(model.actor, ClampedActor)
    assert float(torch.exp(model.log_ent_coef).detach()) == pytest.approx(ENT_COEF_INIT)
    dim = int(np.prod(model.action_space.shape))
    assert dim == 4
    assert model.target_entropy == pytest.approx(dim * TARGET_ENTROPY_PER_DIM)
    encoder = build_sac(
        "encoder", SpacesOnlyEnv("encoder"), buffer_size=1, device="cpu"
    )
    assert encoder.target_entropy == pytest.approx(8 * TARGET_ENTROPY_PER_DIM)


def test_clamped_actor_holds_log_std_inside_the_configured_band():
    model = build_sac("decoder", SpacesOnlyEnv("decoder"), buffer_size=1, device="cpu")
    space = model.observation_space
    obs = {
        k: torch.as_tensor(np.stack([space[k].sample() for _ in range(3)]))
        for k in space.spaces
    }
    with torch.no_grad():
        model.actor.log_std.weight.zero_()
        model.actor.log_std.bias.fill_(WARM_START_LOG_STD)
        _, mid, _ = model.actor.get_action_dist_params(obs)
        model.actor.log_std.bias.fill_(5.0)  # SB3's own cap allows +2
        _, high, _ = model.actor.get_action_dist_params(obs)
        model.actor.log_std.bias.fill_(-10.0)
        _, low, _ = model.actor.get_action_dist_params(obs)
    assert float(mid.min()) == pytest.approx(WARM_START_LOG_STD)
    assert float(high.max()) == pytest.approx(LOG_STD_MAX)
    assert float(low.min()) == pytest.approx(LOG_STD_MIN)


def test_resumed_round_carries_the_saved_alpha(tmp_path):
    model = small_warmup_model(actor_warmup=0)
    with torch.no_grad():
        model.log_ent_coef.fill_(torch.log(torch.tensor(0.003)))
    model.save(tmp_path / "m")
    loaded = WarmupSAC.load(tmp_path / "m.zip", device="cpu", buffer_size=1)
    assert float(torch.exp(loaded.log_ent_coef).detach()) == pytest.approx(0.003)


def test_apply_entropy_regime_overrides_an_old_saved_schedule(tmp_path):
    from fly_drone.sac import apply_entropy_regime

    model = build_sac("decoder", SpacesOnlyEnv("decoder"), buffer_size=1, device="cpu")
    # A model saved before the regime landed: automatic schedule, alpha pinned at 1.0.
    model.ent_coef = "auto"
    model.target_entropy = -4.0
    with torch.no_grad():
        model.log_ent_coef.fill_(0.0)
    model.save(tmp_path / "old")

    loaded = WarmupSAC.load(tmp_path / "old.zip", device="cpu", buffer_size=1)
    assert float(torch.exp(loaded.log_ent_coef).detach()) == pytest.approx(1.0)
    apply_entropy_regime(loaded, 4)
    assert loaded.ent_coef == f"auto_{ENT_COEF_INIT}"
    assert loaded.target_entropy == pytest.approx(4 * TARGET_ENTROPY_PER_DIM)
    assert float(torch.exp(loaded.log_ent_coef).detach()) == pytest.approx(
        ENT_COEF_INIT
    )


def test_probe_logger_records_log_std_and_detects_a_constant_action():
    from fly_drone.sac import ProbeLogger
    from stable_baselines3.common.logger import Logger

    model = build_sac("decoder", SpacesOnlyEnv("decoder"), buffer_size=1, device="cpu")
    model.set_logger(Logger(folder=None, output_formats=[]))
    model.num_timesteps = 0
    callback = ProbeLogger(every=1, batch=8, seed=0)
    callback.init_callback(model)
    with torch.no_grad():
        model.actor.log_std.weight.zero_()
        model.actor.log_std.bias.fill_(WARM_START_LOG_STD)
    assert callback._on_step()
    values = model.logger.name_to_value
    assert values["probe/log_std_mean"] == pytest.approx(WARM_START_LOG_STD)
    assert values["probe/action_state_std"] > 0  # a fresh actor varies with the state
    with torch.no_grad():
        for param in model.actor.mu.parameters():
            param.zero_()
    assert callback._on_step()
    # A mean that no longer depends on the observation is the round-1 collapse signature.
    assert model.logger.name_to_value["probe/action_state_std"] == pytest.approx(
        0.0, abs=1e-6
    )


def test_action_logger_records_lateral_and_vertical_magnitudes():
    from fly_drone.sac import ActionLogger
    from stable_baselines3.common.logger import Logger

    model = build_sac("decoder", SpacesOnlyEnv("decoder"), buffer_size=1, device="cpu")
    model.set_logger(Logger(folder=None, output_formats=[]))
    callback = ActionLogger()
    callback.init_callback(model)
    callback.locals = {
        "actions": np.array([[0.0, 0.3, -0.6, 0.2], [0.0, -0.5, 0.2, 0.0]])
    }
    assert callback._on_step()
    values = model.logger.name_to_value
    assert values["rollout/action_vy_absmean"] == pytest.approx(0.4)
    assert values["rollout/action_vz_absmean"] == pytest.approx(0.4)
    # Encoder-sized actions (8 currents) are not velocity axes and must be ignored.
    callback.locals = {"actions": np.zeros((1, 8))}
    assert callback._on_step()
    assert values["rollout/action_vy_absmean"] == pytest.approx(0.4)
