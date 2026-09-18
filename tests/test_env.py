import hashlib

import numpy as np
import pytest
from fly_drone.env import ConnectomeEnv
from fly_drone.fly import FlyMirror


def test_clocks_reset_and_seed_replay():
    e = ConnectomeEnv(vision=False)
    try:
        a, _ = e.reset(seed=99)
        x, _, _, _, info = e.step(np.zeros(4))
        assert info["tick"] == 48 and info["physics_tick"] == 40
        assert abs(info["time"] - 0.04) < 1e-10
        assert len(e.trace) == 8
        e.reset(seed=99)
        y, *_ = e.step(np.zeros(4))
        np.testing.assert_array_equal(x, y)
        assert e.interventions == {}
    finally:
        e.close()


def test_fly_is_independent_and_deterministic():
    a = FlyMirror()
    b = FlyMirror()
    r = {"power_l": 0.5, "power_r": 0.5, "steer_l": 0.2, "steer_r": 0}
    for _ in range(200):
        a.step(r)
        b.step(r)
    np.testing.assert_array_equal(a.position, b.position)
    assert a.ticks == 200 and np.linalg.norm(a.position - [0, 1, 0]) > 0.01
    a.reset()
    assert a.ticks == 0


def test_target_bearing_wraps_to_pi():
    from fly_drone.env import target_bearing

    assert abs(target_bearing([0, 0, 1], 0.0, [2, 2, 1]) - np.pi / 4) < 1e-12
    # Target behind-left while facing +X after a full extra turn stays in range.
    b = target_bearing([0, 0, 1], 2 * np.pi + 0.1, [-1, 0.01, 1])
    assert -np.pi <= b <= np.pi
    assert abs(target_bearing([0, 0, 1], 3.0, [-1, -0.2, 1])) < 0.5


def test_evaluate_smoke_with_zero_policy(tmp_path):
    import json

    from fly_drone.brain import ENCODER_VERSION, BrainRuntime
    from fly_drone.training import evaluate

    brain = BrainRuntime()
    n = len(brain.feature_ids)
    policy = tmp_path / "zero-actor.json"
    policy.write_text(
        json.dumps(
            {
                "version": 1,
                "encoder_version": ENCODER_VERSION,
                "dataset_hash": brain.dataset_hash,
                "feature_ids": brain.feature_ids,
                "mean": [0.0] * n,
                "scale": [1.0] * n,
                "layers": [{"weights": [[0.0] * n] * 4, "bias": [0.0] * 4}],
                "action_limits": [0.4, 0.4, 0.2, 0.8],
            }
        )
    )
    report = evaluate(
        policy,
        episodes=2,
        output=tmp_path / "evaluation.json",
        seconds=0.2,
        workers=2,
        hover_episodes=1,
        hover_seconds=0.2,
    )
    assert set(report["modes"]) == {"none", "zero", "sensory", "shuffle"}
    assert all(len(m["runs"]) == 2 for m in report["modes"].values())
    assert len(report["hover"]["runs"]) == 1
    none = report["modes"]["none"]
    assert set(none["success_by_side"]) == {"left", "right"}
    assert 0.0 <= none["balanced_success"] <= 1.0
    assert all(r["target_side"] in ("left", "right") for r in none["runs"])
    # A zero decoder cannot steer, so no ablation can be beaten.
    assert report["acceptance"]["passed"] is False
    assert json.loads((tmp_path / "evaluation.json").read_text())["episodes"] == 2


def test_looming_obstacle_launches_and_hits_a_stationary_drone():
    e = ConnectomeEnv(task="looming", vision=False)
    try:
        _, info = e.reset(seed=5)
        assert info["obstacle_side"] in (-1.0, 1.0) and not info["launched"]
        start_distance = info["obstacle_distance"]
        hit = False
        for frame in range(150):
            _, _, done, truncated, info = e.step(np.zeros(4))
            if done:
                hit = info["collision"]
                break
            assert not truncated or frame == 149
        assert info["launched"] and hit, (start_distance, info["obstacle_distance"])
    finally:
        e.close()


def test_evaluate_looming_smoke(tmp_path):
    import json

    from fly_drone.brain import ENCODER_VERSION, BrainRuntime
    from fly_drone.training import evaluate

    brain = BrainRuntime()
    n = len(brain.feature_ids)
    policy = tmp_path / "zero-actor.json"
    policy.write_text(
        json.dumps(
            {
                "version": 1,
                "encoder_version": ENCODER_VERSION,
                "dataset_hash": brain.dataset_hash,
                "feature_ids": brain.feature_ids,
                "mean": [0.0] * n,
                "scale": [1.0] * n,
                "layers": [{"weights": [[0.0] * n] * 4, "bias": [0.0] * 4}],
                "action_limits": [0.4, 0.4, 0.2, 0.8],
            }
        )
    )
    report = evaluate(
        policy,
        episodes=2,
        output=tmp_path / "loom.json",
        seconds=0.2,
        workers=2,
        task="looming",
    )
    assert report["task"] == "looming" and "hover" not in report
    assert "passed" in report["acceptance"]
    assert all("min_obstacle_distance" in r for r in report["modes"]["none"]["runs"])
    assert all("displacement_at_threat" in r for r in report["modes"]["none"]["runs"])
    assert "survival_rate" in report["modes"]["none"]


def test_fly_stays_in_view_under_tonic_power():
    fly = FlyMirror()
    readouts = {"power_l": 0.5, "power_r": 0.5, "steer_l": 0.0, "steer_r": 0.0}
    for _ in range(20 * 200):
        fly.step(readouts)
    assert np.linalg.norm(fly.position - [0, 1, 0]) < 2.0
    assert 0.25 < fly.position[1] < 3.0


V4_LOOMING_REPLAY = "ecec814583cac8bae37ab6e9879fcd68c5f1e85514d3e6a4fbffe8155f48ccd7"
V4_ROAM_REPLAY = "37b6d176c81bfc1d720347b09d15ed6cd9b58eb47a53b2f8c544816087693e92"


def _replay_digest(task, seed, frames, **kwargs):
    env = ConnectomeEnv(task=task, **kwargs)
    try:
        obs, _ = env.reset(seed=seed)
        digest = hashlib.sha256(np.round(obs, 6).tobytes())
        for _ in range(frames):
            obs, *_ = env.step(np.array([0.3, -0.2, 0.1, 0.4]))
            digest.update(np.round(obs, 6).tobytes())
            digest.update(np.round(env.brain.cues, 6).tobytes())
        return digest.hexdigest()
    finally:
        env.close()


def test_v4_looming_replay_is_bit_identical():
    assert _replay_digest("looming", 7, 20) == V4_LOOMING_REPLAY


def test_v4_free_roam_replay_is_bit_identical():
    assert _replay_digest("free_roam", 7, 20, level=3, respawn=True) == V4_ROAM_REPLAY


def test_threat_shaping_rewards_opening_the_gap():
    from fly_drone.env import threat_potential, threat_shaping

    assert threat_potential(0.0) == pytest.approx(1.0)
    assert threat_potential(0.0) > threat_potential(1.0) > threat_potential(3.0)
    assert threat_shaping(None, 1.0) == 0.0
    assert threat_shaping(1.0, 3.0) > 0.0  # moving away
    assert threat_shaping(3.0, 1.0) < 0.0  # closing in


def test_threat_curriculum_level_has_threats_and_no_pillars():
    from fly_drone import arena

    assert arena.LEVELS[4] == (0, True)
    e = ConnectomeEnv(task="free_roam", level=4, respawn=True)
    try:
        e.reset(seed=5)
        assert len(e.plant.pillars) == 0
        assert e.roam["next_threat"] is not None
        if e.launch_threat():
            e.step(np.zeros(4))
            assert e.previous_threat_distance is not None
    finally:
        e.close()
