import numpy as np
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
    # A zero decoder cannot steer, so no ablation can be beaten.
    assert report["acceptance"]["steering_passed"] is False
    assert json.loads((tmp_path / "evaluation.json").read_text())["episodes"] == 2
