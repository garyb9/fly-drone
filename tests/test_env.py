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
