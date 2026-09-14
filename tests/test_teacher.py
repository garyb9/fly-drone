import numpy as np
import pytest
from fly_drone.env import ConnectomeEnv
from fly_drone.teacher import teacher_action


@pytest.fixture(scope="module")
def env():
    e = ConnectomeEnv(task="free_roam", vision=False, level=0)
    e.reset(seed=4)
    yield e
    e.close()


def place(env, pillars=(), beacon=(5.0, 0.0, 1.0), threat=None):
    env.plant.set_pillars(np.asarray(pillars, dtype=float).reshape(-1, 2))
    env.plant.teleport([0.0, 0.0, 1.0], 0.0)
    env.plant.set_objects(target=beacon)
    if threat is None:
        env.roam["threat"] = None
        env.plant.set_objects(obstacle=[0, 0, -20], park_obstacle=True)
    else:
        env.plant.set_objects(obstacle=threat)
        env.roam["threat"] = {"side": 1.0 if threat[1] > 0 else -1.0}


def test_explore_label_ignores_where_the_hidden_beacon_is(env):
    place(env, beacon=(-4.0, 3.0, 1.0))
    left, drive_left = teacher_action(env)
    place(env, beacon=(-4.0, -3.0, 1.0))
    right, drive_right = teacher_action(env)
    assert drive_left == drive_right == "explore"
    np.testing.assert_array_equal(left, right)


def test_occluded_beacon_does_not_drive_approach(env):
    place(env, beacon=(5.0, 0.0, 1.0))
    action, drive = teacher_action(env)
    assert drive == "beacon" and action[0] > 0.5
    place(env, pillars=[[2.5, 0.0]], beacon=(5.0, 0.0, 1.0))
    _, drive = teacher_action(env)
    assert drive == "explore"


def test_beacon_turn_follows_its_side(env):
    place(env, beacon=(3.0, 2.0, 1.0))
    left, _ = teacher_action(env)
    place(env, beacon=(3.0, -2.0, 1.0))
    right, _ = teacher_action(env)
    assert left[3] > 0.5 and right[3] < -0.5


def test_near_pillar_turns_away_and_slows(env):
    place(env, pillars=[[1.0, 0.15]], beacon=(-4.0, 3.0, 1.0))
    action, drive = teacher_action(env)
    assert drive == "avoid" and action[3] < -0.5 and action[0] < 0.4
    place(env, pillars=[[1.0, -0.15]], beacon=(-4.0, 3.0, 1.0))
    action, _ = teacher_action(env)
    assert action[3] > 0.5


@pytest.mark.parametrize(
    "position, yaw",
    [((6.5, 0.0), 0.2), ((-6.5, 0.0), np.pi + 0.2), ((0.0, -6.5), -1.37)],
)
def test_near_wall_turns_along_it_on_either_side_of_the_room(env, position, yaw):
    env.plant.set_pillars(np.zeros((0, 2)))
    env.roam["threat"] = None
    env.plant.teleport([position[0], position[1], 1.0], yaw)
    env.plant.set_objects(target=[-position[0] * 0.2, -position[1] * 0.2 + 3, 1.0])
    from fly_drone.teacher import nearest_obstacle

    distance, into, kind = nearest_obstacle(env)
    assert kind == "wall" and 1.4 < distance < 1.7
    action, drive = teacher_action(env)
    assert drive == "avoid"
    # Yawing in the commanded direction must reduce the approach into the wall.
    env.plant.teleport([position[0], position[1], 1.0], yaw + 0.1 * np.sign(action[3]))
    assert nearest_obstacle(env)[0] > distance


def test_far_pillar_is_ignored_until_loom_could_fire(env):
    place(env, pillars=[[3.5, 0.1]], beacon=(-4.0, 3.0, 1.0))
    _, drive = teacher_action(env)
    assert drive == "explore"


def test_threat_dodge_side_is_committed_once_per_threat(env):
    place(env, beacon=(-4.0, 3.0, 1.0), threat=(1.2, 0.02, 1.0))
    first, drive = teacher_action(env)
    assert drive == "threat" and abs(first[1]) > 0.5 and first[0] < 0
    # Head-on shot drifts across the heading: the label must not flip sides.
    env.plant.set_objects(obstacle=[1.1, -0.02, 1.0])
    second, _ = teacher_action(env)
    assert np.sign(second[1]) == np.sign(first[1])


def test_visible_threat_dodges_away_but_ghost_threat_does_not(env):
    place(env, beacon=(-4.0, 3.0, 1.0), threat=(1.2, 0.4, 1.0))
    action, drive = teacher_action(env)
    assert drive == "threat" and action[1] < -0.5
    env.plant.set_ghost(True)
    try:
        _, drive = teacher_action(env)
        assert drive != "threat"
    finally:
        env.plant.set_ghost(False)
