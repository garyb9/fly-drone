import numpy as np
from fly_drone.arena import ArenaSpec
from fly_drone.plant import LIMITS
from fly_drone.training import _env_kwargs, action_limits


def test_free_roam_ppo_envs_use_the_evaluated_arena_and_respawn():
    assert _env_kwargs("free_roam") == {
        "task": "free_roam",
        "level": 3,
        "respawn": True,
    }
    assert _env_kwargs("visual") == {"task": "visual"}


def test_free_roam_actors_export_arena_limits_not_legacy_room_limits():
    np.testing.assert_array_equal(action_limits("free_roam"), ArenaSpec().limits)
    np.testing.assert_array_equal(action_limits("looming"), LIMITS)
    assert not np.array_equal(action_limits("free_roam"), LIMITS)
