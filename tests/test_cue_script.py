import numpy as np
from fly_drone.feasibility import CueController


def test_rising_loom_makes_the_cue_script_climb_away_from_the_louder_eye():
    script = CueController()
    # Dark area on the right eye saturating within frames: a thrown threat.
    for loom_r in (0.1, 0.5, 1.6):
        action = script.act([0.0, 0.0, 0.1, loom_r])
    assert script.escape_kind == "threat"
    forward, lateral, vertical, _ = action
    assert forward < 0 and lateral > 0.5 and vertical > 0.5


def test_slow_loom_turns_away_without_climbing():
    script = CueController()
    for loom_l in (0.5, 0.6, 0.6):
        action = script.act([0.0, 0.0, loom_l, 0.0])
    assert script.escape_kind == "obstacle"
    assert action[2] == 0.0 and action[3] < 0
    np.testing.assert_array_equal(action[:2], [0.1, 0.0])
