"""A1 audit protocol and matched-input diagnostics, without expensive full runs."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from fly_drone import a1_audit
from fly_drone.adapter import CODEC_PATHS, Adapter


def test_registry_separates_confirmation_and_freezes_battery():
    trials = a1_audit.trial_definitions()
    assert len({row["id"] for row in trials}) == len(trials) == 32
    assert {row["seed"] for row in trials if row["split"] == "development"} == set(
        range(8)
    )
    assert {row["seed"] for row in trials if row["split"] == "confirmation"} == set(
        range(8, 16)
    )
    assert len(a1_audit.scenarios("synthetic")) == 15
    assert len(a1_audit.scenarios("rendered")) == 15
    with pytest.raises(ValueError, match="unregistered"):
        a1_audit.registered_trial("confirmation-00-synthetic")


def test_decision_clock_matches_existing_environment():
    from fly_drone.env import FRAME_SECONDS

    assert a1_audit.FRAME_TICKS == 8
    assert a1_audit.FRAME_SECONDS == FRAME_SECONDS == 0.04
    assert sum(frames for _, frames in a1_audit.PHASES) * FRAME_SECONDS == 2.0


def test_existing_report_is_rejected_before_loading_brain(tmp_path):
    output = tmp_path / "report.json"
    output.write_text("preserve")
    with pytest.raises(FileExistsError):
        a1_audit.run_trial("development-00-synthetic", output)
    assert output.read_text() == "preserve"


def test_camera_history_is_cleared_even_when_neural_state_persists():
    brain = Mock()
    a1_audit.begin_case(brain, 3, "persistent")
    brain.reset.assert_not_called()
    brain.clear_vision_history.assert_called_once_with()
    a1_audit.begin_case(brain, 3, "reset")
    brain.reset.assert_called_once_with(3)
    assert brain.clear_vision_history.call_count == 2


@pytest.mark.parametrize(
    "control,expected",
    [("none", [1, 2, 0.5, 0.25]), ("light", [0, 0, 0.5, 0.25]), ("loom", [1, 2, 0, 0])],
)
def test_input_controls_preserve_other_sense_and_original(control, expected):
    original = np.array([1, 2, 0.5, 0.25])
    assert a1_audit.controlled_currents(original, control).tolist() == expected
    assert original.tolist() == [1, 2, 0.5, 0.25]


def test_control_rejects_incompatible_encoder():
    with pytest.raises(ValueError, match="four-channel"):
        a1_audit.controlled_currents(np.zeros(6), "none")
    with pytest.raises(ValueError, match="unknown control"):
        a1_audit.controlled_currents(np.zeros(4), "typo")


def fixture_brain():
    values = np.array([1, 1, 0.3, 0.2, 0.4, 0.2], dtype=np.float32)
    return SimpleNamespace(
        feature_ids=list(range(6)),
        cells=[{"side": side} for side in ("L", "R", "L", "R", "L", "R")],
        readout_ids={
            "escape": [0, 1],
            "power_l": [2],
            "power_r": [3],
            "steer_l": [4],
            "steer_r": [5],
        },
        features=lambda: values.copy(),
    )


def test_trace_exposes_escape_inversion_and_matches_real_codec():
    brain = fixture_brain()
    adapters = {name: Adapter.load(path) for name, path in CODEC_PATHS.items()}
    row = a1_audit.trace_readout(brain, adapters)
    assert row["steer_left_minus_right"] > 0
    for name, codec in row["codecs"].items():
        assert codec["escape_multiplier"] == -1
        np.testing.assert_array_equal(
            codec["command_normalized"], adapters[name].command(brain)
        )
    assert row["codecs"]["v2"]["pre_escape_steering"] > 0
    assert row["codecs"]["v2"]["command_normalized"][3] < 0


def test_rendered_stimuli_are_registered_for_rotation_and_mixed_threat():
    cases = {case["name"]: case for case in a1_audit.scenarios("rendered")}
    assert cases["turn-left"]["yaw_rate"] == -cases["turn-right"]["yaw_rate"]
    assert not cases["turn-left"]["beacon"]
    assert cases["mixed-left"]["threat"] and cases["mixed-right"]["threat"]
    assert {
        case["distance"]
        for case in cases.values()
        if case["name"].startswith("bearing")
    } == {1, 3}
