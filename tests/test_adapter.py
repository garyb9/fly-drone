import json

import numpy as np
import pytest
from fly_drone import adapter as adapter_module
from fly_drone.adapter import Adapter
from fly_drone.brain import BrainRuntime


@pytest.fixture(scope="module")
def brain():
    return BrainRuntime()


@pytest.fixture(scope="module")
def calibrated(brain):
    return Adapter.calibrate(brain)


def settle(brain, cues, ticks=adapter_module.SETTLE_TICKS):
    brain.reset(11)
    brain.set_currents(np.asarray(cues, dtype=np.float32))
    brain.step(ticks)
    return brain


def test_calibration_is_deterministic(calibrated):
    again = Adapter.calibrate(BrainRuntime())
    assert again.version == calibrated.version
    assert again.params == calibrated.params


def test_version_is_a_content_identity(calibrated):
    assert calibrated.version.startswith("declared-v1:")
    assert calibrated.version == adapter_module._version(
        calibrated.params, calibrated.dataset_hash
    )
    tampered = {**calibrated.params, "g_yaw": calibrated.params["g_yaw"] + 1}
    assert (
        adapter_module._version(tampered, calibrated.dataset_hash) != calibrated.version
    )


def test_command_is_deterministic(brain, calibrated):
    settle(brain, [0, 0, 2, 0])
    first = calibrated.command(brain)
    settle(brain, [0, 0, 2, 0])
    np.testing.assert_array_equal(first, calibrated.command(brain))
    assert first.shape == (4,)


def test_zero_cue_does_not_steer(brain, calibrated):
    # The connectome keeps an endogenous motor tone, so forward may be nonzero; with no
    # cue there is no lateral evidence, so steering must stay near the rest baseline.
    settle(brain, [0, 0, 0, 0], ticks=200)
    command = calibrated.command(brain)
    assert np.all(np.abs(command) <= 1.0)
    np.testing.assert_allclose(command[[1, 3]], np.zeros(2), atol=0.02)


def test_loom_climbs_and_turns_away(brain, calibrated):
    settle(brain, [0, 0, 2, 0])
    command = calibrated.command(brain)
    assert command[2] > 0  # climb
    assert command[3] < 0  # turn away from the left-side loom
    assert command[1] < 0  # sidestep away


def test_light_turns_toward(brain, calibrated):
    settle(brain, [2, 0, 0, 0])
    left = calibrated.command(brain)
    settle(brain, [0, 2, 0, 0])
    right = calibrated.command(brain)
    assert left[3] > 0 and right[3] < 0


def test_zeroed_features_remove_the_visual_drive(brain, calibrated):
    settle(brain, [0, 0, 2, 0])
    zeros = np.zeros(len(brain.feature_ids), dtype=float)
    command = calibrated.command(brain, features=zeros)
    assert command[0] == 0 and command[2] == 0  # no forward, no climb
    assert abs(command[3]) < 0.02  # no lateral evidence


def test_shuffled_features_change_the_command(brain, calibrated):
    settle(brain, [0, 0, 2, 0])
    base = calibrated.command(brain)
    permutation = np.random.default_rng(0).permutation(len(brain.feature_ids))
    shuffled = calibrated.command(brain, features=brain.features()[permutation])
    assert not np.allclose(base, shuffled)


def test_check_rejects_a_different_connectome(calibrated):
    calibrated.dataset_hash = "not-the-runtime"
    with pytest.raises(ValueError, match="dataset_hash"):
        calibrated.check(BrainRuntime())


def test_load_rejects_tampered_constants(tmp_path, calibrated):
    path = calibrated.save(tmp_path / "adapter.json")
    payload = json.loads(path.read_text())
    payload["params"]["g_yaw"] += 1.0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="identity"):
        Adapter.load(path)


def test_declared_path_has_no_teacher():
    import inspect

    source = inspect.getsource(adapter_module)
    assert "from .teacher" not in source
    assert "import teacher" not in source
    assert "teacher_action" not in source
    assert getattr(adapter_module, "teacher", None) is None


def test_declared_command_missing_artifact_fails_loudly(tmp_path, brain):
    with pytest.raises(FileNotFoundError):
        adapter_module.declared_command(brain, path=tmp_path / "absent.json")
