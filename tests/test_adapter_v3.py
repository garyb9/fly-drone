import numpy as np
import pytest
from fly_drone import adapter as adapter_module
from fly_drone.adapter import Adapter


@pytest.fixture(scope="module")
def brain():
    from fly_drone.brain import BrainRuntime

    return BrainRuntime()


@pytest.fixture(scope="module")
def v3(brain):
    return Adapter.calibrate(brain, codec="v3")


def _vector(brain, escape_l, escape_r, power=None, steer=0.0):
    from fly_drone.adapter import feature_positions

    positions = feature_positions(brain)
    out = np.zeros(len(brain.feature_ids), dtype=float)
    sides = {brain.cells[i]["side"]: positions[i] for i in brain.readout_ids["escape"]}
    out[sides["L"]] = escape_l
    out[sides["R"]] = escape_r
    power = 0.5 if power is None else power
    for side in brain.readout_ids["power_l"]:
        out[positions[side]] = power
    for side in brain.readout_ids["steer_l"]:
        out[positions[side]] = steer
    return out


def test_v3_identity_differs_from_v2(brain, v3):
    v2 = Adapter.calibrate(brain, codec="v2")
    assert v3.version.startswith("declared-v3:")
    assert v3.version != v2.version
    assert v3.params["escape_side_gain"] == adapter_module.ESCAPE_SIDE_GAIN
    assert v3.params["climb_fraction"] == adapter_module.CLIMB_FRACTION


def test_v3_strafes_away_from_the_looming_side(v3, brain):
    # Left loom (DNp01 L > R) must strafe right (vy < 0); right loom the other way.
    assert v3.command(brain, features=_vector(brain, 0.90, 0.60))[1] < -0.5
    assert v3.command(brain, features=_vector(brain, 0.60, 0.90))[1] > 0.5


def test_v3_does_not_strafe_on_a_symmetric_or_calm_frame(v3, brain):
    symmetric = v3.command(brain, features=_vector(brain, 0.75, 0.75))
    assert abs(symmetric[1]) < 1e-9  # no side signal -> no lateral escape
    calm = v3.command(brain, features=_vector(brain, 0.0, 0.0))
    assert abs(calm[1]) < 1e-9
    assert abs(calm[2]) < 1e-9  # no loom -> no climb


def test_v3_climbs_only_a_declared_fraction(v3, brain):
    loom_both = v3.command(brain, features=_vector(brain, 0.85, 0.85))
    v2 = Adapter.calibrate(brain, codec="v2")
    v2_same = v2.command(brain, features=_vector(brain, 0.85, 0.85))
    assert 0 < loom_both[2] < v2_same[2]


def test_v3_lateral_escape_is_silenceable(v3, brain):
    from fly_drone.adapter import feature_positions

    positions = feature_positions(brain)
    sides = {brain.cells[i]["side"]: positions[i] for i in brain.readout_ids["escape"]}
    vector = _vector(brain, 0.9, 0.6)
    assert v3.command(brain, features=vector)[1] < -0.5
    vector[sides["L"]] = vector[sides["R"]] = 0.0  # silence the escape cells
    assert abs(v3.command(brain, features=vector)[1]) < 1e-9


def test_calibrating_v3_leaves_the_canonical_artifacts_untouched(brain, tmp_path):
    before = {
        path: path.read_bytes()
        for path in (adapter_module.DEFAULT_PATH, adapter_module.DEFAULT_V2_PATH)
    }
    Adapter.calibrate(brain, codec="v3").save(tmp_path / "adapter-v3.json")
    for path, data in before.items():
        assert path.read_bytes() == data


def test_v3_artifact_is_committed_and_pinned_to_the_bundle(brain):
    adapter = Adapter.load(adapter_module.DEFAULT_V3_PATH)
    adapter.check(brain)  # raises on a bundle/identity mismatch
    assert adapter.version.startswith("declared-v3:")
