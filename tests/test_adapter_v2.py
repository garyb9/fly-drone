import json

import numpy as np
import pytest
from fly_drone import adapter as adapter_module
from fly_drone.adapter import Adapter

BRAIN = None


@pytest.fixture(scope="module")
def brain():
    from fly_drone.brain import BrainRuntime

    return BrainRuntime()


@pytest.fixture(scope="module")
def v2(brain):
    return Adapter.calibrate(brain, codec="v2")


def vector(brain, **groups):
    from fly_drone.adapter import feature_positions

    positions = feature_positions(brain)
    out = np.zeros(len(brain.feature_ids), dtype=float)
    for name, value in groups.items():
        for cell in brain.readout_ids[name]:
            out[positions[cell]] = value
    return out


def test_v2_identity_differs_and_does_not_load_under_v1(brain, v2, tmp_path):
    v1 = Adapter.load(adapter_module.DEFAULT_PATH)
    assert v2.version.startswith("declared-v2:")
    assert v2.version != v1.version
    for other, payload in ((v1, v2.payload), (v2, v1.payload)):
        tampered = dict(payload)
        tampered["adapter_version"] = other.version
        path = tmp_path / "adapter.json"
        path.write_text(json.dumps(tampered))
        with pytest.raises(ValueError, match="identity"):
            Adapter.load(path)


def test_canonical_adapter_stays_byte_identical(brain, tmp_path):
    before = adapter_module.DEFAULT_PATH.read_bytes()
    Adapter.calibrate(brain, codec="v2").save(tmp_path / "adapter-v2.json")
    assert adapter_module.DEFAULT_PATH.read_bytes() == before


def test_shipped_default_bridge_is_codec_v3(brain):
    assert adapter_module.DEFAULT_CODEC == "v3"
    assert adapter_module.DEFAULT_BRIDGE_PATH == adapter_module.DEFAULT_V3_PATH
    # The default path (no --adapter/--codec) loads the v3 identity (C4a lateral escape).
    assert adapter_module.load_default().version.startswith("declared-v3:")


def test_codec_adapter_maps_the_switch_and_rejects_unknown():
    assert adapter_module.codec_adapter("v1") == adapter_module.DEFAULT_PATH
    assert adapter_module.codec_adapter("v2") == adapter_module.DEFAULT_V2_PATH
    assert adapter_module.codec_adapter("v3") == adapter_module.DEFAULT_V3_PATH
    with pytest.raises(ValueError, match="unknown codec"):
        adapter_module.codec_adapter("v9")


def test_forward_drive_is_two_sided(v2, brain):
    p = v2.params
    below = vector(
        brain,
        power_l=p["rest_power"] - 0.05,
        power_r=p["rest_power"] - 0.05,
        steer_l=0.0,
        steer_r=0.0,
    )
    assert v2.command(brain, features=below)[0] < 0
    above = vector(
        brain,
        power_l=p["rest_power"] + 0.05,
        power_r=p["rest_power"] + 0.05,
        steer_l=0.0,
        steer_r=0.0,
    )
    assert v2.command(brain, features=above)[0] > 0


def test_steering_reads_the_features_vector_not_a_side_channel(v2, brain):
    p = v2.params
    asym = vector(
        brain,
        power_l=p["rest_power"],
        power_r=p["rest_power"],
        steer_l=0.05,
        steer_r=0.0,
    )
    yaw = v2.command(brain, features=asym)[3]
    assert yaw > 0
    zeros = np.zeros(len(brain.feature_ids), dtype=float)
    assert abs(v2.command(brain, features=zeros)[3]) < 1e-9


class _TinyBrain:
    """A synthetic brain whose light_both stimulus cannot move the power readout."""

    def __init__(self):
        self.feature_ids = [0, 1, 2]
        self.readout_ids = {
            "escape": [0],
            "power_l": [1],
            "power_r": [2],
            "steer_l": [1],
            "steer_r": [2],
        }
        self.cues = np.zeros(4)

    def reset(self, seed):
        pass

    def set_currents(self, cues):
        self.cues = np.asarray(cues, dtype=float)

    def step(self, ticks):
        pass

    def features(self):
        light_l, light_r = self.cues[0], self.cues[1]
        power_l = 0.2 if (light_l > 0 and light_r == 0) else 0.0
        power_r = 0.2 if (light_r > 0 and light_l == 0) else 0.0
        escape = 1.0 if (self.cues[2] > 0 or self.cues[3] > 0) else 0.0
        return np.array([escape, power_l, power_r])


def test_calibration_falsifier_rejects_a_codec_that_cannot_command_motion():
    with pytest.raises(ValueError, match="cannot command motion"):
        adapter_module._params_v2(_TinyBrain())


def test_motion_and_liveness_paths_have_no_teacher():
    import inspect

    from fly_drone import liveness, motion_stats

    for module in (adapter_module, liveness, motion_stats):
        source = inspect.getsource(module)
        assert "from .teacher" not in source
        assert "import teacher" not in source
