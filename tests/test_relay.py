import importlib.util
import json

import numpy as np
import pytest
from fly_drone import adapter as adapter_module
from fly_drone.adapter import Adapter
from fly_drone.brain import ROOT, BrainRuntime
from fly_drone.relay import (
    FLOW_ROLES,
    RELAY_TYPES,
    RelayState,
    phase_shift,
)

SCRIPT = ROOT / "scripts" / "validate_relay_flow.py"


def _load_validation():
    spec = importlib.util.spec_from_file_location("validate_relay_flow", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def relay_brain():
    return BrainRuntime(relay=True)


def test_phase_shift_recovers_a_known_translation():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (48, 64), dtype=np.uint8).astype(np.float32)
    current = np.roll(base, 3, axis=1)
    dy, dx = phase_shift(current, base)
    assert dx == pytest.approx(3.0, abs=0.5)
    assert abs(dy) < 0.5


def test_flow_is_zero_for_a_still_scene():
    still = np.full((2, 48, 64, 3), 40, dtype=np.uint8)
    state = RelayState()
    state.observe_frame(still)
    state.observe_frame(still)
    assert state.yaw == {"l": 0.0, "r": 0.0}
    assert all(v == 0.0 for v in state.currents().values())


def test_currents_rectify_the_signed_axes():
    state = RelayState()
    state.yaw = {"l": 0.05, "r": -0.05}
    state.pitch = {"l": 0.0, "r": 0.0}
    currents = state.currents()
    pos, neg = FLOW_ROLES["yaw"]
    assert currents[f"{pos}_l"] > 0 and currents[f"{neg}_l"] == 0
    assert currents[f"{neg}_r"] > 0 and currents[f"{pos}_r"] == 0
    assert all(0.0 <= v <= 2.0 for v in currents.values())


def test_relay_roles_target_the_motion_subtypes(relay_brain):
    expected = {f"{key}_{side}" for key in RELAY_TYPES for side in ("l", "r")}
    assert set(relay_brain.relay_ids) == expected
    assert all(relay_brain.relay_ids.values())
    assert set(relay_brain.relay_inputs) == expected
    assert not BrainRuntime().relay_ids  # canonical v4 brain has no relay roles


def test_relay_injection_changes_activity_and_silence_removes_it(relay_brain):
    cues = np.zeros(relay_brain.current_dim, dtype=np.float32)
    relay_brain.reset(3)
    relay_brain.set_currents(cues)
    relay_brain.step(40)
    base = relay_brain.features().copy()

    driven = {role: 1.5 for role in relay_brain.relay_ids}
    relay_brain.reset(3)
    relay_brain.set_currents(cues)
    for _ in range(40):
        relay_brain.inject_relay(driven)
        relay_brain.step(1)
    assert not np.allclose(base, relay_brain.features())

    relay_brain.reset(3)
    relay_brain.set_currents(cues)
    relay_brain.silence_relay()
    for _ in range(40):
        relay_brain.inject_relay(driven)
        relay_brain.step(1)
    assert np.allclose(base, relay_brain.features())


def test_env_relay_ablation_is_wired(relay_brain):
    from fly_drone.env import ConnectomeEnv

    def rollout(ablation):
        brain = BrainRuntime(relay=True)
        env = ConnectomeEnv(task="free_roam", level=0, brain=brain, relay=True)
        env.ablation = ablation
        env.reset(seed=11)
        action = np.array([0.3, 0.0, 0.0, 0.6])
        for _ in range(8):
            env.step(action)
        features = brain.features().copy()
        env.close()
        return features

    assert not np.allclose(rollout("none"), rollout("relay"))


def test_relay_front_end_is_part_of_the_adapter_identity():
    params = {"g_yaw": 1.0}
    assert adapter_module._version(params, "ds") == adapter_module._version(
        params, "ds", None
    )
    assert adapter_module._version(params, "ds") != adapter_module._version(
        params, "ds", "relay"
    )


def test_canonical_adapter_is_unchanged_by_the_relay_identity():
    canonical = Adapter.load(adapter_module.DEFAULT_PATH)
    assert "visual" not in canonical.payload
    committed = json.loads(adapter_module.DEFAULT_PATH.read_text())
    assert committed["adapter_version"] == canonical.version


def test_declared_flow_correlates_with_simulator_egomotion(tmp_path):
    report = _load_validation().run(tmp_path / "validation.json", frames=60, seed=0)
    yaw = report["yaw_segment"]["pearson_yaw_flow_vs_yaw_rate"]
    assert all(r is not None and abs(r) > 0.8 for r in yaw.values())


def test_relay_module_has_no_teacher():
    import inspect

    from fly_drone import relay

    source = inspect.getsource(relay)
    assert "teacher" not in source
    assert "luma_u8" in source  # sanity: the module loaded
