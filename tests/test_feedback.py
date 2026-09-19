import importlib.util

import numpy as np
import pytest
from fly_drone.brain import ROOT, BrainRuntime
from fly_drone.feedback import (
    DEFAULT_MAPPINGS,
    FLOW_BASE,
    FLOW_GAIN,
    HALTERE_BASE,
    PROPRIO_BASE,
    FeedbackState,
    body_rates,
    flow_currents,
    haltere_currents,
    proprio_currents,
)

SCRIPT = ROOT / "scripts" / "make_feedback_bundle.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("make_feedback_bundle", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def feedback_bundle(tmp_path_factory):
    output = tmp_path_factory.mktemp("bundle") / "malecns-feedback"
    _load_builder().build(ROOT / "data" / "malecns", output, DEFAULT_MAPPINGS)
    return output


def test_body_rates_identity_at_level():
    assert np.allclose(body_rates([1, 0, 0, 0], [0, 0, 1]), [0, 0, 1])


def test_haltere_splits_yaw_onto_sides():
    level = haltere_currents([1, 0, 0, 0], [0, 0, 0])
    assert level == {
        "haltere_l": pytest.approx(HALTERE_BASE),
        "haltere_r": pytest.approx(HALTERE_BASE),
    }
    left = haltere_currents([1, 0, 0, 0], [0, 0, 1])
    right = haltere_currents([1, 0, 0, 0], [0, 0, -1])
    assert left["haltere_l"] > left["haltere_r"]
    assert right["haltere_r"] > right["haltere_l"]


def test_flow_grows_with_frame_change():
    still = np.zeros((2, 48, 64, 3), dtype=np.uint8)
    first, luma = flow_currents(still, None)
    assert first == {"flow_l": FLOW_BASE, "flow_r": FLOW_BASE}
    changed = still.copy()
    changed[:, :, :32] = 255
    second, _ = flow_currents(changed, luma)
    assert second["flow_l"] > first["flow_l"]
    assert second["flow_l"] <= FLOW_BASE + FLOW_GAIN


def test_proprio_tracks_rotor_lag():
    currents = proprio_currents([1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], 1.0)
    assert currents["proprio_lf"] > PROPRIO_BASE
    assert currents["proprio_rf"] == pytest.approx(PROPRIO_BASE)


def test_feedback_bundle_is_additive_and_alternate(feedback_bundle):
    canonical = BrainRuntime()
    bundle = BrainRuntime(data=feedback_bundle)
    assert bundle.dataset_hash == canonical.dataset_hash
    assert bundle.bundle_hash != canonical.bundle_hash
    assert bundle.alternate and not canonical.alternate
    assert set(bundle.feedback_ids) == {
        "haltere_l",
        "haltere_r",
        "flow_l",
        "flow_r",
        "proprio_lf",
        "proprio_rf",
        "proprio_lh",
        "proprio_rh",
    }
    assert not canonical.feedback_ids


def test_feedback_injection_changes_activity_and_silence_removes_it(feedback_bundle):
    brain = BrainRuntime(data=feedback_bundle)
    cues = np.zeros(brain.current_dim, dtype=np.float32)
    brain.reset(3)
    brain.set_currents(cues)
    brain.step(40)
    base = brain.features().copy()

    driven_currents = {role: 1.5 for role in brain.feedback_ids}
    brain.reset(3)
    brain.set_currents(cues)
    for _ in range(40):
        brain.inject_feedback(driven_currents)
        brain.step(1)
    driven = brain.features()
    assert not np.allclose(base, driven)

    brain.reset(3)
    brain.set_currents(cues)
    brain.silence_feedback()
    for _ in range(40):
        brain.inject_feedback(driven_currents)
        brain.step(1)
    assert np.allclose(base, brain.features())


def test_env_feedback_ablation_is_wired(feedback_bundle):
    from fly_drone.env import ConnectomeEnv

    def rollout(ablation):
        brain = BrainRuntime(data=feedback_bundle)
        env = ConnectomeEnv(task="free_roam", level=0, brain=brain, feedback=True)
        env.ablation = ablation
        env.reset(seed=11)
        # A moving body is needed for gyro/flow/proprio to differ from their rest values.
        action = np.array([0.3, 0.0, 0.0, 0.6])
        for _ in range(8):
            env.step(action)
        features = brain.features().copy()
        env.close()
        return features

    assert not np.allclose(rollout("none"), rollout("feedback"))


def test_feedback_state_matches_declared_roles(feedback_bundle):
    from fly_drone.plant import DronePlant

    state = FeedbackState(DronePlant(vision=False))
    currents = state.currents()
    assert set(currents) == {
        "haltere_l",
        "haltere_r",
        "flow_l",
        "flow_r",
        "proprio_lf",
        "proprio_rf",
        "proprio_lh",
        "proprio_rh",
    }
    assert all(0.0 <= v <= 2.0 for v in currents.values())
