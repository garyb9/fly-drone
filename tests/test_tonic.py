import importlib.util
import json

import numpy as np
import pytest
from fly_drone.brain import ROOT, BrainRuntime
from fly_drone.identity import ALTERNATE_KEYS, MODEL_KEYS, load_identity

SCRIPT = ROOT / "scripts" / "make_tonic_bundle.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("make_tonic_bundle", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tonic_bundle(tmp_path_factory):
    output = tmp_path_factory.mktemp("bundle") / "malecns-tonic"
    _load_builder().build(ROOT / "data" / "malecns", output)
    return output


def test_tonic_is_a_model_identity_key():
    assert "tonic" in MODEL_KEYS
    assert "tonic" in ALTERNATE_KEYS


def test_canonical_bundle_hash_is_unchanged():
    _, bundle_hash, alternate = load_identity(ROOT / "data" / "malecns")
    assert bundle_hash.startswith("edc5439e")
    assert not alternate


def test_tonic_bundle_is_additive_and_alternate(tonic_bundle):
    canonical = BrainRuntime()
    bundle = BrainRuntime(data=tonic_bundle)
    assert bundle.dataset_hash == canonical.dataset_hash
    assert bundle.bundle_hash != canonical.bundle_hash
    assert bundle.alternate and not canonical.alternate
    assert bundle.tonic_roles == {
        "power_l": 0.85,
        "power_r": 0.85,
        "steer_l": 0.85,
        "steer_r": 0.85,
    }
    # Only the steering roles are the C3a addition; the power pair is the historical bias.
    assert set(bundle.tonic_added) == {"steer_l", "steer_r"}
    assert canonical.tonic_added == {}


def test_absent_marker_keeps_the_power_only_bias():
    brain = BrainRuntime()
    assert brain.tonic_roles == {"power_l": 0.85, "power_r": 0.85}
    assert brain.tonic_added == {}


def test_tonic_steering_rests_mid_rate_and_silence_restores_it(tonic_bundle):
    brain = BrainRuntime(data=tonic_bundle)
    brain.reset(11)
    brain.step(400)  # let the activity trace converge
    rest = brain.read()
    assert rest["steer_l"] > 0.4 and rest["steer_r"] > 0.4

    brain.reset(11)
    brain.silence_tonic()
    brain.step(400)
    silenced = brain.read()
    assert silenced["steer_l"] < 0.05 and silenced["steer_r"] < 0.05
    # The pre-existing flight-power tone is untouched by the C3a gate.
    assert silenced["power_l"] > 0.4 and silenced["power_r"] > 0.4


def test_env_tonic_ablation_is_wired(tonic_bundle):
    from fly_drone.env import ConnectomeEnv

    def rollout(ablation):
        brain = BrainRuntime(data=tonic_bundle)
        env = ConnectomeEnv(task="free_roam", level=0, brain=brain)
        env.ablation = ablation
        env.reset(seed=11)
        action = np.array([0.3, 0.0, 0.0, 0.6])
        for _ in range(8):
            env.step(action)
        features = brain.features().copy()
        env.close()
        return features

    assert not np.allclose(rollout("none"), rollout("tonic"))


def test_builder_marker_is_declared_not_fitted(tonic_bundle):
    manifest = json.loads((tonic_bundle / "manifest.json").read_text())
    marker = manifest["tonic"]
    assert marker["version"] == "declared-tonic-v1"
    assert marker["roles"]["steer_l"] == marker["roles"]["steer_r"] == 0.85
