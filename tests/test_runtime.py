import json

import numpy as np
import pytest
from fly_drone._brain import Brain
from fly_drone.brain import ROOT, BrainRuntime


@pytest.fixture(scope="module")
def brain():
    return BrainRuntime()


def test_full_graph_and_roles(brain):
    assert brain.core.neuron_count() == 166700
    assert len(brain.feature_ids) == 2022
    assert all(brain.input_ids.values())


def test_reset_replays_and_interventions_clear(brain):
    images = np.zeros((2, 48, 64, 3), dtype=np.uint8)
    images[0] = 255

    def run():
        brain.reset(812)
        brain.sense(images)
        brain.step(20)
        return brain.features()

    first = run()
    brain.core.silence(brain.feature_ids, True)
    np.testing.assert_array_equal(first, run())


def test_binding_rejects_invalid_ids(brain):
    with pytest.raises(ValueError):
        brain.core.silence([166700], True)
    with pytest.raises(ValueError):
        brain.core.encode(b"", b"", 64, 48)
    with pytest.raises(ValueError):
        brain.core.inject(0, float("nan"))


def test_fixture_instances_are_independent():
    p = ROOT / "pipeline/out/fixture"
    a = Brain((p / "neurons.bin").read_bytes(), (p / "graph.bin").read_bytes(), 42)
    b = Brain((p / "neurons.bin").read_bytes(), (p / "graph.bin").read_bytes(), 42)
    role = a.input_role("test", [0])
    a.inject(role, 1.5)
    a.step(5)
    assert a.snapshot() != b.snapshot()
    b.step(5)
    assert a.snapshot() != b.snapshot()


def test_export_inference_and_manifest_validation(brain, tmp_path):
    from fly_drone.env import ConnectomeEnv
    from fly_drone.training import export_actor
    from stable_baselines3 import PPO

    env = ConnectomeEnv(brain=brain, vision=False)
    try:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=8,
            batch_size=8,
            policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}},
            device="cpu",
            seed=42,
        )
        out = tmp_path / "actor.json"
        assert export_actor(model, brain, out) < 1e-5
        data = json.loads(out.read_text())
        data["dataset_hash"] = "wrong"
        with pytest.raises(ValueError):
            brain.core.load_policy(
                json.dumps(data), brain.dataset_hash, brain.feature_ids
            )
    finally:
        env.close()
