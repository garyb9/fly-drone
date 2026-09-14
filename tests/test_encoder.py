import json

import numpy as np
import torch
from fly_drone import encoder
from fly_drone.brain import ENCODER_VERSION, BrainRuntime
from fly_drone.encoder import LearnedEncoder, v4_targets


def test_v4_targets_copy_light_to_mi1_tm3_and_loom_to_lc4_lplc2():
    cues = np.array([[0.1, 0.2, 0.3, 0.4]])
    assert v4_targets(cues).tolist() == [[0.1, 0.2, 0.1, 0.2, 0.3, 0.4, 0.3, 0.4]]


def test_fresh_encoder_outputs_eight_bounded_currents_and_128_features():
    enc = LearnedEncoder.fresh(seed=1)
    stack = np.random.default_rng(0).integers(0, 256, (6, 48, 64), dtype=np.uint8)
    currents = enc.currents(stack)
    assert currents.shape == (8,) and currents.dtype == np.float32
    assert (currents >= 0).all() and (currents <= 2).all()
    x = torch.zeros(2, 6, 48, 64)
    assert enc.extractor({"eyes": x}).shape == (2, 2 * encoder.FEATURES)


def test_save_load_round_trip_keeps_currents_and_version(tmp_path):
    enc = LearnedEncoder.fresh(seed=2)
    version = enc.save(tmp_path / "e.pt")
    assert version.startswith("learned-v5:") and len(version) == len("learned-v5:") + 16
    loaded = LearnedEncoder.load(tmp_path / "e.pt")
    stack = np.full((6, 48, 64), 77, np.uint8)
    np.testing.assert_array_equal(enc.currents(stack), loaded.currents(stack))
    assert loaded.version == version
    with torch.no_grad():
        loaded.mu.bias[0] += 0.1
    assert loaded.version != version


def test_brain_runtime_loads_a_learned_encoder_and_encodes_its_stack(tmp_path):
    enc = LearnedEncoder.fresh(seed=3)
    enc.save(tmp_path / "e.pt")
    brain = BrainRuntime(encoder=tmp_path / "e.pt")
    assert brain.learned and brain.encoder_version == enc.version
    brain.push_frame(np.full((2, 48, 64, 3), 90, np.uint8))
    np.testing.assert_allclose(brain.encode_stack(), enc.currents(brain.stack.array()))


def test_clone_job_records_one_stack_and_v4_cue_per_frame():
    stacks, cues, flights = encoder._clone_job(([5], 0.2, 3, 0.0))
    assert len(stacks) == len(cues) == len(flights) == 5
    assert stacks[0].shape == (6, 48, 64) and stacks[0].dtype == np.uint8
    assert cues[0].shape == (4,) and set(flights) == {5}


def test_fit_clone_writes_a_loadable_encoder_and_per_channel_report(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    path = tmp_path / "clone.npz"
    np.savez(
        path,
        stacks=rng.integers(0, 256, (n, 6, 48, 64), dtype=np.uint8),
        cues=rng.uniform(0, 2, (n, 4)).astype(np.float32),
        flight=np.repeat(np.arange(20), n // 20).astype(np.int32),
        encoder_version=ENCODER_VERSION,
    )
    report = encoder.fit_clone(
        [path], tmp_path / "fit", steps=5, batch=16, device="cpu"
    )
    assert set(report["held_out"]) == set(encoder.V5_CHANNELS)
    saved = json.loads((tmp_path / "fit" / "clone.json").read_text())
    assert (
        saved["version"] == LearnedEncoder.load(tmp_path / "fit" / "encoder.pt").version
    )
