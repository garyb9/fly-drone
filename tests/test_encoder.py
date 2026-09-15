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
    diffs = saved["held_out_differences"]
    assert set(diffs) == {"mi1", "tm3", "lc4", "lplc2"}
    for stats in diffs.values():
        assert set(stats) == {"r", "rmse", "gain"}
        assert all(np.isfinite(v) for v in stats.values())


def test_clone_batches_draw_a_third_each_uniform_loom_and_light_side_frames():
    n = 3000
    targets = np.zeros((n, 8), np.float32)
    targets[:100, 4] = 1.0  # loom-active frames
    targets[100:200, 0] = 1.0  # light off to the left: mi1_l - mi1_r = 1
    train_ids = np.arange(n)
    rng = np.random.default_rng(0)
    pools = encoder.clone_pools(train_ids, targets)
    ids = np.concatenate(
        [encoder.clone_batch_ids(rng, train_ids, pools, 256) for _ in range(20)]
    )
    assert len(ids) == 20 * 256
    # Uniform draws land on either pool only ~3% of the time each.
    assert 0.30 < np.mean(ids < 100) < 0.40
    assert 0.30 < np.mean((ids >= 100) & (ids < 200)) < 0.40


def test_clone_batches_fall_back_to_uniform_when_a_pool_is_empty():
    targets = np.zeros((1000, 8), np.float32)
    rng = np.random.default_rng(0)
    train_ids = np.arange(1000)
    pools = encoder.clone_pools(train_ids, targets)
    assert all(len(pool) == 0 for pool in pools)
    ids = encoder.clone_batch_ids(rng, train_ids, pools, 100)
    assert len(ids) == 100 and ids.min() >= 0 and ids.max() < 1000


def test_clone_loss_includes_the_left_right_difference_term(monkeypatch):
    target = torch.ones(4, 8)
    # Same per-channel error, but left and right err in opposite directions.
    pred = target + 0.1 * torch.tensor([1.0, -1.0] * 4)
    monkeypatch.setattr(encoder, "DIFF_WEIGHT", 0.0)
    without = encoder.clone_loss(pred, target).item()
    monkeypatch.setattr(encoder, "DIFF_WEIGHT", 1.0)
    with_diff = encoder.clone_loss(pred, target).item()
    assert without > 0 and with_diff > without
