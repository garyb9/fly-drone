import json

import numpy as np
import pytest
from fly_drone.brain import (
    ENCODER_VERSION,
    LEARNED_EXTERNAL,
    V4_TO_V5,
    V5_CHANNELS,
    BrainRuntime,
    FrameStack,
    luma_u8,
)


@pytest.fixture(scope="module")
def v4():
    return BrainRuntime()


@pytest.fixture(scope="module")
def v5():
    return BrainRuntime(encoder="external")


def test_luma_matches_the_rust_encoder_weights():
    images = np.zeros((2, 48, 64, 3), np.uint8)
    images[0, ..., 0] = 255
    images[1, ..., 1] = 100
    y = luma_u8(images)
    assert y.shape == (2, 48, 64) and y.dtype == np.uint8
    assert y[0, 0, 0] == round(0.2126 * 255) and y[1, 0, 0] == round(0.7152 * 100)


def test_frame_stack_fills_with_the_first_frame_then_shifts_and_clears():
    stack = FrameStack()
    assert stack.array().shape == (6, 48, 64) and not stack.array().any()
    frames = [np.full((2, 48, 64, 3), v, np.uint8) for v in (10, 20, 30, 40)]
    stack.push(frames[0])
    assert (stack.array()[[0, 1, 2]] == 10).all() and (stack.array()[3:] == 10).all()
    for f in frames[1:]:
        stack.push(f)
    assert [int(stack.array()[i, 0, 0]) for i in range(6)] == [20, 30, 40, 20, 30, 40]
    stack.clear()
    assert not stack.array().any()


def test_v5_roles_split_v4_roles_by_cell_type(v4, v5):
    assert tuple(v5.input_ids) == V5_CHANNELS
    for side in "lr":
        light = v5.input_ids[f"mi1_{side}"] + v5.input_ids[f"tm3_{side}"]
        loom = v5.input_ids[f"lc4_{side}"] + v5.input_ids[f"lplc2_{side}"]
        assert sorted(light) == sorted(v4.input_ids[f"light_{side}"])
        assert sorted(loom) == sorted(v4.input_ids[f"looming_{side}"])
    ids = [i for v in v5.input_ids.values() for i in v]
    assert len(ids) == len(set(ids))
    assert [len(v5.input_ids[c]) for c in V5_CHANNELS] == [
        886,
        887,
        1017,
        1037,
        71,
        55,
        94,
        91,
    ]


def test_equal_split_currents_reproduce_the_v4_brain_exactly(v4, v5):
    rng = np.random.default_rng(3)
    v4.reset(11)
    v5.reset(11)
    for _ in range(5):
        cues = rng.uniform(0, 2, 4).astype(np.float32)
        v4.cues = cues
        v5.set_currents(cues[list(V4_TO_V5)])
        np.testing.assert_array_equal(v4.step(8), v5.step(8))


def test_versions_and_actor_encoder_check(v4, v5, tmp_path):
    assert v4.encoder_version == ENCODER_VERSION and not v4.learned
    assert v5.encoder_version == LEARNED_EXTERNAL and v5.learned
    n = len(v4.feature_ids)
    actor = {
        "version": 1,
        "encoder_version": ENCODER_VERSION,
        "dataset_hash": v4.dataset_hash,
        "feature_ids": v4.feature_ids,
        "mean": [0.0] * n,
        "scale": [1.0] * n,
        "layers": [{"weights": [[0.0] * n] * 4, "bias": [0.0] * 4}],
        "action_limits": [1.0] * 4,
    }
    path = tmp_path / "v4.json"
    path.write_text(json.dumps(actor))
    v4.load_policy(path)
    with pytest.raises(ValueError, match="encoder"):
        v5.load_policy(path)
    v5.load_policy(path, check_encoder=False)
    actor["encoder_version"] = "learned-v5:0123456789abcdef"
    path.write_text(json.dumps(actor))
    with pytest.raises(ValueError, match="encoder"):
        v4.load_policy(path)


def test_set_currents_clips_and_checks_shape_and_sense_is_v4_only(v5):
    v5.set_currents(np.array([-1, 0.5, 3, 1, 1, 1, 1, 1]))
    assert v5.cues.tolist() == [0.0, 0.5, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        v5.set_currents(np.zeros(4))
    with pytest.raises(RuntimeError):
        v5.sense(np.zeros((2, 48, 64, 3), np.uint8))
    assert v5.encode_stack() is v5.cues  # external: the learner owns the currents


def test_pathway_silencing_names_work_on_v5(v5):
    v5.silence_inputs(("light_l", "looming_r"))
    v5.core.restore()
