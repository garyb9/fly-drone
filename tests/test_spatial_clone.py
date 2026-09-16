import numpy as np
import pytest
from fly_drone import spatial_clone
from fly_drone.brain import BrainRuntime
from fly_drone.retinotopy import build_default_maps


@pytest.fixture(scope="module")
def maps():
    return build_default_maps(BrainRuntime().cells)


def test_single_cue_seeds_light_and_loom_maps(maps):
    cues = np.array([1.5, 0.25, 0.75, 0.0], np.float32)
    targets = spatial_clone.v4_cues_to_targets(cues, maps)
    assert set(targets) == set(maps)
    for channel in ("mi1_l", "tm3_l"):
        np.testing.assert_allclose(targets[channel], 1.5)
    for channel in ("mi1_r", "tm3_r"):
        np.testing.assert_allclose(targets[channel], 0.25)
    for channel in ("lc4_l", "lplc2_l"):
        np.testing.assert_allclose(targets[channel], 0.75)
    for channel in ("lc4_r", "lplc2_r"):
        np.testing.assert_allclose(targets[channel], 0.0)
    for channel in spatial_clone.UNSEEDED:
        np.testing.assert_allclose(targets[channel], 1.0)


def test_target_shapes_follow_the_map_grids(maps):
    targets = spatial_clone.v4_cues_to_targets(np.zeros(4), maps)
    for channel, block in targets.items():
        assert block.shape == (maps[channel].nx, maps[channel].ny)


def test_batch_targets_carry_the_per_frame_values(maps):
    cues = np.array([[1.0, 2.0, 0.0, 0.5], [0.0, 0.0, 2.0, 2.0]], np.float32)
    targets = spatial_clone.v4_cues_to_targets(cues, maps)
    assert targets["mi1_l"].shape == (2, maps["mi1_l"].nx, maps["mi1_l"].ny)
    np.testing.assert_allclose(targets["mi1_l"][0], 1.0)
    np.testing.assert_allclose(targets["mi1_l"][1], 0.0)
    np.testing.assert_allclose(targets["lc4_r"][1], 2.0)
    assert all(block.min() >= 0 and block.max() <= 2 for block in targets.values())


def test_neutral_baseline_is_configurable(maps):
    targets = spatial_clone.v4_cues_to_targets(np.zeros(4), maps, neutral=0.5)
    for channel in spatial_clone.UNSEEDED:
        np.testing.assert_allclose(targets[channel], 0.5)


def test_seed_motion_broadcasts_the_loom_cue(maps):
    cues = np.array([1.5, 0.25, 0.75, 0.1], np.float32)
    seeded = spatial_clone.v4_cues_to_targets(cues, maps, seed_motion=True)
    for channel, cue in spatial_clone.MOTION_TO_CUE.items():
        np.testing.assert_allclose(seeded[channel], cues[cue])


def test_mirror_targets_swaps_eyes_and_flips_width(maps):
    cues = np.array([1.5, 0.25, 0.75, 0.0], np.float32)
    targets = spatial_clone.v4_cues_to_targets(cues, maps)
    mirrored = spatial_clone.mirror_targets(targets)
    for channel in targets:
        other = channel[:-1] + ("r" if channel.endswith("_l") else "l")
        np.testing.assert_array_equal(mirrored[channel], targets[other][..., ::-1])


def test_fit_spatial_clone_writes_a_loadable_v6_encoder(tmp_path):
    from fly_drone.brain import ENCODER_VERSION
    from fly_drone.spatial_encoder import CHANNEL_ORDER, SpatialEncoder

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
    from fly_drone.sac import fit_spatial_clone

    report = fit_spatial_clone(
        [path], tmp_path / "fit", steps=5, batch=16, device="cpu"
    )
    assert set(report["held_out"]) == set(CHANNEL_ORDER)
    assert set(report["groups"]) == {"light", "motion", "loom"}
    saved = SpatialEncoder.load(tmp_path / "fit" / "encoder.pt")
    assert saved.version == report["version"]
