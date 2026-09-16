import numpy as np
import torch
from fly_drone import spatial_encoder
from fly_drone.brain import STACK_FRAMES
from fly_drone.spatial_encoder import CHANNEL_ORDER, SpatialEncoder


def _stack(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (2 * STACK_FRAMES, 48, 64), dtype=np.uint8)


def test_channel_layout_matches_the_retinotopic_grids():
    assert CHANNEL_ORDER == (
        "mi1_l",
        "mi1_r",
        "tm3_l",
        "tm3_r",
        "tm4_l",
        "tm4_r",
        "t2_l",
        "t2_r",
        "lc4_l",
        "lc4_r",
        "lplc2_l",
        "lplc2_r",
    )
    assert spatial_encoder.channel_grid("mi1_r") == (12, 8)
    assert spatial_encoder.channel_grid("tm4_r") == (12, 8)
    assert spatial_encoder.channel_grid("lplc2_l") == (4, 3)
    # 4 spatial x 12*8 + 2 direct x 4*3, per eye.
    assert spatial_encoder.flat_dim() == (4 * 12 * 8 + 2 * 4 * 3) * 2


def test_eye_inputs_add_frame_differences_and_a_side_flag():
    n, frames = 2, STACK_FRAMES
    rng = np.random.default_rng(0)
    eyes = torch.as_tensor(
        rng.integers(0, 256, (n, 2 * frames, 48, 64)), dtype=torch.float32
    )
    left_in, right_in = spatial_encoder.eye_inputs(eyes)
    assert left_in.shape == (n, 2 * frames, 48, 64)
    left, right = eyes[:, :frames], eyes[:, frames:]
    assert torch.equal(left_in[:, :frames], left)
    assert torch.allclose(
        left_in[:, frames : 2 * frames - 1], left[:, 1:] - left[:, :-1]
    )
    assert bool((left_in[:, -1] == 1).all()) and bool((right_in[:, -1] == -1).all())
    assert torch.equal(right_in[:, :frames], torch.flip(right, dims=[3]))


def test_fresh_encoder_currents_have_layout_shapes_and_bounds():
    enc = SpatialEncoder.fresh(seed=1)
    currents = enc.currents(_stack(2))
    assert set(currents) == set(CHANNEL_ORDER)
    assert enc.version.startswith("learned-v6:")
    for name, arr in currents.items():
        nx, ny = spatial_encoder.channel_grid(name)
        assert arr.shape == (nx, ny)
        assert arr.dtype == np.float32
        assert (arr >= 0).all() and (arr <= 2).all()


def test_save_load_round_trip_keeps_currents_and_version(tmp_path):
    enc = SpatialEncoder.fresh(seed=2)
    stack = _stack(3)
    version = enc.save(tmp_path / "e.pt")
    assert version.startswith("learned-v6:")
    assert len(version) == len("learned-v6:") + 16
    loaded = SpatialEncoder.load(tmp_path / "e.pt")
    for name in CHANNEL_ORDER:
        np.testing.assert_array_equal(
            enc.currents(stack)[name], loaded.currents(stack)[name]
        )
    assert loaded.version == version
    with torch.no_grad():
        loaded.net.spatial.bias[0] += 0.1
    assert loaded.version != version


def test_encoder_uses_temporal_order_of_the_stack():
    # Same three frames, different order: only the difference channels see it.
    a = np.zeros((48, 64), np.uint8)
    b = np.full((48, 64), 200, np.uint8)
    eye_ab = np.stack([a, a, b])
    eye_bb = np.stack([a, b, b])
    early = np.concatenate([eye_ab, eye_ab])
    late = np.concatenate([eye_bb, eye_bb])
    enc = SpatialEncoder.fresh(seed=3)
    first, second = enc.currents(early), enc.currents(late)
    assert any(not np.allclose(first[n], second[n]) for n in CHANNEL_ORDER)


def test_flatten_unflatten_round_trip():
    enc = SpatialEncoder.fresh(seed=4)
    x = torch.as_tensor(_stack(5)[None], dtype=torch.float32) / 255.0
    with torch.no_grad():
        raw = enc.net(x)
    flat = spatial_encoder.flatten(raw)
    assert flat.shape == (1, spatial_encoder.flat_dim())
    back = spatial_encoder.unflatten(flat)
    for name in CHANNEL_ORDER:
        assert torch.allclose(back[name], raw[name])


def test_group_slices_partition_the_flat_vector():
    slices = spatial_encoder.group_slices()
    merged = np.concatenate([slices[group] for group in slices])
    assert sorted(merged.tolist()) == list(range(spatial_encoder.flat_dim()))
    assert set(slices) == {"light", "motion", "loom"}
    assert len(slices["light"]) == 4 * 12 * 8
    assert len(slices["motion"]) == 4 * 12 * 8
    assert len(slices["loom"]) == 4 * 4 * 3


def test_mirror_maps_swaps_and_flips_eyes():
    out = {
        name: np.arange(
            int(np.prod(spatial_encoder.channel_grid(name))), dtype=np.float32
        ).reshape(spatial_encoder.channel_grid(name))
        for name in CHANNEL_ORDER
    }
    mirrored = spatial_encoder.mirror_maps(out)
    for name in CHANNEL_ORDER:
        other = name[:-1] + ("r" if name.endswith("_l") else "l")
        np.testing.assert_array_equal(mirrored[name], out[other][:, ::-1])


def test_numpy_flatten_unflatten_round_trip_and_is_file(tmp_path):
    enc = SpatialEncoder.fresh(seed=6)
    maps = enc.currents(_stack(7))
    flat = spatial_encoder.flatten_np(maps)
    assert flat.shape == (spatial_encoder.flat_dim(),)
    back = spatial_encoder.unflatten_np(flat)
    for name in CHANNEL_ORDER:
        np.testing.assert_array_equal(back[name], maps[name])
    version = enc.save(tmp_path / "v6.pt")
    assert version.startswith("learned-v6:") and SpatialEncoder.is_file(
        tmp_path / "v6.pt"
    )


def test_mirror_flat_matches_mirror_maps():
    out = {
        name: np.arange(
            int(np.prod(spatial_encoder.channel_grid(name))), dtype=np.float32
        ).reshape(spatial_encoder.channel_grid(name))
        for name in CHANNEL_ORDER
    }
    flat = spatial_encoder.flatten_np(out)
    np.testing.assert_array_equal(
        spatial_encoder.mirror_flat(flat),
        spatial_encoder.flatten_np(spatial_encoder.mirror_maps(out)),
    )


def test_mirror_eyes_swaps_halves_and_flips_width():
    rng = np.random.default_rng(8)
    stacks = rng.integers(0, 256, (4, 6, 48, 64), dtype=np.uint8)
    mask = np.array([True, False, True, False])
    out = spatial_encoder.mirror_eyes(stacks, mask)
    mirrored = stacks[:, :, :, ::-1]
    expected = np.concatenate([mirrored[mask][:, 3:], mirrored[mask][:, :3]], axis=1)
    np.testing.assert_array_equal(out[mask], expected)
    np.testing.assert_array_equal(out[~mask], stacks[~mask])


def test_spatial_maps_summary_covers_every_channel(tmp_path):
    from fly_drone import maps

    result = maps.run(tmp_path / "maps", None)
    assert len(result["rows"]) == len(CHANNEL_ORDER)
    assert result["flat_dim"] == spatial_encoder.flat_dim()
    assert (tmp_path / "maps" / "summary.json").exists()
