import numpy as np
import pytest
from fly_drone.brain import BrainRuntime
from fly_drone.spatial_encoder import SpatialEncoder, flat_dim, flatten_np


@pytest.fixture(scope="module")
def v6(tmp_path_factory):
    path = tmp_path_factory.mktemp("v6") / "encoder.pt"
    enc = SpatialEncoder.fresh(seed=0)
    enc.save(path)
    return BrainRuntime(encoder=path)


def test_v6_roles_cover_every_injectable_cell_in_its_channel(v6):
    from fly_drone import spatial_encoder as se

    for channel in se.CHANNEL_ORDER:
        assert set(v6.roles[channel]) == set(v6.maps[channel].patches)
        assert sorted(v6.input_ids[channel]) == sorted(v6.maps[channel].all_cells())


def test_v6_pathway_ids_match_the_group_cells(v6):
    assert v6.pathway_ids["light_l"] == (v6.input_ids["mi1_l"] + v6.input_ids["tm3_l"])
    assert v6.pathway_ids["looming_r"] == (
        v6.input_ids["lc4_r"] + v6.input_ids["lplc2_r"]
    )
    assert v6.pathway_ids["motion_l"] == (v6.input_ids["tm4_l"] + v6.input_ids["t2_l"])


def test_v6_equal_currents_reach_the_right_cells(v6):
    v6.reset(3)
    vector = np.full(flat_dim(), 0.7, np.float32)
    v6.set_currents(vector)
    assert v6.current_maps["tm4_l"].shape == (12, 8)
    np.testing.assert_allclose(v6.current_maps["tm4_l"], 0.7)
    v6.step(1)
    assert v6.cues.shape == (flat_dim(),)


def test_v6_set_currents_clips_and_checks_shape(v6):
    with pytest.raises(ValueError):
        v6.set_currents(np.zeros(8, np.float32))
    out = v6.set_currents(np.full(flat_dim(), 3.0, np.float32))
    assert out.max() == 2.0 and v6.current_maps["mi1_r"].max() == 2.0


def test_v6_encoder_round_trip_through_the_runtime():
    enc = SpatialEncoder.fresh(seed=1)
    brain = BrainRuntime(encoder=enc)
    assert brain.encoder_version == enc.version
    brain.push_frame(np.full((2, 48, 64, 3), 90, np.uint8))
    before = brain.cues.copy()
    brain.push_frame(np.full((2, 48, 64, 3), 130, np.uint8))
    brain.encode_stack()
    assert not np.array_equal(before, brain.cues)
    np.testing.assert_allclose(brain.cues, flatten_np(brain.current_maps))


def test_v4_role_set_is_unchanged():
    brain = BrainRuntime()
    assert tuple(brain.input_ids) == ("light_l", "light_r", "looming_l", "looming_r")
    assert brain.current_maps is None and brain.cues.shape == (4,)
