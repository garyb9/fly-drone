import numpy as np
import pytest
from fly_drone import retinotopy
from fly_drone.brain import BrainRuntime


@pytest.fixture(scope="module")
def brain():
    return BrainRuntime()


@pytest.fixture(scope="module")
def maps(brain):
    return retinotopy.build_default_maps(brain.cells)


def test_default_maps_cover_the_expected_channels_and_every_cell(maps):
    assert set(maps) == {
        "tm4_l",
        "tm4_r",
        "t2_l",
        "t2_r",
        "lc4_l",
        "lc4_r",
        "lplc2_l",
        "lplc2_r",
    }
    tm4_l = maps["tm4_l"]
    assert tm4_l.population == "Tm4" and tm4_l.side == "l"
    assert tm4_l.nx == retinotopy.DEFAULT_GRID[0]
    assert maps["lc4_l"].nx == retinotopy.DIRECT_GRID[0]
    # Every cell of the population is assigned exactly once.
    assert len(tm4_l.all_cells()) == tm4_l.n_cells
    assert len(set(tm4_l.all_cells())) == tm4_l.n_cells


def test_build_map_is_deterministic(brain):
    a = retinotopy.build_map(brain.cells, "Tm4", "l", 12, 8)
    b = retinotopy.build_map(brain.cells, "Tm4", "l", 12, 8)
    assert a.patches == b.patches
    np.testing.assert_allclose(a.basis, b.basis)


def test_patch_of_reproduces_the_binned_assignment(brain):
    m = retinotopy.build_map(brain.cells, "Tm4", "l", 12, 8)
    for (bx, by), ids in m.patches.items():
        for i in ids:
            assert m.patch_of(brain.cells[i]["position"]) == (bx, by)


def test_build_map_rejects_an_unknown_population(brain):
    with pytest.raises(ValueError):
        retinotopy.build_map(brain.cells, "NoSuchCell", "l", 4, 3)


def test_maps_round_trip_through_json(tmp_path, maps):
    path = retinotopy.save_maps(maps, tmp_path / "maps.json")
    loaded = retinotopy.load_maps(path)
    assert set(loaded) == set(maps)
    assert loaded["tm4_l"].patches == maps["tm4_l"].patches
    np.testing.assert_allclose(loaded["tm4_l"].basis, maps["tm4_l"].basis)


def test_apply_map_injects_bounded_currents(brain, maps):
    m = maps["tm4_l"]
    roles = retinotopy.define_roles(brain.core, m)
    assert len(roles) == len(m.patches)
    # Out-of-range values are clipped, not rejected; inject() itself enforces [0, 2].
    retinotopy.apply_map(brain.core, roles, np.full((m.nx, m.ny), 5.0))
    retinotopy.apply_map(brain.core, roles, np.full((m.nx, m.ny), -1.0))


def test_spatial_tm4_injection_lateralises_the_loom_circuit(brain, maps):
    left, right = maps["tm4_l"], maps["tm4_r"]
    roles_l = retinotopy.define_roles(brain.core, left)
    roles_r = retinotopy.define_roles(brain.core, right)
    lc4_l = brain._cells("l", ("LC4",))
    lc4_r = brain._cells("r", ("LC4",))

    def mean(ids):
        return float(np.mean(brain.core.activity(list(ids))))

    brain.reset(7)
    for _ in range(80):
        retinotopy.apply_map(brain.core, roles_l, np.full((left.nx, left.ny), 1.6))
        retinotopy.apply_map(brain.core, roles_r, np.zeros((right.nx, right.ny)))
        brain.core.step(1)
    ipsi, contra = mean(lc4_l), mean(lc4_r)
    assert ipsi > 0.05
    assert contra < 0.2 * ipsi
