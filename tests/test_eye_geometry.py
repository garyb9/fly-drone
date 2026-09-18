import numpy as np
from fly_drone.eye_geometry import ACCEPTANCE_DEG, HFOV_DEG, SPLAY_DEG, VFOV_DEG, EyeMap

EYEMAP = EyeMap()


def test_counts_and_unit_directions():
    assert EYEMAP.n == 1398
    assert EYEMAP.left_mask.sum() == EYEMAP.right_mask.sum() == 699
    np.testing.assert_allclose(np.linalg.norm(EYEMAP.dirs, axis=1), 1.0, atol=1e-9)


def test_eyes_are_mirror_split():
    assert EYEMAP.azimuth[EYEMAP.left_mask].mean() > 0
    assert EYEMAP.azimuth[EYEMAP.right_mask].mean() < 0


def test_visible_is_the_right_eye_inside_the_frustum():
    for side, center in (("l", SPLAY_DEG), ("r", -SPLAY_DEG)):
        vis = EYEMAP.visible(side)
        assert vis.sum() > 0
        assert np.all(EYEMAP.side_mask(side)[vis])
        assert np.all(np.abs(EYEMAP.azimuth[vis] - center) <= HFOV_DEG / 2 + 1e-9)
        assert np.all(np.abs(EYEMAP.elevation[vis]) <= VFOV_DEG / 2 + 1e-9)


def test_projection_is_in_bounds_and_monotonic_in_azimuth():
    cols, rows, vis = EYEMAP.project_to_pixels((48, 64), "l")
    assert np.all((cols >= 0) & (cols <= 63)) and np.all((rows >= 0) & (rows <= 47))
    az, c = EYEMAP.azimuth[vis], cols[vis]
    order = np.argsort(az)
    assert np.all(np.diff(c[order]) <= 1e-6)  # more left => smaller column


def test_sample_constant_and_gradient_frames():
    frame = np.full((48, 64), 128, np.uint8)
    vis = EYEMAP.visible("l")
    out = EYEMAP.sample(frame, "l")
    np.testing.assert_allclose(out[vis], 128 / 255.0, atol=1e-6)
    assert np.all(out[~vis] == 0)

    gradient = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (48, 1))
    intensities = EYEMAP.sample(gradient, "l")
    az = EYEMAP.azimuth[vis]
    corr = np.corrcoef(az, intensities[vis])[0, 1]
    assert abs(corr) > 0.5  # the sampler really is position-dependent


def test_visible_ommatidial_spread_is_mostly_horizontal():
    vis = EYEMAP.visible("l")
    centred = EYEMAP.dirs[vis] - EYEMAP.dirs[vis].mean(0)
    vals, vecs = np.linalg.eigh(centred.T @ centred)
    pc1 = vecs[:, np.argsort(vals)[-1]]
    assert abs(pc1[2]) < 0.7  # first spread axis is not vertical
    assert ACCEPTANCE_DEG > 0 and SPLAY_DEG > 0
