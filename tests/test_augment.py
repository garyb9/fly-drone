import numpy as np
from fly_drone import augment
from fly_drone.augment import AugmentConfig


def _images(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (2, 48, 64, 3), dtype=np.uint8)


def test_default_config_is_the_identity():
    images = _images()
    out = augment.randomize(images, np.random.default_rng(0))
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, images)
    assert AugmentConfig().is_identity


def test_brightness_raises_the_mean():
    images = _images()
    out = augment.randomize(
        images, np.random.default_rng(1), AugmentConfig(brightness=0.3)
    )
    assert out.dtype == np.uint8 and out.shape == images.shape
    assert out.mean() > images.mean()


def test_contrast_scales_around_zero_and_clips():
    images = np.full((2, 8, 8, 3), 200, np.uint8)
    out = augment.randomize(
        images, np.random.default_rng(2), AugmentConfig(contrast=0.5)
    )
    assert out.max() <= 255 and out.min() >= 0
    assert not np.array_equal(out, images)


def test_noise_changes_pixels_but_not_shape():
    images = _images()
    out = augment.randomize(images, np.random.default_rng(3), AugmentConfig(noise=0.1))
    assert out.shape == images.shape and out.dtype == np.uint8
    assert not np.array_equal(out, images)


def test_eye_gain_is_independent_per_eye():
    images = np.full((2, 8, 8, 3), 100, np.uint8)
    out = augment.randomize(
        images, np.random.default_rng(4), AugmentConfig(eye_gain=0.5)
    )
    # The two eyes must differ somewhere; a shared gain would keep them equal.
    assert not np.array_equal(out[0], out[1])


def test_transform_is_deterministic_for_a_seed():
    images = _images()
    a = augment.make_transform(AugmentConfig(brightness=0.2, noise=0.05), seed=9)
    b = augment.make_transform(AugmentConfig(brightness=0.2, noise=0.05), seed=9)
    np.testing.assert_array_equal(a(images), b(images))
