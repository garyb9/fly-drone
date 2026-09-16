"""Domain randomisation for the eye frames (standalone; additive).

The v6 sensing spec (§4.2) wants brightness, contrast and noise perturbed so the
encoder keys on structure rather than absolute luma, which the goal's "unseen
arenas" clause needs. This is the pure transform; the env hook that calls it
(``ConnectomeEnv.push_frame``) lands with the M2 wiring, so nothing here touches
the running v5 path.

The default config is the identity, so an unset hook changes nothing. The
transform accepts any ``uint8`` array whose first axis is the eye (or frame)
axis: the env's ``(2, H, W, 3)`` RGB images or a ``(2*n, H, W)`` luma stack.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AugmentConfig:
    """Per-call randomisation strengths; all zero means identity."""

    brightness: float = 0.0  # additive luma offset, sampled in [-b, +b]
    contrast: float = 0.0  # shared multiplicative gain, sampled in [1-c, 1+c]
    eye_gain: float = 0.0  # independent per-eye gain deviation (calibration drift)
    noise: float = 0.0  # gaussian noise std in [0, 1] luma units

    @property
    def is_identity(self):
        return not (self.brightness or self.contrast or self.eye_gain or self.noise)


NONE = AugmentConfig()


def randomize(images, rng, config=NONE):
    """Apply one random sample of the configured perturbations to a uint8 array."""
    if config.is_identity:
        return np.asarray(images, dtype=np.uint8)
    x = np.asarray(images, dtype=np.float32) / 255.0
    if config.brightness:
        x = x + rng.uniform(-config.brightness, config.brightness)
    if config.contrast:
        x = x * (1.0 + rng.uniform(-config.contrast, config.contrast))
    if config.eye_gain:
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        x = x * (1.0 + rng.uniform(-config.eye_gain, config.eye_gain, size=shape))
    if config.noise:
        x = x + rng.normal(0.0, config.noise, size=x.shape)
    return np.clip(x * 255.0, 0.0, 255.0).round().astype(np.uint8)


def make_transform(config=NONE, seed=0):
    """Return a stateful ``images -> images`` transform for the env hook."""
    rng = np.random.default_rng(seed)

    def transform(images):
        return randomize(images, rng, config)

    return transform
