"""P3 declared optic-flow relay: rendered eye frames -> direction-selective motion currents.

Declared, deterministic, no learning. Two consecutive eye frames give a signed per-eye
horizontal (yaw) and vertical (pitch) global translation via phase correlation; the signed
axes are rectified into the direction-selective motion populations (T4/T5 by subtype:
``a``/``b`` horizontal, ``c``/``d`` vertical). The mapping is a declared prior, not a fit.

The relay is an *additive* sensory front-end: it injects its own input roles each brain tick
and never changes ``cues``, the canonical bundle, or ``roam_eval.ACCEPTANCE``. It is
validated offline against the simulator's own camera egomotion (labels only; never injected)
by ``scripts/validate_relay_flow.py``.
"""

import numpy as np

RELAY_VERSION = "optic-flow-relay-v1"

# Camera geometry (matches brain.EYE_SHAPE and the two splayed plant cameras).
HFOV_DEG = 91.3
VFOV_DEG = 75.0

# Declared current encoding: zero flow -> zero current; FLOW_GAIN scales rad/frame to current.
FLOW_BASE = 0.0
FLOW_GAIN = 10.0

# Signed flow axis -> (positive-direction, negative-direction) role prefixes.
FLOW_ROLES = {"yaw": ("yaw_pos", "yaw_neg"), "pitch": ("pitch_up", "pitch_dn")}

# Direction-selective target populations, by subtype, as a declared prior. T4a/T4b and
# T5a/T5b are the horizontal (front-to-back / back-to-front) subtypes; T4c/T4d and T5c/T5d
# are the vertical ones. The connectome has no cell<->ommatidium join, so this is the
# anatomical handle used instead; the direct drive is a declared shortcut past the
# photoreceptor->lamina relay (see docs/external-prior-art.md).
RELAY_TYPES = {
    "yaw_pos": ("T4a", "T5a"),
    "yaw_neg": ("T4b", "T5b"),
    "pitch_up": ("T4c", "T5c"),
    "pitch_dn": ("T4d", "T5d"),
}
SIDES = ("l", "r")


def luma_u8(images):
    """Rec. 709 luma of packed RGB eyes, matching the v4 encoder weights."""
    rgb = np.asarray(images, dtype=np.float32)
    y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return np.round(y).astype(np.uint8)


def _parabolic(corr, index, axis, size):
    """Sub-pixel peak offset along ``axis`` by fitting a parabola to the three samples."""
    lo = list(index)
    hi = list(index)
    lo[axis] = (index[axis] - 1) % size
    hi[axis] = (index[axis] + 1) % size
    before = corr[tuple(lo)]
    centre = corr[tuple(index)]
    after = corr[tuple(hi)]
    denom = before - 2.0 * centre + after
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (before - after) / denom, -0.5, 0.5))


def phase_shift(current, previous):
    """Signed sub-pixel shift ``(dy, dx)`` aligning ``previous`` onto ``current``.

    FFT phase correlation with a Hann window and a parabolic sub-pixel peak fit; the
    declared global-translation motion model. Sub-pixel refinement matters because the
    eye only spans 64x48, so small head motion is well under a pixel per frame.
    """
    height, width = current.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    a = (current - current.mean()) * window
    b = (previous - previous.mean()) * window
    cross = np.fft.rfft2(a) * np.conj(np.fft.rfft2(b))
    cross /= np.abs(cross) + 1e-9
    corr = np.fft.irfft2(cross, s=(height, width))
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    dy = peak[0] if peak[0] <= height // 2 else peak[0] - height
    dx = peak[1] if peak[1] <= width // 2 else peak[1] - width
    dy += _parabolic(corr, peak, 0, height)
    dx += _parabolic(corr, peak, 1, width)
    return float(dy), float(dx)


def pixel_to_angle(dy, dx, shape):
    """Convert a pixel shift to (yaw, pitch) radians using the camera field of view."""
    height, width = shape
    yaw = dx / width * np.radians(HFOV_DEG)
    pitch = -dy / height * np.radians(VFOV_DEG)
    return float(yaw), float(pitch)


class RelayState:
    """Carries the previous eye frame and turns signed optic flow into declared currents."""

    def __init__(self, gain=FLOW_GAIN, base=FLOW_BASE):
        self.gain = float(gain)
        self.base = float(base)
        self.previous = None
        self.yaw = {side: 0.0 for side in SIDES}
        self.pitch = {side: 0.0 for side in SIDES}

    def reset(self):
        self.previous = None
        self.yaw = {side: 0.0 for side in SIDES}
        self.pitch = {side: 0.0 for side in SIDES}

    def observe_frame(self, images):
        """Update signed per-eye flow from the rendered stereo frames (one per env step)."""
        gray = luma_u8(images).astype(np.float32)
        if self.previous is not None:
            for index, side in enumerate(SIDES):
                dy, dx = phase_shift(gray[index], self.previous[index])
                self.yaw[side], self.pitch[side] = pixel_to_angle(
                    dy, dx, gray.shape[1:]
                )
        self.previous = gray
        return self.yaw, self.pitch

    def currents(self):
        """The eight rectified direction-selective currents for this tick."""
        out = {}
        for side in SIDES:
            yaw, pitch = self.yaw[side], self.pitch[side]
            pos, neg = FLOW_ROLES["yaw"]
            out[f"{pos}_{side}"] = self.base + self.gain * max(0.0, yaw)
            out[f"{neg}_{side}"] = self.base + self.gain * max(0.0, -yaw)
            up, down = FLOW_ROLES["pitch"]
            out[f"{up}_{side}"] = self.base + self.gain * max(0.0, pitch)
            out[f"{down}_{side}"] = self.base + self.gain * max(0.0, -pitch)
        return {k: float(np.clip(v, 0.0, 2.0)) for k, v in out.items()}
