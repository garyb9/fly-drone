"""Measured Drosophila ommatidial directions and a camera sampler.

Ports only the **eye geometry** from ``dylankainth/flybrain`` ``flybrain_eye_map.py``:
the Buchner-1971 ommatidial viewing directions digitized by the Straw lab, a pinhole
projection and an acceptance-window sampler. The cell↔ommatidium identity is *not*
ported — the source itself marks it as modelled, and this project injects at
Mi1/Tm3/LC4/LPLC2 through a learned encoder rather than a photoreceptor luminance map.

Used to measure the v6 retinotopic map's visual axis; see
``docs/superpowers/specs/2026-09-18-prior-art-harvest-design.md`` workstream A.

Data licence: BSD (Straw lab), vendored with ``vendor/eye_map/NOTICE.md``.
"""

from pathlib import Path

import numpy as np

EYE_MAP_CSV = (
    Path(__file__).resolve().parent / "vendor/eye_map/receptor_directions_buchner71.csv"
)

# Camera geometry from docs/sensory-model.md section 1: two 64x48 eyes, fovy 75 deg,
# yaw splay +/-0.75 rad. The eye map's frame is +X frontal, +Y left, +Z dorsal.
HFOV_DEG = 91.3
VFOV_DEG = 75.0
SPLAY_DEG = 43.0
ACCEPTANCE_DEG = 5.0


class EyeMap:
    """Buchner-1971 ommatidial directions with a pinhole camera sampler."""

    def __init__(self, csv_path=EYE_MAP_CSV):
        dirs, eye = [], []
        with open(csv_path) as handle:
            next(handle)
            for line in handle:
                dx, dy, dz, e = line.strip().split(",")
                dirs.append((float(dx), float(dy), float(dz)))
                eye.append(e)
        self.dirs = np.asarray(dirs, dtype=float)
        self.dirs /= np.linalg.norm(self.dirs, axis=1, keepdims=True)
        self.eye = np.asarray(eye)
        self.n = len(self.dirs)
        self.left_mask = self.eye == "left"
        self.right_mask = self.eye == "right"
        self.azimuth = np.degrees(np.arctan2(self.dirs[:, 1], self.dirs[:, 0]))
        self.elevation = np.degrees(np.arcsin(np.clip(self.dirs[:, 2], -1, 1)))

    def side_mask(self, side):
        return self.left_mask if side == "l" else self.right_mask

    def visible(self, side, hfov_deg=HFOV_DEG, vfov_deg=VFOV_DEG, splay_deg=SPLAY_DEG):
        """Ommatidia of one eye inside that eye's camera frustum."""
        center = splay_deg if side == "l" else -splay_deg
        in_h = np.abs(self.azimuth - center) <= hfov_deg / 2
        in_v = np.abs(self.elevation) <= vfov_deg / 2
        return self.side_mask(side) & in_h & in_v

    def project_to_pixels(
        self, shape, side, hfov_deg=HFOV_DEG, vfov_deg=VFOV_DEG, splay_deg=SPLAY_DEG
    ):
        """Pinhole-project each direction into the eye's 64x48 image (cols, rows, vis).

        Image right is more forward (decreasing azimuth away from the camera centre).
        Non-visible ommatidia are clipped and must be ignored via the mask.
        """
        height, width = shape[:2]
        center = splay_deg if side == "l" else -splay_deg
        fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        fy = (height / 2.0) / np.tan(np.radians(vfov_deg) / 2.0)
        beta = np.radians(self.azimuth - center)
        cols = width / 2.0 - fx * np.tan(beta)
        rows = height / 2.0 - fy * np.tan(np.radians(self.elevation))
        vis = self.visible(side, hfov_deg, vfov_deg, splay_deg)
        return np.clip(cols, 0, width - 1), np.clip(rows, 0, height - 1), vis

    def sample(
        self,
        gray,
        side,
        hfov_deg=HFOV_DEG,
        vfov_deg=VFOV_DEG,
        splay_deg=SPLAY_DEG,
        accept_deg=ACCEPTANCE_DEG,
    ):
        """Sample a single-channel frame at each ommatidium of one eye.

        Returns intensities in [0, 1]; ommatidia outside the camera FOV are 0.
        """
        gray = np.asarray(gray, dtype=float)
        if gray.max() > 1.0:
            gray /= 255.0
        height, width = gray.shape
        cols, rows, vis = self.project_to_pixels(
            (height, width), side, hfov_deg, vfov_deg, splay_deg
        )
        fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        rad = max(1, int(round(np.tan(np.radians(accept_deg)) * fx)))
        out = np.zeros(self.n)
        ci = np.clip(np.round(cols).astype(int), 0, width - 1)
        ri = np.clip(np.round(rows).astype(int), 0, height - 1)
        for k in np.nonzero(vis)[0]:
            r0, r1 = max(0, ri[k] - rad), min(height, ri[k] + rad + 1)
            c0, c1 = max(0, ci[k] - rad), min(width, ci[k] + rad + 1)
            out[k] = gray[r0:r1, c0:c1].mean()
        return out
