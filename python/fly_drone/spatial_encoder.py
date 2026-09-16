"""Learned retinotopic eye encoder (v6): two eyes -> per-population current maps.

Standalone and additive; nothing here is wired into ``brain.py`` yet, so the v5
free-roam path is untouched. It is the network the v6 sensing spec describes
(``docs/superpowers/specs/2026-09-16-retinotopic-sensing-v6-design.md``):

- input is each eye's 3-frame luma stack plus explicit **frame differences**
  (motion is first-class, not left for the CNN to rediscover);
- one shared conv trunk runs per eye, the right eye mirrored with a side flag;
- a spatial head emits a ``nx x ny`` current map for Tm4 and T2, a coarse head
  emits LC4/LPLC2 maps, all squashed to ``[0, 2]``;
- the version is ``learned-v6:<hash>`` so it cannot be confused with v5.

The mapping from patch to neuron lives in ``retinotopy.py``; this module only
produces the per-patch currents.
"""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as fn

from .brain import STACK_FRAMES
from .retinotopy import (
    DEFAULT_GRID,
    DIRECT_CHANNELS,
    DIRECT_GRID,
    LEARNED_PREFIX_V6,
    SPATIAL_CHANNELS,
)

# Canonical channel order and per-channel grid, used by flatten/unflatten.
CHANNEL_ORDER = tuple(
    f"{prefix}_{side}"
    for prefix in (*SPATIAL_CHANNELS, *DIRECT_CHANNELS)
    for side in ("l", "r")
)


def channel_grid(name):
    prefix = name.rsplit("_", 1)[0]
    return DIRECT_GRID if prefix in DIRECT_CHANNELS else DEFAULT_GRID


def flat_dim():
    return sum(int(np.prod(channel_grid(name))) for name in CHANNEL_ORDER)


def eye_inputs(eyes, frames=STACK_FRAMES):
    """Build the two eyes' conv inputs: frames + frame differences + side flag.

    ``eyes`` is ``(N, 2*frames, 48, 64)``; returns two ``(N, 2*frames, 48, 64)``
    tensors. The right eye is mirrored on width and flagged -1 (as in v5).
    """
    left = eyes[:, :frames]
    right = eyes[:, frames:]
    left_diff = left[:, 1:] - left[:, :-1]
    right_diff = right[:, 1:] - right[:, :-1]
    n, _, h, w = left.shape
    side = eyes.new_ones(n, 1, h, w)
    left_in = torch.cat([left, left_diff, side], dim=1)
    right_in = torch.cat(
        [torch.flip(right, dims=[3]), torch.flip(right_diff, dims=[3]), -side], dim=1
    )
    return left_in, right_in


class EyeNetV6(nn.Module):
    """One eye's conv trunk: 2*frames x 48 x 64 -> 32 x 6 x 8."""

    def __init__(self, frames=STACK_FRAMES):
        super().__init__()
        in_ch = 2 * frames  # frames + (frames - 1) differences + 1 side flag
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class SpatialEncoderNet(nn.Module):
    """Eye stack -> one raw-logit current map per (channel, eye)."""

    def __init__(self, frames=STACK_FRAMES):
        super().__init__()
        self.frames = frames
        self.eye = EyeNetV6(frames)
        self.spatial = nn.Conv2d(32, len(SPATIAL_CHANNELS), 1)
        self.direct = nn.Conv2d(32, len(DIRECT_CHANNELS), 1)

    def forward(self, eyes):
        left_in, right_in = eye_inputs(eyes, self.frames)
        out = {}
        for side, feature in (("l", self.eye(left_in)), ("r", self.eye(right_in))):
            spatial = fn.interpolate(
                self.spatial(feature), size=DEFAULT_GRID, mode="bilinear"
            )
            direct = fn.adaptive_avg_pool2d(self.direct(feature), DIRECT_GRID)
            for p, prefix in enumerate(SPATIAL_CHANNELS):
                out[f"{prefix}_{side}"] = spatial[:, p]
            for p, prefix in enumerate(DIRECT_CHANNELS):
                out[f"{prefix}_{side}"] = direct[:, p]
        return out


def _hash_state(state):
    digest = hashlib.sha256()
    for part in sorted(state):
        for key in sorted(state[part]):
            digest.update(f"{part}.{key}".encode())
            digest.update(
                state[part][key].detach().cpu().numpy().astype(np.float32).tobytes()
            )
    return digest.hexdigest()[:16]


def flatten(outputs):
    """Stack a channel dict into a canonical ``(N, flat_dim())`` tensor."""
    return torch.cat(
        [outputs[name].reshape(outputs[name].shape[0], -1) for name in CHANNEL_ORDER],
        dim=1,
    )


def unflatten(vector):
    """Inverse of ``flatten``."""
    out, start = {}, 0
    for name in CHANNEL_ORDER:
        nx, ny = channel_grid(name)
        out[name] = vector[:, start : start + nx * ny].reshape(-1, nx, ny)
        start += nx * ny
    return out


class SpatialEncoder:
    """Deployable encoder: eye stack -> current maps in [0, 2]."""

    def __init__(self, net):
        self.net = net.cpu().eval()

    @property
    def version(self):
        return LEARNED_PREFIX_V6 + _hash_state({"net": self.net.state_dict()})

    @classmethod
    def fresh(cls, seed=0):
        torch.manual_seed(seed)
        return cls(SpatialEncoderNet())

    @classmethod
    def load(cls, path):
        torch.set_num_threads(1)
        state = torch.load(path, map_location="cpu", weights_only=True)
        enc = cls.fresh()
        enc.net.load_state_dict(state["net"])
        return enc

    def state(self):
        return {"net": self.net.state_dict()}

    def currents(self, stack):
        """stack: (2*STACK_FRAMES, 48, 64) uint8 -> {channel: (nx, ny) float32}."""
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(stack)[None], dtype=torch.float32) / 255.0
            raw = self.net(x)
            return {
                name: (torch.tanh(v) + 1.0)[0].numpy().astype(np.float32)
                for name, v in raw.items()
            }

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state(), path)
        return self.version
