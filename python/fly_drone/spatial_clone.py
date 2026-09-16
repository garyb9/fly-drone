"""v4-clone target maps for the v6 spatial encoder (standalone; additive).

The v6 clone warm start is the v4 encoder's cues broadcast onto the spatial
channels (sensing spec §4.1): the v4 light cue seeds the Mi1/Tm3 maps and the v4
loom cue seeds the direct LC4/LPLC2 maps. Tm4/T2 have no v4 analogue, so they
start at a neutral baseline and are left to SAC.

This is only the target construction; the fit needs frames collected under
spatial injection and runs in the M2 window, so nothing here touches the v5 path.
"""

import numpy as np

# v4 cue order (sensory-model.md §2).
CUE_LIGHT_L, CUE_LIGHT_R, CUE_LOOM_L, CUE_LOOM_R = 0, 1, 2, 3

# Which v6 channels each v4 cue seeds (v4 pools Mi1+Tm3 into one light cue).
CUE_TO_CHANNELS = {
    CUE_LIGHT_L: ("mi1_l", "tm3_l"),
    CUE_LIGHT_R: ("mi1_r", "tm3_r"),
    CUE_LOOM_L: ("lc4_l", "lplc2_l"),
    CUE_LOOM_R: ("lc4_r", "lplc2_r"),
}

# Channels with no v4 analogue; left at the neutral baseline.
UNSEEDED = ("tm4_l", "tm4_r", "t2_l", "t2_r")


def v4_cues_to_targets(cues, maps, neutral=1.0):
    """Build per-patch target maps from v4 cues.

    ``cues`` is ``(4,)`` for one frame or ``(N, 4)`` for a batch; returns a dict
    ``channel -> (nx, ny)`` or ``(N, nx, ny)`` array in ``[0, 2]``.
    """
    cues = np.asarray(cues, dtype=np.float32)
    batch = cues.ndim == 2
    if not batch:
        cues = cues[None]
    n = len(cues)
    out = {}
    for index, channels in CUE_TO_CHANNELS.items():
        for channel in channels:
            nx, ny = maps[channel].nx, maps[channel].ny
            value = cues[:, index].reshape(n, 1, 1)
            block = np.broadcast_to(value, (n, nx, ny)).astype(np.float32).copy()
            out[channel] = block if batch else block[0]
    for channel in UNSEEDED:
        nx, ny = maps[channel].nx, maps[channel].ny
        block = np.full((n, nx, ny), float(neutral), np.float32)
        out[channel] = block if batch else block[0]
    return out
