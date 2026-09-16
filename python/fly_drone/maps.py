"""Inspect the v6 retinotopic maps and, optionally, an encoder's current maps.

Pure helpers behind ``scripts/spatial_maps.py`` and ``fly-drone spatial-maps``.
Read-only: no brain wiring, no training.
"""

import json
from pathlib import Path

import numpy as np

from .brain import STACK_FRAMES, BrainRuntime
from .retinotopy import build_default_maps
from .spatial_encoder import CHANNEL_ORDER, flat_dim


def map_summary(maps):
    rows = []
    for name in CHANNEL_ORDER:
        m = maps[name]
        rows.append(
            {
                "channel": name,
                "population": m.population,
                "side": m.side,
                "grid": [m.nx, m.ny],
                "occupied_patches": len(m.patches),
                "cells": m.n_cells,
                "occupancy": round(len(m.patches) / (m.nx * m.ny), 3),
            }
        )
    return rows


def render_grid(values, path, scale=24):
    """Render a (nx, ny) array as a grayscale heatmap, one pixel block per patch."""
    from PIL import Image

    values = np.asarray(values, dtype=float)
    nx, ny = values.shape
    lo, hi = float(values.min()), float(values.max())
    norm = (values - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(values)
    img = Image.new("L", (nx * scale, ny * scale), 0)
    px = img.load()
    for bx in range(nx):
        for by in range(ny):
            v = int(255 * norm[bx, by])
            for x in range(bx * scale, (bx + 1) * scale):
                for y in range(by * scale, (by + 1) * scale):
                    px[x, y] = v
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def run(output="runs/probe/maps", encoder=None):
    """Write the map summary and, with an encoder, its current maps. Returns the summary rows."""
    maps = build_default_maps(BrainRuntime().cells)
    rows = map_summary(maps)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    if encoder:
        from .spatial_encoder import SpatialEncoder

        enc = SpatialEncoder.load(encoder)
        rng = np.random.default_rng(0)
        stack = rng.integers(0, 256, (2 * STACK_FRAMES, 48, 64), dtype=np.uint8)
        currents = enc.currents(stack)
        for name in CHANNEL_ORDER:
            render_grid(currents[name], out / f"current_{name}.png")
    return {"rows": rows, "flat_dim": flat_dim(), "output": str(out)}
