"""Inspect the v6 retinotopic maps and, optionally, an encoder's current maps.

    env -u PYTHONPATH .venv/bin/python scripts/spatial_maps.py
    env -u PYTHONPATH .venv/bin/python scripts/spatial_maps.py --encoder runs/v6/encoder.pt

Read-only: it builds the maps from cells.json and (with --encoder) runs the net
on a random eye stack. No brain, no training, no env wiring.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from fly_drone.brain import STACK_FRAMES, BrainRuntime
from fly_drone.retinotopy import build_default_maps
from fly_drone.spatial_encoder import CHANNEL_ORDER, flat_dim
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


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
    values = np.asarray(values, dtype=float)
    nx, ny = values.shape
    lo, hi = float(values.min()), float(values.max())
    span = hi - lo if hi - lo > 1e-9 else 1.0
    norm = (values - lo) / span
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(ROOT / "runs" / "probe" / "maps"))
    ap.add_argument("--encoder", default=None, help="learned-v6 .pt to visualise")
    args = ap.parse_args()

    maps = build_default_maps(BrainRuntime().cells)
    rows = map_summary(maps)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    print(
        f"{'channel':10s} {'pop':6s} {'side':4s} {'grid':7s} {'occ':>5s} {'cells':>6s}"
    )
    for r in rows:
        print(
            f"{r['channel']:10s} {r['population']:6s} {r['side']:4s} "
            f"{r['grid'][0]:>2d}x{r['grid'][1]:<2d}  {r['occupancy']:5.2f} {r['cells']:6d}"
        )
    print(f"\nflat_dim = {flat_dim()} currents/frame")

    if args.encoder:
        from fly_drone.spatial_encoder import SpatialEncoder

        enc = SpatialEncoder.load(args.encoder)
        rng = np.random.default_rng(0)
        stack = rng.integers(0, 256, (2 * STACK_FRAMES, 48, 64), dtype=np.uint8)
        currents = enc.currents(stack)
        for name in CHANNEL_ORDER:
            render_grid(currents[name], out / f"current_{name}.png")
        print(f"wrote current maps for {enc.version} to {out}")


if __name__ == "__main__":
    main()
