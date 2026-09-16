"""M1 retinotopy feasibility probe (analysis only, no training).

Question: do the candidate sensory input populations form a usable 2D spatial
(retinotopic) map, per side, that we can inject a structured image into?

We cannot know the true receptive-field direction of each neuron from the
extracted centroids, but for an 80/20 agent we do not need it: we need the
injected neurons to tile space so that neighbouring neurons sample neighbouring
image patches. So this probe measures, per population and side:

  - how planar the position cloud is (PCA explained variance),
  - the 2D projected extent and aspect ratio,
  - how uniformly a coarse grid over that plane is occupied (tiling quality),
  - a PNG scatter coloured by depth, to eyeball columnar structure.

Outputs a JSON report + PNGs under runs/probe/retinotopy/.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "malecns"
OUT = ROOT / "runs" / "probe" / "retinotopy"

POPS = [
    "L1",
    "L2",
    "L3",
    "Mi1",
    "Tm3",
    "T4a",
    "T4b",
    "T5a",
    "T5b",
    "LC4",
    "LPLC2",
    "T2",
    "Tm4",
    "Tm2",
    "Tm1",
    "T3",
    "Lawf1",
]
GRID = 24  # occupancy grid resolution per axis


def pca(coords):
    """Return (projected 3xN scores, components 3x3 rows, explained variance ratio)."""
    c = coords - coords.mean(0)
    cov = c.T @ c / max(len(coords) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)  # ascending eigenvalues
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    scores = c @ vecs
    evr = vals / vals.sum()
    return scores, vecs.T, evr


def occupancy(scores2d, grid=GRID):
    """Fraction of grid cells (over the data's bounding box) holding >= 1 point."""
    lo = scores2d.min(0)
    hi = scores2d.max(0)
    span = np.where(hi - lo > 1e-9, hi - lo, 1.0)
    idx = ((scores2d - lo) / span * grid).astype(int)
    idx = np.clip(idx, 0, grid - 1)
    filled = np.zeros((grid, grid), bool)
    filled[idx[:, 0], idx[:, 1]] = True
    return float(filled.mean()), filled


def scatter_png(scores2d, depth, path, size=560):
    lo = scores2d.min(0)
    hi = scores2d.max(0)
    span = np.where(hi - lo > 1e-9, hi - lo, 1.0)
    xy = (scores2d - lo) / span
    pad = 12
    w = h = size - 2 * pad
    img = Image.new("RGB", (size, size), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    d = depth - depth.min()
    d = d / (d.max() + 1e-9)  # 0..1 depth, red -> blue
    order = np.argsort(d)  # draw shallow first
    for i in order:
        x = pad + xy[i, 0] * w
        y = pad + (1 - xy[i, 1]) * h  # flip y so up is up
        t = d[i]
        col = (int(255 * t), 40, int(255 * (1 - t)))
        dr.ellipse([x - 2, y - 2, x + 2, y + 2], fill=col)
    img.save(path)


def main():
    cells = json.loads((DATA / "cells.json").read_text())
    pos = np.array([c["position"] for c in cells], dtype=np.float64)
    types = np.array([c["type"] for c in cells])
    sides = np.array([str(c["side"]).lower() for c in cells])

    OUT.mkdir(parents=True, exist_ok=True)
    report = {}

    for pop in POPS:
        for side in ("l", "r"):
            mask = (types == pop) & (sides == side)
            if mask.sum() == 0:
                continue
            p = pos[mask]
            scores, vecs, evr = pca(p)
            s2 = scores[:, :2]
            occ, _ = occupancy(s2)
            depth = scores[:, 2]  # third PC = out-of-plane
            key = f"{pop}_{side}"
            report[key] = {
                "n": int(mask.sum()),
                "pc_explained_variance": [round(float(v), 4) for v in evr],
                "planarity": round(float(evr[0] + evr[1]), 4),
                "out_of_plane_std": round(float(depth.std()), 4),
                "in_plane_range": [round(float(x), 4) for x in s2.min(0)]
                + [round(float(x), 4) for x in s2.max(0)],
                "aspect_ratio": round(
                    float((np.ptp(s2[:, 0]) + 1e-9) / (np.ptp(s2[:, 1]) + 1e-9)), 3
                ),
                "grid_occupancy": round(occ, 3),
            }
            scatter_png(s2, depth, OUT / f"{key}.png")

    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    for key, r in report.items():
        print(
            f"{key:10s} n={r['n']:5d} planar={r['planarity']:.3f} "
            f"oospread={r['out_of_plane_std']:.3f} aspect={r['aspect_ratio']:.2f} "
            f"occupancy={r['grid_occupancy']:.3f} evr={r['pc_explained_variance']}"
        )
    print("\nWrote", OUT)


if __name__ == "__main__":
    main()
