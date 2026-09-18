"""Measure the visual-axis alignment of the v6 retinotopic maps (workstream A).

Diagnostic, not a hard assert. The Buchner eye map is in the fly head frame while
cell positions are in the MaleCNS browser scene frame, and the source project marks
the cell<->ommatidium join as modelled. We therefore report the principal angles
between a population sheet's positional PCA basis and the visible ommatidial
direction PCA, plus the angle between the sheet normal and the mean viewing
direction. A small angle is supporting evidence for the v6 map's visual axis, not
proof of an anatomical join.

See ``docs/superpowers/specs/2026-09-18-prior-art-harvest-design.md`` workstream A.

Usage:
    python scripts/eye_alignment.py
"""

import json
from pathlib import Path

import numpy as np
from fly_drone.eye_geometry import EyeMap
from fly_drone.retinotopy import _pca2d, build_map

ROOT = Path(__file__).resolve().parents[1]
POPS = ("Tm4", "T2", "Mi1", "Tm3")


def direction_basis(eye, side):
    vis = eye.visible(side)
    _, basis = _pca2d(eye.dirs[vis])
    return basis, eye.dirs[vis].mean(0)


def principal_angles(a, b):
    """Angles (deg) between two orthonormal 2x3 bases: arccos of the SVD values."""
    values = np.linalg.svd(np.asarray(a) @ np.asarray(b).T, compute_uv=False)
    return np.degrees(np.arccos(np.clip(values, -1.0, 1.0)))


def main():
    eye = EyeMap()
    cells = json.loads((ROOT / "data/malecns/cells.json").read_text())
    report = {}
    for population in POPS:
        for side in ("l", "r"):
            mapping = build_map(cells, population, side, 12, 8)
            positions = np.array([cells[i]["position"] for i in mapping.all_cells()])
            _, sheet = _pca2d(positions)
            directions, mean_dir = direction_basis(eye, side)
            normal = np.cross(sheet[0], sheet[1])
            angles = principal_angles(sheet, directions)
            normal_angle = float(
                np.degrees(np.arccos(np.clip(abs(normal @ mean_dir), -1.0, 1.0)))
            )
            report[f"{population}_{side}"] = {
                "n_cells": mapping.n_cells,
                "principal_angles_deg": [float(a) for a in angles],
                "normal_to_mean_direction_deg": normal_angle,
            }
            print(
                f"{population}_{side:1s} n={mapping.n_cells:4d} "
                f"principal angles {angles[0]:5.1f}/{angles[1]:5.1f} deg, "
                f"normal {normal_angle:5.1f} deg"
            )
    out = ROOT / "runs/diagnostics/eye-alignment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "note": (
                    "Head-frame eye map vs scene-frame cell positions; modelled "
                    "cell<->ommatidium join. Diagnostic, not an assert."
                ),
                "populations": report,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
