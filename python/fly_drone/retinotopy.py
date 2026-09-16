"""Retinotopic input mapping for learned encoder v6 (additive; v5 untouched).

The v6 encoder does not inject one scalar per anatomical population. It injects a
low-resolution spatial current map, patch by patch, so the connectome's own optic
lobe computes motion and looming. This module computes the cell-position -> patch
assignment offline and defines the Rust input roles.

Populations follow the 2026-09-16 feasibility gate
(``docs/superpowers/specs/2026-09-16-retinotopic-sensing-v6-design.md``): Tm4 and
T2 carry a spatial map because they drive the loom/escape circuit and lateralise;
LC4/LPLC2 are injected directly, v4's guaranteed route.

Nothing here builds or flies a brain; it is pure mapping plus role definition, so
it cannot affect the frozen v4/v5 paths.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LEARNED_PREFIX_V6 = "learned-v6:"

# A spatial map is injected into these populations (cell type -> channel prefix).
SPATIAL_CHANNELS = {"tm4": "Tm4", "t2": "T2"}
# These are injected directly, v4's guaranteed loom route; coarser by necessity.
DIRECT_CHANNELS = {"lc4": "LC4", "lplc2": "LPLC2"}
# (nx, ny) patches per eye.
DEFAULT_GRID = (12, 8)
DIRECT_GRID = (4, 3)


@dataclass
class RetinotopicMap:
    """One population's per-side position -> patch assignment."""

    channel: str
    population: str
    side: str
    nx: int
    ny: int
    center: tuple
    basis: tuple  # 2x3 rows (PC1, PC2), canonical sign
    lo: tuple  # 2 in-plane minima
    hi: tuple  # 2 in-plane maxima
    patches: dict = field(default_factory=dict)  # (bx, by) -> tuple of neuron ids

    @property
    def n_cells(self):
        return sum(len(v) for v in self.patches.values())

    def in_plane(self, positions):
        """Project world positions onto the map's 2D basis (un-normalised)."""
        p = np.asarray(positions, dtype=float) - np.asarray(self.center)
        return p @ np.asarray(self.basis, dtype=float).T

    def patch_of(self, position):
        """Return the (bx, by) patch a single world position falls in."""
        s = self.in_plane([position])[0]
        span = np.where(
            np.asarray(self.hi) - np.asarray(self.lo) > 1e-9,
            np.asarray(self.hi) - np.asarray(self.lo),
            1.0,
        )
        norm = (s - np.asarray(self.lo)) / span
        bx = int(min(max(norm[0] * self.nx, 0), self.nx - 1))
        by = int(min(max(norm[1] * self.ny, 0), self.ny - 1))
        return bx, by

    def cells(self, bx, by):
        return list(self.patches.get((bx, by), ()))

    def all_cells(self):
        return [i for ids in self.patches.values() for i in ids]

    def patch_centres(self):
        """Normalised (0..1) centre of every occupied patch, keyed by (bx, by)."""
        return {
            key: ((key[0] + 0.5) / self.nx, (key[1] + 0.5) / self.ny)
            for key in self.patches
        }

    def to_dict(self):
        return {
            "channel": self.channel,
            "population": self.population,
            "side": self.side,
            "nx": self.nx,
            "ny": self.ny,
            "center": list(self.center),
            "basis": [list(row) for row in self.basis],
            "lo": list(self.lo),
            "hi": list(self.hi),
            "patches": {
                f"{bx},{by}": list(ids) for (bx, by), ids in self.patches.items()
            },
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            channel=d["channel"],
            population=d["population"],
            side=d["side"],
            nx=d["nx"],
            ny=d["ny"],
            center=tuple(d["center"]),
            basis=tuple(tuple(r) for r in d["basis"]),
            lo=tuple(d["lo"]),
            hi=tuple(d["hi"]),
            patches={
                tuple(int(x) for x in k.split(",")): tuple(v)
                for k, v in d["patches"].items()
            },
        )


def _pca2d(positions):
    """2D PCA with a canonical sign: the largest-|loading| entry of each axis is +."""
    c = positions - positions.mean(0)
    cov = c.T @ c / max(len(positions) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    vecs = vecs[:, np.argsort(vals)[::-1]][:, :2].T  # 2x3
    for k in range(2):
        j = int(np.argmax(np.abs(vecs[k])))
        if vecs[k, j] < 0:
            vecs[k] = -vecs[k]
    scores = c @ vecs.T
    return scores, vecs


def build_map(cells, population, side, nx, ny, channel=None):
    """Assign each (population, side) cell to one of nx x ny patches by 2D PCA."""
    ids = [
        i
        for i, c in enumerate(cells)
        if c["side"].lower() == side and c["type"] == population
    ]
    if not ids:
        raise ValueError(f"no cells for {population} side {side}")
    positions = np.array([cells[i]["position"] for i in ids], dtype=float)
    scores, basis = _pca2d(positions)
    lo, hi = scores.min(0), scores.max(0)
    span = np.where(hi - lo > 1e-9, hi - lo, 1.0)
    norm = (scores - lo) / span
    patches = {}
    for i, xy in zip(ids, norm, strict=True):
        bx = int(min(max(xy[0] * nx, 0), nx - 1))
        by = int(min(max(xy[1] * ny, 0), ny - 1))
        patches.setdefault((bx, by), []).append(i)
    return RetinotopicMap(
        channel=channel or f"{population.lower()}_{side}",
        population=population,
        side=side,
        nx=nx,
        ny=ny,
        center=tuple(positions.mean(0)),
        basis=tuple(tuple(row) for row in basis),
        lo=tuple(lo),
        hi=tuple(hi),
        patches={k: tuple(v) for k, v in patches.items()},
    )


def build_default_maps(cells):
    """All v6 channels x both sides, keyed by channel name (e.g. "tm4_l")."""
    maps = {}
    for prefix, population in SPATIAL_CHANNELS.items():
        for side in ("l", "r"):
            maps[f"{prefix}_{side}"] = build_map(cells, population, side, *DEFAULT_GRID)
    for prefix, population in DIRECT_CHANNELS.items():
        for side in ("l", "r"):
            maps[f"{prefix}_{side}"] = build_map(cells, population, side, *DIRECT_GRID)
    return maps


def define_roles(core, mapping, namespace="v6"):
    """Define one Rust input role per occupied patch. Returns {(bx, by): role id}."""
    roles = {}
    for (bx, by), ids in mapping.patches.items():
        roles[(bx, by)] = core.input_role(
            f"{namespace}_{mapping.channel}_{bx}_{by}", list(ids)
        )
    return roles


def apply_map(core, roles, values):
    """Inject a patch current map: values[bx, by] -> current in [0, 2]."""
    values = np.asarray(values, dtype=float)
    for (bx, by), role in roles.items():
        core.inject(role, float(np.clip(values[bx, by], 0.0, 2.0)))


def save_maps(maps, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: m.to_dict() for k, m in maps.items()}, indent=2))
    return path


def load_maps(path):
    data = json.loads(Path(path).read_text())
    return {k: RetinotopicMap.from_dict(v) for k, v in data.items()}
