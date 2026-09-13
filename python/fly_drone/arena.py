"""Free-roam arena: seeded layout, beacon placement, threat planning, respawn."""

from collections import deque
from dataclasses import dataclass

import numpy as np

# Pillars never placed at runtime are parked below the floor, out of view and contact.
PARK_Z = -20.0
# Half of the 177 deg binocular field is 1.546 rad; margins keep "in view" unambiguous.
IN_VIEW = 1.4
OUT_OF_VIEW = 1.8
DRONE_RADIUS = 0.15


@dataclass(frozen=True)
class ArenaSpec:
    half_size: float = 8.0
    wall_height: float = 3.0
    # A 1.2 m band read as looming in 55% of turning frames near corners; 0.4 m in 17%.
    band: tuple = (0.8, 1.2)
    wall_luma: float = 0.35
    floor_luma: float = 0.35
    pillar_slots: int = 16
    pillar_radius: float = 0.3
    # None: whole pillar near-black. A height: grey pillar with a dark ring at eye level,
    # so distant pillars sweeping past during turns stay below the loom threshold.
    # Measured under flat lighting (roam-feasibility): a fully dark pillar fires loom in
    # 64-70% of turning frames; a 0.3 m ring gives 0-19% and still warns at ~1.1 m.
    pillar_ring: float | None = 0.3
    pillar_luma: float = 0.35
    pillar_height: float = 3.0
    pillar_spacing: float = 2.5
    spawn_clear: float = 2.0
    beacon_radius: float = 0.3
    beacon_range: tuple = (3.0, 12.0)
    collect_radius: float = 0.5
    threat_radius: float = 0.25
    # 0.7 m/s keeps >= 1.6 s of pillar loom warning; yaw above 0.8 rad/s pushes
    # rotation-induced false loom past 15% of frames.
    limits: tuple = (0.7, 0.5, 0.3, 0.8)
    altitude: tuple = (0.4, 2.5)
    wall_margin: float = 0.6

    @property
    def hold_low(self):
        return np.array([-self.inner, -self.inner, self.altitude[0]])

    @property
    def hold_high(self):
        return np.array([self.inner, self.inner, self.altitude[1]])

    @property
    def inner(self):
        return self.half_size - self.wall_margin


# level -> (pillar count, threats enabled)
LEVELS = {0: (0, False), 1: (6, False), 2: (16, False), 3: (16, True)}


def bearing_to(pos, yaw, point):
    angle = np.arctan2(point[1] - pos[1], point[0] - pos[0]) - yaw
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def clearance(spec, pillars, xy):
    """Horizontal surface distance from a point to the nearest pillar or wall."""
    xy = np.asarray(xy, dtype=float)[:2]
    walls = spec.half_size - float(np.max(np.abs(xy)))
    if len(pillars) == 0:
        return walls
    d = np.linalg.norm(np.asarray(pillars) - xy, axis=1) - spec.pillar_radius
    return float(min(walls, d.min()))


def _grid(spec, step=0.25):
    ticks = np.arange(-spec.inner, spec.inner + 1e-9, step)
    return ticks, np.stack(np.meshgrid(ticks, ticks, indexing="ij"), axis=-1)


def free_mask(spec, pillars, margin=DRONE_RADIUS + 0.15, step=0.25):
    ticks, cells = _grid(spec, step)
    mask = np.ones(cells.shape[:2], dtype=bool)
    for p in pillars:
        mask &= np.linalg.norm(cells - p, axis=-1) > spec.pillar_radius + margin
    return ticks, cells, mask


def connected(mask):
    """True when every free grid cell is reachable from every other (4-neighbour)."""
    free = np.argwhere(mask)
    if len(free) == 0:
        return False
    seen = np.zeros_like(mask)
    queue = deque([tuple(free[0])])
    seen[tuple(free[0])] = True
    while queue:
        i, j = queue.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if (
                0 <= a < mask.shape[0]
                and 0 <= b < mask.shape[1]
                and mask[a, b]
                and not seen[a, b]
            ):
                seen[a, b] = True
                queue.append((a, b))
    return int(seen.sum()) == len(free)


def generate_layout(rng, spec=None, level=3, spawn=(0.0, 0.0), attempts=200):
    """Pillar centres (k, 2): spaced, clear of walls and spawn, free space connected."""
    spec = spec or ArenaSpec()
    count = LEVELS[level][0]
    spawn = np.asarray(spawn, dtype=float)
    limit = spec.half_size - 1.0 - spec.pillar_radius
    centre_gap = spec.pillar_spacing + 2 * spec.pillar_radius
    for _ in range(attempts):
        pillars = []
        for _ in range(count * 60):
            if len(pillars) == count:
                break
            p = rng.uniform(-limit, limit, 2)
            if np.linalg.norm(p - spawn) < spec.spawn_clear + spec.pillar_radius:
                continue
            if any(np.linalg.norm(p - q) < centre_gap for q in pillars):
                continue
            pillars.append(p)
        if len(pillars) < count:
            continue
        pillars = np.array(pillars).reshape(-1, 2)
        if connected(free_mask(spec, pillars)[2]):
            return pillars
    raise RuntimeError("could not generate a connected arena layout")


def next_beacon(rng, spec, pillars, pos, yaw, attempts=400):
    """A free beacon position; half the time deliberately outside the field of view."""
    hidden = bool(rng.random() < 0.5)
    lo, hi = spec.beacon_range
    limit = spec.inner - 0.2
    fallback = None
    for _ in range(attempts):
        xy = rng.uniform(-limit, limit, 2)
        if clearance(spec, pillars, xy) < spec.beacon_radius + 0.6:
            continue
        distance = float(np.linalg.norm(xy - pos[:2]))
        if not lo <= distance <= hi:
            continue
        point = np.array([xy[0], xy[1], rng.uniform(0.8, 1.4)])
        fallback = point
        off = abs(bearing_to(pos, yaw, xy))
        if (hidden and off > OUT_OF_VIEW) or (not hidden and off < IN_VIEW):
            return point, hidden
    if fallback is None:
        raise RuntimeError("no free beacon position")
    return fallback, bool(abs(bearing_to(pos, yaw, fallback)) > IN_VIEW)


def plan_threat(rng, spec, pos, yaw):
    """Launch from 3-4 m ahead (within +-30 deg of heading), aimed at the drone now."""
    angle = yaw + rng.uniform(-0.52, 0.52)
    distance = rng.uniform(3.0, 4.0)
    limit = spec.inner
    origin = np.array(
        [
            np.clip(pos[0] + distance * np.cos(angle), -limit, limit),
            np.clip(pos[1] + distance * np.sin(angle), -limit, limit),
            float(np.clip(pos[2] + rng.uniform(-0.1, 0.1), 0.6, 2.0)),
        ]
    )
    direction = np.asarray(pos, dtype=float) - origin
    norm = float(np.linalg.norm(direction))
    side = float(np.sign(bearing_to(pos, yaw, origin)) or 1.0)
    return {
        "origin": origin,
        "direction": direction / max(norm, 1e-9),
        "speed": float(rng.uniform(0.8, 1.4)),
        "side": side,
        "range": norm,
    }


def safe_respawn(spec, pillars, pos, needed=1.0):
    """Nearest grid point with at least `needed` m clearance to pillars and walls."""
    _, cells, _ = free_mask(spec, pillars, margin=0.0)
    flat = cells.reshape(-1, 2)
    ok = np.array([clearance(spec, pillars, c) >= needed for c in flat])
    if not ok.any():
        raise RuntimeError("no safe respawn point")
    candidates = flat[ok]
    best = candidates[np.argmin(np.linalg.norm(candidates - pos[:2], axis=1))]
    return np.array([best[0], best[1], 1.0])
