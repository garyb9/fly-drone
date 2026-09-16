"""M1b spatial propagation assay (analysis only, no training).

Gate for retinotopic sensing: can a spatially structured current injected into
Mi1/Tm3 (the medulla ON sheet that the M1 probe found is a genuine 2D map) reach
the connectome's own looming / escape circuitry with spatial selectivity?

We bin Mi1+Tm3 per eye into a small grid using the 2D PCA of their positions,
drive the grid with spatio-temporal stimuli, and read LC4 / LPLC2 / escape.

Conditions
  uniform        all patches high           (baseline drive)
  v4_loom        the existing uniform loom  (positive control for the readout)
  dark_loom_l/r  expanding low-current disc in one eye   (OFF-like loom)
  bright_loom_l/r expanding high-current disc in one eye (ON-like loom)
  edge_l2r_l     bright/dark edge translating across the left eye (motion)

The decisive number is the lateralisation of the loom readout: a left-eye
stimulus must drive left LC4/LPLC2 (and escape) more than right, and vice versa.
"""

import json
from pathlib import Path

import numpy as np
from fly_drone.brain import BrainRuntime

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "probe"
NX, NY = 12, 8
FRAMES = 30
TICKS = 8
HIGH, LOW = 1.6, 0.2
SEED = 7


def build_bins(br, side, types=("Mi1", "Tm3"), nx=NX, ny=NY):
    ids = [
        i
        for i, c in enumerate(br.cells)
        if c["side"].lower() == side and c["type"] in types
    ]
    pos = np.array([br.cells[i]["position"] for i in ids], float)
    centered = pos - pos.mean(0)
    cov = centered.T @ centered / max(len(pos) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    vecs = vecs[:, np.argsort(vals)[::-1]]
    scores = centered @ vecs[:, :2]
    lo, hi = scores.min(0), scores.max(0)
    span = np.where(hi - lo > 1e-9, hi - lo, 1.0)
    norm = (scores - lo) / span
    roles = {}
    for i, xy in zip(ids, norm, strict=True):
        bx = min(int(xy[0] * nx), nx - 1)
        by = min(int(xy[1] * ny), ny - 1)
        roles.setdefault((bx, by), []).append(i)
    role_ids = {
        key: br.core.input_role(f"sp_{side}_{key[0]}_{key[1]}", v)
        for key, v in roles.items()
    }
    return role_ids, {k: len(v) for k, v in roles.items()}


def apply_map(br, role_ids, value_fn):
    for (bx, by), role in role_ids.items():
        br.core.inject(role, float(np.clip(value_fn(bx, by), 0.0, 2.0)))


def mean_activity(br, ids):
    return float(np.mean(br.core.activity(list(ids)))) if ids else 0.0


def run(br, role_ids_l, role_ids_r, value_fn_l, value_fn_r):
    br.reset(SEED)
    lc4_l, lc4_r = br._cells("l", ("LC4",)), br._cells("r", ("LC4",))
    lp_l, lp_r = br._cells("l", ("LPLC2",)), br._cells("r", ("LPLC2",))
    acc = {k: [] for k in ("lc4_l", "lc4_r", "lp_l", "lp_r", "escape")}
    for _ in range(FRAMES):
        apply_map(br, role_ids_l, value_fn_l)
        apply_map(br, role_ids_r, value_fn_r)
        br.core.step(TICKS)
        acc["lc4_l"].append(mean_activity(br, lc4_l))
        acc["lc4_r"].append(mean_activity(br, lc4_r))
        acc["lp_l"].append(mean_activity(br, lp_l))
        acc["lp_r"].append(mean_activity(br, lp_r))
        acc["escape"].append(float(br.core.readout(br.readouts["escape"])))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def loom(centre, t):
    """Expanding disc: high background, low disc (OFF-like loom)."""

    def fn(bx, by):
        cx, cy = centre
        x = (bx + 0.5) / NX
        y = (by + 0.5) / NY
        r = 0.05 + 0.65 * (t / (FRAMES - 1))
        return LOW if (x - cx) ** 2 + (y - cy) ** 2 < r * r else HIGH

    return fn


def static(value):
    return lambda bx, by: value


def edge(t):
    """Vertical boundary sweeping left->right; left of it high."""

    def fn(bx, by):
        x = (bx + 0.5) / NX
        return HIGH if x < (t / (FRAMES - 1)) else LOW

    return fn


def main():
    br = BrainRuntime()
    rl, n_l = build_bins(br, "l")
    rr, n_r = build_bins(br, "r")
    print(
        f"bins: left={len(rl)} right={len(rr)} cells L={sum(n_l.values())} R={sum(n_r.values())}"
    )

    none = static(1.0)
    results = {"uniform": run(br, rl, rr, none, none)}
    results["dark_loom_l"] = run_dynamic(br, rl, rr, "dark_l")
    results["dark_loom_r"] = run_dynamic(br, rl, rr, "dark_r")
    results["bright_loom_l"] = run_dynamic(br, rl, rr, "bright_l")
    results["bright_loom_r"] = run_dynamic(br, rl, rr, "bright_r")

    # v4 positive control: the existing uniform loom route
    role = br.inputs["looming_l"]
    br.reset(SEED)
    acc = []
    for _ in range(FRAMES):
        br.core.inject(role, 1.5)
        br.core.step(TICKS)
        acc.append(float(br.core.readout(br.readouts["escape"])))
    results["v4_loom_escape"] = float(np.mean(acc))

    for k, v in results.items():
        print(k, json.dumps(v, default=str))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "spatial_assay.json").write_text(json.dumps(results, indent=2))
    print("wrote", OUT / "spatial_assay.json")


def run_dynamic(br, rl, rr, kind):
    """Stateful per-frame stimulus (loom radius / polarity depends on frame index)."""
    br.reset(SEED)
    lc4_l, lc4_r = br._cells("l", ("LC4",)), br._cells("r", ("LC4",))
    lp_l, lp_r = br._cells("l", ("LPLC2",)), br._cells("r", ("LPLC2",))
    acc = {k: [] for k in ("lc4_l", "lc4_r", "lp_l", "lp_r", "escape")}
    side = kind.split("_")[-1]
    bright = kind.startswith("bright")
    bg, disc = (LOW, HIGH) if bright else (HIGH, LOW)
    for t in range(FRAMES):
        r = 0.05 + 0.65 * (t / (FRAMES - 1))

        def fn(bx, by, r=r):
            x = (bx + 0.5) / NX
            y = (by + 0.5) / NY
            return disc if (x - 0.5) ** 2 + (y - 0.5) ** 2 < r * r else bg

        if side == "l":
            apply_map(br, rl, fn)
            apply_map(br, rr, static(bg))
        else:
            apply_map(br, rl, static(bg))
            apply_map(br, rr, fn)
        br.core.step(TICKS)
        acc["lc4_l"].append(mean_activity(br, lc4_l))
        acc["lc4_r"].append(mean_activity(br, lc4_r))
        acc["lp_l"].append(mean_activity(br, lp_l))
        acc["lp_r"].append(mean_activity(br, lp_r))
        acc["escape"].append(float(br.core.readout(br.readouts["escape"])))
    return {k: float(np.mean(v)) for k, v in acc.items()}


if __name__ == "__main__":
    main()
