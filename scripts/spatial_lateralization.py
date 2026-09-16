"""Spatial selectivity of the candidate injection populations.

If left Tm4/T2 drives left LC4/LPLC2 (ipsilateral) more than right, a spatial map
on that population can encode direction. Also tests half-map activation to check
that *sub-regions* of the sheet have separable effects.
"""

import numpy as np
from fly_drone.brain import BrainRuntime

TICKS = 120
POPS = ("Tm4", "T2")


def pca2d(pos):
    c = pos - pos.mean(0)
    cov = c.T @ c / max(len(pos) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    vecs = vecs[:, np.argsort(vals)[::-1]]
    return c @ vecs[:, :2]


def measure(br, lc4, lp, esc):
    return (
        float(np.mean(br.core.activity(lc4["l"]))),
        float(np.mean(br.core.activity(lc4["r"]))),
        float(np.mean(br.core.activity(lp["l"]))),
        float(np.mean(br.core.activity(lp["r"]))),
        float(br.core.readout(esc)),
    )


def run_uniform(br, pop, stim_side, lc4, lp, esc):
    br.reset(7)
    ids = [
        i
        for i, c in enumerate(br.cells)
        if c["side"].lower() == stim_side and c["type"] == pop
    ]
    role = br.core.input_role(f"{pop}_{stim_side}", ids)
    for _ in range(TICKS):
        br.core.inject(role, 1.6)
        br.core.step(1)
    return measure(br, lc4, lp, esc)


def run_half(br, pop, side, half, lc4, lp, esc):
    """Activate only the low/high half of the side's 2D PCA sheet."""
    ids = [
        i
        for i, c in enumerate(br.cells)
        if c["side"].lower() == side and c["type"] == pop
    ]
    pos = np.array([br.cells[i]["position"] for i in ids], float)
    s = pca2d(pos)[:, 0]
    thr = np.median(s)
    sel = [i for i, v in zip(ids, s, strict=True) if (v < thr) == (half == "low")]
    br.reset(7)
    role = br.core.input_role(f"{pop}_{side}_{half}", sel)
    for _ in range(TICKS):
        br.core.inject(role, 1.6)
        br.core.step(1)
    return measure(br, lc4, lp, esc), len(sel)


def main():
    br = BrainRuntime()
    lc4 = {s: br._cells(s, ("LC4",)) for s in ("l", "r")}
    lp = {s: br._cells(s, ("LPLC2",)) for s in ("l", "r")}
    esc = br.readouts["escape"]

    print("uniform whole-side injection: ipsi vs contra")
    for pop in POPS:
        for stim in ("l", "r"):
            l4l, l4r, lpl, lpr, e = run_uniform(br, pop, stim, lc4, lp, esc)
            ipsi = l4l + lpl if stim == "l" else l4r + lpr
            contra = l4r + lpr if stim == "l" else l4l + lpl
            lat = (ipsi - contra) / (ipsi + contra + 1e-9)
            print(
                f"  {pop} {stim}: lc4_l={l4l:.3f} lc4_r={l4r:.3f} "
                f"lplc2_l={lpl:.3f} lplc2_r={lpr:.3f} esc={e:.3f} lat={lat:+.2f}"
            )

    print("\nhalf-sheet injection (left side), low vs high PC1")
    for pop in POPS:
        for half in ("low", "high"):
            (l4l, l4r, lpl, lpr, e), n = run_half(br, pop, "l", half, lc4, lp, esc)
            print(
                f"  {pop} l/{half:4s} n={n:4d}: lc4_l={l4l:.3f} lplc2_l={lpl:.3f} "
                f"lc4_r={l4r:.3f} esc={e:.3f}"
            )


if __name__ == "__main__":
    main()
