"""Fixed-stimulus calibration battery on the compact readouts (workstream D).

Adapted from FlyDrones ``src/flydrones/calibrate.py``: show the brain a fixed set of
visual situations, record the motor readouts, and fit a baseline-centred ridge from
those readouts to the command a fly would need. Unlike FlyDrones' optic-flow
features, this project's v4 cues are only light and loom per side, so the battery
uses those; up/down flow and vertical thrust are not representable here.

Diagnostic only: no decoder, no acceptance threshold. See
``docs/superpowers/specs/2026-09-18-prior-art-harvest-design.md`` workstream D.

Usage:
    python scripts/stimulus_battery.py --out runs/diagnostics/battery-v4.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from fly_drone.brain import BrainRuntime

ROOT = Path(__file__).resolve().parents[1]

# (name, [light_l, light_r, loom_l, loom_r], {"turn": target, "escape": target}).
# Turn is signed toward the light and away from the loom on the same side.
STIMULI = [
    ("rest", [0, 0, 0, 0], {"turn": 0.0, "escape": 0.0}),
    ("light_l", [2, 0, 0, 0], {"turn": 0.8, "escape": 0.0}),
    ("light_r", [0, 2, 0, 0], {"turn": -0.8, "escape": 0.0}),
    ("loom_l", [0, 0, 2, 0], {"turn": -0.8, "escape": 0.8}),
    ("loom_r", [0, 0, 0, 2], {"turn": 0.8, "escape": 0.8}),
    ("light_both", [2, 2, 0, 0], {"turn": 0.0, "escape": 0.0}),
    ("loom_both", [0, 0, 2, 2], {"turn": 0.0, "escape": 0.8}),
]


def collect(brain, settle=80, repeats=3):
    names, rows, targets = [], [], []
    for _ in range(repeats):
        for name, cues, target in STIMULI:
            brain.reset(11)
            brain.set_currents(np.asarray(cues, dtype=np.float32))
            brain.step(settle)
            rows.append(brain.read())
            targets.append(target)
            names.append(name)
    keys = sorted(rows[0])
    x = np.array([[r[k] for k in keys] for r in rows], dtype=float)
    return names, keys, x, targets


def fit_readout(names, keys, x, targets, ridge=1.0):
    rest = x[[i for i, n in enumerate(names) if n == "rest"]].mean(0)
    xc = x - rest
    scale = np.abs(xc).max(0) + 1e-6
    xn = xc / scale
    out = {
        "readouts": keys,
        "baseline": dict(zip(keys, rest.round(4).tolist(), strict=True)),
    }
    for axis in ("turn", "escape"):
        y = np.array([t[axis] for t in targets])
        w = np.linalg.solve(xn.T @ xn + ridge * np.eye(xn.shape[1]), xn.T @ y) / scale
        pred = xc @ w
        r2 = 1 - ((y - pred) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-9)
        terms = {
            k: round(float(v), 6) for k, v in zip(keys, w, strict=True) if abs(v) > 1e-6
        }
        out[axis] = {"r2": round(float(r2), 3), "terms": terms}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "runs/diagnostics/battery-v4.json"))
    ap.add_argument("--settle", type=int, default=80)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    brain = BrainRuntime()
    names, keys, x, targets = collect(brain, args.settle, args.repeats)
    report = fit_readout(names, keys, x, targets)
    report["stimuli"] = [n for n, _, _ in STIMULI]
    report["dataset_hash"] = brain.dataset_hash
    report["note"] = "Fixed-stimulus readout diagnostic; not an acceptance run."
    for axis in ("turn", "escape"):
        print(f"{axis:6s} R^2 = {report[axis]['r2']}")
    print("readouts:", ", ".join(keys))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
