"""C4a A0: does the connectome carry a *sided* escape signal?

The declared bridge's ``escape`` readout is the mean over both ``DNp01`` giant-fibre descending
neurons, so it discards the side. This free probe asks whether the two cells separately encode the
side of a looming object, using the declared stimulus battery (``adapter.STIMULI``): a lateralised
loom must raise the left ``DNp01`` more than the right (and vice versa), above the rest noise floor.
If it does, C4a (command a lateral escape along the neural side) is implementable from neurons; if
not, the descending readout (C4c) would be required first.

Run: ``env -u PYTHONPATH .venv/bin/python scripts/escape_side_probe.py``
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "results" / "adapter" / "escape-side-probe.json"
SEEDS = 8


def run(output=OUTPUT, seeds=SEEDS):
    from fly_drone.adapter import SETTLE_TICKS, STIMULI, feature_positions
    from fly_drone.brain import BrainRuntime

    brain = BrainRuntime()
    positions = feature_positions(brain)
    cells = brain.readout_ids["escape"]
    side_index = {brain.cells[i]["side"]: positions[i] for i in cells}

    def activity(cues, seed):
        brain.reset(seed)
        brain.set_currents(np.asarray(cues, dtype=np.float32))
        brain.step(SETTLE_TICKS)
        features = brain.features()
        return float(features[side_index["L"]]), float(features[side_index["R"]])

    rest = [activity([0, 0, 0, 0], seed) for seed in range(seeds)]
    rest_diffs = np.array([left - right for left, right in rest])
    rows = {}
    for name, cues, _target in STIMULI:
        diffs = np.array([activity(cues, seed) for seed in range(seeds)])
        mean_l, mean_r = float(diffs[:, 0].mean()), float(diffs[:, 1].mean())
        delta = diffs[:, 0] - diffs[:, 1]
        rows[name] = {
            "left": round(mean_l, 6),
            "right": round(mean_r, 6),
            "left_minus_right": round(float(delta.mean()), 6),
            "left_minus_right_std": round(float(delta.std()), 6),
        }
    noise = float(np.max(np.abs(rest_diffs)))
    loom_l = rows["loom_l"]["left_minus_right"]
    loom_r = rows["loom_r"]["left_minus_right"]
    both = rows["loom_both"]["left_minus_right"]
    # A usable side signal is opposite-signed for the two lateralised looms, zero when symmetric,
    # and far above the rest noise floor.
    separated = bool(
        loom_l > 0 > loom_r
        and abs(both) < 1e-6
        and min(abs(loom_l), abs(loom_r)) > 5 * noise
    )
    report = {
        "cells": {
            "L": int(side_index["L"]),
            "R": int(side_index["R"]),
            "type": "DNp01",
        },
        "seeds": seeds,
        "rest_noise_floor": round(noise, 8),
        "stimuli": rows,
        "side_signal_separated": separated,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def _print(report):
    print(
        f"DNp01 L(idx {report['cells']['L']}) / R(idx {report['cells']['R']}); "
        f"rest noise floor {report['rest_noise_floor']}"
    )
    print(f"{'stimulus':12} {'L':>9} {'R':>9} {'L-R':>9} {'std':>8}")
    for name, row in report["stimuli"].items():
        print(
            f"{name:12} {row['left']:9.4f} {row['right']:9.4f} "
            f"{row['left_minus_right']:9.4f} {row['left_minus_right_std']:8.4f}"
        )
    print(f"\nside signal separated: {report['side_signal_separated']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--seeds", type=int, default=SEEDS)
    args = parser.parse_args()
    report = run(args.output, args.seeds)
    _print(report)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
