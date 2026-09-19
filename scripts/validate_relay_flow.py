"""Validate the P3 declared optic-flow feature against simulator camera egomotion.

Offline, labels-only: the plant's own yaw rate and vertical velocity are used **only** to check
that the declared flow estimator is monotonic with real motion. Nothing from geometry is ever
injected into the brain. This stands in for FlyView ground truth until that dataset is added.

Run: ``env -u PYTHONPATH .venv/bin/python scripts/validate_relay_flow.py``
"""

import argparse
import json
from pathlib import Path

import numpy as np

FRAME_DT = 0.04
SUBSTEPS = 8


def _pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def run(output="docs/results/liveness/relay-flow-validation.json", frames=120, seed=0):
    from fly_drone.plant import DronePlant
    from fly_drone.relay import RELAY_VERSION, RelayState

    plant = DronePlant(vision=True)
    plant.reset(seed=seed)
    relay = RelayState()

    est_yaw = {"l": [], "r": []}
    est_pitch = {"l": [], "r": []}
    act_yaw, act_vz = [], []
    for frame in range(frames):
        yaw0 = float(plant.rpy[0, 2])
        vz0 = float(plant.vel[0][2])
        if frame < frames // 2:
            command = [
                0.0,
                0.0,
                0.0,
                plant.limits[3] * 0.8 * np.sin(2 * np.pi * frame / 20),
            ]
        else:
            command = [
                0.0,
                0.0,
                plant.limits[2] * 0.8 * np.sin(2 * np.pi * frame / 20),
                0.0,
            ]
        for _ in range(SUBSTEPS):
            plant.advance(command)
        images = plant.camera()
        relay.observe_frame(images)
        for side in ("l", "r"):
            est_yaw[side].append(relay.yaw[side])
            est_pitch[side].append(relay.pitch[side])
        act_yaw.append((float(plant.rpy[0, 2]) - yaw0) / FRAME_DT)
        act_vz.append((float(plant.vel[0][2]) - vz0) / FRAME_DT)

    half = frames // 2
    report = {
        "relay_version": RELAY_VERSION,
        "frames": frames,
        "seed": seed,
        "yaw_segment": {
            "pearson_yaw_flow_vs_yaw_rate": {
                side: _pearson(est_yaw[side][:half], act_yaw[:half])
                for side in ("l", "r")
            }
        },
        "pitch_segment": {
            "pearson_pitch_flow_vs_vertical_accel": {
                side: _pearson(est_pitch[side][half:], act_vz[half:])
                for side in ("l", "r")
            }
        },
        "note": "Declared translation model; sign convention is declared, correlation is magnitude.",
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="docs/results/liveness/relay-flow-validation.json"
    )
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = run(args.output, args.frames, args.seed)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
