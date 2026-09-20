"""P4 command audit: what body command does the codec declare, offline, for free?

The P3 readout audit (``scripts/readout_audit.py``) measures how far each declared stimulus moves
the descending/VNC readouts. This one carries those measured deltas through the codec and reports
the physical ``[vx, vy, vz, yaw]`` the body would be commanded, in m/s and rad/s, against the
0.05 m/s threshold the rollout uses to score a frame as stuck. It is the free falsifier for codec
v2: if a bright non-loom stimulus still commands < 0.05 m/s here, no seed needs to be spent.

Reconstruction. For each stimulus the audit rebuilds the 2,022-vector the codec would see by
placing the measured per-readout deltas (from ``readout-audit.json``) on the cells of each named
readout, with the adapter's declared rest baselines. Population laterality (the v1 steering
quantity, a left-minus-right mean over all 2,022 traces) is **not** measured by the readout audit,
so the reconstructed vector leaves it at zero; v1 yaw is therefore not meaningful here and only
vx / the v2 steer column should be read. Everything else (power, escape, steer_l/steer_r) is
exactly the quantity the audit measured.

Run: ``env -u PYTHONPATH .venv/bin/python scripts/command_audit.py``
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
READOUT_AUDIT = ROOT / "docs" / "results" / "liveness" / "readout-audit.json"
OUTPUT = ROOT / "docs" / "results" / "liveness" / "command-audit.json"


def _delta(row, name):
    return float(row["readout_delta"].get(name, 0.0))


def feature_vector(brain, adapter, row):
    """The 2,022-vector implied by one readout-audit row, on the adapter's rest baselines."""
    from fly_drone.adapter import feature_positions

    positions = feature_positions(brain)
    vector = np.zeros(len(brain.feature_ids), dtype=float)
    params = adapter.params
    rest = {
        "escape": float(params.get("rest_escape", 0.0)),
        "power_l": float(params.get("rest_power", 0.0)),
        "power_r": float(params.get("rest_power", 0.0)),
        "steer_l": float(params.get("rest_steer", 0.0)),
        "steer_r": float(params.get("rest_steer", 0.0)),
        "wing_l": 0.0,
        "wing_r": 0.0,
        "thrust": 0.0,
    }
    for name, ids in brain.readout_ids.items():
        if name not in rest:
            continue
        value = rest[name] + _delta(row, name)
        for cell in ids:
            if cell in positions:
                vector[positions[cell]] = value
    return vector


def run(readout=READOUT_AUDIT, output=OUTPUT):
    from fly_drone.adapter import DEFAULT_PATH, DEFAULT_V2_PATH, Adapter
    from fly_drone.brain import BrainRuntime
    from fly_drone.plant import LIMITS

    audit = json.loads(Path(readout).read_text())
    brain = BrainRuntime()
    thresholds = float(audit.get("moving_threshold_ms", 0.05))

    codecs = {}
    for name, path in (("v1", DEFAULT_PATH), ("v2", DEFAULT_V2_PATH)):
        path = Path(path)
        if not path.is_file():
            print(f"{name}: {path} absent, skipped")
            continue
        adapter = Adapter.load(path)
        adapter.check(brain)
        stimuli = {}
        for stimulus, row in audit["results"].items():
            command = np.asarray(
                adapter.command(brain, features=feature_vector(brain, adapter, row)),
                dtype=float,
            )
            physical = command * LIMITS
            stimuli[stimulus] = {
                "delta_power": round(
                    0.5 * (_delta(row, "power_l") + _delta(row, "power_r")), 5
                ),
                "command": [round(float(v), 6) for v in command],
                "physical": [round(float(v), 6) for v in physical],
                "vx_ge_threshold": bool(physical[0] >= thresholds),
            }
        codecs[name] = {
            "path": str(path),
            "adapter_version": adapter.version,
            "stimuli": stimuli,
        }

    report = {
        "source": str(readout),
        "moving_threshold_ms": thresholds,
        "plant_limits": [float(v) for v in LIMITS],
        "codecs": codecs,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def _print_table(report):
    names = list(report["codecs"])
    threshold = report["moving_threshold_ms"]
    header = f"{'stimulus':22} {'dpower':>8} " + " ".join(
        f"{name + ' vx':>10} {'>= ' + format(threshold, 'g'):>7}" for name in names
    )
    print(header)
    first = next(iter(report["codecs"].values()))
    for stimulus in first["stimuli"]:
        row = first["stimuli"][stimulus]
        cells = []
        for name in names:
            entry = report["codecs"][name]["stimuli"][stimulus]
            cells.append(
                f"{entry['physical'][0]:10.4f} {str(entry['vx_ge_threshold']):>7}"
            )
        print(f"{stimulus:22} {row['delta_power']:8.3f} " + " ".join(cells))
    print(f"\ncommanded vx in m/s; threshold {threshold} m/s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout", default=str(READOUT_AUDIT))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    report = run(args.readout, args.output)
    _print_table(report)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
