"""C4a A2: does codec v3 command the lateral escape on the correct side, offline, for free?

Feeds the declared stimulus battery through the v3 codec and checks the *lateral* command before any
seed is spent: a left loom must strafe right (``vy < 0``), a right loom must strafe left (``vy > 0``),
a symmetric (both-side) loom must not strafe (``vy ≈ 0``), and a calm/light frame must not either.
This is the free falsifier for C4a; failing here means no seed should be run.

Run: ``env -u PYTHONPATH .venv/bin/python scripts/escape_command_audit.py``
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "results" / "adapter" / "escape-command-audit.json"
# A lateral loom must command at least this much sideways motion, and a symmetric one at most this.
MIN_LATERAL = 0.7
MAX_SYMMETRIC = 0.1
MAX_CALM = 0.05


def run(output=OUTPUT):
    from fly_drone.adapter import (
        DEFAULT_V3_PATH,
        ESCAPE_SIDE_GAIN,
        SETTLE_TICKS,
        STIMULI,
        Adapter,
    )
    from fly_drone.brain import BrainRuntime

    brain = BrainRuntime()
    adapter = Adapter.load(DEFAULT_V3_PATH)
    adapter.check(brain)
    rows = {}
    for name, cues, _target in STIMULI:
        brain.reset(11)
        brain.set_currents(np.asarray(cues, dtype=np.float32))
        brain.step(SETTLE_TICKS)
        command = np.asarray(adapter.command(brain, features=brain.features()), float)
        rows[name] = {
            "vy": round(float(command[1]), 6),
            "vz": round(float(command[2]), 6),
            "yaw": round(float(command[3]), 6),
        }
    calm = ("rest", "light_l", "light_r", "light_both")
    checks = {
        "loom_l_left": rows["loom_l"]["vy"] < -MIN_LATERAL,
        "loom_r_right": rows["loom_r"]["vy"] > MIN_LATERAL,
        "loom_both_symmetric": abs(rows["loom_both"]["vy"]) <= MAX_SYMMETRIC,
        "calm_no_strafe": all(abs(rows[n]["vy"]) <= MAX_CALM for n in calm),
    }
    report = {
        "artifact": str(DEFAULT_V3_PATH),
        "adapter_version": adapter.version,
        "escape_side_gain": ESCAPE_SIDE_GAIN,
        "min_lateral": MIN_LATERAL,
        "stimuli": rows,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def _print(report):
    print(f"{report['adapter_version']}  escape_side_gain {report['escape_side_gain']}")
    print(f"{'stimulus':12} {'vy':>9} {'vz':>9} {'yaw':>9}")
    for name, row in report["stimuli"].items():
        print(f"{name:12} {row['vy']:9.4f} {row['vz']:9.4f} {row['yaw']:9.4f}")
    for name, ok in report["checks"].items():
        print(f"  {'ok ' if ok else 'FAIL'} {name}")
    print(f"\npassed: {report['passed']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    report = run(args.output)
    _print(report)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
