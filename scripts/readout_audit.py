"""P3 readout audit: does the declared optic-flow relay actually drive the descending readout?

Cheap, offline diagnostic (no arena, no long run). It measures how far each declared stimulus moves
the 2,022 descending/VNC-motor traces and the named readouts, relative to a matched rest run of the
same duration. Canonical v4 cues and the relay's direction-selective currents are compared on the
same footing. If the relay is near the readout's motion floor while the known loom pathway is not,
the bottleneck is downstream of sensation.

Run: ``env -u PYTHONPATH .venv/bin/python scripts/readout_audit.py``
"""

import argparse
import json
from pathlib import Path

import numpy as np

SETTLE_TICKS = 40
STIM_TICKS = 80
SEED = 11


def _rollout(brain, cues=None, relay=None):
    """Settle, then apply one condition for STIM_TICKS; return the post-state."""
    brain.reset(SEED)
    if cues is not None:
        brain.set_currents(np.asarray(cues, dtype=np.float32))
    brain.step(SETTLE_TICKS)
    for _ in range(STIM_TICKS):
        if relay:
            brain.inject_relay(relay)
        brain.step(1)
    return brain.features().copy(), dict(brain.read())


def _compare(base_feat, base_read, feat, read):
    drive = np.abs(feat - base_feat)
    return {
        "readout_delta": {k: round(read[k] - base_read[k], 5) for k in read},
        "descending_drive_max": round(float(drive.max()), 5),
        "descending_drive_mean": round(float(drive.mean()), 6),
        "descending_changed_0p02": int((drive > 0.02).sum()),
    }


def run(output="docs/results/liveness/readout-audit.json"):
    from fly_drone.brain import BrainRuntime

    v4 = BrainRuntime()
    relay_brain = BrainRuntime(relay=True)
    rest_feat, rest_read = _rollout(v4)

    canonical = {
        "light_l": [2, 0, 0, 0],
        "light_r": [0, 2, 0, 0],
        "loom_l": [0, 0, 2, 0],
        "loom_r": [0, 0, 0, 2],
        "light_both": [2, 2, 0, 0],
    }
    results = {
        f"v4:{name}": _compare(rest_feat, rest_read, *_rollout(v4, cues=cues))
        for name, cues in canonical.items()
    }

    results["relay:all_0.6"] = _compare(
        rest_feat,
        rest_read,
        *_rollout(relay_brain, relay={r: 0.6 for r in relay_brain.relay_ids}),
    )
    # Realistic magnitudes: yaw flow during a turn is ~0.05 rad/frame -> current ~0.5
    # (relay.FLOW_GAIN=10); vertical flow is ~0.001 rad -> ~0.01.
    results["relay:cruise_yaw"] = _compare(
        rest_feat,
        rest_read,
        *_rollout(relay_brain, relay={"yaw_pos_l": 0.5, "yaw_pos_r": 0.5}),
    )
    results["relay:cruise_pitch"] = _compare(
        rest_feat,
        rest_read,
        *_rollout(relay_brain, relay={"pitch_up_l": 0.01, "pitch_up_r": 0.01}),
    )
    for role in sorted(relay_brain.relay_ids):
        results[f"relay:{role}"] = _compare(
            rest_feat, rest_read, *_rollout(relay_brain, relay={role: 1.0})
        )

    report = {
        "settle_ticks": SETTLE_TICKS,
        "stim_ticks": STIM_TICKS,
        "seed": SEED,
        "n_features": len(v4.feature_ids),
        "results": results,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def _print_table(report):
    print(
        f"{'condition':22} {'desc_max':>9} {'desc_mean':>10} {'#chg':>5}  top readout deltas"
    )
    for name, row in report["results"].items():
        deltas = sorted(row["readout_delta"].items(), key=lambda kv: -abs(kv[1]))[:3]
        top = ", ".join(f"{k}={v:+.3f}" for k, v in deltas if abs(v) > 3e-4)
        print(
            f"{name:22} {row['descending_drive_max']:9.4f} "
            f"{row['descending_drive_mean']:10.5f} {row['descending_changed_0p02']:5d}  {top}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/results/liveness/readout-audit.json")
    args = parser.parse_args()
    report = run(args.output)
    _print_table(report)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
