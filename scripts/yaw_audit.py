"""P4 follow-up audit: can this body ever register a saccade under the L7 detector?

L7 scores the free-flight saccade rate (~0.5 Hz, Schnell et al. 2017) with a declared detector
that fires when the heading rate exceeds ``motion_stats.SACCADE_THRESHOLD`` (3 rad/s), chosen to sit
below the fly's own peak (~35 rad/s, Fry et al. 2005) and above smooth optomotor steering. The
simulated quadrotor, however, has a yaw-rate command limit of ``plant.LIMITS[3]`` (0.8 rad/s), so a
controlled turn cannot reach the detector threshold. This audit measures the heading rate actually
achieved in free roam and separates real turns from the position/heading discontinuities a crash
respawn introduces, which the detector counts as saccades.

It is the free falsifier for L7 on this body: if the clean-frame heading rate stays below the
threshold on every seed, no controller can pass L7 and no seed should be spent on a C4 saccade
generator until the body or the criterion changes.

Run: ``env -u PYTHONPATH .venv/bin/python scripts/yaw_audit.py``
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "results" / "liveness" / "yaw-audit.json"
DT = 0.04
TELEPORT_JUMP_M = 0.1


def _resolve_adapter(name):
    from fly_drone.adapter import DEFAULT_PATH, DEFAULT_V2_PATH

    return {"v1": DEFAULT_PATH, "v2": DEFAULT_V2_PATH}.get(name, Path(name))


def _seed_run(brain, env, adapter_path, seed, frames):
    from fly_drone.adapter import declared_command

    obs, info = env.reset(seed=seed)
    headings, positions = [], []
    for _ in range(frames):
        action = declared_command(brain, features=obs, path=adapter_path)
        obs, _, _, _, info = env.step(action)
        headings.append(float(env.plant.rpy[0, 2]))
        # Copy: ``env.plant.pos`` is a view into MuJoCo's buffer and mutates in place.
        positions.append(np.array(env.plant.pos[0][:2], dtype=float))
    headings = np.asarray(headings)
    positions = np.asarray(positions)
    omega = np.diff(np.unwrap(headings)) / DT
    jump = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    dirty = np.zeros(omega.size, dtype=bool)
    for i in range(omega.size):
        lo, hi = max(0, i - 2), min(jump.size, i + 3)
        dirty[i] = bool(np.any(jump[lo:hi] > TELEPORT_JUMP_M))
    det = np.abs(omega) >= 3.0
    return {
        "seed": int(seed),
        "max_abs_yaw_rate": round(float(np.max(np.abs(omega))), 4)
        if omega.size
        else 0.0,
        "p999_abs_yaw_rate": round(float(np.percentile(np.abs(omega), 99.9)), 4)
        if omega.size
        else 0.0,
        "clean_max_abs_yaw_rate": round(float(np.max(np.abs(omega[~dirty]))), 4)
        if (~dirty).any()
        else 0.0,
        "detections": int(np.sum(det)),
        "detections_at_teleport": int(np.sum(det & dirty)),
        "teleport_frames": int(np.sum(dirty)),
    }


def run(adapter="v2", seeds=(0, 1, 2), seconds=60, level=3, output=OUTPUT):
    from fly_drone.brain import BrainRuntime
    from fly_drone.distill import _roam_env
    from fly_drone.motion_stats import SACCADE_THRESHOLD
    from fly_drone.plant import LIMITS

    adapter_path = _resolve_adapter(adapter)
    brain = BrainRuntime()
    env = _roam_env(level, brain)
    frames = int(seconds / DT)
    runs = []
    try:
        for seed in seeds:
            runs.append(_seed_run(brain, env, adapter_path, seed, frames))
    finally:
        env.close()

    clean_ceiling = max(r["clean_max_abs_yaw_rate"] for r in runs)
    report = {
        "adapter": str(adapter_path),
        "level": level,
        "seconds": seconds,
        "frames": frames,
        "plant_yaw_limit_rad_s": float(LIMITS[3]),
        "saccade_threshold_rad_s": float(SACCADE_THRESHOLD),
        "threshold_reachable": bool(clean_ceiling >= SACCADE_THRESHOLD),
        "clean_ceiling_rad_s": clean_ceiling,
        "runs": runs,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def _print(report):
    print(
        f"plant yaw limit {report['plant_yaw_limit_rad_s']} rad/s; "
        f"L7 detector {report['saccade_threshold_rad_s']} rad/s; "
        f"clean ceiling {report['clean_ceiling_rad_s']} rad/s "
        f"-> threshold reachable: {report['threshold_reachable']}"
    )
    for r in report["runs"]:
        print(
            f"  seed {r['seed']}: all max {r['max_abs_yaw_rate']:.3f}, "
            f"clean max {r['clean_max_abs_yaw_rate']:.3f}, "
            f"detections {r['detections']} "
            f"({r['detections_at_teleport']} at teleport), "
            f"teleport frames {r['teleport_frames']}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="v2", help="v1, v2 or a path")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    report = run(args.adapter, args.seeds, args.seconds, args.level, args.output)
    _print(report)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
