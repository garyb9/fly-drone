"""Closed-loop step-response measurement for the arena stabiliser.

Answers the question behind free-roam blocker #1 (thrown balls aren't dodgeable): from a
steady forward cruise, how fast does each translational axis (lateral, vertical) actually
reach a commanded step, under the PID gains `plant.py`/`pid_control.py` already use? See
`docs/superpowers/plans/2026-09-14-fly-drone-04-free-roam-handoff.md` and the follow-up
plan for the decision this feeds.
"""

import numpy as np

from .arena import ArenaSpec
from .plant import DronePlant

DT = 0.005  # one plant.advance() call == one 200 Hz control tick


def _run(plant, commands, ticks):
    """Advance `plant` under `commands` for `ticks` steps, recording world velocity."""
    trace = np.zeros((ticks, 3))
    for i in range(ticks):
        plant.advance(commands)
        trace[i] = plant.vel[0]
    return trace


def measure_step_response(
    axis: int,
    cruise_forward: float = 0.56,
    warmup_seconds: float = 2.0,
    step_seconds: float = 2.0,
    arena: ArenaSpec | None = None,
) -> dict:
    """Step `axis` (1=lateral, 2=vertical) from a forward cruise to its arena limit.

    Returns the raw velocity trace plus time-to-80%-of-target, so a caller can also check
    for overshoot/oscillation rather than trusting a single scalar.
    """
    if axis not in (1, 2):
        raise ValueError("axis must be 1 (lateral) or 2 (vertical)")
    arena = arena or ArenaSpec()
    plant = DronePlant(vision=False, arena=arena)
    try:
        warmup_ticks = int(round(warmup_seconds / DT))
        step_ticks = int(round(step_seconds / DT))
        cruise = np.array([cruise_forward, 0.0, 0.0, 0.0])
        _run(plant, cruise, warmup_ticks)
        pre_step_axis_vel = plant.vel[0][axis]

        target = float(arena.limits[axis])
        step_command = cruise.copy()
        step_command[axis] = target
        trace = _run(plant, step_command, step_ticks)

        axis_vel = trace[:, axis]
        delta = target - pre_step_axis_vel
        threshold = pre_step_axis_vel + 0.8 * delta
        reached = np.where(axis_vel >= threshold) if delta > 0 else np.where(axis_vel <= threshold)
        time_to_80pct = float(reached[0][0] * DT) if reached[0].size else None
        return {
            "axis": "lateral" if axis == 1 else "vertical",
            "target_mps": target,
            "pre_step_mps": float(pre_step_axis_vel),
            "peak_mps": float(np.max(np.abs(axis_vel))),
            "final_mps": float(axis_vel[-1]),
            "time_to_80pct_s": time_to_80pct,
            "reached_80pct": reached[0].size > 0,
            "trace_mps": axis_vel.tolist(),
            "dt_s": DT,
        }
    finally:
        plant.close()


def run(output: str | None = None) -> dict:
    report = {
        "cruise_forward_mps": 0.56,
        "lateral": measure_step_response(1),
        "vertical": measure_step_response(2),
    }
    if output:
        import json
        from pathlib import Path

        Path(output).write_text(json.dumps(report, indent=2))
    return report
