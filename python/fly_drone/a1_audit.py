"""Registered A1 transmission diagnostic; no fitting or capability acceptance.

Run ``python -m fly_drone.a1_audit --list`` for the fixed seed/family registry.
Rendered stimuli are open-loop: three independent bodies replay the three codecs'
commands from identical neural traces. Their headings measure execution, not seeking.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

AUDIT_VERSION = "a1-transmission-v1"
CONTROLS = ("none", "light", "loom")
HISTORIES = ("reset", "persistent")
FRAME_TICKS = 8
FRAME_SECONDS = FRAME_TICKS * 0.005
# Baseline, onset/sustain, disappearance, reacquisition. Total 2 seconds.
PHASES = (("baseline", 5), ("on", 25), ("off", 10), ("reacquire", 10))


def scenarios(family):
    """Fixed before confirmation; no adaptive choice using behavioral outcomes."""
    if family == "synthetic":
        result = [{"name": "absent", "cues": [0, 0, 0, 0]}]
        for strength in (0.25, 0.5, 1.0, 2.0):
            for side, pair in (("left", (1, 0)), ("right", (0, 1)), ("both", (1, 1))):
                result.append(
                    {
                        "name": f"{side}-{strength:g}",
                        "cues": [strength * x for x in pair] + [0, 0],
                    }
                )
        result.extend(
            [
                {"name": "mixed-left", "cues": [1, 0, 1, 0]},
                {"name": "mixed-right", "cues": [0, 1, 0, 1]},
            ]
        )
        return result
    if family == "rendered":
        result = [{"name": "absent", "bearing": 0, "distance": 2, "beacon": False}]
        for brightness in (0.65, 0.95):
            for bearing in (-0.6, 0.6):
                result.append(
                    {
                        "name": f"brightness-{brightness}-bearing-{bearing}",
                        "bearing": bearing,
                        "distance": 2,
                        "beacon": True,
                        "brightness": brightness,
                    }
                )
        for distance in (1.0, 3.0):
            for bearing in (-0.6, 0.0, 0.6):
                result.append(
                    {
                        "name": f"bearing-{bearing:g}-distance-{distance:g}",
                        "bearing": bearing,
                        "distance": distance,
                        "beacon": True,
                    }
                )
        result.extend(
            [
                {
                    "name": "turn-left",
                    "bearing": 0,
                    "distance": 2,
                    "beacon": False,
                    "yaw_rate": 0.8,
                },
                {
                    "name": "turn-right",
                    "bearing": 0,
                    "distance": 2,
                    "beacon": False,
                    "yaw_rate": -0.8,
                },
                {
                    "name": "mixed-left",
                    "bearing": 0.6,
                    "distance": 2,
                    "beacon": True,
                    "threat": True,
                },
                {
                    "name": "mixed-right",
                    "bearing": -0.6,
                    "distance": 2,
                    "beacon": True,
                    "threat": True,
                },
            ]
        )
        return result
    raise ValueError(f"unknown family: {family}")


def trial_definitions():
    return [
        {
            "id": f"{'development' if seed < 8 else 'confirmation'}-{seed:02d}-{family}",
            "seed": seed,
            "split": "development" if seed < 8 else "confirmation",
            "family": family,
        }
        for seed in range(16)
        for family in ("synthetic", "rendered")
    ]


def registered_trial(trial_id):
    for trial in trial_definitions():
        if trial["id"] == trial_id:
            return trial
    raise ValueError(f"unregistered trial: {trial_id}")


def controlled_currents(currents, control):
    currents = np.asarray(currents, dtype=np.float32).copy()
    if currents.shape != (4,):
        raise ValueError("this audit requires canonical four-channel v4 currents")
    if control == "light":
        currents[:2] = 0
    elif control == "loom":
        currents[2:] = 0
    elif control != "none":
        raise ValueError(f"unknown control: {control}")
    return currents


def trace_readout(brain, adapters):
    from .adapter import _latch, _laterality, _readouts, _steer_diff, feature_positions

    features = brain.features()
    positions = feature_positions(brain)
    populations = {
        name: float(np.mean([features[positions[i]] for i in ids]))
        for name, ids in brain.readout_ids.items()
        if ids and all(i in positions for i in ids)
    }
    escape, power = _readouts(brain, features)
    diff = _steer_diff(brain, features)
    laterality = _laterality(brain, features)
    codecs = {}
    for name, adapter in adapters.items():
        p = adapter.params
        raw = (
            (diff - p["rest_steer"]) / p["steer_span"]
            if name != "v1"
            else laterality - p["rest_lat"]
        )
        latch = _latch(escape, p["rest_escape"])
        codecs[name] = {
            "raw_steering": float(raw),
            "pre_escape_steering": float(np.clip(raw, -1, 1))
            if name != "v1"
            else float(raw),
            "escape_latch": float(latch),
            "escape_multiplier": float(1 - 2 * latch),
            "command_normalized": adapter.command(brain, features).tolist(),
        }
    return {
        "populations": populations,
        "steer_left_minus_right": diff,
        "population_laterality": laterality,
        "escape": escape,
        "power": power,
        "codecs": codecs,
    }


def render_stimulus(plant, scenario, active, active_time):
    """Geometry is stimulus truth only; never passed to the codec."""
    import mujoco

    bearing = scenario["bearing"]
    distance = scenario["distance"]
    yaw = scenario.get("yaw_rate", 0) * active_time if active else 0
    plant.teleport([0, 0, 1], yaw=yaw)
    target = [distance * np.cos(bearing), distance * np.sin(bearing), 1]
    # Hide target completely between presentations; alpha does not change geometry.
    geom = mujoco.mj_name2id(plant.model, mujoco.mjtObj.mjOBJ_GEOM, "target")
    plant.model.geom_rgba[geom, 3] = float(active and scenario["beacon"])
    plant.model.geom_rgba[geom, :3] = scenario.get("brightness", 0.95)
    threat_distance = max(0.6, 3.0 - 5 * active_time)
    obstacle = (
        [threat_distance * np.cos(bearing), threat_distance * np.sin(bearing), 1]
        if active and scenario.get("threat")
        else [0, 0, -10]
    )
    plant.set_objects(target=target, obstacle=obstacle, park_obstacle=True)
    return plant.camera()


def summarize_case(case):
    """Descriptive phase means, never a capability or causal pass decision."""
    summaries = {}
    for phase, _ in PHASES:
        rows = [row for row in case["frames"] if row["phase"] == phase]
        summaries[phase] = {
            "raw_currents": np.mean(
                [row["raw_currents"] for row in rows], axis=0
            ).tolist(),
            "applied_currents": np.mean(
                [row["applied_currents"] for row in rows], axis=0
            ).tolist(),
            "steer_left_minus_right": float(
                np.mean([row["steer_left_minus_right"] for row in rows])
            ),
            "codecs": {
                codec: {
                    key: float(np.mean([row["codecs"][codec][key] for row in rows]))
                    for key in ("raw_steering", "escape_latch", "heading_change_rad")
                }
                | {
                    "yaw_command": float(
                        np.mean(
                            [
                                row["codecs"][codec]["command_normalized"][3]
                                for row in rows
                            ]
                        )
                    )
                }
                for codec in rows[0]["codecs"]
            },
        }
    return summaries


def begin_case(brain, seed, history):
    if history == "reset":
        brain.reset(seed)
    elif history != "persistent":
        raise ValueError(f"unknown history: {history}")
    # Preserve neural state only: camera jumps are not within-scene looming.
    brain.clear_vision_history()


def run_trial(trial_id, output):
    if Path(output).exists():
        raise FileExistsError(f"refusing to overwrite existing audit report: {output}")
    from .adapter import CODEC_PATHS, Adapter
    from .brain import BrainRuntime
    from .plant import DronePlant

    trial = registered_trial(trial_id)
    battery = scenarios(trial["family"])
    brain = BrainRuntime(seed=trial["seed"])
    if brain.alternate or brain.learned or brain.cues.shape != (4,):
        raise ValueError("audit requires the canonical declared v4 brain")
    adapters = {name: Adapter.load(path) for name, path in CODEC_PATHS.items()}
    for adapter in adapters.values():
        adapter.check(brain)
    protocol = {
        "version": AUDIT_VERSION,
        "trial": trial,
        "scenarios": battery,
        "controls": CONTROLS,
        "histories": HISTORIES,
        "phases": PHASES,
        "frame_ticks": FRAME_TICKS,
        "frame_seconds": FRAME_SECONDS,
        "timing": "Readouts/commands at frame start; then inject current frame and advance brain/body eight ticks, matching env.step. Headings are frame-end measurements.",
    }
    report = {
        "protocol": protocol,
        "protocol_hash": hashlib.sha256(
            json.dumps(protocol, sort_keys=True).encode()
        ).hexdigest(),
        "dataset_hash": brain.dataset_hash,
        "bundle_hash": brain.bundle_hash,
        "adapters": {name: adapter.payload for name, adapter in adapters.items()},
        "interpretation": "Open-loop transmission and mechanical replay; not closed-loop seeking or capability acceptance.",
        "cases": [],
    }
    bodies = {}
    visual = None
    started = time.monotonic()
    try:
        for name in adapters:
            bodies[name] = DronePlant(vision=False)
        if trial["family"] == "rendered":
            visual = DronePlant(vision=True)
        for control in CONTROLS:
            for history in HISTORIES:
                brain.reset(trial["seed"])
                brain.clear_vision_history()
                for scenario in battery:
                    begin_case(brain, trial["seed"], history)
                    for body in bodies.values():
                        body.reset(seed=trial["seed"])
                    case = {
                        "scenario": scenario["name"],
                        "control": control,
                        "history": history,
                        "frames": [],
                    }
                    for phase, frames in PHASES:
                        active = phase in ("on", "reacquire")
                        for frame in range(frames):
                            if visual is None:
                                raw = np.asarray(
                                    scenario["cues"] if active else [0, 0, 0, 0],
                                    dtype=np.float32,
                                )
                            else:
                                raw = brain.sense(
                                    render_stimulus(
                                        visual, scenario, active, frame * FRAME_SECONDS
                                    )
                                ).copy()
                            applied = controlled_currents(raw, control)
                            brain.set_currents(applied)
                            row = trace_readout(brain, adapters)
                            row.update(
                                phase=phase,
                                frame=frame,
                                brain_tick=brain.tick,
                                raw_currents=raw.tolist(),
                                applied_currents=applied.tolist(),
                            )
                            # Commands use the preceding neural response, as in env.step.
                            # Independent replay has no body feedback, so serial brain/body
                            # integration is equivalent to interleaving their eight ticks.
                            brain.step(FRAME_TICKS)
                            end_readout = trace_readout(brain, adapters)
                            row["frame_end_response"] = {
                                "brain_tick": brain.tick,
                                "steer_left_minus_right": end_readout[
                                    "steer_left_minus_right"
                                ],
                                "escape": end_readout["escape"],
                            }
                            for name, body in bodies.items():
                                codec = row["codecs"][name]
                                command = (
                                    np.asarray(codec["command_normalized"])
                                    * body.limits
                                )
                                before = float(body.rpy[0, 2])
                                for _ in range(FRAME_TICKS):
                                    body.advance(command)
                                after = float(body.rpy[0, 2])
                                codec.update(
                                    command_physical=command.tolist(),
                                    heading_rad=after,
                                    heading_change_rad=float(
                                        np.arctan2(
                                            np.sin(after - before),
                                            np.cos(after - before),
                                        )
                                    ),
                                )
                            case["frames"].append(row)
                    case["summary"] = summarize_case(case)
                    report["cases"].append(case)
                    print(
                        json.dumps(
                            {
                                "trial": trial_id,
                                "completed_cases": len(report["cases"]),
                                "total_cases": len(battery)
                                * len(CONTROLS)
                                * len(HISTORIES),
                                "elapsed_seconds": time.monotonic() - started,
                            }
                        ),
                        flush=True,
                    )
    finally:
        for body in bodies.values():
            body.close()
        if visual is not None:
            visual.close()
    report.update(completed=True, elapsed_seconds=time.monotonic() - started)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        help="print fixed trial definitions without loading the brain",
    )
    parser.add_argument("--trial")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.list:
        print(json.dumps(trial_definitions(), indent=2))
    elif args.trial and args.output:
        run_trial(args.trial, args.output)
    else:
        parser.error("supply --list or both --trial and --output")


if __name__ == "__main__":
    main()
