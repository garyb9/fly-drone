"""Free-roam sensory feasibility gate: can the four v4 cues support roaming at all?

Nothing here is learned or touches the connectome's wiring. A controller that reads only
the cues entering the brain sets what a decoder could possibly achieve.
"""

import json
import time
from collections import deque
from pathlib import Path

import numpy as np

FIRE = 0.22  # an LIF input cell fires once its held cue exceeds this (neuron-model §3)


class CueController:
    """Braitenberg-style script over the 4 cues and a 5-frame history; no simulator state."""

    name = "cue_script"

    def __init__(self):
        self.history = deque(maxlen=5)
        self.escape = 0
        self.escape_dir = 1.0
        self.search_dir = 1.0

    def _escape_action(self):
        # Brake and sidestep; only a little yaw, because turning itself reads as loom.
        return np.array([-0.4, self.escape_dir, 0.0, 0.3 * self.escape_dir])

    def act(self, cues):
        light_l, light_r, loom_l, loom_r = (float(c) for c in cues)
        self.history.append((loom_l, loom_r))
        if self.escape > 0:
            self.escape -= 1
            return self._escape_action()
        recent = [max(a, b) > FIRE for a, b in list(self.history)[-3:]]
        if sum(recent) >= 2 and max(loom_l, loom_r) > 0.3:
            if abs(loom_l - loom_r) > 0.1:
                # +yaw and +lateral are left: move away from the louder eye.
                self.escape_dir = 1.0 if loom_r > loom_l else -1.0
            self.escape = 15
            return self._escape_action()
        light = light_l + light_r
        if light > 0.3:
            side = (light_l - light_r) / light
            if side:
                self.search_dir = float(np.sign(side))
            forward = 1.0 if abs(side) < 0.3 else 0.2
            return np.array([forward, 0.0, 0.0, float(np.clip(1.5 * side, -1, 1))])
        return np.array([0.6, 0.0, 0.0, 0.25 * self.search_dir])


class RandomController:
    """Ornstein-Uhlenbeck action noise: the chance baseline."""

    name = "random"

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.action = np.zeros(4)

    def act(self, cues):
        self.action += -0.15 * self.action + 0.3 * self.rng.normal(size=4)
        self.action = np.clip(self.action, -1, 1)
        return self.action.copy()


def _episode_job(job):
    from .env import FRAME_SECONDS, ConnectomeEnv

    controller_name, seeds, seconds, level = job
    env = ConnectomeEnv(task="free_roam", level=level, respawn=True)
    runs = []
    try:
        for seed in seeds:
            controller = (
                CueController()
                if controller_name == "cue_script"
                else RandomController(seed)
            )
            _, info = env.reset(seed=int(seed))
            frames = int(seconds / FRAME_SECONDS)
            loom_frames = 0
            start = time.perf_counter()
            for _ in range(frames):
                cues = env.brain.cues
                loom_frames += max(cues[2], cues[3]) > FIRE
                _, _, _, _, info = env.step(controller.act(cues))
            threats = info["threats"]
            minutes = seconds / 60
            runs.append(
                {
                    "seed": int(seed),
                    "beacons_per_min": info["beacons_collected"] / minutes,
                    "collisions_per_min": info["collisions"] / minutes,
                    "collision_kinds": info["collision_kinds"],
                    "threats": len(threats),
                    "threats_dodged": sum(t["dodged"] for t in threats),
                    "threats_hit": sum(t["hit"] for t in threats),
                    "visited_cells": info["visited_cells"],
                    "loom_frame_fraction": loom_frames / frames,
                    "wall_seconds": time.perf_counter() - start,
                }
            )
    finally:
        env.close()
    return controller_name, runs


def sensor_scan():
    """Kinematic cue measurements in the arena (teleported poses, no dynamics)."""
    from .arena import ArenaSpec, generate_layout
    from .brain import BrainRuntime
    from .plant import DronePlant

    brain = BrainRuntime()
    spec = ArenaSpec()
    plant = DronePlant(arena=spec)
    park = [0.0, 0.0, -20.0]
    far = [-7.5, -7.5, 0.8]

    def sense():
        return brain.sense(plant.camera())

    def pose(x, y, yaw):
        plant.teleport([x, y, 1.0], yaw)

    def warning(speed, start, stop, surface):
        brain.clear_vision_history()
        run, x = 0, start
        while surface(x) > stop:
            pose(x, 0.0, 0.0)
            run = run + 1 if max(sense()[2:]) > FIRE else 0
            if run == 3:
                return round(surface(x) + 2 * FRAME * speed, 2)
            x += FRAME * speed
        return 0.0

    def yaw_fraction(rate):
        brain.clear_vision_history()
        fires = []
        for i in range(int(2 * np.pi / rate / FRAME)):
            pose(0.0, 0.0, rate * FRAME * i)
            cues = sense()
            if i:
                fires.append(max(cues[2:]) > FIRE)
        return round(float(np.mean(fires)), 3)

    FRAME = 0.04
    try:
        plant.set_objects(obstacle=park, park_obstacle=True)
        beacon = {}
        for d in (2, 4, 6, 8, 10, 12):
            row = {}
            for beta in (0.0, 0.3, -0.3):
                pose(-7.0, 0.0, 0.0)
                plant.set_objects(target=[-7 + d * np.cos(beta), d * np.sin(beta), 1.0])
                brain.clear_vision_history()
                sense()
                row[str(beta)] = [round(float(c), 2) for c in sense()[:2]]
            beacon[str(d)] = row
        plant.set_objects(target=far)
        plant.set_pillars([[5.0, 0.0]])
        pillar = {
            str(v): warning(v, 5.0 - 0.3 - 4.5, 0.35, lambda x: 5.0 - 0.3 - x)
            for v in (0.5, 1.0)
        }
        plant.set_pillars(np.zeros((0, 2)))
        wall = {str(v): warning(v, 1.0, 0.4, lambda x: 8.0 - x) for v in (0.5, 1.0)}
        brain.clear_vision_history()
        fires = []
        for i in range(150):
            pose(-3 + FRAME * i, 6.0, 0.0)
            cues = sense()
            if i:
                fires.append(max(cues[2:]) > FIRE)
        forward = round(float(np.mean(fires)), 3)
        yaw_walls = {str(r): yaw_fraction(r) for r in (0.4, 0.8, 1.2)}
        plant.set_pillars(generate_layout(np.random.default_rng(3), spec, 3))
        yaw_pillars = {str(r): yaw_fraction(r) for r in (0.4, 0.8, 1.2)}
    finally:
        plant.close()
    return {
        "beacon_light_cues_by_distance_and_bearing": beacon,
        "pillar_warning_surface_distance_m": pillar,
        "wall_warning_surface_distance_m": wall,
        "forward_flight_false_loom_fraction": forward,
        "yaw_false_loom_fraction_walls_only": yaw_walls,
        "yaw_false_loom_fraction_16_pillars": yaw_pillars,
    }


def run(
    output="runs/roam-feasibility.json", episodes=20, seconds=120, workers=16, level=3
):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    seeds = np.arange(3000, 3000 + episodes)
    chunks = np.array_split(seeds, max(1, min(workers // 2, episodes)))
    jobs = [
        (name, chunk.tolist(), seconds, level)
        for name in ("cue_script", "random")
        for chunk in chunks
    ]
    start = time.perf_counter()
    collected = {"cue_script": [], "random": []}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for name, runs in pool.map(_episode_job, jobs):
            collected[name] += runs
    summary = {}
    for name, runs in collected.items():
        runs.sort(key=lambda r: r["seed"])
        threats = sum(r["threats"] for r in runs)
        summary[name] = {
            "beacons_per_min": float(np.mean([r["beacons_per_min"] for r in runs])),
            "collisions_per_min": float(
                np.mean([r["collisions_per_min"] for r in runs])
            ),
            "threat_dodge_rate": sum(r["threats_dodged"] for r in runs) / threats
            if threats
            else None,
            "mean_loom_frame_fraction": float(
                np.mean([r["loom_frame_fraction"] for r in runs])
            ),
            "runs": runs,
        }
    cue, rnd = summary["cue_script"], summary["random"]
    gate = {
        "beacons_vs_random": cue["beacons_per_min"] / max(rnd["beacons_per_min"], 1e-9),
        "collisions_vs_random": cue["collisions_per_min"]
        / max(rnd["collisions_per_min"], 1e-9),
    }
    gate["passed"] = bool(
        gate["beacons_vs_random"] > 3.0 and gate["collisions_vs_random"] < 0.5
    )
    report = {
        "episodes": episodes,
        "seconds": seconds,
        "level": level,
        "seeds": [int(seeds[0]), int(seeds[-1])],
        "wall_seconds": time.perf_counter() - start,
        "sensors": sensor_scan(),
        "controllers": summary,
        "gate": gate,
        "note": "Gate: a controller reading only the 4 cues (+5-frame history) must "
        "collect > 3x the beacons of OU-noise random actions with < 0.5x its "
        "collisions (docs plan Phase 2).",
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2))
    return report
