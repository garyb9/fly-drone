import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from . import arena
from .brain import BrainRuntime
from .plant import DronePlant

FRAME_SECONDS = 0.04
TASKS = (
    "visual",
    "looming",
    "approach",
    "track",
    "steer_dodge",
    "escape",
    "free_roam",
)
HORIZON_FRAMES = {
    "hover": 750,
    "visual": 750,
    "looming": 150,
    "approach": 375,
    "track": 750,
    "steer_dodge": 375,
    "escape": 150,
    "free_roam": 1500,
}
EVAL_SECONDS = {
    "visual": 10,
    "looming": 6,
    "approach": 15,
    "track": 30,
    "steer_dodge": 15,
    "escape": 6,
    "free_roam": 120,
}
OBSTACLE_PARK = np.array([3.8, -3.8, 0.4])
TARGET_BEHIND = np.array([-3.8, 0.0, 1.0])
THREAT_RANGE = 2.0
MAX_DISPLACEMENT_AT_THREAT = 0.25
APPROACH_RADIUS = 0.6
TRACK_SETTLE_SECONDS = 5.0
TRACK_MAX_MEAN_BEARING = 0.3
ESCAPE_MIN_CLIMB = 0.1
SETTLE_SECONDS = 2.0
# Full binocular field is +-88.6 deg; beyond this the beacon cannot reach either eye.
FIELD_HALF_ANGLE = 1.546
BEACON_VISIBLE_RANGE = 12.0
PATHWAYS = {"light": ("light_l", "light_r"), "loom": ("looming_l", "looming_r")}


def target_bearing(pos, yaw, target):
    """Signed target angle relative to heading, wrapped to [-pi, pi]."""
    delta = np.asarray(target, dtype=float) - np.asarray(pos, dtype=float)
    angle = np.arctan2(delta[1], delta[0]) - yaw
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


class ConnectomeEnv(gym.Env):
    """PPO observes frozen neural activity; action is normalized motion intent."""

    metadata = {}

    def __init__(
        self,
        task="visual",
        vision=True,
        brain=None,
        ablation="none",
        level=3,
        respawn=False,
        spec=None,
    ):
        if task not in HORIZON_FRAMES:
            raise ValueError(f"unknown task {task!r}")
        if level not in arena.LEVELS:
            raise ValueError(f"unknown arena level {level!r}")
        self.brain = brain or BrainRuntime()
        self.vision = vision
        self.spec = spec or arena.ArenaSpec()
        self.level = level
        self.respawn = respawn
        self.task = task
        self.plant = DronePlant(vision=vision, arena=self._wanted_arena())
        self.ablation = ablation
        self.action_space = spaces.Box(-1, 1, (4,), np.float32)
        self.observation_space = spaces.Box(
            0, 1, (len(self.brain.feature_ids),), np.float32
        )
        self.frames = 0
        self.command = np.zeros(4)
        self.trace = []
        self.interventions = {}
        self.previous_bearing = 0.0
        self.previous_distance = 0.0
        self.launch = None
        self.orbit = None
        self.roam = None
        self.side = 0.0
        self.start = np.array([0.0, 0.0, 1.0])

    def _wanted_arena(self):
        return self.spec if self.task == "free_roam" else None

    def _ensure_plant(self):
        """The live server switches tasks on one env; the arena needs its own model."""
        wanted = self._wanted_arena()
        if (self.plant.arena is None) != (wanted is None):
            self.plant.close()
            self.plant = DronePlant(vision=self.vision, arena=wanted)
            self._geom_kinds = None

    def observe(self):
        x = self.brain.features()
        if self.ablation == "zero":
            return np.zeros_like(x)
        if self.ablation == "shuffle":
            return x[self.permutation]
        return x

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        seed = int(seed if seed is not None else self.np_random.integers(0, 2**31))
        self._ensure_plant()
        self.brain.reset(seed)
        self.plant.reset(seed=seed)
        self.frames = 0
        self.command[:] = 0
        self.trace = []
        self.interventions = {}
        rng = self.np_random
        # Draw order for visual/looming is frozen: accepted evaluations replay by seed.
        self.permutation = rng.permutation(len(self.brain.feature_ids))
        side = float(rng.choice([-1, 1]))
        self.start = self.plant.pos[0].copy()
        self.launch = None
        self.orbit = None
        self.roam = None
        self.side = side
        if self.task == "free_roam":
            self._reset_roam(rng)
        else:
            self._reset_trial(rng, side)
        if self.ablation == "sensory":
            self.brain.silence_sensors()
        elif self.ablation in PATHWAYS:
            self.brain.silence_inputs(PATHWAYS[self.ablation])
        if self.plant.arena is not None:
            self.plant.set_ghost(self.ablation == "ghost")
        if self.brain.learned:
            self.brain.push_frame(self.plant.camera())
            # Settle on zero input: identical for a deployed and a learning encoder.
            self.brain.set_currents(np.zeros_like(self.brain.cues))
        else:
            self.brain.sense(self.plant.camera())
        # Deterministic neural settling, no hidden body time advancement.
        self.brain.step(40)
        info = self.info()
        self.previous_bearing = abs(info["bearing"])
        self.previous_distance = info["target_distance"]
        return self.observe(), info

    def _sense(self):
        """v4 renders now; v5 encodes the stack rendered at the end of the previous step."""
        if self.brain.learned:
            self.brain.encode_stack()
        else:
            self.brain.sense(self.plant.camera())

    def _reset_trial(self, rng, side):
        target, obstacle = TARGET_BEHIND, OBSTACLE_PARK
        if self.task in ("hover", "visual", "steer_dodge"):
            target = np.array([2, side * rng.uniform(0.6, 1.5), 1])
            obstacle = np.array([2, -side, 1])
        elif self.task == "approach":
            bearing = side * rng.uniform(0.1, 0.5)
            distance = rng.uniform(2.2, 3.0)
            target = np.array(
                [distance * np.cos(bearing), distance * np.sin(bearing), 1.0]
            )
        elif self.task == "track":
            self.orbit = {
                "angle": float(rng.uniform(-0.3, 0.3)),
                "rate": side * float(rng.uniform(0.15, 0.3)),
                "reverse": float(rng.uniform(8.0, 20.0)),
                "radius": 2.0,
            }
            target = self._orbit_position(0.0)
        if self.task in ("looming", "escape", "steer_dodge"):
            obstacle_side = side
            if self.task == "steer_dodge":
                obstacle_side = float(rng.choice([-1, 1]))
            escape = self.task == "escape"
            origin = np.array(
                [
                    2.5,
                    obstacle_side
                    * rng.uniform(0.0 if escape else 0.05, 0.15 if escape else 0.3),
                    1.0 + (rng.uniform(0.0, 0.1) if escape else rng.uniform(-0.1, 0.1)),
                ]
            )
            # Escape aims below centre so a short climb (0.2 m/s limit) can clear it.
            aim = self.start + (np.array([0.0, 0.0, -0.12]) if escape else 0.0)
            direction = aim - origin
            delay = (
                rng.uniform(1.5, 4.0)
                if self.task == "steer_dodge"
                else rng.uniform(0.4, 2.0)
            )
            self.launch = {
                "side": obstacle_side,
                "origin": origin,
                "direction": direction / np.linalg.norm(direction),
                "delay": float(delay),
                "speed": float(rng.uniform(0.8, 1.2)),
            }
            obstacle = origin
            if self.task != "steer_dodge":
                self.side = obstacle_side
        if self.task == "steer_dodge":
            self.side = self.launch["side"]
        self.plant.set_objects(target=target, obstacle=obstacle)

    def _reset_roam(self, rng):
        spec = self.spec
        yaw = float(rng.uniform(-np.pi, np.pi))
        self.plant.teleport([0.0, 0.0, 1.0], yaw)
        pillars = arena.generate_layout(rng, spec, self.level)
        self.plant.set_pillars(pillars)
        beacon, hidden = arena.next_beacon(rng, spec, pillars, self.plant.pos[0], yaw)
        self.plant.set_objects(
            target=beacon, obstacle=[0.0, 0.0, arena.PARK_Z], park_obstacle=True
        )
        self.start = self.plant.pos[0].copy()
        self.side = 0.0
        threats = arena.LEVELS[self.level][1]
        self.roam = {
            "beacons": 0,
            "beacon_hidden": hidden,
            "beacon_was_visible": False,
            "collisions": 0,
            "collision_kinds": {},
            "threat": None,
            "next_threat": float(rng.uniform(3.0, 8.0)) if threats else None,
            "threats": [],
            "visited": {self._cell(self.start)},
            "events": [],
            # Teacher-only state for the explore drive's yaw cast (teacher.py); resampled
            # lazily on first use so a resumed/short episode still starts unbiased.
            "explore": {"yaw_bias": 0.0, "next_change": 0.0},
        }

    @staticmethod
    def _cell(pos):
        return (int(np.floor(pos[0])), int(np.floor(pos[1])))

    def _orbit_position(self, t):
        o = self.orbit
        swept = o["rate"] * min(t, o["reverse"]) - o["rate"] * max(
            0.0, t - o["reverse"]
        )
        angle = o["angle"] + swept
        return np.array(
            [
                self.start[0] + o["radius"] * np.cos(angle),
                self.start[1] + o["radius"] * np.sin(angle),
                1.0,
            ]
        )

    def launched(self):
        if self.roam is not None:
            return self.roam["threat"] is not None
        return (
            self.launch is not None
            and self.frames * FRAME_SECONDS >= self.launch["delay"]
        )

    def _move_objects(self):
        t = self.frames * FRAME_SECONDS
        if self.roam is not None:
            self._move_threat(t)
            return
        if self.orbit is not None:
            self.plant.set_objects(target=self._orbit_position(t))
        if not self.launched():
            return
        travelled = (t - self.launch["delay"]) * self.launch["speed"]
        position = self.launch["origin"] + self.launch["direction"] * travelled
        if position[0] < -1.5:
            position = OBSTACLE_PARK
        self.plant.set_objects(obstacle=position)

    def _move_threat(self, t):
        r, spec, plant = self.roam, self.spec, self.plant
        threat = r["threat"]
        if threat is not None:
            travelled = (t - threat["t0"]) * threat["speed"]
            position = threat["origin"] + threat["direction"] * travelled
            if (
                travelled > threat["range"] + 2.0
                or np.any(np.abs(position[:2]) > spec.inner)
                or position[2] < 0.3
            ):
                self._finish_threat(hit=False)
            else:
                plant.set_objects(obstacle=position)
        elif r["next_threat"] is not None and t >= r["next_threat"]:
            if not self.launch_threat():
                r["next_threat"] = t + 0.5

    def launch_threat(self):
        """Throw a threat from ahead of the drone; False if one is flying or blocked."""
        r, spec, plant = self.roam, self.spec, self.plant
        if r is None or r["threat"] is not None:
            return False
        pos = plant.pos[0]
        plan = arena.plan_threat(
            self.np_random, spec, pos, plant.rpy[0, 2], plant.vel[0]
        )
        if (
            plan["range"] < arena.MIN_LAUNCH_RANGE
            or arena.clearance(spec, plant.pillars, plan["origin"])
            <= spec.threat_radius + 0.3
        ):
            return False
        plan.update(t0=self.frames * FRAME_SECONDS, min_distance=plan["range"])
        r["threat"] = plan
        r["events"].append({"type": "threat_launched", "side": plan["side"]})
        plant.set_objects(obstacle=plan["origin"])
        return True

    def _finish_threat(self, hit):
        r = self.roam
        threat, r["threat"] = r["threat"], None
        outcome = {
            "side": threat["side"],
            "min_distance": float(threat["min_distance"]),
            "hit": bool(hit),
            # Only a threat that actually came within range can count as dodged.
            "dodged": bool(not hit and threat["min_distance"] < THREAT_RANGE),
        }
        r["threats"].append(outcome)
        r["events"].append(
            {"type": "threat_hit" if hit else "threat_passed", **outcome}
        )
        t = self.frames * FRAME_SECONDS
        # A manually launched threat must not start the scheduler on threat-free levels.
        if arena.LEVELS[self.level][1]:
            r["next_threat"] = t + float(self.np_random.uniform(8.0, 20.0))
        self.plant.set_objects(obstacle=[0.0, 0.0, arena.PARK_Z], park_obstacle=True)

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("action must contain four finite values")
        self.command = np.clip(action, -1, 1) * self.plant.limits
        self._move_objects()
        self._sense()
        self.trace = []
        for _ in range(8):
            for idx, value in self.interventions.items():
                self.brain.core.stimulate([idx], value)
            self.brain.step()
            self.plant.advance(self.command)
            self.trace.append(self.brain.read())
        self.frames += 1
        if self.roam is not None:
            result = self._step_roam(action)
        else:
            result = self._step_room(action)
        if self.brain.learned:
            # The next step's currents come from what the eyes see now, after any respawn.
            self.brain.push_frame(self.plant.camera())
        return result

    def _step_room(self, action):
        pos = self.plant.pos[0]
        delta = self.plant.target - pos
        distance = float(np.linalg.norm(delta[:2]))
        bearing = target_bearing(pos, self.plant.rpy[0, 2], self.plant.target)
        task = self.task
        reward = 1.0 - 0.05 * float(np.dot(action, action))
        reward -= (1.0 if task == "escape" else 4.0) * (pos[2] - 1.0) ** 2
        if task in ("visual", "steer_dodge"):
            # Potential-based turning progress: sums to 10 * total bearing reduction.
            reward += 10 * (self.previous_bearing - abs(bearing))
            reward += 0.5 * np.cos(bearing) - 0.2 * distance
        elif task == "track":
            reward += 5 * (self.previous_bearing - abs(bearing)) + np.cos(bearing)
        elif task == "approach":
            reward += 10 * (self.previous_distance - distance)
            reward += 5 * (self.previous_bearing - abs(bearing)) - 0.1 * distance
        self.previous_bearing = abs(bearing)
        self.previous_distance = distance
        if task in ("looming", "steer_dodge", "escape"):
            # Dodging costs a little drift; contact costs the terminal penalty.
            reward -= 0.5 * float(np.sum((pos[:2] - self.start[:2]) ** 2))
            reward -= 2 * np.exp(
                -(np.linalg.norm(pos - self.plant.obstacle) ** 2) / 0.2
            )
        collision = self.plant.data.ncon > 0
        terminated = bool(
            collision
            or pos[2] < 0.15
            or pos[2] > 3
            or np.any(np.abs(pos[:2]) > 3.5)
            or np.any(np.abs(self.plant.rpy[0, :2]) > 1.2)
        )
        if terminated:
            reward -= 20
        return (
            self.observe(),
            float(reward),
            terminated,
            self.frames >= HORIZON_FRAMES[task],
            self.info(),
        )

    def _step_roam(self, action):
        r, spec, plant = self.roam, self.spec, self.plant
        pos = plant.pos[0].copy()
        reward = 0.05 - 0.05 * float(np.dot(action, action))
        kinds = self.contact_kinds()
        threat = r["threat"]
        if threat is not None:
            distance = float(np.linalg.norm(pos - plant.obstacle))
            threat["min_distance"] = min(threat["min_distance"], distance)
            if "threat" in kinds:
                self._finish_threat(hit=True)
        beacon_distance = float(np.linalg.norm((plant.target - pos)[:2]))
        visible = self.beacon_visible()
        if (
            beacon_distance < spec.collect_radius
            and abs(plant.target[2] - pos[2]) < 0.5
        ):
            r["beacons"] += 1
            reward += 20.0
            r["events"].append({"type": "beacon_collected", "count": r["beacons"]})
            beacon, hidden = arena.next_beacon(
                self.np_random, spec, plant.pillars, pos, plant.rpy[0, 2]
            )
            r["beacon_hidden"] = hidden
            plant.set_objects(target=beacon)
            beacon_distance = float(np.linalg.norm((plant.target - pos)[:2]))
            visible = self.beacon_visible()
        elif visible and r["beacon_was_visible"]:
            # Progress only while the beacon is seen: no reward for unseen luck.
            reward += 2.0 * (self.previous_distance - beacon_distance)
        r["beacon_was_visible"] = visible
        self.previous_distance = beacon_distance
        cell = self._cell(pos)
        if cell not in r["visited"]:
            r["visited"].add(cell)
            reward += 0.5
        reward -= 2.0 * np.exp(-(self.clearance() ** 2) / 0.2)
        reward -= 2.0 * max(0.0, abs(pos[2] - 1.1) - 0.5) ** 2
        tilted = bool(np.any(np.abs(plant.rpy[0, :2]) > 1.2))
        if pos[2] < 0.15 or pos[2] > 3.0:
            kinds.add("altitude")
        if np.any(np.abs(pos[:2]) > spec.half_size - 0.05):
            kinds.add("bounds")
        if tilted:
            kinds.add("tilt")
        terminated = False
        if kinds:
            reward -= 20.0
            r["collisions"] += 1
            for kind in kinds:
                r["collision_kinds"][kind] = r["collision_kinds"].get(kind, 0) + 1
            r["events"].append({"type": "collision", "kinds": sorted(kinds)})
            if self.respawn:
                spot = arena.safe_respawn(spec, plant.pillars, pos)
                plant.teleport(spot, float(plant.rpy[0, 2]))
                self.brain.clear_vision_history()
                self.previous_distance = float(
                    np.linalg.norm((plant.target - plant.pos[0])[:2])
                )
                r["beacon_was_visible"] = False
            else:
                terminated = True
        info = self.info()
        # Events since the previous step, including probes issued between steps.
        r["events"] = []
        return (
            self.observe(),
            float(reward),
            terminated,
            self.frames >= HORIZON_FRAMES["free_roam"],
            info,
        )

    def contact_kinds(self):
        plant = self.plant
        if getattr(self, "_geom_kinds", None) is None:
            kinds = {}
            for g in range(plant.model.ngeom):
                name = mujoco.mj_id2name(plant.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                if name == "obstacle":
                    kinds[g] = "threat"
                elif name.startswith("pillar_"):
                    kinds[g] = "pillar"
                elif name.startswith(("wall_", "back", "left", "right")):
                    kinds[g] = "wall"
                elif name == "floor":
                    kinds[g] = "floor"
            self._geom_kinds = kinds
        found = set()
        for i in range(plant.data.ncon):
            contact = plant.data.contact[i]
            for g in (contact.geom1, contact.geom2):
                if g in self._geom_kinds:
                    found.add(self._geom_kinds[g])
        return found

    def beacon_visible(self):
        """Geometric visibility: inside the binocular field, in range, unoccluded."""
        plant = self.plant
        pos = plant.pos[0]
        if abs(arena.bearing_to(pos, plant.rpy[0, 2], plant.target)) > FIELD_HALF_ANGLE:
            return False
        eye = pos + np.array([0.0, 0.0, 0.008])
        ray = plant.target - eye
        distance = float(np.linalg.norm(ray))
        if distance > BEACON_VISIBLE_RANGE:
            return False
        geom = np.array([-1], dtype=np.int32)
        body = mujoco.mj_name2id(plant.model, mujoco.mjtObj.mjOBJ_BODY, "drone0")
        # mj_ray ignores alpha-0 geoms, so ghost objects do not occlude either.
        hit = mujoco.mj_ray(
            plant.model, plant.data, eye, ray / distance, None, 1, body, geom
        )
        target = mujoco.mj_name2id(plant.model, mujoco.mjtObj.mjOBJ_GEOM, "target")
        return bool(hit < 0 or geom[0] == target or hit >= distance - 1e-6)

    def clearance(self):
        """Surface distance from the drone centre to the nearest pillar, wall or threat."""
        pos = self.plant.pos[0]
        nearest = arena.clearance(self.spec, self.plant.pillars, pos)
        if self.roam is not None and self.roam["threat"] is not None:
            nearest = min(
                nearest,
                float(np.linalg.norm(pos - self.plant.obstacle))
                - self.spec.threat_radius,
            )
        return float(nearest)

    def info(self):
        pos = self.plant.pos[0]
        info = {
            "tick": self.brain.tick,
            "physics_tick": self.plant.step_counter,
            "time": self.plant.data.time,
            "state": self.plant.state(),
            "bearing": target_bearing(pos, self.plant.rpy[0, 2], self.plant.target),
            "target_distance": float(np.linalg.norm((self.plant.target - pos)[:2])),
            "collision": bool(self.plant.data.ncon > 0),
            "obstacle_distance": float(np.linalg.norm(pos - self.plant.obstacle)),
            "obstacle_side": self.launch["side"] if self.launch else 0.0,
            "side": self.side,
            "launched": self.launched(),
            "displacement": float(np.linalg.norm(pos[:2] - self.start[:2])),
            "climb": float(pos[2] - self.start[2]),
        }
        if self.roam is not None:
            r = self.roam
            threat = r["threat"]
            info.update(
                {
                    "obstacle_side": threat["side"] if threat else 0.0,
                    "level": self.level,
                    "beacons_collected": r["beacons"],
                    "beacon_visible": self.beacon_visible(),
                    "beacon_hidden": r["beacon_hidden"],
                    "clearance": self.clearance(),
                    "speed": float(np.linalg.norm(self.plant.vel[0][:2])),
                    "collisions": r["collisions"],
                    "collision_kinds": dict(r["collision_kinds"]),
                    "threats": list(r["threats"]),
                    "visited_cells": len(r["visited"]),
                    "events": list(r["events"]),
                }
            )
        return info

    def close(self):
        self.plant.close()


class EpisodeTracker:
    """Per-episode success rules shared by evaluation and the live viewer."""

    def __init__(self, task, info):
        self.task = task
        self.side = info["side"]
        self.initial_bearing = abs(info["bearing"])
        self.frames = 0
        self.threat_displacement = None
        self.pre_launch_displacement = 0.0
        self.min_obstacle_distance = info["obstacle_distance"]
        self.peak_climb = 0.0
        self.tracking_errors = []
        self.altitudes = []
        self.last = info

    def update(self, info):
        self.frames += 1
        self.last = info
        self.altitudes.append(float(info["state"]["position"][2]))
        self.min_obstacle_distance = min(
            self.min_obstacle_distance, info["obstacle_distance"]
        )
        if not info["launched"]:
            self.pre_launch_displacement = max(
                self.pre_launch_displacement, info["displacement"]
            )
        elif (
            self.threat_displacement is None
            and info["obstacle_distance"] < THREAT_RANGE
        ):
            self.threat_displacement = info["displacement"]
        self.peak_climb = max(self.peak_climb, info["climb"])
        if self.frames * FRAME_SECONDS >= TRACK_SETTLE_SECONDS:
            self.tracking_errors.append(abs(info["bearing"]))

    def finish(self, terminated):
        info = self.last
        threat = (
            info["displacement"]
            if self.threat_displacement is None
            else self.threat_displacement
        )
        # Survival alone rewards blind fleeing; require the drone to still be near its
        # start when the obstacle first comes within threat range.
        specific = threat < MAX_DISPLACEMENT_AT_THREAT
        final_bearing = abs(info["bearing"])
        steered = final_bearing < 0.5 * self.initial_bearing
        tracking = (
            float(np.mean(self.tracking_errors)) if self.tracking_errors else None
        )
        rules = {
            "hover": True,
            "visual": steered,
            "looming": specific,
            "approach": info["target_distance"] < APPROACH_RADIUS,
            "track": tracking is not None and tracking < TRACK_MAX_MEAN_BEARING,
            "steer_dodge": steered and specific,
            "escape": specific and self.peak_climb >= ESCAPE_MIN_CLIMB,
            # Free roam is judged by RoamTracker rates, not a per-episode verdict.
            "free_roam": True,
        }
        zs = np.asarray(self.altitudes) if self.altitudes else np.ones(1)
        settled = zs[int(SETTLE_SECONDS / FRAME_SECONDS) :]
        side = "left" if self.side > 0 else "right"
        return {
            "success": bool(not terminated and rules[self.task]),
            "terminated": bool(terminated),
            "survived": not terminated,
            "collision": bool(info["collision"]),
            "side": side,
            "target_side": side,
            "initial_bearing": self.initial_bearing,
            "final_bearing": final_bearing,
            "final_target_distance": info["target_distance"],
            "mean_tracking_error": tracking,
            "min_obstacle_distance": float(self.min_obstacle_distance),
            "pre_launch_displacement": float(self.pre_launch_displacement),
            "displacement_at_threat": float(threat),
            "final_displacement": float(info["displacement"]),
            "peak_climb": float(self.peak_climb),
            "altitude_rms": float(np.sqrt(np.mean((zs - 1) ** 2))),
            "settled_altitude_rms": float(np.sqrt(np.mean((settled - 1) ** 2)))
            if len(settled)
            else None,
            "frames": self.frames,
        }
