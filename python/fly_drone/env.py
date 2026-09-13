import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .brain import BrainRuntime
from .plant import LIMITS, DronePlant

FRAME_SECONDS = 0.04
TASKS = ("visual", "looming", "approach", "track", "steer_dodge", "escape")
HORIZON_FRAMES = {
    "hover": 750,
    "visual": 750,
    "looming": 150,
    "approach": 375,
    "track": 750,
    "steer_dodge": 375,
    "escape": 150,
}
EVAL_SECONDS = {
    "visual": 10,
    "looming": 6,
    "approach": 15,
    "track": 30,
    "steer_dodge": 15,
    "escape": 6,
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


def target_bearing(pos, yaw, target):
    """Signed target angle relative to heading, wrapped to [-pi, pi]."""
    delta = np.asarray(target, dtype=float) - np.asarray(pos, dtype=float)
    angle = np.arctan2(delta[1], delta[0]) - yaw
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


class ConnectomeEnv(gym.Env):
    """PPO observes frozen neural activity; action is normalized motion intent."""

    metadata = {}

    def __init__(self, task="visual", vision=True, brain=None, ablation="none"):
        if task not in HORIZON_FRAMES:
            raise ValueError(f"unknown task {task!r}")
        self.brain = brain or BrainRuntime()
        self.plant = DronePlant(vision=vision)
        self.task = task
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
        self.side = 0.0
        self.start = np.array([0.0, 0.0, 1.0])

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
        self.side = side
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
        if self.ablation == "sensory":
            self.brain.silence_sensors()
        self.brain.sense(self.plant.camera())
        # Deterministic neural settling, no hidden body time advancement.
        self.brain.step(40)
        info = self.info()
        self.previous_bearing = abs(info["bearing"])
        self.previous_distance = info["target_distance"]
        return self.observe(), info

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
        return (
            self.launch is not None
            and self.frames * FRAME_SECONDS >= self.launch["delay"]
        )

    def _move_objects(self):
        t = self.frames * FRAME_SECONDS
        if self.orbit is not None:
            self.plant.set_objects(target=self._orbit_position(t))
        if not self.launched():
            return
        travelled = (t - self.launch["delay"]) * self.launch["speed"]
        position = self.launch["origin"] + self.launch["direction"] * travelled
        if position[0] < -1.5:
            position = OBSTACLE_PARK
        self.plant.set_objects(obstacle=position)

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("action must contain four finite values")
        self.command = np.clip(action, -1, 1) * LIMITS
        self._move_objects()
        self.brain.sense(self.plant.camera())
        self.trace = []
        for _ in range(8):
            for idx, value in self.interventions.items():
                self.brain.core.stimulate([idx], value)
            self.brain.step()
            self.plant.advance(self.command)
            self.trace.append(self.brain.read())
        self.frames += 1
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

    def info(self):
        pos = self.plant.pos[0]
        return {
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
