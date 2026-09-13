import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .brain import BrainRuntime
from .plant import LIMITS, DronePlant

FRAME_SECONDS = 0.04
HORIZON_FRAMES = {"hover": 750, "visual": 750, "looming": 150}
OBSTACLE_PARK = np.array([3.8, -3.8, 0.4])
TARGET_BEHIND = np.array([-3.8, 0.0, 1.0])


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
        self.launch = None

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
        self.permutation = self.np_random.permutation(len(self.brain.feature_ids))
        side = float(self.np_random.choice([-1, 1]))
        self.start = self.plant.pos[0].copy()
        if self.task == "looming":
            # Obstacle waits ahead, then flies at the drone's start position.
            origin = np.array(
                [
                    2.5,
                    side * self.np_random.uniform(0.05, 0.3),
                    1.0 + self.np_random.uniform(-0.1, 0.1),
                ]
            )
            direction = self.start - origin
            self.launch = {
                "side": side,
                "origin": origin,
                "direction": direction / np.linalg.norm(direction),
                "delay": float(self.np_random.uniform(0.4, 2.0)),
                "speed": float(self.np_random.uniform(0.8, 1.2)),
            }
            self.plant.set_objects(target=TARGET_BEHIND, obstacle=origin)
        else:
            self.launch = None
            self.plant.set_objects(
                target=[2, side * self.np_random.uniform(0.6, 1.5), 1],
                obstacle=[2, -side, 1],
            )
        if self.ablation == "sensory":
            self.brain.silence_sensors()
        self.brain.sense(self.plant.camera())
        # Deterministic neural settling, no hidden body time advancement.
        self.brain.step(40)
        info = self.info()
        self.previous_bearing = abs(info["bearing"])
        return self.observe(), info

    def launched(self):
        return (
            self.launch is not None
            and self.frames * FRAME_SECONDS >= self.launch["delay"]
        )

    def _move_obstacle(self):
        if not self.launched():
            return
        travelled = (self.frames * FRAME_SECONDS - self.launch["delay"]) * self.launch[
            "speed"
        ]
        position = self.launch["origin"] + self.launch["direction"] * travelled
        if position[0] < -1.5:
            position = OBSTACLE_PARK
        self.plant.set_objects(obstacle=position)

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("action must contain four finite values")
        self.command = np.clip(action, -1, 1) * LIMITS
        self._move_obstacle()
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
        bearing = target_bearing(pos, self.plant.rpy[0, 2], self.plant.target)
        reward = 1.0 - 4.0 * (pos[2] - 1.0) ** 2 - 0.05 * float(np.dot(action, action))
        if self.task == "visual":
            # Potential-based turning progress: sums to 10 * total bearing reduction.
            reward += 10 * (self.previous_bearing - abs(bearing))
            reward += 0.5 * np.cos(bearing) - 0.2 * np.linalg.norm(delta[:2])
        self.previous_bearing = abs(bearing)
        if self.task == "looming":
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
            self.frames >= HORIZON_FRAMES[self.task],
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
            "collision": bool(self.plant.data.ncon > 0),
            "obstacle_distance": float(np.linalg.norm(pos - self.plant.obstacle)),
            "obstacle_side": self.launch["side"] if self.launch else 0.0,
            "launched": self.launched(),
            "displacement": float(np.linalg.norm(pos[:2] - self.start[:2])),
        }

    def close(self):
        self.plant.close()
