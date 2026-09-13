import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .brain import BrainRuntime
from .plant import LIMITS, DronePlant


def target_bearing(pos, yaw, target):
    """Signed target angle relative to heading, wrapped to [-pi, pi]."""
    delta = np.asarray(target, dtype=float) - np.asarray(pos, dtype=float)
    angle = np.arctan2(delta[1], delta[0]) - yaw
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


class ConnectomeEnv(gym.Env):
    """PPO observes frozen neural activity; action is normalized motion intent."""

    metadata = {}

    def __init__(self, task="visual", vision=True, brain=None, ablation="none"):
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
        side = self.np_random.choice([-1, 1])
        self.plant.set_objects(
            target=[2, float(side) * self.np_random.uniform(0.6, 1.5), 1],
            obstacle=[2, -float(side), 1],
        )
        if self.ablation == "sensory":
            self.brain.silence_sensors()
        self.brain.sense(self.plant.camera())
        # Deterministic neural settling, no hidden body time advancement.
        self.brain.step(40)
        return self.observe(), self.info()

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("action must contain four finite values")
        self.command = np.clip(action, -1, 1) * LIMITS
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
        if self.task != "hover":
            reward += 2 * np.cos(bearing) - 0.2 * np.linalg.norm(delta[:2])
        if self.task == "looming":
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
            self.frames >= 750,
            self.info(),
        )

    def info(self):
        return {
            "tick": self.brain.tick,
            "physics_tick": self.plant.step_counter,
            "time": self.plant.data.time,
            "state": self.plant.state(),
            "bearing": target_bearing(
                self.plant.pos[0], self.plant.rpy[0, 2], self.plant.target
            ),
        }

    def close(self):
        self.plant.close()
