"""Port of fly-playground wrench/integrate equations, Y-up illustrative units.
Noise and movement assistance omitted; neural power/steering and escape retained.
"""

import mujoco
import numpy as np


class FlyMirror:
    def __init__(self):
        self.reset()

    def reset(self):
        self.position = np.array([0.0, 1.0, 0.0])
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.velocity = np.zeros(3)
        self.angular = np.zeros(3)
        self.armed = True
        self.lockout = 0.0
        self.ticks = 0

    def step(self, r, dt=0.005):
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, self.quaternion)
        rot = rot.reshape(3, 3)
        forward = rot[:, 0]
        up = rot[:, 1]
        s = (r.get("power_l", 0) + r.get("power_r", 0)) / 2
        a = r.get("steer_l", 0) - r.get("steer_r", 0)
        escape = r.get("escape", 0)
        self.lockout = max(0.0, self.lockout - dt)
        if self.armed and escape >= 0.5:
            direction = up + forward
            self.velocity += 9 * direction / np.linalg.norm(direction)
            self.armed = False
            self.lockout = 0.35
        if not self.armed and escape < 0.35:
            self.armed = True
        force = np.zeros(3)
        torque = np.zeros(3)
        if self.lockout == 0:
            axis = up + 0.12 * forward
            force = axis / np.linalg.norm(axis) * 9.81 * max(0, s) / 0.5
            torque = forward * 2.2 * a + up * 1.4 * a
        self.velocity = (
            self.velocity + (force + np.array([0.0, -9.81, 0.0])) * dt
        ) * np.exp(-1.4 * dt)
        self.position += self.velocity * dt
        self.angular = (self.angular + torque * dt / 0.05) * np.exp(-6 * dt)
        # mj quaternion integration takes local angular velocity.
        mujoco.mju_quatIntegrate(self.quaternion, rot.T @ self.angular, dt)
        for i, (lo, hi) in enumerate([(-3, 3), (0.25, 3), (-3, 3)]):
            if self.position[i] < lo:
                self.position[i] = lo
                self.velocity[i] = abs(self.velocity[i]) * 0.35
            if self.position[i] > hi:
                self.position[i] = hi
                self.velocity[i] = -abs(self.velocity[i]) * 0.35
        self.ticks += 1

    def state(self):
        return {
            "position": self.position.tolist(),
            "quaternion": self.quaternion.tolist(),
            "ticks": self.ticks,
        }
