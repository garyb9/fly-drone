"""Pinned upstream force model; project-owned cameras, motor lag and scheduling."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from multi_drone_mujoco.control.pid_control import PIDControl
from multi_drone_mujoco.envs.base_aviary import BaseAviary, _generate_aviary_xml
from multi_drone_mujoco.utils.enums import DroneModel

LIMITS = np.array([0.4, 0.4, 0.2, 0.8])


class DronePlant(BaseAviary):
    def __init__(self, vision=True, motor_tau=0.025, drag=True, eye_splay=0.75):
        self.eye_splay = float(eye_splay)
        self.motor_tau = float(motor_tau)
        if self.motor_tau < 0:
            raise ValueError("motor_tau must be nonnegative")
        self.use_drag = drag
        super().__init__(
            drone_model=DroneModel.CF2X,
            sim_freq=1000,
            ctrl_freq=200,
            initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
        )
        xml = ET.fromstring(
            _generate_aviary_xml(
                1, DroneModel.CF2X, self.INIT_XYZS, self.INIT_RPYS, timestep=0.001
            )
        )
        world = xml.find("worldbody")
        body = world.find("body[@name='drone0']")
        for name, angle in [("eye_l", self.eye_splay), ("eye_r", -self.eye_splay)]:
            # Camera looks down local -Z, local Y is world up. Body +X forward, +Y left.
            ET.SubElement(
                body,
                "camera",
                name=name,
                pos="0.035 0 0.008",
                fovy="75",
                xyaxes=f"{math.sin(angle)} {-math.cos(angle)} 0 0 0 1",
            )
        ET.SubElement(
            world,
            "geom",
            name="target",
            type="sphere",
            pos="2 1 1",
            size="0.22",
            rgba="1 0.95 0.55 1",
            contype="0",
            conaffinity="0",
        )
        ET.SubElement(
            world,
            "geom",
            name="obstacle",
            type="sphere",
            pos="2 -1 1",
            size="0.25",
            rgba="0.04 0.06 0.08 1",
        )
        # High contrast walls make lateral visual steering observable.
        for name, pos, size in [
            ("back", "4 0 2", "0.1 4 2"),
            ("left", "0 4 2", "4 0.1 2"),
            ("right", "0 -4 2", "4 0.1 2"),
        ]:
            ET.SubElement(
                world,
                "geom",
                name=name,
                type="box",
                pos=pos,
                size=size,
                rgba="0.22 0.28 0.35 1",
            )
        self.model = mujoco.MjModel.from_xml_string(
            ET.tostring(xml, encoding="unicode")
        )
        # Eyes render at 64x48 on CPU GL: a full-size MSAA buffer and floor reflection
        # cost ~5x more than the image itself. Changing these alters pixels (encoder id).
        self.model.vis.global_.offwidth = 64
        self.model.vis.global_.offheight = 48
        self.model.vis.quality.offsamples = 0
        self.data = mujoco.MjData(self.model)
        self.controller = PIDControl(self)
        self.commanded = np.zeros(4)
        self.actual = np.zeros(4)
        self.phase = np.zeros(4)
        self.target = np.array([2.0, 1.0, 1.0])
        self.obstacle = np.array([2.0, -1.0, 1.0])
        self.images = np.zeros((2, 48, 64, 3), dtype=np.uint8)
        self.vision = vision
        self.renderer = None
        self.hold = np.array([0.0, 0.0, 1.0])
        self.yaw_target = 0.0
        self.reset()

    def reset(self, seed=None, options=None):
        obs = super().reset(seed=seed, options=options)
        if hasattr(self, "controller"):
            self.controller.reset()
            self.commanded[:] = self.HOVER_RPM
            self.actual[:] = self.HOVER_RPM
            self.phase[:] = 0
            self.hold = self.pos[0].copy()
            self.yaw_target = float(self.rpy[0, 2])
        return obs

    def set_objects(self, target=None, obstacle=None):
        for name, value in [("target", target), ("obstacle", obstacle)]:
            if value is None:
                continue
            v = np.asarray(value, dtype=float)
            if (
                v.shape != (3,)
                or not np.isfinite(v).all()
                or np.any(np.abs(v) > 4)
                or v[2] < 0.3
            ):
                raise ValueError(
                    "object position must be finite, inside room, z >= 0.3"
                )
            setattr(self, name, v.copy())
            self.model.geom_pos[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            ] = v
        mujoco.mj_forward(self.model, self.data)

    def camera(self):
        if not self.vision:
            return self.images
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=48, width=64)
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        for i, name in enumerate(["eye_l", "eye_r"]):
            self.renderer.update_scene(self.data, camera=name)
            self.images[i] = self.renderer.render()
        return self.images

    def advance(self, command):
        command = np.asarray(command, dtype=float)
        if command.shape != (4,) or not np.isfinite(command).all():
            raise ValueError("four finite motion commands required")
        command = np.clip(command, -LIMITS, LIMITS)
        yaw = self.rpy[0, 2]
        c, s = np.cos(yaw), np.sin(yaw)
        velocity = np.array(
            [
                c * command[0] - s * command[1],
                s * command[0] + c * command[1],
                command[2],
            ]
        )
        self.hold += velocity * 0.005
        self.hold = np.clip(self.hold, [-3, -3, 0.4], [3, 3, 2.5])
        self.yaw_target += command[3] * 0.005
        rpm, _, _ = self.controller.computeControl(
            0.005,
            self.pos[0],
            self.quat[0],
            self.vel[0],
            self.ang_v[0],
            self.hold,
            np.array([0.0, 0.0, self.yaw_target]),
            velocity,
        )
        self.advance_rpm(rpm)

    def advance_rpm(self, rpm):
        rpm = np.asarray(rpm, dtype=float)
        if rpm.shape != (4,) or not np.isfinite(rpm).all():
            raise ValueError("four finite rotor RPMs required")
        self.commanded = np.clip(rpm, 0, self.MAX_RPM)
        alpha = 1 if self.motor_tau == 0 else -np.expm1(-0.001 / self.motor_tau)
        for _ in range(5):
            self.actual += alpha * (self.commanded - self.actual)
            self._physics(self.actual, 0)
            if self.use_drag:
                self._drag(self.actual, 0)
            mujoco.mj_step(self.model, self.data)
            self._updateAndStoreKinematicInformation()
            self.phase = (
                self.phase
                + self.actual * (2 * np.pi / 60) * 0.001 * np.array([1, -1, 1, -1])
            ) % (2 * np.pi)
            self.step_counter += 1
        self.last_clipped_action = self.actual[None, :].copy()

    def state(self):
        return {
            "position": self.pos[0].tolist(),
            "quaternion": self.quat[0].tolist(),
            "velocity": self.vel[0].tolist(),
            "commanded_rpm": self.commanded.tolist(),
            "actual_rpm": self.actual.tolist(),
            "rotor_phase": self.phase.tolist(),
            "target": self.target.tolist(),
            "obstacle": self.obstacle.tolist(),
        }

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None
        super().close()
