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

from . import arena
from .arena import PARK_Z

LIMITS = np.array([0.4, 0.4, 0.2, 0.8])
LEGACY_HOLD = (np.array([-3.0, -3.0, 0.4]), np.array([3.0, 3.0, 2.5]))


def _grey(luma):
    return f"{luma} {luma} {luma} 1"


class DronePlant(BaseAviary):
    def __init__(
        self, vision=True, motor_tau=0.025, drag=True, eye_splay=0.75, arena=None
    ):
        self.eye_splay = float(eye_splay)
        # Camera extrinsics/intrinsics, kept as the authored strings so build_xml() can
        # interpolate them without changing the generated MJCF, and exposed as numbers
        # through cameras() for the viewer's FOV cones.
        self.eye_pos = "0.035 0 0.008"
        self.eye_fovy = "75"
        self.motor_tau = float(motor_tau)
        if self.motor_tau < 0:
            raise ValueError("motor_tau must be nonnegative")
        self.use_drag = drag
        self.arena = arena
        super().__init__(
            drone_model=DroneModel.CF2X,
            sim_freq=1000,
            ctrl_freq=200,
            initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
        )
        if arena is None:
            self.limits = LIMITS.copy()
            self.hold_low, self.hold_high = (b.copy() for b in LEGACY_HOLD)
            self.object_range = 4.0
        else:
            self.limits = np.array(arena.limits, dtype=float)
            self.hold_low, self.hold_high = arena.hold_low, arena.hold_high
            self.object_range = arena.half_size
        self.model = mujoco.MjModel.from_xml_string(self.build_xml())
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
        self.pillars = np.zeros((0, 2))
        self.ghost = False
        self.images = np.zeros((2, 48, 64, 3), dtype=np.uint8)
        self.vision = vision
        self.renderer = None
        self.hold = np.array([0.0, 0.0, 1.0])
        self.yaw_target = 0.0
        self.reset()

    def build_xml(self):
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
                pos=self.eye_pos,
                fovy=self.eye_fovy,
                xyaxes=f"{math.sin(angle)} {-math.cos(angle)} 0 0 0 1",
            )
        if self.arena is None:
            self._legacy_room(world)
        else:
            self._arena_room(xml, world, self.arena)
        return ET.tostring(xml, encoding="unicode")

    def _legacy_room(self, world):
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
        # Mocap body: world-body geoms moved via model.geom_pos keep their compile-time
        # collision bounds, so a relocated static obstacle would never register contact.
        obstacle = ET.SubElement(
            world, "body", name="obstacle_body", mocap="true", pos="2 -1 1"
        )
        ET.SubElement(
            obstacle,
            "geom",
            name="obstacle",
            type="sphere",
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

    def _arena_room(self, xml, world, spec):
        # Uniform floor: the upstream checker has tiles at the dark threshold and bright
        # edge marks, which would turn forward flight into false looming.
        texture = xml.find("asset/texture[@name='groundplane']")
        texture.attrib.update(
            builtin="flat",
            rgb1=_grey(spec.floor_luma)[:-2],
            rgb2=_grey(spec.floor_luma)[:-2],
        )
        texture.attrib.pop("mark", None)
        texture.attrib.pop("markrgb", None)
        # Camera near clip is znear * extent. Bodies parked under the floor would inflate
        # the computed extent and blind the eyes within ~0.5-1.8 m; pin it to the legacy
        # room's 17 m (near clip 0.17 m).
        ET.SubElement(xml, "statistic", extent="17", center="0 0 1")
        # Flat lighting: with 0.3 ambient / 0.6 diffuse, grey walls and pillars seen at
        # grazing angles render below the 0.18 dark threshold, and every turn read as
        # looming (59% of frames with nothing dark in the room; 0% with this setting).
        xml.find("visual/headlight").attrib.update(
            ambient="0.6 0.6 0.6", diffuse="0.3 0.3 0.3"
        )
        xml.find("asset/material[@name='groundplane']").set("reflectance", "0")
        ET.SubElement(
            xml.find("asset"),
            "material",
            name="beacon",
            rgba="1 1 0.9 1",
            emission="1",
            specular="0",
        )
        ET.SubElement(
            world,
            "geom",
            name="target",
            type="sphere",
            pos="0 3 1",
            size=str(spec.beacon_radius),
            material="beacon",
            contype="0",
            conaffinity="0",
        )
        obstacle = ET.SubElement(
            world, "body", name="obstacle_body", mocap="true", pos=f"0 0 {PARK_Z}"
        )
        ET.SubElement(
            obstacle,
            "geom",
            name="obstacle",
            type="sphere",
            size=str(spec.threat_radius),
            rgba="0.04 0.06 0.08 1",
        )
        h, t = spec.half_size, 0.1
        lo, hi = spec.band
        for name, x, y, sx, sy in [
            ("wall_px", h + t, 0, t, h + 2 * t),
            ("wall_nx", -h - t, 0, t, h + 2 * t),
            ("wall_py", 0, h + t, h + 2 * t, t),
            ("wall_ny", 0, -h - t, h + 2 * t, t),
        ]:
            ET.SubElement(
                world,
                "geom",
                name=name,
                type="box",
                pos=f"{x} {y} {spec.wall_height / 2}",
                size=f"{sx} {sy} {spec.wall_height / 2}",
                rgba=_grey(spec.wall_luma),
            )
            # Dark band at eye height: approaching a wall expands a dark region.
            inset = 0.01
            bx = x - np.sign(x) * (t + inset) if x else 0
            by = y - np.sign(y) * (t + inset) if y else 0
            ET.SubElement(
                world,
                "geom",
                name=name + "_band",
                type="box",
                pos=f"{bx} {by} {(lo + hi) / 2}",
                size=f"{max(sx, inset) if not x else inset} "
                f"{max(sy, inset) if not y else inset} {(hi - lo) / 2}",
                rgba="0.03 0.03 0.04 1",
                contype="0",
                conaffinity="0",
            )
        for i in range(spec.pillar_slots):
            pillar = ET.SubElement(
                world,
                "body",
                name=f"pillar_{i}",
                mocap="true",
                pos=f"0 0 {PARK_Z}",
            )
            r = arena.pillar_radius(spec, i)
            dark = "0.03 0.03 0.04 1"
            ET.SubElement(
                pillar,
                "geom",
                name=f"pillar_{i}",
                type="cylinder",
                size=f"{r} {spec.pillar_height / 2}",
                rgba=dark if spec.pillar_ring is None else _grey(spec.pillar_luma),
            )
            if spec.pillar_ring is not None:
                # Visual only; the body offset puts the ring centre at z = 1 m.
                ET.SubElement(
                    pillar,
                    "geom",
                    name=f"pillar_{i}_ring",
                    type="cylinder",
                    pos=f"0 0 {1.0 - spec.pillar_height / 2}",
                    size=f"{r + 0.005} {spec.pillar_ring / 2}",
                    rgba=dark,
                    contype="0",
                    conaffinity="0",
                )

    def reset(self, seed=None, options=None):
        obs = super().reset(seed=seed, options=options)
        if hasattr(self, "controller"):
            self.controller.reset()
            self.commanded[:] = self.HOVER_RPM
            self.actual[:] = self.HOVER_RPM
            self.phase[:] = 0
            self.hold = self.pos[0].copy()
            self.yaw_target = float(self.rpy[0, 2])
            # Restores existing state; a free-roam threat may be parked below the floor.
            self.set_objects(
                target=self.target, obstacle=self.obstacle, park_obstacle=True
            )
        return obs

    def _body_mocap(self, name):
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.model.body_mocapid[body]

    def set_objects(self, target=None, obstacle=None, park_obstacle=False):
        for name, value in [("target", target), ("obstacle", obstacle)]:
            if value is None:
                continue
            v = np.asarray(value, dtype=float)
            parked = name == "obstacle" and park_obstacle
            if (
                v.shape != (3,)
                or not np.isfinite(v).all()
                or (
                    not parked and (np.any(np.abs(v) > self.object_range) or v[2] < 0.3)
                )
            ):
                raise ValueError(
                    "object position must be finite, inside room, z >= 0.3"
                )
            setattr(self, name, v.copy())
            if name == "obstacle":
                self.data.mocap_pos[self._body_mocap("obstacle_body")] = v
            else:
                self.model.geom_pos[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                ] = v
        mujoco.mj_forward(self.model, self.data)

    def set_pillars(self, centres):
        if self.arena is None:
            raise ValueError("pillars exist only in the free-roam arena")
        centres = np.asarray(centres, dtype=float).reshape(-1, 2)
        if len(centres) > self.arena.pillar_slots:
            raise ValueError("too many pillars")
        z = self.arena.pillar_height / 2
        for i in range(self.arena.pillar_slots):
            slot = self._body_mocap(f"pillar_{i}")
            if i < len(centres):
                self.data.mocap_pos[slot] = [centres[i, 0], centres[i, 1], z]
            else:
                self.data.mocap_pos[slot] = [50 + 2 * i, 50, PARK_Z]
        self.pillars = centres.copy()
        mujoco.mj_forward(self.model, self.data)

    def set_ghost(self, ghost):
        """Invisible to the eyes but still collidable: a causal control for vision."""
        if self.arena is None:
            raise ValueError("ghost objects exist only in the free-roam arena")
        self.ghost = bool(ghost)
        names = ["obstacle"] + [f"pillar_{i}" for i in range(self.arena.pillar_slots)]
        if self.arena.pillar_ring is not None:
            names += [f"pillar_{i}_ring" for i in range(self.arena.pillar_slots)]
        for name in names:
            geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            self.model.geom_rgba[geom, 3] = 0.0 if self.ghost else 1.0

    def teleport(self, position, yaw=None):
        """Place the body at rest (respawn); controller and motors restart from hover."""
        position = np.asarray(position, dtype=float)
        self.data.qpos[0:3] = position
        if yaw is not None:
            self.data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self._updateAndStoreKinematicInformation()
        self.controller.reset()
        self.commanded[:] = self.HOVER_RPM
        self.actual[:] = self.HOVER_RPM
        self.hold = self.pos[0].copy()
        self.yaw_target = float(self.rpy[0, 2])

    def cameras(self):
        """Eye extrinsics/intrinsics for the viewer, so its FOV cones match the sim."""
        return {
            "count": 2,
            "splay": self.eye_splay,
            "fovy_deg": float(self.eye_fovy),
            "pos": [float(v) for v in self.eye_pos.split()],
        }

    def room(self):
        """Static room geometry for the viewer, read from the compiled model (Z-up m)."""
        m = self.model
        walls, bands, pillars = [], [], []
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            box = {
                "center": m.geom_pos[g].tolist(),
                "half_extents": m.geom_size[g].tolist(),
            }
            if name.endswith("_band"):
                bands.append(box)
            elif name.startswith("wall_") or name in ("back", "left", "right"):
                walls.append(box)
        for i, (x, y) in enumerate(self.pillars):
            geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"pillar_{i}")
            radius, half_height = m.geom_size[geom][:2]
            pillars.append(
                {
                    "center": [float(x), float(y), float(half_height)],
                    "radius": float(radius),
                    "height": float(2 * half_height),
                    "ring_height": self.arena.pillar_ring,
                }
            )

        def radius(name):
            return float(
                m.geom_size[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)][0]
            )

        return {
            "kind": "legacy" if self.arena is None else "arena",
            "half_size": None if self.arena is None else self.arena.half_size,
            "walls": walls,
            "bands": bands,
            "pillars": pillars,
            "beacon_radius": radius("target"),
            "threat_radius": radius("obstacle"),
        }

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
        command = np.clip(command, -self.limits, self.limits)
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
        self.hold = np.clip(self.hold, self.hold_low, self.hold_high)
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
            # World-frame rad/s (qvel[3:6] of the drone's free joint; base_aviary.py
            # converts it back to body frame elsewhere, so it's stored world-frame here).
            "angular_velocity": self.ang_v[0].tolist(),
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
