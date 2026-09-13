"""Public SI-unit boundary types for simulator and eventual onboard transport."""

from typing import TypedDict


class MotionCommand(TypedDict):
    # Body +X forward, +Y left, +Z up, yaw counterclockwise about +Z.
    velocity_body_m_s: list[float]
    yaw_rate_rad_s: float


class MotorState(TypedDict):
    # Index positions: (+x,+y), (-x,+y), (-x,-y), (+x,-y).
    commanded_rpm: list[float]
    actual_rpm: list[float]
    rotor_phase: list[float]


class TelemetryFrame(TypedDict):
    version: int
    seq: int
    episode: int
    tick: int
    physics_tick: int
    time: float
    paused: bool
    activity: list[float]
    command: list[float]  # [vx,vy,vz,yaw_rate], SI, body frame.
