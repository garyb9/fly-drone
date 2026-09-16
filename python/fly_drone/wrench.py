"""Wing-level control math: body wrench <-> rotor RPMs (standalone; additive).

The v6 action spec (``docs/superpowers/specs/2026-09-16-wing-level-action-design.md``)
deepens the decoder's command from a velocity intent to a body wrench
``[roll, pitch, yaw, thrust]``. This module is the pure mixer and reflex rate
loop for that. It does not touch ``plant.py``; the ``advance_wrench`` entry point
and the env wiring land in the W0 window, so the running v5 path is unaffected.

The airframe constants mirror ``docs/control-and-physics.md`` §2-3 and must stay
in step with the pinned simulator.
"""

import math
from dataclasses import dataclass, field

import numpy as np

MASS = 0.027  # kg
ARM = 0.0397  # motor-to-centre, m
KF = 3.16e-10  # N / RPM^2
KM = 7.94e-12  # N*m / RPM^2
G = 9.81  # m/s^2
THRUST_TO_WEIGHT = 2.25

HOVER_THRUST = MASS * G
HOVER_RPM = math.sqrt(HOVER_THRUST / (4.0 * KF))
MAX_RPM = math.sqrt(THRUST_TO_WEIGHT) * HOVER_RPM
MAX_THRUST = 4.0 * KF * MAX_RPM**2  # total, all four rotors
YAW_ARM = KM / KF

_S = ARM / math.sqrt(2.0)
_MIXER = np.array(
    [
        [1.0, 1.0, 1.0, 1.0],
        [_S, _S, -_S, -_S],
        [-_S, _S, _S, -_S],
        [-YAW_ARM, YAW_ARM, -YAW_ARM, YAW_ARM],
    ]
)
_MIXER_INV = np.linalg.inv(_MIXER)


def mixer_matrix():
    return _MIXER.copy()


def mix(thrust, torques):
    """Body ``(thrust, [roll, pitch, yaw] torques)`` -> four rotor RPMs.

    Thrusts are clipped to the actuator envelope, so an infeasible wrench
    saturates instead of producing NaNs.
    """
    wrench = np.array([thrust, *np.asarray(torques, dtype=float)], dtype=float)
    forces = _MIXER_INV @ wrench
    forces = np.clip(forces, 0.0, KF * MAX_RPM**2)
    return np.sqrt(forces / KF)


def rotor_wrench(rpm):
    """Forward mixer: four rotor RPMs -> ``(thrust, torques)``."""
    forces = KF * np.asarray(rpm, dtype=float) ** 2
    wrench = _MIXER @ forces
    return float(wrench[0]), wrench[1:]


def hover_rpm():
    return np.full(4, HOVER_RPM)


@dataclass
class RateController:
    """Reflex body-rate loop: rate error -> torques (P-D).

    Plays the role the fly's VNC/halteres play — the fast stabilisation the
    connectome gets no vestibular input for. Defaults are first-guess gains to be
    calibrated in W0/W1, not final.
    """

    kp: tuple = (2e-3, 2e-3, 1e-3)
    kd: tuple = (5e-4, 5e-4, 2e-4)
    dt: float = 0.005
    _prev: np.ndarray = field(default=None, repr=False)

    def torques(self, rate_error):
        error = np.asarray(rate_error, dtype=float)
        torques = np.asarray(self.kp) * error
        if self._prev is not None:
            torques = torques + np.asarray(self.kd) * (error - self._prev) / self.dt
        self._prev = error
        return torques

    def reset(self):
        self._prev = None


def thrust_for_vertical(vz_target, vz_measured, kp=0.15):
    """Collective thrust to track a vertical velocity (hover + P on error)."""
    return float(
        np.clip(HOVER_THRUST + kp * (vz_target - vz_measured), 0.0, MAX_THRUST)
    )


def intent_to_wrench(intent, gains=(0.4, 0.4, 0.4, 0.4)):
    """Approximate body wrench for a ``[vx, vy, vz, yaw_rate]`` intent.

    A deterministic starting point for the W1 DAgger labels; the roll/pitch signs
    and gains are to be calibrated against the plant, so only the zero map and
    antisymmetry are load-bearing here.
    """
    vx, vy, vz, yaw_rate = (float(v) for v in intent)
    g_roll, g_pitch, g_vert, g_yaw = gains
    torques = np.array([-g_roll * vy, g_pitch * vx, g_yaw * yaw_rate])
    return HOVER_THRUST + g_vert * vz, torques
