"""P1 (C1): declared ascending/proprioceptive feedback for the frozen connectome.

A real fly closes the loop on its own body: halteres sense rotation, the optic
lobes sense flow, and leg chordotonal/campaniform organs sense joint angle and
load. This module turns the *simulated body's* state into declared currents onto
a small set of annotated input cells, so the brain is told what the body is
doing. It never decides anything: the encoding is a fixed, declared map and the
wiring is untouched.

Design: ``docs/superpowers/specs/2026-09-19-body-agnostic-fidelity-cyborg-design.md``
§5. The roles and their cells live in the bundle's ``feedback-mappings.json``;
every confidence label there is a claim about the *cell mapping*, not about the
drone analogue.

The three channels, and the signal each carries:

- **haltere_l/r** -> the named ascending candidates (AN07B037_a/b, AN06B009).
  Body-frame yaw and roll rates from the gyro, split by sign onto the two sides.
- **flow_l/r** -> a fixed subsample of T4/T5. Mean absolute luma change between
  consecutive rendered frames of each eye, a coarse flow proxy.
- **proprio_lf/rf/lh/rh** -> chordotonal/campaniform cells. Rotor lag
  ``(commanded - actual) / MAX_RPM``, one rotor per limb (a declared,
  arbitrary rotor-to-limb convention; a quadrotor has no legs).

All currents are clipped to ``[0, 2]``, the range the runtime injects.
"""

import json
from pathlib import Path

import numpy as np

from .brain import luma_u8

# Declared encoding constants. Fixed at build time; not fitted to anything.
HALTERE_BASE = 0.1
HALTERE_GAIN = 1.0
HALTERE_ROLL_WEIGHT = 0.5
FLOW_BASE = 0.0
FLOW_GAIN = 4.0
PROPRIO_BASE = 0.1
PROPRIO_GAIN = 1.0
# Rotor index -> body-mappings limb. Declared convention, not anatomy.
ROTOR_LIMBS = ("lf", "rf", "lh", "rh")

DEFAULT_MAPPINGS = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "results"
    / "feedback"
    / "feedback-mappings.json"
)


def load_mappings(path=DEFAULT_MAPPINGS):
    """The declared role -> cell table; validated against the bundle it is used with."""
    payload = json.loads(Path(path).read_text())
    roles = payload["roles"]
    if not roles:
        raise ValueError("feedback mappings declare no roles")
    return payload


def body_rates(quaternion, angular_velocity):
    """Body-frame angular rate ``(p, q, r)`` rad/s from a world-frame gyro.

    ``quaternion`` is MuJoCo ``(w, x, y, z)``; the rotation matrix maps body to
    world, so its transpose maps the world-frame rate back to the body.
    """
    w, x, y, z = (float(v) for v in quaternion)
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return rotation.T @ np.asarray(angular_velocity, dtype=float)


def _relu(value):
    return max(0.0, float(value))


def haltere_currents(quaternion, angular_velocity):
    """Left/right descending rate onto the ascending candidates (declared)."""
    _, roll, yaw = body_rates(quaternion, angular_velocity)
    left = HALTERE_BASE + HALTERE_GAIN * (
        _relu(yaw) + HALTERE_ROLL_WEIGHT * _relu(-roll)
    )
    right = HALTERE_BASE + HALTERE_GAIN * (
        _relu(-yaw) + HALTERE_ROLL_WEIGHT * _relu(roll)
    )
    return {"haltere_l": left, "haltere_r": right}


def flow_currents(images, previous):
    """Per-eye mean absolute luma change between frames, scaled to a current.

    Returns ``(currents, luma)`` so the caller can keep ``luma`` as the next
    frame's reference. A missing previous frame yields resting current.
    """
    luma = luma_u8(images)
    if previous is None:
        change = np.zeros(2, dtype=float)
    else:
        change = (
            np.abs(luma.astype(np.int16) - previous.astype(np.int16)).mean(axis=(1, 2))
            / 255.0
        )
    return (
        {
            "flow_l": FLOW_BASE + FLOW_GAIN * float(change[0]),
            "flow_r": FLOW_BASE + FLOW_GAIN * float(change[1]),
        },
        luma,
    )


def proprio_currents(commanded_rpm, actual_rpm, max_rpm):
    """Rotor lag per limb, the drone's actuator-proprioception analogue."""
    commanded = np.asarray(commanded_rpm, dtype=float)
    actual = np.asarray(actual_rpm, dtype=float)
    lag = np.clip((commanded - actual) / max(float(max_rpm), 1e-9), 0.0, 1.0)
    return {
        f"proprio_{limb}": PROPRIO_BASE + PROPRIO_GAIN * float(lag[i])
        for i, limb in enumerate(ROTOR_LIMBS)
    }


class FeedbackState:
    """Carries the previous frame and turns body state into declared currents."""

    def __init__(self, plant):
        self.plant = plant
        self.previous_luma = None
        self.flow = {"flow_l": FLOW_BASE, "flow_r": FLOW_BASE}

    def reset(self):
        self.previous_luma = None
        self.flow = {"flow_l": FLOW_BASE, "flow_r": FLOW_BASE}

    def observe_frame(self, images):
        """Advance the optic-flow channel once per rendered environment frame."""
        self.flow, self.previous_luma = flow_currents(images, self.previous_luma)
        return self.flow

    def currents(self):
        """The full role -> current map for one body instant (holds flow between frames)."""
        out = {}
        state = self.plant.state()
        out.update(haltere_currents(state["quaternion"], state["angular_velocity"]))
        out.update(self.flow)
        out.update(
            proprio_currents(
                state["commanded_rpm"], state["actual_rpm"], self.plant.MAX_RPM
            )
        )
        return {k: float(np.clip(v, 0.0, 2.0)) for k, v in out.items()}
