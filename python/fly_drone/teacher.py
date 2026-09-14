"""Composite free-roam teacher: privileged geometry, gated on what the eyes can see.

Labels may use simulator state only for things visible to the cameras (inside the
binocular field, unoccluded, within the measured cue range). Anything else would ask the
decoder to read information that never entered the connectome.
"""

import mujoco
import numpy as np

from . import arena
from .env import FIELD_HALF_ANGLE, FRAME_SECONDS

# Measured with roam-feasibility (encoder v4, ringed pillars, flat lighting).
PILLAR_LOOM_RANGE = 1.2
WALL_LOOM_RANGE = 2.0
THREAT_LOOM_RANGE = 2.0
BEACON_RANGE = 12.0
AVOID_CONE = 0.61  # +-35 deg
EXPLORE_FORWARD = 0.8
# Bounds and pacing for the explore drive's yaw cast (see `_explore_action`): a fixed
# constant here was cloned onto every decoder as a scripted forward-drift-plus-turn,
# reading as "flying on its own" rather than reacting to anything (raised 2026-09-14).
EXPLORE_YAW_RANGE = 0.35
EXPLORE_CHANGE_SECONDS = (1.5, 3.5)
DRIVES = ("threat", "avoid", "beacon", "explore")


def _explore_action(env):
    """Forward cruise with a slowly-varying, randomized yaw — a cast pattern rather than
    a fixed constant, so idle flight looks like undirected search and no two flights
    trace the same arc. Still fully privileged/scripted (this is a label, not sensing);
    the point is only to stop baking one specific scripted turn into every rollout."""
    state = env.roam["explore"]
    t = env.frames * FRAME_SECONDS
    if t >= state["next_change"]:
        state["yaw_bias"] = float(
            env.np_random.uniform(-EXPLORE_YAW_RANGE, EXPLORE_YAW_RANGE)
        )
        state["next_change"] = t + float(env.np_random.uniform(*EXPLORE_CHANGE_SECONDS))
    return np.array([EXPLORE_FORWARD, 0.0, 0.0, state["yaw_bias"]])


def _sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-x)))


def visible(env, point, geom_name):
    """Inside the field of view and the first thing a ray from the eye hits."""
    plant = env.plant
    pos = plant.pos[0]
    if abs(arena.bearing_to(pos, plant.rpy[0, 2], point)) > FIELD_HALF_ANGLE:
        return False
    eye = pos + np.array([0.0, 0.0, 0.008])
    ray = np.asarray(point, dtype=float) - eye
    distance = float(np.linalg.norm(ray))
    geom = np.array([-1], dtype=np.int32)
    body = mujoco.mj_name2id(plant.model, mujoco.mjtObj.mjOBJ_BODY, "drone0")
    target = mujoco.mj_name2id(plant.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    # mj_ray skips alpha-0 geoms: ghost objects are invisible here too.
    hit = mujoco.mj_ray(
        plant.model, plant.data, eye, ray / distance, None, 1, body, geom
    )
    return bool(geom[0] == target or (hit < 0 and geom_name != "obstacle"))


def nearest_obstacle(env):
    """Closest pillar or wall surface inside the forward cone: (distance, bearing, kind)."""
    plant, spec = env.plant, env.spec
    pos, yaw = plant.pos[0], plant.rpy[0, 2]
    best = (np.inf, 0.0, None)
    for centre in plant.pillars:
        bearing = arena.bearing_to(pos, yaw, centre)
        distance = float(np.linalg.norm(centre - pos[:2])) - spec.pillar_radius
        if abs(bearing) < AVOID_CONE and distance < best[0]:
            best = (distance, bearing, "pillar")
    heading = np.array([np.cos(yaw), np.sin(yaw)])
    # d(heading)/d(yaw): turning left (+yaw) moves the heading this way.
    left_turn = np.array([-np.sin(yaw), np.cos(yaw)])
    for axis in (0, 1):
        if abs(heading[axis]) < 1e-6:
            continue
        normal = np.zeros(2)
        normal[axis] = np.sign(heading[axis])
        # Ray distance along the heading to the wall this heading points at.
        along = float((spec.half_size - pos[axis] * normal[axis]) / abs(heading[axis]))
        if along >= best[0]:
            continue
        # Positive when a left turn steers further into the wall: treat as "on the left".
        into = float(left_turn @ normal)
        best = (along, into if abs(into) > 1e-3 else 1e-3, "wall")
    return best


def teacher_action(env):
    """Normalised action and dominant drive for the current free-roam state."""
    plant = env.plant
    pos, yaw = plant.pos[0], plant.rpy[0, 2]
    roam = env.roam
    # Explore: identical wherever an unseen beacon is (the cast's yaw bias only depends
    # on elapsed time, not on anything positional).
    action = _explore_action(env)
    drive = "explore"

    if (
        env.beacon_visible()
        and np.linalg.norm(plant.target[:2] - pos[:2]) < BEACON_RANGE
    ):
        bearing = arena.bearing_to(pos, yaw, plant.target)
        facing = max(0.0, 1.0 - abs(bearing) / 0.35)
        distance = float(np.linalg.norm(plant.target[:2] - pos[:2]))
        forward = float(np.clip(distance / 1.0, 0.3, 1.0)) * facing
        action = np.array(
            [forward, 0.0, 0.0, float(np.clip(1.5 * bearing / 0.8, -1, 1))]
        )
        drive = "beacon"

    distance, bearing, kind = nearest_obstacle(env)
    if kind is not None:
        reach = PILLAR_LOOM_RANGE if kind == "pillar" else WALL_LOOM_RANGE
        weight = _sigmoid((reach - distance) / 0.15)
        if weight > 1e-3:
            # Turn away from the side the obstacle is on (+yaw is left).
            away = -1.0 if bearing > 0 else 1.0
            forward = float(np.clip((distance - 0.45) / 0.8, -0.3, 0.6))
            avoid = np.array([forward, 0.0, 0.0, away])
            action = weight * avoid + (1 - weight) * action
            if weight > 0.5:
                drive = "avoid"

    threat = roam["threat"] if roam else None
    if threat is not None:
        gap = float(np.linalg.norm(plant.obstacle - pos))
        if gap < THREAT_LOOM_RANGE + 0.5 and visible(env, plant.obstacle, "obstacle"):
            weight = _sigmoid((THREAT_LOOM_RANGE - gap) / 0.15)
            if "evade_dir" not in threat:
                # Commit once per threat: re-deciding from the bearing sign every frame
                # dithers on head-on shots (55% dodged) and never clears the path.
                side = arena.bearing_to(pos, yaw, plant.obstacle)
                threat["evade_dir"] = -1.0 if side > 0 else 1.0
                # Vertical is the primary escape axis: roam-step-response measured it
                # reaching 80% of a commanded step in ~0.69s vs ~1.48s for lateral (direct
                # thrust vs. tilt-dependent), and the threat is typically only ~1.9m away
                # when first evaded — lateral alone was too slow to matter (docs/results/
                # roam-step-response.json). Climb by default; dive only near the ceiling.
                threat["evade_vertical"] = (
                    -1.0 if pos[2] > env.spec.altitude[1] - 0.5 else 1.0
                )
            # Brake while climbing/diving; lateral and vertical are independent axes, so
            # commanding both at full authority costs nothing (each is clipped to its own
            # arena limit downstream, not a shared budget).
            evade = np.array([-0.5, threat["evade_dir"], threat["evade_vertical"], 0.0])
            action = weight * evade + (1 - weight) * action
            if weight > 0.5:
                drive = "threat"
    return np.clip(action, -1, 1), drive
