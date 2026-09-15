import hashlib

import numpy as np
import pytest
from fly_drone import arena
from fly_drone.arena import ArenaSpec
from fly_drone.brain import BrainRuntime
from fly_drone.env import ConnectomeEnv
from fly_drone.plant import DronePlant

# Accepted visual/looming results were measured in this exact room.
LEGACY_MJCF_SHA256 = "3c829f83964a2dd331727abc7abd93078d8fe85b15a8643f9135cb8e368e9426"


def luma(images):
    return (
        0.2126 * images[..., 0] + 0.7152 * images[..., 1] + 0.0722 * images[..., 2]
    ) / 255


def test_legacy_room_mjcf_unchanged():
    plant = DronePlant(vision=False)
    try:
        digest = hashlib.sha256(plant.build_xml().encode()).hexdigest()
        assert digest == LEGACY_MJCF_SHA256
        assert plant.limits.tolist() == [0.4, 0.4, 0.2, 0.8]
    finally:
        plant.close()


@pytest.mark.parametrize("seed", range(5))
def test_layout_deterministic_spaced_clear_and_connected(seed):
    spec = ArenaSpec()
    a = arena.generate_layout(np.random.default_rng(seed), spec, 3)
    b = arena.generate_layout(np.random.default_rng(seed), spec, 3)
    np.testing.assert_array_equal(a, b)
    assert len(a) == 16
    gaps = [
        np.linalg.norm(p - q) - 2 * spec.pillar_radius
        for i, p in enumerate(a)
        for q in a[i + 1 :]
    ]
    assert min(gaps) >= spec.pillar_spacing - 1e-9
    assert np.linalg.norm(a, axis=1).min() >= spec.spawn_clear + spec.pillar_radius
    assert np.abs(a).max() <= spec.half_size - 1.0
    assert arena.connected(arena.free_mask(spec, a)[2])


@pytest.mark.parametrize("seed", range(5))
def test_layout_with_varied_pillar_radii_stays_spaced_clear_and_connected(seed):
    spec = ArenaSpec(pillar_radii=arena.VARIED_PILLAR_RADII)
    a = arena.generate_layout(np.random.default_rng(seed), spec, 3)
    b = arena.generate_layout(np.random.default_rng(seed), spec, 3)
    np.testing.assert_array_equal(a, b)
    assert len(a) == 16
    radii = [arena.pillar_radius(spec, i) for i in range(len(a))]
    assert len(set(radii)) > 1
    gaps = [
        np.linalg.norm(p - q) - radii[i] - radii[j]
        for i, p in enumerate(a)
        for j, q in enumerate(a[i + 1 :], start=i + 1)
    ]
    assert min(gaps) >= spec.pillar_spacing - 1e-9
    for i, p in enumerate(a):
        assert np.linalg.norm(p) >= spec.spawn_clear + radii[i]
    assert np.abs(a).max() <= spec.half_size - 1.0
    assert arena.connected(arena.free_mask(spec, a)[2])


def test_varied_pillar_radii_render_at_their_own_size():
    spec = ArenaSpec(pillar_radii=arena.VARIED_PILLAR_RADII)
    plant = DronePlant(vision=False, arena=spec)
    try:
        plant.set_pillars([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        room = plant.room()
        radii = [p["radius"] for p in room["pillars"]]
        assert radii == [arena.pillar_radius(spec, i) for i in range(3)]
        assert len(set(radii)) > 1
    finally:
        plant.close()


def test_half_of_beacons_spawn_out_of_view():
    spec = ArenaSpec()
    rng = np.random.default_rng(0)
    pillars = arena.generate_layout(rng, spec, 3)
    pos = np.array([0.0, 0.0, 1.0])
    hidden = []
    for _ in range(200):
        beacon, is_hidden = arena.next_beacon(rng, spec, pillars, pos, 0.3)
        off = abs(arena.bearing_to(pos, 0.3, beacon))
        assert off > arena.OUT_OF_VIEW if is_hidden else off < arena.IN_VIEW
        assert arena.clearance(spec, pillars, beacon) >= spec.beacon_radius + 0.6
        assert 3.0 <= np.linalg.norm(beacon[:2]) <= 12.0
        hidden.append(is_hidden)
    assert 0.35 < np.mean(hidden) < 0.65


def test_room_geometry_describes_active_objects():
    legacy = DronePlant(vision=False)
    plant = DronePlant(vision=False, arena=ArenaSpec())
    try:
        assert legacy.room()["kind"] == "legacy" and len(legacy.room()["walls"]) == 3
        plant.set_pillars([[2.0, 1.0], [-3.0, 4.0]])
        room = plant.room()
        assert room["kind"] == "arena" and room["half_size"] == 8.0
        assert len(room["walls"]) == 4 and len(room["bands"]) == 4
        assert [p["center"][:2] for p in room["pillars"]] == [[2.0, 1.0], [-3.0, 4.0]]
        assert room["pillars"][0]["height"] == 3.0
        assert room["beacon_radius"] == 0.3 and room["threat_radius"] == 0.25
        assert all(b["center"][2] == 1.0 for b in room["bands"])
    finally:
        legacy.close()
        plant.close()


def test_free_roam_env_resets_repeatedly_with_a_parked_threat():
    env = ConnectomeEnv(task="free_roam", vision=False, level=3, respawn=True)
    try:
        for seed in (1, 2, 1):
            _, info = env.reset(seed=seed)
            assert env.plant.obstacle[2] < 0 and info["beacons_collected"] == 0
            for _ in range(3):
                env.step(np.zeros(4))
        env.plant.set_objects(obstacle=[1.0, 1.0, 1.0])
        env.reset(seed=3)
    finally:
        env.close()


def test_threats_lead_a_moving_drone_and_hit_a_still_one():
    spec = ArenaSpec()
    rng = np.random.default_rng(5)
    pos = np.array([0.0, 0.0, 1.0])
    velocity = np.array([0.5, 0.2, 0.0])
    for _ in range(20):
        plan = arena.plan_threat(rng, spec, pos, 0.0, velocity)
        assert 0.9 <= plan["speed"] <= 1.3
        assert 2.9 <= np.linalg.norm(plan["origin"][:2] - pos[:2]) <= 4.1
        time = plan["range"] / plan["speed"]
        shot = plan["origin"] + plan["direction"] * plan["speed"] * time
        np.testing.assert_allclose(shot, pos + velocity * time, atol=1e-6)
        still = arena.plan_threat(rng, spec, pos, 0.0)
        shot = still["origin"] + still["direction"] * still["range"]
        np.testing.assert_allclose(shot, pos, atol=1e-9)


def test_pillar_contact_registers():
    plant = DronePlant(vision=False, arena=ArenaSpec())
    try:
        plant.set_pillars([[1.0, 0.0]])
        plant.teleport([1.0 - 0.32, 0.0, 1.0])
        plant.advance(np.zeros(4))
        assert plant.data.ncon > 0
        plant.set_pillars([])
        plant.teleport([0.0, 0.0, 1.0])
        plant.advance(np.zeros(4))
        assert plant.data.ncon == 0
    finally:
        plant.close()


def test_ghost_threat_is_invisible_but_collides():
    plant = DronePlant(arena=ArenaSpec())
    try:
        plant.teleport([0.0, 0.0, 1.0], 0.0)
        plant.set_objects(obstacle=[0.6, 0.0, 1.0])
        seen = (luma(plant.camera()) < 0.18).mean()
        plant.set_ghost(True)
        ghost = (luma(plant.camera()) < 0.18).mean()
        assert seen - ghost > 0.02
        plant.set_objects(obstacle=[0.0, 0.0, 1.0])
        plant.advance(np.zeros(4))
        assert plant.data.ncon > 0
    finally:
        plant.close()


def _silenced_features(brain, first, second, roles):
    brain.reset(11)
    if roles:
        brain.silence_inputs(roles)
    brain.sense(first)
    brain.sense(second)
    return brain.step(200)


def test_pathway_silencing_is_specific_and_vision_history_clears():
    brain = BrainRuntime()
    bright = np.full((2, 48, 64, 3), 255, np.uint8)
    dark = np.zeros_like(bright)
    baseline = _silenced_features(brain, dark, dark, None)
    light_only = (bright, bright)  # cues [2, 2, 0, 0]
    loom_only = (bright, dark)  # cues [0, 0, 2, 2]
    light, loom = ("light_l", "light_r"), ("looming_l", "looming_r")
    assert np.abs(_silenced_features(brain, *light_only, light) - baseline).max() < 1e-6
    assert np.abs(_silenced_features(brain, *light_only, loom) - baseline).max() > 1e-4
    assert np.abs(_silenced_features(brain, *loom_only, loom) - baseline).max() < 1e-6
    assert np.abs(_silenced_features(brain, *loom_only, light) - baseline).max() > 1e-4
    brain.reset(1)
    brain.sense(bright)
    assert brain.sense(dark)[2] > 0
    brain.sense(bright)
    brain.clear_vision_history()
    assert brain.sense(dark)[2] == 0


def test_free_roam_collects_beacon_and_respawns_it():
    env = ConnectomeEnv(task="free_roam", vision=False, level=0)
    try:
        _, info = env.reset(seed=7)
        assert info["beacons_collected"] == 0 and info["clearance"] > 1.0
        env.plant.set_objects(target=[*env.plant.pos[0][:2], 1.0])
        _, reward, done, _, info = env.step(np.zeros(4))
        assert info["beacons_collected"] == 1 and not done and reward > 10
        assert any(e["type"] == "beacon_collected" for e in info["events"])
        assert np.linalg.norm(env.plant.target[:2] - env.plant.pos[0][:2]) >= 3.0
    finally:
        env.close()


def test_free_roam_crash_respawns_without_resetting_the_brain():
    env = ConnectomeEnv(task="free_roam", vision=False, level=1, respawn=True)
    try:
        env.reset(seed=3)
        pillar = env.plant.pillars[0]
        env.plant.teleport([pillar[0] - 0.32, pillar[1], 1.0])
        tick = env.brain.tick
        _, reward, done, _, info = env.step(np.zeros(4))
        assert not done and info["collisions"] == 1 and reward < -10
        assert info["collision_kinds"] == {"pillar": 1}
        assert env.brain.tick == tick + 8
        assert info["clearance"] >= 1.0
    finally:
        env.close()


def test_free_roam_threats_launch_and_env_switches_rooms():
    env = ConnectomeEnv(
        task="free_roam", vision=False, level=3, respawn=True, ablation="ghost"
    )
    try:
        env.reset(seed=1)
        assert env.plant.ghost and len(env.plant.pillars) == 16
        launched = False
        for _ in range(int(9 / 0.04)):
            _, _, done, _, info = env.step(np.zeros(4))
            launched |= any(e["type"] == "threat_launched" for e in info["events"])
            assert not done
        assert launched
        env.task, env.ablation = "visual", "none"
        env.reset(seed=2)
        assert env.plant.arena is None and "beacons_collected" not in env.info()
    finally:
        env.close()
