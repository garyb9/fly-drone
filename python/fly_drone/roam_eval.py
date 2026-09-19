"""Free-roam evaluation and its pre-registered acceptance (plan Phase 7).

These thresholds were committed before any free-roam decoder was trained. Changing them
after seeing results requires the user's agreement.
"""

import json
from pathlib import Path

import numpy as np

CONDITIONS = ("none", "zero", "shuffle", "sensory", "light", "loom", "ghost")
BASELINES = ("teacher", "cue_script", "random")
ACCEPTANCE = {
    "A1_min_fraction_of_teacher_beacons": 0.6,
    "A1_min_ratio_over_best_ablation": 2.0,
    "A2_max_ratio_of_loom_or_ghost_collisions": 0.5,
    "A2_max_collisions_per_min": 0.5,
    "A3_min_dodge_rate": 0.8,
    "A3_min_balanced_dodge_rate": 0.8,
    "A3_max_ghost_dodge_rate": 0.3,
    "A4_light_min_beacon_cut": 0.5,
    "A4_light_max_collision_increase_per_min": 0.25,
    "A4_loom_min_collision_ratio": 2.0,
    "A4_loom_min_beacon_fraction": 0.5,
    "A5_min_balanced": 0.8,
    "A6_min_coverage": 0.4,
    "A6_max_slow_fraction": 0.1,
    "A6_max_abs_yaw_bias": 0.25,
}
FREE_CELLS = 256  # 16 x 16 one-metre cells
LOSS_OF_CONTROL = ("tilt", "bounds", "altitude")


def paired_bootstrap(a, b, n=4000, seed=0):
    """95% interval of mean(a - b) over seeds resampled with replacement."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (n, len(diff)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _per_seed(summary, key):
    return [r[key] for r in summary["runs"]]


def dodge_rates(summary):
    log = [t for r in summary["runs"] for t in r.get("threat_log", [])]
    near = [t for t in log if t["hit"] or t["min_distance"] < 2.0]
    if not near:
        return None, None
    rate = float(np.mean([t["dodged"] for t in near]))
    sides = [
        [t["dodged"] for t in near if (t["side"] > 0) == left] for left in (True, False)
    ]
    balanced = min(float(np.mean(s)) for s in sides) if all(sides) else None
    return rate, balanced


def acceptance(results, policy, probes=None):
    """results: distill.screen results keyed "controller|ablation"."""
    t = ACCEPTANCE
    r = {c: results[f"{policy}|{c}"] for c in CONDITIONS}
    base = {b: results[f"{b}|none"] for b in BASELINES}
    none = r["none"]
    rate = {k: v["beacons_per_min"] for k, v in {**r, **base}.items()}
    crash = {k: v["collisions_per_min"] for k, v in {**r, **base}.items()}
    rivals = ("zero", "shuffle", "sensory", "light")
    best_rival = max(rivals + ("random",), key=lambda k: rate[k])
    rival_summary = r.get(best_rival) or base[best_rival]
    ci_a1 = paired_bootstrap(
        _per_seed(none, "beacons_per_min"), _per_seed(rival_summary, "beacons_per_min")
    )
    a1 = {
        "beacons_per_min": rate["none"],
        "teacher": rate["teacher"],
        "best_rival": best_rival,
        "best_rival_rate": rate[best_rival],
        "ci_vs_best_rival": ci_a1,
        "passed": rate["none"]
        >= t["A1_min_fraction_of_teacher_beacons"] * rate["teacher"]
        and rate["none"] >= t["A1_min_ratio_over_best_ablation"] * rate[best_rival]
        and ci_a1[0] > 0,
    }
    a2 = {
        "collisions_per_min": crash["none"],
        "loom": crash["loom"],
        "ghost": crash["ghost"],
        "passed": crash["none"]
        <= t["A2_max_ratio_of_loom_or_ghost_collisions"]
        * min(crash["loom"], crash["ghost"])
        and crash["none"] <= t["A2_max_collisions_per_min"],
    }
    dodge, balanced = dodge_rates(none)
    ghost_dodge, _ = dodge_rates(r["ghost"])
    a3 = {
        "dodge_rate": dodge,
        "balanced": balanced,
        "ghost_dodge_rate": ghost_dodge,
        "passed": dodge is not None
        and balanced is not None
        and dodge >= t["A3_min_dodge_rate"]
        and balanced >= t["A3_min_balanced_dodge_rate"]
        and (ghost_dodge is None or ghost_dodge <= t["A3_max_ghost_dodge_rate"]),
    }
    light_collision_ci = paired_bootstrap(
        _per_seed(r["light"], "collisions_per_min"),
        _per_seed(none, "collisions_per_min"),
    )
    a4 = {
        "light_beacon_cut": 1 - rate["light"] / rate["none"] if rate["none"] else None,
        "light_collision_increase_ci": light_collision_ci,
        "loom_collision_ratio": crash["loom"] / crash["none"]
        if crash["none"]
        else None,
        "loom_beacon_fraction": rate["loom"] / rate["none"] if rate["none"] else None,
    }
    a4["passed"] = bool(
        rate["none"] > 0
        and a4["light_beacon_cut"] >= t["A4_light_min_beacon_cut"]
        # Upper bound: light silencing must not plausibly add collisions.
        and light_collision_ci[1] <= t["A4_light_max_collision_increase_per_min"]
        and crash["loom"] >= t["A4_loom_min_collision_ratio"] * max(crash["none"], 1e-9)
        and a4["loom_beacon_fraction"] >= t["A4_loom_min_beacon_fraction"]
    )
    kinds = {}
    for run in none["runs"]:
        for k, v in run["collision_kinds"].items():
            kinds[k] = kinds.get(k, 0) + v
    coverage = none["mean_visited_cells"] / FREE_CELLS
    a6 = {
        "coverage": coverage,
        "slow_fraction": none["slow_fraction"],
        "abs_yaw_bias": none["mean_abs_yaw_bias"],
        "loss_of_control_events": {k: kinds.get(k, 0) for k in LOSS_OF_CONTROL},
        "passed": coverage >= t["A6_min_coverage"]
        and none["slow_fraction"] <= t["A6_max_slow_fraction"]
        and none["mean_abs_yaw_bias"] <= t["A6_max_abs_yaw_bias"]
        and not any(kinds.get(k, 0) for k in LOSS_OF_CONTROL),
    }
    out = {"A1": a1, "A2": a2, "A3": a3, "A4": a4, "A6": a6}
    for entry in out.values():
        entry["passed"] = bool(entry["passed"])
    if probes is None:
        out["A5"] = {"passed": None, "note": "skill probes not run"}
    else:
        out["A5"] = {
            name: {"success_rate": p["success_rate"], "balanced": p["balanced"]}
            for name, p in probes.items()
        }
        out["A5"]["passed"] = all(
            p["balanced"] is not None and p["balanced"] >= t["A5_min_balanced"]
            for p in probes.values()
        )
    out["A7"] = {
        "passed": None,
        "note": "live real-time factor is measured on the server",
    }
    out["passed"] = (
        all(out[k]["passed"] for k in ("A1", "A2", "A3", "A4", "A6"))
        and out["A5"]["passed"] is not False
    )
    out["thresholds"] = ACCEPTANCE
    return out


PROBES = {"steer": 10.0, "approach": 15.0, "dodge": 8.0, "wall": 8.0}
PROBE_WALL_REACH = 2.5


def _probe_setup(env, probe, rng):
    """Place drone and objects for one skill probe in an empty arena; returns the side."""
    spec = env.spec
    side = float(rng.choice([-1.0, 1.0]))
    yaw = float(rng.uniform(-np.pi, np.pi))
    position = [0.0, 0.0, 1.0]
    if probe == "wall":
        yaw = side * float(rng.uniform(0.1, 0.4))
        position = [spec.half_size - 3.0, 0.0, 1.0]
    env.plant.teleport(position, yaw)
    env.brain.clear_vision_history()
    if env.brain.learned:
        env.brain.push_frame(env._frame())

    def at(distance, bearing):
        angle = yaw + bearing
        return [
            position[0] + distance * np.cos(angle),
            position[1] + distance * np.sin(angle),
            1.0,
        ]

    if probe == "steer":
        target = at(2.0, side * rng.uniform(0.6, 1.5))
    elif probe == "approach":
        target = at(rng.uniform(2.2, 3.0), side * rng.uniform(0.1, 0.5))
    elif probe == "dodge":
        target = at(6.0, np.pi)
    else:
        target = [-6.0, 0.0, 1.0]
    env.plant.set_objects(target=target)
    return side


def _probe_job(job):
    from .brain import BrainRuntime
    from .env import FRAME_SECONDS, ConnectomeEnv
    from .teacher import teacher_action

    controller, probe, seeds, vision, seconds, *rest = job
    encoder = rest[0] if rest and controller.startswith("policy:") else None
    brain = BrainRuntime(encoder=encoder)
    adapter_path = None
    if controller.startswith("policy:"):
        brain.load_policy(controller.split(":", 1)[1])
    elif controller == "adapter" or controller.startswith("adapter:"):
        adapter_path = controller.split(":", 1)[1] if ":" in controller else None
    env = ConnectomeEnv(
        task="free_roam", level=0, respawn=False, brain=brain, vision=vision
    )
    runs = []
    try:
        for seed in seeds:
            env.reset(seed=int(seed))
            side = _probe_setup(env, probe, np.random.default_rng(int(seed) + 104729))
            obs = env.observe()
            info = env.info()
            initial = abs(info["bearing"])
            launch_frame = int(1.0 / FRAME_SECONDS)
            collected = crashed = False
            closest_wall = env.spec.half_size - env.plant.pos[0][0]
            for frame in range(int((seconds or PROBES[probe]) / FRAME_SECONDS)):
                if probe == "dodge" and frame == launch_frame and env.launch_threat():
                    side = env.roam["threat"]["side"]
                if controller == "teacher":
                    action = teacher_action(env)[0]
                elif controller == "adapter" or controller.startswith("adapter:"):
                    from .adapter import declared_command

                    action = declared_command(brain, path=adapter_path)
                else:
                    action = brain.infer(obs) / env.plant.limits
                obs, _, done, _, info = env.step(action)
                collected |= any(
                    e["type"] == "beacon_collected" for e in info["events"]
                )
                closest_wall = min(
                    closest_wall, env.spec.half_size - env.plant.pos[0][0]
                )
                if done:
                    crashed = True
                    break
                if probe in ("steer", "approach") and collected:
                    break
            threats = env.roam["threats"] + (
                [env.roam["threat"]] if env.roam["threat"] else []
            )
            hit = any(t.get("hit") for t in env.roam["threats"])
            near = min((t["min_distance"] for t in threats), default=np.inf)
            success = {
                "steer": not crashed
                and (collected or abs(info["bearing"]) < 0.5 * initial),
                "approach": not crashed and collected,
                "dodge": not crashed and not hit and near < 2.0,
                "wall": not crashed and closest_wall < PROBE_WALL_REACH,
            }[probe]
            runs.append(
                {
                    "seed": int(seed),
                    "side": "left" if side > 0 else "right",
                    "success": bool(success),
                    "crashed": bool(crashed),
                    "collected": bool(collected),
                    "closest_threat": float(near),
                    "closest_wall": float(closest_wall),
                }
            )
    finally:
        env.close()
    return probe, runs


def skill_probes(
    controller,
    episodes=50,
    workers=6,
    vision=True,
    seconds=None,
    seed_base=1000,
    encoder=None,
):
    """A5: steer, approach, dodge and wall probes, balanced by side."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    seeds = np.arange(seed_base, seed_base + episodes)
    per = max(1, workers // len(PROBES))
    jobs = [
        (controller, probe, chunk.tolist(), vision, seconds, encoder)
        for probe in PROBES
        for chunk in np.array_split(seeds, min(per, episodes))
        if len(chunk)
    ]
    collected = {probe: [] for probe in PROBES}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for probe, runs in pool.map(_probe_job, jobs):
            collected[probe] += runs
    report = {}
    for probe, runs in collected.items():
        by_side = {
            s: [r["success"] for r in runs if r["side"] == s] for s in ("left", "right")
        }
        rates = {s: float(np.mean(v)) if v else None for s, v in by_side.items()}
        report[probe] = {
            "success_rate": float(np.mean([r["success"] for r in runs])),
            "success_by_side": rates,
            "balanced": min(rates.values())
            if all(v is not None for v in rates.values())
            else None,
            "runs": sorted(runs, key=lambda r: r["seed"]),
        }
    return report


def evaluate_free_roam(
    policy=None,
    output="runs/evaluation.json",
    episodes=50,
    seconds=120,
    workers=6,
    level=3,
    seed_base=1000,
    probes=True,
    encoder=None,
    bridge="learned",
):
    from .distill import screen

    declared = bridge == "declared"
    if declared:
        key = "adapter"
    else:
        if not policy:
            raise ValueError("the learned free-roam bridge needs a policy path")
        key = f"policy:{Path(policy).resolve()}"
    combos = [(key, c) for c in CONDITIONS] + [(b, "none") for b in BASELINES]
    report = screen(
        None,
        output,
        seeds=episodes,
        seconds=seconds,
        level=level,
        workers=workers,
        seed_base=seed_base,
        combos=combos,
        encoder=encoder,
    )
    report["policy"] = key
    report["task"] = "free_roam"
    report["bridge"] = bridge
    report["teacher_in_behaviour_path"] = False if declared else None
    report["encoder"] = str(encoder) if encoder else None
    report["probes"] = (
        skill_probes(key, episodes, workers, seed_base=seed_base, encoder=encoder)
        if probes
        else None
    )
    report["acceptance"] = acceptance(report["results"], key, report["probes"])
    Path(output).write_text(json.dumps(report, indent=2))
    return report


def adapter_check(
    output,
    episodes=50,
    seconds=120,
    level=3,
    workers=6,
    seed_base=1000,
    probes=True,
):
    """P0 gate: the declared adapter under the standard causal conditions, teacher-free.

    The teacher is a simulator-geometry baseline comparator only; it is never in the
    declared bridge's behaviour path. Reuses ``ACCEPTANCE`` unchanged, so a pass here
    means the same thing it means for a learned actor.
    """
    from .distill import screen

    key = "adapter"
    combos = [(key, c) for c in CONDITIONS] + [(b, "none") for b in BASELINES]
    report = screen(
        None,
        output,
        seeds=episodes,
        seconds=seconds,
        level=level,
        workers=workers,
        seed_base=seed_base,
        combos=combos,
    )
    report["policy"] = key
    report["bridge"] = "declared"
    report["task"] = "free_roam"
    report["teacher_in_behaviour_path"] = False
    report["probes"] = (
        skill_probes(key, episodes, workers, seed_base=seed_base) if probes else None
    )
    report["acceptance"] = acceptance(report["results"], key, report["probes"])
    Path(output).write_text(json.dumps(report, indent=2))
    return report


# Encoder v5 checks, pre-registered 2026-09-14 before any v5 result (spec §5). Additive:
# they never change ACCEPTANCE.
# Both ranges measure the gap to the threat's centre. E1/E2 score the loom group (LC4,
# LPLC2) by max over its four channels and the light group (Mi1, Tm3) by mean over its four.
THREAT_POSITIVE_RANGE = 3.0
THREAT_NEGATIVE_RANGE = 6.0
ENCODER_CHECKS = {
    "E1_min_loom_auc": 0.8,
    "E2_min_light_margin": 0.05,
    "E2_min_loom_margin": 0.1,
}


def roc_auc(scores, labels):
    """Mann-Whitney AUC with tied scores ranked at their average; None without both classes."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    pos, neg = int(labels.sum()), int((~labels).sum())
    if not pos or not neg:
        return None
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2.0)[inverse]
    return float((ranks[labels].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def encoder_scores(currents, threat, beacon):
    from .sac import LIGHT, LOOM
    from .spatial_encoder import flat_dim, group_slices

    currents = np.asarray(currents, dtype=float)
    threat = np.asarray(threat)
    keep = threat >= 0
    motion = None
    if currents.shape[1] == flat_dim():  # v6 spatial: score the named groups
        groups = group_slices()
        light_ids, loom_ids = groups["light"], groups["loom"]
        motion_ids = groups["motion"]
    else:
        light_ids, loom_ids = LIGHT, LOOM
        motion_ids = None
    light = currents[keep][:, light_ids].mean(1)
    loom = currents[keep][:, loom_ids].max(1)
    if motion_ids is not None:
        motion = currents[keep][:, motion_ids].max(1)
    is_threat = threat[keep] == 1
    seen = np.asarray(beacon, dtype=bool)[keep]
    auc = {
        "loom_threat": roc_auc(loom, is_threat),
        "loom_beacon": roc_auc(loom, seen),
        "light_beacon": roc_auc(light, seen),
        "light_threat": roc_auc(light, is_threat),
    }
    # v6 also scores the primary M1b channel (motion = Tm4/T2) and the union with the direct loom
    # route, reported alongside the pre-registered loom number. The bar (0.8) and the
    # pre-registered `passed` (on the loom group, the v5 definition) are unchanged; this only lets
    # E1 be read on the spatial channels the v6 spec re-bases it on.
    extra = {}
    if motion is not None:
        extra = {
            "motion_auc": roc_auc(motion, is_threat),
            "union_auc": roc_auc(np.maximum(loom, motion), is_threat),
        }
    t = ENCODER_CHECKS
    complete = None not in auc.values()
    light_margin = auc["light_beacon"] - auc["light_threat"] if complete else None
    loom_margin = auc["loom_threat"] - auc["loom_beacon"] if complete else None
    return {
        "E1": {
            "loom_auc": auc["loom_threat"],
            **extra,
            "positives": int(is_threat.sum()),
            "negatives": int((~is_threat).sum()),
            "passed": bool(
                auc["loom_threat"] is not None
                and auc["loom_threat"] >= t["E1_min_loom_auc"]
            ),
        },
        "E2": {
            "auc": auc,
            "light_margin": light_margin,
            "loom_margin": loom_margin,
            "passed": bool(
                complete
                and light_margin >= t["E2_min_light_margin"]
                and loom_margin >= t["E2_min_loom_margin"]
            ),
        },
        "thresholds": ENCODER_CHECKS,
    }


def _threat_label(flying, gap, previous_gap, visible):
    """Pure E1 labelling rule, factored out of `_checks_job` for direct testing.

    Positive (1): an active threat closing inside THREAT_POSITIVE_RANGE and visible.
    Negative (0): no active threat, or the threat is beyond THREAT_NEGATIVE_RANGE.
    Excluded (-1): everything else (the band between the two ranges, not closing, or
    closing-but-unseen inside THREAT_POSITIVE_RANGE).
    """
    if not flying or gap > THREAT_NEGATIVE_RANGE:
        return 0
    if (
        gap < THREAT_POSITIVE_RANGE
        and previous_gap is not None
        and gap < previous_gap
        and visible
    ):
        return 1
    return -1


def _checks_job(job):
    """Fly a controller; per frame, the currents applied and labels of the state they saw.

    `controller="teacher"` (fixed probe): the geometry-driven teacher flies and only the encoder
    varies, so trajectories and threat/beacon labels are identical across encoders and only the
    applied currents differ. `controller="policy"`: the loaded decoder flies, so labels mix in the
    policy's own behaviour (the pre-2026-09-16 mode, kept as a secondary number).
    """
    from .brain import V4_TO_V5, BrainRuntime
    from .distill import _roam_env
    from .teacher import teacher_action, visible

    policy, encoder, seeds, seconds, level, *rest = job
    controller = rest[0] if rest else "policy"
    brain = BrainRuntime(encoder=encoder)
    brain.load_policy(policy)
    env = _roam_env(level, brain)
    currents, threat, beacon = [], [], []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=int(seed))
            previous_gap = None
            for _ in range(int(seconds / 0.04)):
                flying = env.roam["threat"] is not None
                gap = float(np.linalg.norm(env.plant.obstacle - env.plant.pos[0]))
                label = _threat_label(
                    flying,
                    gap,
                    previous_gap,
                    visible(env, env.plant.obstacle, "obstacle"),
                )
                previous_gap = gap if flying else None
                seen = env.beacon_visible()
                if controller == "teacher":
                    action = teacher_action(env)[0]
                else:
                    action = brain.infer(obs) / env.plant.limits
                obs, *_ = env.step(action)
                applied = brain.cues if brain.learned else brain.cues[list(V4_TO_V5)]
                currents.append(np.asarray(applied, dtype=np.float32).copy())
                threat.append(label)
                beacon.append(bool(seen))
    finally:
        env.close()
    return currents, threat, beacon


def encoder_checks(
    policy,
    output,
    encoder=None,
    episodes=50,
    seconds=120,
    workers=6,
    seed_base=1000,
    level=3,
    controller="teacher",
):
    """E1-E2 on held-out seeds, intact brain. encoder=None measures the v4 baseline.

    Defaults to the fixed probe (`controller="teacher"`): the teacher flies, so the frame labels
    are policy-independent and E1/E2 compare encoders rather than behaviours. Pass
    `controller="policy"` for the confounded, policy-driven number.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    seeds = np.arange(seed_base, seed_base + episodes)
    policy = str(Path(policy).resolve())
    encoder = str(Path(encoder).resolve()) if encoder else None
    jobs = [
        (policy, encoder, c.tolist(), seconds, level, controller)
        for c in np.array_split(seeds, min(workers, episodes))
        if len(c)
    ]
    currents, threat, beacon = [], [], []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for c, t, b in pool.map(_checks_job, jobs):
            currents += c
            threat += t
            beacon += b
    report = {
        "policy": policy,
        "encoder": encoder,
        "controller": controller,
        "seeds": [int(seeds[0]), int(seeds[-1])],
        "seconds": seconds,
        "frames": len(currents),
        **encoder_scores(currents, threat, beacon),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2))
    return report


def bypass_comparison(full, bypass):
    """E3, reported without a bar: can a decoder reading the 8 currents do better than the brain?"""
    keys = ("beacons_per_min", "collisions_per_min", "near_dodge_rate")
    pick = {
        name: {k: s.get(k) for k in keys}
        for name, s in (("full", full), ("bypass", bypass))
    }
    f, b = pick["full"], pick["bypass"]
    better = (
        None not in (*f.values(), *b.values())
        and b["beacons_per_min"] >= f["beacons_per_min"]
        and b["collisions_per_min"] <= f["collisions_per_min"]
        and b["near_dodge_rate"] >= f["near_dodge_rate"]
    )
    return {
        **pick,
        "bypass_better": bool(better),
        "note": "If bypass_better, stop and report to the user before any stage 4 conclusions.",
    }


# Round selection guard (added 2026-09-16 after the encoder v5 round-1 collapse). Task 13's original
# rule ranked rounds by near_dodge_rate alone; a blind, erratic policy scores high on it while
# collecting zero beacons and scoring the same with the eyes ablated. A round is eligible only if
# it kept foraging, its dodging is causal (ghost), and channel semantics survive (E2). Round 0 is
# eligible by construction: it defines the baseline and is the fallback.
ROUND_GATE = {
    "min_beacon_fraction": 0.5,  # of round 0's validated beacons/min
    "max_ghost_dodge": ACCEPTANCE["A3_max_ghost_dodge_rate"],
}


def round_eligible(validation, baseline, gate=ROUND_GATE):
    """True when a SAC round kept foraging, dodges on sight, and kept E2 semantics.

    `validation` and `baseline` are `sac.validate` summaries (round 0's is the baseline). A
    missing or non-positive baseline beacon rate makes every round ineligible: there is nothing to
    measure progress against.
    """
    base = baseline.get("beacons_per_min")
    beacons = validation.get("beacons_per_min")
    ghost = validation.get("ghost_near_dodge_rate")
    if base is None or base <= 0 or beacons is None or ghost is None:
        return False
    return bool(
        beacons >= gate["min_beacon_fraction"] * base
        and ghost <= gate["max_ghost_dodge"]
        and (validation.get("E2") or {}).get("passed", False)
    )


def pick_best_round(validations, gate=ROUND_GATE):
    """Index of the eligible round with the best near-dodge rate; ties to more beacons/min.

    `validations[0]` is the round-0 baseline and is always eligible, so the result is always 0 for
    an all-degenerate run. Earlier rounds win exact ties.
    """
    if not validations:
        raise ValueError("no round validations to choose from")
    base = validations[0]
    rows = []
    for k, v in enumerate(validations):
        if k > 0 and not round_eligible(v, base, gate):
            continue
        nd = v.get("near_dodge_rate")
        rows.append(
            (
                -1.0 if nd is None else float(nd),
                float(v.get("beacons_per_min") or 0.0),
                -k,
                k,
            )
        )
    return max(rows)[3]


def round_gate_report(validations, gate=ROUND_GATE):
    """Per-round eligibility columns for the Task 13 report table."""
    base = validations[0]
    return [
        {
            "round": k,
            "eligible": k == 0 or round_eligible(v, base, gate),
            "near_dodge_rate": v.get("near_dodge_rate"),
            "balanced_dodge_rate": v.get("balanced_dodge_rate"),
            "ghost_near_dodge_rate": v.get("ghost_near_dodge_rate"),
            "beacons_per_min": v.get("beacons_per_min"),
            "collisions_per_min": v.get("collisions_per_min"),
            "E2_passed": (v.get("E2") or {}).get("passed"),
        }
        for k, v in enumerate(validations)
    ]
