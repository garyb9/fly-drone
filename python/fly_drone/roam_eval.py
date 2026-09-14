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


def acceptance(results, policy):
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
        and light_collision_ci[0] <= t["A4_light_max_collision_increase_per_min"]
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
    out["A5"] = {"passed": None, "note": "in-arena skill probes are a separate report"}
    out["A7"] = {
        "passed": None,
        "note": "live real-time factor is measured on the server",
    }
    out["passed"] = all(out[k]["passed"] for k in ("A1", "A2", "A3", "A4", "A6"))
    out["thresholds"] = ACCEPTANCE
    return out


def evaluate_free_roam(
    policy, output, episodes=50, seconds=120, workers=16, level=3, seed_base=1000
):
    from .distill import screen

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
    )
    report["policy"] = str(Path(policy).resolve())
    report["task"] = "free_roam"
    report["acceptance"] = acceptance(report["results"], key)
    Path(output).write_text(json.dumps(report, indent=2))
    return report
