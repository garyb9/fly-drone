import numpy as np
from fly_drone.roam_eval import BASELINES, CONDITIONS, acceptance, paired_bootstrap

POLICY = "policy:/x/actor.json"


def summary(beacons, collisions, dodges=(), visited=150, slow=0.02, yaw=0.05):
    runs = []
    for seed in range(20):
        runs.append(
            {
                "seed": seed,
                "beacons_per_min": beacons + 0.1 * (seed % 3),
                "collisions_per_min": collisions + 0.05 * (seed % 2),
                "collision_kinds": {"pillar": 1} if collisions else {},
                "threat_log": [
                    {"side": side, "dodged": ok, "hit": not ok, "min_distance": 0.5}
                    for side, ok in dodges
                ],
            }
        )
    return {
        "beacons_per_min": float(np.mean([r["beacons_per_min"] for r in runs])),
        "collisions_per_min": float(np.mean([r["collisions_per_min"] for r in runs])),
        "mean_visited_cells": visited,
        "slow_fraction": slow,
        "mean_abs_yaw_bias": yaw,
        "runs": runs,
    }


def results(**overrides):
    dodge_all = [(1.0, True), (-1.0, True)]
    dodge_none = [(1.0, False), (-1.0, False)]
    base = {
        "none": summary(4.0, 0.2, dodge_all),
        "zero": summary(0.2, 1.0, dodge_none),
        "shuffle": summary(0.5, 1.0, dodge_none),
        "sensory": summary(0.2, 1.5, dodge_none),
        "light": summary(0.8, 0.2, dodge_all),
        "loom": summary(3.5, 1.2, dodge_none),
        "ghost": summary(3.8, 1.0, dodge_none),
    }
    base.update(overrides)
    out = {f"{POLICY}|{c}": base[c] for c in CONDITIONS}
    out.update(
        {
            f"{b}|none": summary(v, c)
            for b, v, c in zip(BASELINES, (5.0, 2.0, 0.3), (0.1, 0.4, 2.0), strict=True)
        }
    )
    return out


def test_bootstrap_interval_contains_zero_for_identical_samples():
    lo, hi = paired_bootstrap([1, 2, 3, 4], [1, 2, 3, 4])
    assert lo == hi == 0.0
    lo, hi = paired_bootstrap([3, 4, 5, 6], [1, 2, 3, 4])
    assert lo == hi == 2.0


def test_a_dissociating_brain_driven_policy_passes():
    report = acceptance(results(), POLICY)
    assert report["A1"]["passed"] and report["A2"]["passed"] and report["A3"]["passed"]
    assert report["A4"]["passed"] and report["A6"]["passed"] and report["passed"]


def test_a_blind_forager_fails_the_dissociation():
    # Silencing light barely changes foraging: the beacons are not found by sight.
    report = acceptance(
        results(light=summary(3.9, 0.2, [(1.0, True), (-1.0, True)])), POLICY
    )
    assert not report["A4"]["passed"] and not report["A1"]["passed"]
    assert not report["passed"]


def test_one_sided_dodging_fails_balance_and_spinning_fails_a6():
    one_sided = summary(4.0, 0.2, [(1.0, True), (-1.0, False)])
    assert not acceptance(results(none=one_sided), POLICY)["A3"]["passed"]
    spinner = summary(4.0, 0.2, [(1.0, True), (-1.0, True)], yaw=0.9)
    assert not acceptance(results(none=spinner), POLICY)["A6"]["passed"]
