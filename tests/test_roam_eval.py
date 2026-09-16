import json

import numpy as np
import pytest
from fly_drone import distill
from fly_drone.brain import BrainRuntime
from fly_drone.encoder import LearnedEncoder
from fly_drone.roam_eval import (
    BASELINES,
    CONDITIONS,
    PROBES,
    ROUND_GATE,
    THREAT_NEGATIVE_RANGE,
    THREAT_POSITIVE_RANGE,
    _checks_job,
    _probe_job,
    _threat_label,
    acceptance,
    bypass_comparison,
    encoder_scores,
    paired_bootstrap,
    pick_best_round,
    roc_auc,
    round_eligible,
    round_gate_report,
)
from fly_drone.sac import SpacesOnlyEnv, build_sac
from test_sac import zero_actor

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


def test_a5_requires_every_probe_balanced():
    good = {p: {"success_rate": 0.95, "balanced": 0.9} for p in PROBES}
    assert acceptance(results(), POLICY, good)["A5"]["passed"]
    weak = {**good, "dodge": {"success_rate": 0.6, "balanced": 0.4}}
    report = acceptance(results(), POLICY, weak)
    assert report["A5"]["passed"] is False and not report["passed"]


def test_skill_probe_jobs_run_and_report_sides():
    for probe in PROBES:
        name, runs = _probe_job(("teacher", probe, [1, 2], False, 1.2))
        assert name == probe and len(runs) == 2
        assert all(r["side"] in ("left", "right") for r in runs)
        assert all(isinstance(r["success"], bool) for r in runs)


def test_light_silencing_that_also_causes_crashes_is_not_a_dissociation():
    crashes = summary(0.8, 1.0, [(1.0, True), (-1.0, True)])
    assert not acceptance(results(light=crashes), POLICY)["A4"]["passed"]


def test_one_sided_dodging_fails_balance_and_spinning_fails_a6():
    one_sided = summary(4.0, 0.2, [(1.0, True), (-1.0, False)])
    assert not acceptance(results(none=one_sided), POLICY)["A3"]["passed"]
    spinner = summary(4.0, 0.2, [(1.0, True), (-1.0, True)], yaw=0.9)
    assert not acceptance(results(none=spinner), POLICY)["A6"]["passed"]


def test_roc_auc_counts_ties_as_half_and_needs_both_classes():
    assert roc_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
    assert roc_auc([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert roc_auc([0.5] * 4, [0, 1, 0, 1]) == 0.5
    assert roc_auc([0.1, 0.2], [1, 1]) is None


def _synthetic(loom_selective):
    rng = np.random.default_rng(0)
    n = 2000
    threat = rng.choice([-1, 0, 1], n, p=[0.1, 0.8, 0.1])
    beacon = rng.random(n) < 0.4
    currents = rng.uniform(0, 0.2, (n, 8)).astype(np.float32)
    currents[:, :4] += np.where(beacon, 1.0, 0.0)[:, None]
    if loom_selective:
        currents[:, 4:] += np.where(threat == 1, 1.5, 0.0)[:, None]
    else:
        currents[:, 4:] += rng.uniform(0, 1.5, (n, 1))
    return currents, threat, beacon


def test_a_selective_encoder_passes_e1_and_e2():
    report = encoder_scores(*_synthetic(True))
    assert report["E1"]["passed"] and report["E2"]["passed"]
    assert report["E1"]["positives"] > 0 and report["E1"]["negatives"] > 0


def test_loom_that_ignores_threats_fails_e1_and_e2():
    report = encoder_scores(*_synthetic(False))
    assert not report["E1"]["passed"] and not report["E2"]["passed"]


def test_checks_job_labels_frames_and_maps_v4_cues_to_eight_channels(tmp_path):
    actor = zero_actor(tmp_path / "a.json", BrainRuntime())
    currents, threat, beacon = _checks_job((str(actor), None, [3], 0.4, 3))
    assert len(currents) == len(threat) == len(beacon) == 10
    assert currents[0].shape == (8,) and set(threat) <= {-1, 0, 1}


def test_screen_job_flies_a_v5_policy_with_its_encoder_and_baselines_on_v4(tmp_path):
    enc = LearnedEncoder.fresh(seed=6)
    enc.save(tmp_path / "e.pt")
    path = zero_actor(tmp_path / "a.json", BrainRuntime())
    actor = json.loads(path.read_text())
    actor["encoder_version"] = enc.version
    path.write_text(json.dumps(actor))
    job = (f"policy:{path}", [3], 0.4, 3, "loom", str(tmp_path / "e.pt"))
    _, _, runs = distill._screen_job(job)
    assert len(runs) == 1
    _, _, runs = distill._screen_job(("cue_script", [3], 0.4, 3, "none", None))
    assert len(runs) == 1


def test_bypass_comparison_flags_only_a_bypass_better_on_all_three():
    full = {"beacons_per_min": 1.0, "collisions_per_min": 0.4, "near_dodge_rate": 0.8}
    worse = {"beacons_per_min": 1.2, "collisions_per_min": 0.6, "near_dodge_rate": 0.9}
    better = {"beacons_per_min": 1.2, "collisions_per_min": 0.3, "near_dodge_rate": 0.9}
    assert not bypass_comparison(full, worse)["bypass_better"]
    assert bypass_comparison(full, better)["bypass_better"]


def test_threat_label_covers_e1_positive_negative_and_excluded_cases():
    # ~2 m ahead, inside the field of view, closing.
    assert _threat_label(True, 2.0, 2.5, visible=True) == 1
    # Same position but not closing (gap growing, not shrinking).
    assert _threat_label(True, 2.0, 1.5, visible=True) != 1
    # Behind the drone (not visible) at < 3 m: excluded, not a negative.
    assert _threat_label(True, 2.0, 2.5, visible=False) == -1
    # Between the positive and negative ranges: excluded.
    assert _threat_label(True, 4.0, 4.5, visible=True) == -1
    assert THREAT_POSITIVE_RANGE < 4.0 < THREAT_NEGATIVE_RANGE
    # Inactive threat, or beyond the negative range: negative.
    assert _threat_label(False, 2.0, None, visible=False) == 0
    assert _threat_label(True, 7.0, 7.5, visible=True) == 0
    assert THREAT_NEGATIVE_RANGE < 7.0


def test_screen_job_runs_a_bypass_controller_reading_encoder_currents(tmp_path):
    enc = LearnedEncoder.fresh(seed=6)
    enc.save(tmp_path / "e.pt")
    model = build_sac("bypass", SpacesOnlyEnv("bypass"), buffer_size=1, device="cpu")
    model.save(tmp_path / "bypass.zip")
    job = (
        f"bypass:{tmp_path / 'bypass.zip'}",
        [3],
        0.4,
        3,
        "none",
        str(tmp_path / "e.pt"),
    )
    controller, ablation, runs = distill._screen_job(job)
    assert controller.startswith("bypass:") and ablation == "none"
    assert len(runs) == 1
    assert {"seed", "beacons_per_min", "collisions_per_min"} <= runs[0].keys()


def test_bypass_controller_without_a_learned_encoder_raises(tmp_path):
    model = build_sac("bypass", SpacesOnlyEnv("bypass"), buffer_size=1, device="cpu")
    model.save(tmp_path / "bypass.zip")
    job = (f"bypass:{tmp_path / 'bypass.zip'}", [3], 0.4, 3, "none", None)
    with pytest.raises(ValueError):
        distill._screen_job(job)


def validation(near, beacons, ghost, e2=True, balanced=None, collisions=1.0):
    return {
        "near_dodge_rate": near,
        "balanced_dodge_rate": near if balanced is None else balanced,
        "ghost_near_dodge_rate": ghost,
        "beacons_per_min": beacons,
        "collisions_per_min": collisions,
        "E2": {"passed": e2},
    }


# The recorded round 0..2 validations that motivated the guard (see
# docs/results/encoder-v5/PHASE0-DIAGNOSTICS-2026-09-16.md).
ROUND0 = validation(0.269, 1.9, 0.296, True)
ROUND1 = validation(0.926, 0.0, 0.815, False, balanced=0.909, collisions=6.1)
ROUND2 = validation(1.0, 0.0, 1.0, False, balanced=1.0, collisions=10.3)


def test_round_gate_rejects_the_recorded_degenerate_rounds_and_falls_back_to_round_0():
    assert round_eligible(ROUND0, ROUND0)
    assert not round_eligible(ROUND1, ROUND0)  # zero beacons, ghost 0.815, E2 fail
    assert not round_eligible(ROUND2, ROUND0)
    assert pick_best_round([ROUND0, ROUND1, ROUND2]) == 0


def test_round_gate_accepts_a_round_that_keeps_foraging_and_dodges_on_sight():
    good = validation(0.9, 1.5, 0.1, True, balanced=0.85)
    assert round_eligible(good, ROUND0)
    assert pick_best_round([ROUND0, good]) == 1


def test_round_gate_requires_foraging_above_half_of_round_0():
    assert ROUND_GATE["min_beacon_fraction"] == 0.5
    assert not round_eligible(validation(0.9, 0.9, 0.1, True), ROUND0)  # < 0.95
    assert round_eligible(validation(0.9, 0.95, 0.1, True), ROUND0)


def test_round_gate_requires_causal_dodging_and_e2():
    assert not round_eligible(validation(0.9, 1.5, 0.5, True), ROUND0)  # ghost too high
    assert not round_eligible(validation(0.9, 1.5, 0.1, False), ROUND0)  # E2 fail


def test_round_gate_missing_baseline_never_advances_past_round_0():
    assert not round_eligible(ROUND1, validation(0.0, 0.0, 0.0))
    assert pick_best_round([ROUND0, ROUND1]) == 0
    with pytest.raises(ValueError):
        pick_best_round([])


def test_round_gate_report_lists_every_round_with_its_verdict():
    report = round_gate_report([ROUND0, ROUND1, ROUND2])
    assert [r["round"] for r in report] == [0, 1, 2]
    assert [r["eligible"] for r in report] == [True, False, False]
    assert (
        report[1]["ghost_near_dodge_rate"] == 0.815 and report[1]["E2_passed"] is False
    )
