import sys

import pytest
from fly_drone import cli
from fly_drone.liveness import liveness
from fly_drone.roam_eval import FREE_CELLS

POLICY = "adapter"


def summary(slow, visited, yaw=0.05, n=20, **motion):
    runs = [
        {"seed": seed, "slow_fraction": slow, "collision_kinds": {}}
        for seed in range(n)
    ]
    # Default to a fly-like motion structure so L1-L5 tests exercise only their own metric;
    # structure-specific tests override these.
    values = {
        "bouts_per_min": 30.0,
        "mean_move_bout_s": 1.0,
        "mean_pause_bout_s": 0.5,
        "saccade_rate_hz": 0.5,
        "mean_saccade_amplitude_rad": 1.57,
        "msd_exponent": 1.8,
        **motion,
    }
    return {
        "slow_fraction": slow,
        "mean_visited_cells": visited,
        "mean_abs_yaw_bias": yaw,
        **values,
        "runs": runs,
    }


def results(none, sensory, ghost, random=None, cue=None):
    return {
        f"{POLICY}|none": none,
        f"{POLICY}|sensory": sensory,
        f"{POLICY}|ghost": ghost,
        "random|none": random or summary(0.10, 5),
        "cue_script|none": cue or summary(0.05, 20),
    }


def alive():
    # Moves, explores, senses keep it moving, controls fail, no spin.
    return results(summary(0.10, 60), summary(0.90, 5), summary(0.60, 10))


def test_an_alive_brain_driven_body_passes_every_criterion():
    report = liveness(alive(), POLICY)
    assert report["passed"]
    assert all(report[k]["passed"] for k in report if k.startswith("L"))


def test_the_current_dead_adapter_fails_mobility_and_exploration():
    dead = results(summary(0.62, 8), summary(1.0, 2), summary(0.62, 8))
    report = liveness(dead, POLICY)
    assert not report["L1_mobility"]["passed"]
    assert not report["L2_exploration"]["passed"]
    assert not report["passed"]


def test_a_spinner_fails_non_degeneracy():
    spinner = results(summary(0.10, 60, yaw=0.9), summary(0.90, 5), summary(0.60, 10))
    report = liveness(spinner, POLICY)
    assert not report["L5_non_degenerate"]["passed"] and not report["passed"]


def test_a_control_that_is_also_alive_fails_anti_luck():
    # random happens to move and explore -> liveness is not attributable to the connectome.
    lucky = results(
        summary(0.10, 60),
        summary(0.90, 5),
        summary(0.60, 10),
        random=summary(0.05, 80),
    )
    report = liveness(lucky, POLICY)
    assert report["L4_anti_luck"]["controls"]["random"] is True
    assert not report["L4_anti_luck"]["passed"] and not report["passed"]


def test_senses_must_keep_it_moving_for_causality():
    # Silencing senses does not make it more stationary -> no sense-causality.
    flat = results(summary(0.10, 60), summary(0.10, 60), summary(0.60, 10))
    report = liveness(flat, POLICY)
    assert not report["L3_sense_causality"]["passed"] and not report["passed"]


def test_a_brick_with_one_endless_pause_fails_intermittency():
    brick = results(
        summary(0.95, 5, bouts_per_min=0.0, mean_move_bout_s=None),
        summary(1.0, 2),
        summary(0.6, 10),
    )
    report = liveness(brick, POLICY)
    assert not report["L6_intermittency"]["passed"] and not report["passed"]


def test_an_endless_cruise_fails_structure_and_intermittency():
    cruise = results(
        summary(0.0, 60, bouts_per_min=0.5, mean_move_bout_s=120.0, msd_exponent=2.0),
        summary(0.9, 5),
        summary(0.6, 10),
    )
    report = liveness(cruise, POLICY)
    assert not report["L6_intermittency"]["passed"]
    assert not report["L8_exploration_structure"]["passed"]
    assert not report["passed"]


def test_missing_structure_is_not_evaluable_and_never_passes():
    runs = [
        {"seed": seed, "slow_fraction": 0.1, "collision_kinds": {}}
        for seed in range(20)
    ]
    bare = {
        "slow_fraction": 0.1,
        "mean_visited_cells": 60,
        "mean_abs_yaw_bias": 0.05,
        "runs": runs,
    }
    report = liveness(results(bare, summary(0.9, 5), summary(0.6, 10)), POLICY)
    assert report["L6_intermittency"]["passed"] is None
    assert report["L7_saccadic_turning"]["passed"] is None
    assert report["L8_exploration_structure"]["passed"] is None
    assert report["not_evaluable"] == ["L6", "L7", "L8"]
    assert not report["passed"]


def test_liveness_never_needs_a_teacher_row():
    report = results(summary(0.10, 60), summary(0.90, 5), summary(0.60, 10))
    assert not any(k.startswith("teacher|") for k in report)
    assert liveness(report, POLICY)["passed"]


def test_coverage_uses_the_same_free_cell_count_as_acceptance():
    report = liveness(
        results(summary(0.10, FREE_CELLS // 2), summary(0.9, 5), summary(0.6, 10)),
        POLICY,
    )
    assert report["L2_exploration"]["coverage"] == pytest.approx(0.5)


def _report(passed):
    return {"liveness": {"passed": passed}, "results": {}}


def test_liveness_check_exits_nonzero_on_failure(monkeypatch, tmp_path):
    from fly_drone import roam_eval

    monkeypatch.setattr(roam_eval, "liveness_check", lambda *a, **k: _report(False))
    monkeypatch.setattr(
        sys,
        "argv",
        ["fly-drone", "liveness-check", "--output", str(tmp_path / "l.json")],
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 1


def test_liveness_check_exits_zero_when_the_bar_passes(monkeypatch, tmp_path):
    from fly_drone import roam_eval

    monkeypatch.setattr(roam_eval, "liveness_check", lambda *a, **k: _report(True))
    monkeypatch.setattr(
        sys,
        "argv",
        ["fly-drone", "liveness-check", "--output", str(tmp_path / "l.json")],
    )
    cli.main()  # no SystemExit


def test_liveness_bar_is_far_below_the_skill_bars():
    from fly_drone.liveness import LIVENESS
    from fly_drone.roam_eval import ACCEPTANCE

    assert LIVENESS["L1_max_slow_fraction"] > ACCEPTANCE["A6_max_slow_fraction"]
    assert LIVENESS["L2_min_coverage"] < ACCEPTANCE["A6_min_coverage"]
    assert LIVENESS["L3_causal_metric"] == "slow_fraction"
