"""Evidence omissions cannot be promoted through permissive legacy score fields."""

from copy import deepcopy

import pytest
from fly_drone.report_completeness import report_completeness
from fly_drone.roam_eval import BASELINES, CONDITIONS, PROBES


def complete_report():
    keys = [f"adapter|{c}" for c in CONDITIONS] + [f"{b}|none" for b in BASELINES]
    provenance = {
        "source": {"revision": "revision", "fingerprint": "source-hash"},
        "files": {
            name: "hash"
            for name in (
                "controller",
                "bundle/manifest.json",
                "bundle/graph.bin",
                "bundle/neurons.bin",
                "bundle/cells.json",
                "bundle/groups.json",
                "bundle/sensory-mappings.json",
            )
        },
        "sensory": "v4",
        "errors": [],
    }
    return {
        "policy": "adapter",
        "results": {
            key: {
                "beacons_per_min": 1.0,
                "collisions_per_min": 0.0,
                "mean_visited_cells": 150.0,
                "slow_fraction": 0.01,
                "mean_abs_yaw_bias": 0.01,
                "runs": [
                    {
                        "seed": seed,
                        "beacons_per_min": 1.0,
                        "collisions_per_min": 0.0,
                        "collision_kinds": {},
                        "threat_log": [
                            {
                                "side": side,
                                "hit": False,
                                "min_distance": 1,
                                "dodged": True,
                            }
                            for side in (-1, 1)
                        ],
                    }
                    for seed in (10, 11)
                ],
            }
            for key in keys
        },
        "probes": {
            name: {
                "balanced": 1.0,
                "runs": [
                    {"seed": 10, "side": "left", "success": True},
                    {"seed": 11, "side": "right", "success": True},
                ],
            }
            for name in PROBES
        },
        "provenance": {"before": provenance, "after": deepcopy(provenance)},
        "acceptance": {"passed": True},
    }


def check(report, realtime=None):
    return report_completeness(
        report,
        expected_conditions=[f"adapter|{c}" for c in CONDITIONS]
        + [f"{b}|none" for b in BASELINES],
        expected_seeds=[10, 11],
        probe_names=PROBES,
        realtime=realtime,
    )


def live_timing(wall=1):
    return {
        "mode": "live_end_to_end",
        "simulated_seconds": 2,
        "wall_seconds": wall,
        "evidence": "runs/live/timing.json",
    }


def test_offline_report_is_behaviorally_complete_but_not_promotable():
    report = complete_report()
    original = deepcopy(report)
    result = check(report)
    assert result["behavioral_complete"] and result["provenance_complete"]
    assert not result["realtime_complete"]
    assert not result["eligible_for_promotion"]
    assert result["A7"]["passed"] is None
    assert report == original


def test_missing_metrics_rejects_an_otherwise_complete_seed_list():
    report = complete_report()
    report["results"]["adapter|none"]["beacons_per_min"] = float("nan")
    assert not check(report, live_timing())["behavioral_complete"]


def test_live_report_requires_both_timing_and_behavioral_pass():
    report = complete_report()
    assert check(report, live_timing())["eligible_for_promotion"]
    slow = check(report, live_timing(wall=3))
    assert slow["complete"] and not slow["eligible_for_promotion"]
    report["acceptance"]["passed"] = False
    assert not check(report, live_timing())["eligible_for_promotion"]


@pytest.mark.parametrize("missing", ["condition", "seed", "duplicate", "unfinished"])
def test_incomplete_trials_rejected(missing):
    report = complete_report()
    if missing == "condition":
        del report["results"]["adapter|light"]
    else:
        runs = report["results"]["adapter|light"]["runs"]
        if missing == "seed":
            runs.pop()
        elif missing == "duplicate":
            runs[1]["seed"] = 10
        else:
            runs[0]["status"] = "cancelled"
    result = check(report, live_timing())
    assert not result["behavioral_complete"]
    assert not result["eligible_for_promotion"]


@pytest.mark.parametrize("missing", ["all", "one", "side", "seed", "success"])
def test_probe_omissions_rejected_even_with_legacy_acceptance_true(missing):
    report = complete_report()
    if missing == "all":
        report["probes"] = None
    elif missing == "one":
        del report["probes"]["wall"]
    elif missing == "side":
        report["probes"]["wall"]["runs"][1]["side"] = "left"
    elif missing == "seed":
        report["probes"]["wall"]["runs"][1]["seed"] = 999
    else:
        del report["probes"]["wall"]["runs"][1]["success"]
    assert not check(report, live_timing())["eligible_for_promotion"]


@pytest.mark.parametrize("condition", ["none", "ghost"])
@pytest.mark.parametrize(
    "threats",
    [
        [],
        [
            {"side": 1, "hit": False, "min_distance": 1, "dodged": True},
            {"side": -1, "hit": False, "min_distance": 3, "dodged": True},
        ],
    ],
)
def test_near_threat_evidence_requires_both_sides(condition, threats):
    report = complete_report()
    for run in report["results"][f"adapter|{condition}"]["runs"]:
        run["threat_log"] = threats
    assert not check(report, live_timing())["behavioral_complete"]


@pytest.mark.parametrize("failure", ["missing", "changed", "errors"])
def test_missing_or_changed_provenance_blocks_promotion(failure):
    report = complete_report()
    if failure == "missing":
        del report["provenance"]
    elif failure == "changed":
        report["provenance"]["after"]["files"]["controller"] = "new-hash"
    else:
        report["provenance"]["before"]["errors"] = ["missing bundle"]
    assert not check(report, live_timing())["provenance_complete"]


@pytest.mark.parametrize(
    "timing",
    [
        {"mode": "offline", "simulated_seconds": 2, "wall_seconds": 1, "evidence": "x"},
        {
            "mode": "live_end_to_end",
            "simulated_seconds": 2,
            "wall_seconds": 0,
            "evidence": "x",
        },
        {
            "mode": "live_end_to_end",
            "simulated_seconds": 2,
            "wall_seconds": float("nan"),
            "evidence": "x",
        },
        {"mode": "live_end_to_end", "simulated_seconds": 2, "wall_seconds": 1},
    ],
)
def test_invalid_or_offline_timing_cannot_satisfy_a7(timing):
    assert not check(complete_report(), timing)["realtime_complete"]


@pytest.mark.parametrize("entrypoint", ["evaluate_free_roam", "adapter_check"])
def test_entrypoints_attach_gate_without_changing_acceptance(
    monkeypatch, tmp_path, entrypoint
):
    from fly_drone import distill, roam_eval
    from fly_drone import report_completeness as module

    report = complete_report()
    provenance = report["provenance"]["before"]
    monkeypatch.setattr(distill, "screen", lambda *a, **kw: deepcopy(report))
    monkeypatch.setattr(roam_eval, "skill_probes", lambda *a, **kw: report["probes"])
    monkeypatch.setattr(roam_eval, "acceptance", lambda *a: {"passed": True})
    monkeypatch.setattr(module, "capture_provenance", lambda *a, **kw: provenance)
    kwargs = {"bridge": "declared"} if entrypoint == "evaluate_free_roam" else {}
    result = getattr(roam_eval, entrypoint)(
        output=tmp_path / "report.json", episodes=2, seed_base=10, **kwargs
    )
    assert result["acceptance"] == {"passed": True}
    assert result["completeness"]["behavioral_complete"]
    assert not result["completeness"]["eligible_for_promotion"]


def test_provenance_fingerprints_actual_inputs_and_reports_missing_files(
    monkeypatch, tmp_path
):
    from fly_drone import adapter, brain
    from fly_drone import report_completeness as module

    names = (
        "manifest.json",
        "graph.bin",
        "neurons.bin",
        "cells.json",
        "groups.json",
        "sensory-mappings.json",
        "codec.json",
        "source.py",
    )
    for name in names:
        (tmp_path / name).write_bytes(b"original")
    monkeypatch.setattr(brain, "DATA", tmp_path)
    monkeypatch.setattr(brain, "ROOT", tmp_path)
    monkeypatch.setattr(adapter, "DEFAULT_BRIDGE_PATH", tmp_path / "codec.json")
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda cmd, **kw: b"source.py\0" if "ls-files" in cmd else b"revision\n",
    )
    first = module.capture_provenance("adapter")
    assert not first["errors"]
    assert first == module.capture_provenance("adapter")
    (tmp_path / "graph.bin").write_bytes(b"changed")
    second = module.capture_provenance("adapter")
    assert second["files"]["bundle/graph.bin"] != first["files"]["bundle/graph.bin"]
    (tmp_path / "source.py").write_bytes(b"changed")
    assert module.capture_provenance("adapter")["source"] != first["source"]
    (tmp_path / "codec.json").unlink()
    assert module.capture_provenance("adapter")["errors"]
