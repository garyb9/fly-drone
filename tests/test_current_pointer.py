import json

from fly_drone.brain import ENCODER_VERSION
from fly_drone.current_pointer import current_entry, current_policies, scan, update


def _write_actor(path, encoder_version=ENCODER_VERSION):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "encoder_version": encoder_version,
                "layers": [],
                "action_limits": [1, 1, 1, 1],
                "feature_ids": [],
            }
        )
    )


def test_scan_falls_back_to_dagger_it0_when_nothing_else_exists(tmp_path):
    _write_actor(tmp_path / "runs" / "v5" / "dagger" / "it0" / "warm-actor.json")
    entry = scan(root=tmp_path)
    assert entry["status"] == "interim"
    assert entry["decoder"] == "runs/v5/dagger/it0/warm-actor.json"
    assert entry["encoder"] is None
    assert entry["encoder_version"] == ENCODER_VERSION


def test_scan_prefers_the_frozen_v6_pair_over_the_v5_fallback(tmp_path):
    _write_actor(tmp_path / "runs" / "v5" / "dagger" / "it0" / "warm-actor.json")
    _write_actor(
        tmp_path / "runs" / "v6" / "round0" / "decoder.json",
        encoder_version="learned-v6:01280e414169ff9c",
    )
    (tmp_path / "runs" / "v6" / "clone").mkdir(parents=True)
    (tmp_path / "runs" / "v6" / "clone" / "encoder.pt").write_bytes(b"x")
    entry = scan(root=tmp_path)
    assert entry["status"] == "interim"
    assert entry["encoder"] == "runs/v6/clone/encoder.pt"
    assert entry["decoder"] == "runs/v6/round0/decoder.json"
    assert entry["encoder_version"] == "learned-v6:01280e414169ff9c"


def test_scan_reports_none_when_nothing_usable_exists(tmp_path):
    entry = scan(root=tmp_path)
    assert entry == {
        "status": "none",
        "encoder": None,
        "encoder_version": None,
        "decoder": None,
        "source_run": None,
        "gate_report": None,
    }


def test_scan_rejects_unsupported_task(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        scan(task="visual", root=tmp_path)


def test_scan_prefers_a_fully_passing_evaluation_over_the_fallback(tmp_path):
    _write_actor(tmp_path / "runs" / "v5" / "dagger" / "it0" / "warm-actor.json")
    run_dir = tmp_path / "runs" / "v5" / "roundN"
    _write_actor(
        run_dir / "decoder.json", encoder_version="learned-v5:deadbeefcafef00d"
    )
    report = run_dir / "evaluation-free-roam.json"
    report.write_text(
        json.dumps(
            {
                "task": "free_roam",
                "policy": str(run_dir / "decoder.json"),
                "encoder": "runs/v5/clone/encoder.pt",
                "acceptance": {
                    "a1": {"passed": True},
                    "a2": {"passed": True},
                },
            }
        )
    )
    entry = scan(root=tmp_path)
    assert entry["status"] == "gated"
    assert entry["decoder"] == str(run_dir / "decoder.json")
    assert entry["encoder"] == "runs/v5/clone/encoder.pt"


def test_scan_ignores_an_evaluation_with_a_failing_criterion(tmp_path):
    _write_actor(tmp_path / "runs" / "v5" / "dagger" / "it0" / "warm-actor.json")
    run_dir = tmp_path / "runs" / "v5" / "roundN"
    _write_actor(
        run_dir / "decoder.json", encoder_version="learned-v5:deadbeefcafef00d"
    )
    report = run_dir / "evaluation-free-roam.json"
    report.write_text(
        json.dumps(
            {
                "task": "free_roam",
                "policy": str(run_dir / "decoder.json"),
                "acceptance": {
                    "a1": {"passed": True},
                    "a2": {"passed": False},
                },
            }
        )
    )
    entry = scan(root=tmp_path)
    assert entry["status"] == "interim"


def test_scan_skips_a_malformed_report_instead_of_crashing(tmp_path):
    _write_actor(tmp_path / "runs" / "v5" / "dagger" / "it0" / "warm-actor.json")
    bad = tmp_path / "runs" / "v5" / "roundN" / "evaluation-free-roam.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{not json")
    entry = scan(root=tmp_path)
    assert entry["status"] == "interim"


def test_update_writes_the_manifest_and_preserves_other_tasks(tmp_path):
    _write_actor(tmp_path / "runs" / "v5" / "dagger" / "it0" / "warm-actor.json")
    manifest = tmp_path / "current-policies.json"
    manifest.write_text(
        json.dumps({"note": "x", "policies": {"other_task": {"status": "interim"}}})
    )
    payload = update(root=tmp_path, manifest=manifest)
    assert payload["policies"]["free_roam"]["status"] == "interim"
    assert payload["policies"]["other_task"]["status"] == "interim"
    on_disk = json.loads(manifest.read_text())
    assert set(on_disk["policies"]) == {"free_roam", "other_task"}


def test_update_dry_run_does_not_write(tmp_path):
    _write_actor(tmp_path / "runs" / "v5" / "dagger" / "it0" / "warm-actor.json")
    manifest = tmp_path / "current-policies.json"
    update(root=tmp_path, manifest=manifest, dry_run=True)
    assert not manifest.is_file()


def test_current_policies_splits_present_and_missing(tmp_path):
    manifest = tmp_path / "current-policies.json"
    (tmp_path / "runs" / "present").mkdir(parents=True)
    (tmp_path / "runs" / "present" / "actor.json").write_text("{}")
    manifest.write_text(
        json.dumps(
            {
                "policies": {
                    "free_roam": {
                        "status": "interim",
                        "decoder": "runs/present/actor.json",
                    },
                    "other": {"status": "none", "decoder": None},
                }
            }
        )
    )
    present, missing = current_policies(manifest, tmp_path)
    assert list(present) == ["free_roam"]
    assert missing == {}


def test_current_entry_returns_none_without_a_manifest(tmp_path):
    assert current_entry("free_roam", tmp_path / "missing.json", tmp_path) is None


def test_current_entry_resolves_decoder_and_encoder_paths(tmp_path):
    manifest = tmp_path / "current-policies.json"
    (tmp_path / "runs" / "clone").mkdir(parents=True)
    (tmp_path / "runs" / "clone" / "encoder.pt").write_bytes(b"x")
    (tmp_path / "runs" / "dec").mkdir(parents=True)
    (tmp_path / "runs" / "dec" / "actor.json").write_text("{}")
    manifest.write_text(
        json.dumps(
            {
                "policies": {
                    "free_roam": {
                        "status": "gated",
                        "decoder": "runs/dec/actor.json",
                        "encoder": "runs/clone/encoder.pt",
                    }
                }
            }
        )
    )
    entry = current_entry("free_roam", manifest, tmp_path)
    assert entry["decoder"].endswith("runs/dec/actor.json")
    assert entry["encoder"].endswith("runs/clone/encoder.pt")
