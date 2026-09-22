import json
import subprocess
import sys

import pytest
from fly_drone import experiments


@pytest.fixture
def setup(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "source.txt").write_text("frozen")
    subprocess.run(["git", "add", "source.txt"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repo,
        check=True,
    )
    root = tmp_path / "runs"
    manifest = {
        "run_id": "test",
        "stage": "S0",
        "hypothesis": "test",
        "decision_rules": "score >= 1",
        "identities": ["source.txt"],
        "cwd": str(repo),
        "max_seconds": 10,
        "workers": 1,
        "trials": [
            {
                "id": "one",
                "argv": [
                    sys.executable,
                    "-c",
                    "import pathlib; pathlib.Path('out.json').write_text('{\"score\": 1}')",
                ],
                "output": "out.json",
                "criterion": {"field": "score", "op": "gte", "value": 1},
            }
        ],
    }
    path = tmp_path / "manifest.json"

    def save():
        path.write_text(json.dumps(manifest))
        return path

    return repo, root, manifest, save


def test_success_and_resume_do_not_repeat(setup):
    repo, root, _, save = setup
    result = experiments.run(save(), root=root)
    assert result["state"] == "passed"
    original = (repo / "out.json").stat().st_mtime_ns
    result = experiments.run(run_id="test", root=root)
    assert result["state"] == "passed"
    assert (repo / "out.json").stat().st_mtime_ns == original
    assert (root / "test/events.jsonl").exists()


def test_process_success_is_not_scientific_pass(setup):
    _, root, manifest, save = setup
    del manifest["trials"][0]["criterion"]
    result = experiments.run(save(), root=root)
    assert result["state"] == "incomplete"
    assert result["completed_count"] == 1


def test_declared_failure(setup):
    _, root, manifest, save = setup
    manifest["trials"][0]["criterion"]["value"] = 2
    assert experiments.run(save(), root=root)["state"] == "failed"


def test_resume_rejects_changed_identity_and_completed_output(setup):
    repo, root, _, save = setup
    experiments.run(save(), root=root)
    (repo / "source.txt").write_text("changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        experiments.run(run_id="test", root=root)
    (repo / "source.txt").write_text("frozen")
    (repo / "out.json").write_text("{}")
    with pytest.raises(ValueError, match="completed output changed"):
        experiments.run(run_id="test", root=root)


def test_run_budget_is_cumulative(setup):
    _, root, manifest, save = setup
    manifest["max_seconds"] = 0.1
    manifest["trials"][0]["argv"] = [
        sys.executable,
        "-c",
        "import time; time.sleep(10)",
    ]
    result = experiments.run(save(), root=root)
    assert result["state"] == "budget_exhausted"
    result = experiments.run(run_id="test", root=root)
    assert result["state"] == "budget_exhausted"
    assert result["completed_count"] == 0


def test_stop_preserves_completed_trials_and_resume(setup):
    repo, root, manifest, save = setup
    # Second trial cancels once, then succeeds when explicitly resumed.
    manifest["trials"].append(
        {
            "id": "two",
            "output": "two.json",
            "argv": [
                sys.executable,
                "-c",
                "import pathlib,time; "
                "marker=pathlib.Path('marker'); "
                f"request=pathlib.Path({str(root / 'test/stop.request')!r}); "
                "first=not marker.exists(); marker.touch(); "
                "request.touch() if first else None; "
                "time.sleep(2) if first else None; "
                "pathlib.Path('two.json').write_text('{}')",
            ],
        }
    )
    assert experiments.run(save(), root=root)["state"] == "cancelled"
    assert len(experiments.status("test", root)["completed"]) == 1
    original = (repo / "out.json").stat().st_mtime_ns
    result = experiments.run(run_id="test", root=root)
    assert result["completed_count"] == 2
    assert (repo / "out.json").stat().st_mtime_ns == original


def test_stage_budget_and_manifest_limits(setup):
    _, root, manifest, save = setup
    experiments.run(save(), root=root)
    state_path = root / "test/status.json"
    state = json.loads(state_path.read_text())
    state["elapsed_seconds"] = 21600
    state_path.write_text(json.dumps(state))
    manifest["run_id"] = "second"
    assert experiments.run(save(), root=root)["state"] == "budget_exhausted"
    manifest["workers"] = 7
    with pytest.raises(ValueError, match="workers"):
        experiments.validate(manifest)


def test_existing_output_never_receives_credit(setup):
    repo, root, _, save = setup
    (repo / "out.json").write_text('{"score": 1}')
    result = experiments.run(save(), root=root)
    assert result["state"] == "incomplete"
    assert result["completed_count"] == 0
    assert "existing output" in result["reason"]


def test_resume_rejects_manifest_edit(setup):
    _, root, _, save = setup
    experiments.run(save(), root=root)
    path = root / "test/manifest.json"
    data = json.loads(path.read_text())
    data["max_seconds"] = 7200
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="manifest changed"):
        experiments.run(run_id="test", root=root)


def test_stage_budget_cannot_reset_with_another_root(setup):
    _, root, manifest, save = setup
    experiments.run(save(), root=root)
    path = root / "test/status.json"
    data = json.loads(path.read_text())
    data["elapsed_seconds"] = 21600
    path.write_text(json.dumps(data))
    manifest["run_id"] = "different"
    result = experiments.run(save(), root=root.parent / "different-root")
    assert result["state"] == "budget_exhausted"


def test_source_change_during_trial_is_not_accepted(setup):
    _, root, manifest, save = setup
    manifest["trials"][0]["argv"][-1] += (
        "; pathlib.Path('source.txt').write_text('modified')"
    )
    result = experiments.run(save(), root=root)
    assert result["state"] == "incomplete"
    assert "identity changed during execution" in result["reason"]


def test_untracked_source_is_fingerprinted_but_ignored_secrets_are_not(setup):
    repo, root, _, save = setup
    (repo / "python").mkdir()
    source = repo / "python/new_module.py"
    source.write_text("value = 1")
    experiments.run(save(), root=root)
    (repo / "ignored-token.txt").write_text("secret")
    assert experiments.run(run_id="test", root=root)["state"] == "passed"
    source.write_text("value = 2")
    with pytest.raises(ValueError, match="identity mismatch"):
        experiments.run(run_id="test", root=root)


def test_exited_leader_does_not_leave_live_descendants(setup):
    repo, root, manifest, save = setup
    manifest["trials"][0]["argv"] = [
        sys.executable,
        "-c",
        "import subprocess,sys,os,pathlib,json; "
        "subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "pathlib.Path('out.json').write_text(json.dumps({'score':1,'group':os.getpid()}))",
    ]
    assert experiments.run(save(), root=root)["state"] == "passed"
    group = json.loads((repo / "out.json").read_text())["group"]
    assert not experiments._group_live(group)


def test_cancelled_partial_output_stays_on_its_filesystem(setup):
    repo, root, manifest, save = setup
    manifest["trials"][0]["argv"] = [
        sys.executable,
        "-c",
        "import pathlib,time; pathlib.Path('out.json').write_text('{}'); "
        f"pathlib.Path({str(root / 'test/stop.request')!r}).touch(); time.sleep(30)",
    ]
    result = experiments.run(save(), root=root)
    assert result["state"] == "cancelled"
    assert not (repo / "out.json").exists()
    assert len(list(repo.glob("out.json.partial.*"))) == 1
    assert len(result["partial_outputs"]) == 1
