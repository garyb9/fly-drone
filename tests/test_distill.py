import json

import numpy as np
import pytest
from fly_drone import distill
from fly_drone.arena import ArenaSpec
from fly_drone.brain import ENCODER_VERSION, BrainRuntime
from fly_drone.encoder import LearnedEncoder
from fly_drone.teacher import DRIVES


def fake_data(path, n=40, encoder_version=ENCODER_VERSION):
    rng = np.random.default_rng(0)
    np.savez(
        path,
        x=rng.uniform(0, 1, (n, 2022)).astype(np.float16),
        y=rng.uniform(-1, 1, (n, 4)).astype(np.float32),
        drive=rng.integers(0, len(DRIVES), n).astype(np.int8),
        flight=np.repeat(np.arange(20), n // 20).astype(np.int32),
        dataset_hash=BrainRuntime().dataset_hash,
        encoder_version=encoder_version,
        student="None",
        beta=1.0,
    )
    return path


def test_collect_fit_and_export_a_free_roam_decoder(tmp_path):
    xs, ys, drives, flights = distill._collect_job(([11], 0.2, 1, None, 1.0, 0.2, 1))
    assert len(xs) == 5 and len(ys) == 5 and set(flights) == {11}
    assert all(x.shape == (2022,) and x.dtype == np.float16 for x in xs)
    assert all(0 <= d < len(DRIVES) for d in drives)
    rng = np.random.default_rng(0)
    n = 400
    path = tmp_path / "data.npz"
    np.savez(
        path,
        x=rng.uniform(0, 1, (n, 2022)).astype(np.float16),
        y=rng.uniform(-1, 1, (n, 4)).astype(np.float32),
        drive=rng.integers(0, len(DRIVES), n).astype(np.int8),
        flight=np.repeat(np.arange(20), n // 20).astype(np.int32),
        dataset_hash=BrainRuntime().dataset_hash,
        encoder_version=ENCODER_VERSION,
        student="None",
        beta=1.0,
    )
    report = distill.fit([path], tmp_path / "fit", net_arch=(16, 16), steps=20)
    assert report["held_out_flights"] == 2
    assert set(report["drives"]) == set(DRIVES)
    assert report["export_max_error"] <= 1e-4
    actor = json.loads((tmp_path / "fit" / "warm-actor.json").read_text())
    assert actor["action_limits"] == list(ArenaSpec().limits)
    assert [len(layer["bias"]) for layer in actor["layers"]] == [16, 16, 4]


def test_collect_with_a_learned_encoder_records_its_version(tmp_path):
    version = LearnedEncoder.fresh(seed=4).save(tmp_path / "e.pt")
    out = tmp_path / "data.npz"
    report = distill.collect(
        out,
        flights=1,
        seconds=0.2,
        workers=1,
        seed_base=11,
        levels=[1],
        encoder=tmp_path / "e.pt",
    )
    data = np.load(out)
    assert str(data["encoder_version"]) == version
    assert data["x"].shape == (report["samples"], 2022) and report["samples"] > 0
    with pytest.raises(ValueError, match="external"):
        distill.collect(out, flights=1, seconds=0.2, workers=1, encoder="external")


def test_load_rejects_data_from_a_different_encoder(tmp_path):
    learned = "learned-v5:" + "0" * 16
    digest = BrainRuntime().dataset_hash
    v4 = fake_data(tmp_path / "v4.npz")
    v5 = fake_data(tmp_path / "v5.npz", encoder_version=learned)
    assert len(distill._load([v4], digest)[0]) == 40
    assert len(distill._load([v5], digest, learned)[0]) == 40
    for path, expected, got in (
        (v4, learned, ENCODER_VERSION),
        (v5, ENCODER_VERSION, learned),
    ):
        with pytest.raises(ValueError) as error:
            distill._load([path], digest, expected)
        assert expected in str(error.value) and got in str(error.value)
        assert str(path) in str(error.value)


def test_screen_job_reports_rates_for_teacher_and_baselines():
    for controller in ("teacher", "cue_script", "random"):
        name, ablation, runs = distill._screen_job((controller, [3], 0.4, 3, "none"))
        assert name == controller and ablation == "none" and len(runs) == 1
        run = runs[0]
        assert run["beacons_per_min"] >= 0 and run["collisions_per_min"] >= 0
        assert 0 <= run["slow_fraction"] <= 1
    summary = distill.summarise(runs)
    assert "threat_dodge_rate" in summary and summary["runs"][0]["seed"] == 3


def test_screen_warns_when_no_controller_would_fly_the_given_encoder(tmp_path, capsys):
    distill.screen(
        ["teacher"],
        tmp_path / "screen.json",
        seeds=1,
        seconds=0.2,
        level=0,
        workers=1,
        seed_base=1,
        encoder="unused.pt",
    )
    assert "ignored" in capsys.readouterr().err
