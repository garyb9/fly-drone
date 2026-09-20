import numpy as np
import pytest
from fly_drone import motion_stats as ms

DT = ms.FRAME_SECONDS


def test_clean_square_wave_gives_known_bouts():
    # 4 move bouts of 10 frames (0.4 s) and 4 pause bouts of 10 frames.
    speeds = np.tile(np.concatenate([np.ones(10), np.zeros(10)]), 4)
    stats = ms.episode_stats(speeds, np.zeros(len(speeds)), np.zeros((len(speeds), 2)))
    assert stats["mean_move_bout_s"] == pytest.approx(10 * DT)
    assert stats["median_move_bout_s"] == pytest.approx(10 * DT)
    assert stats["mean_pause_bout_s"] == pytest.approx(10 * DT)
    assert stats["bouts_per_min"] == pytest.approx(4 / (len(speeds) * DT / 60.0))


def test_endless_move_is_one_bout():
    speeds = np.ones(200)
    stats = ms.episode_stats(speeds, np.zeros(200), np.zeros((200, 2)))
    assert stats["mean_move_bout_s"] == pytest.approx(200 * DT)
    assert stats["mean_pause_bout_s"] is None
    assert stats["bouts_per_min"] == pytest.approx(1 / (200 * DT / 60.0))


def test_impulse_train_gives_known_saccades():
    # 4 saccades of exactly 1 rad, 0.8 s apart, over 4 s.
    n = 100
    headings = np.zeros(n)
    for count, frame in enumerate((20, 40, 60, 80), start=1):
        headings[frame:] = count
    stats = ms.episode_stats(np.ones(n), headings, np.zeros((n, 2)))
    assert stats["saccade_rate_hz"] == pytest.approx(4 / (n * DT))
    assert stats["mean_saccade_amplitude_rad"] == pytest.approx(1.0)
    assert stats["median_isi_s"] == pytest.approx(20 * DT)


def test_straight_line_msd_is_ballistic():
    n = 300
    positions = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    stats = ms.episode_stats(np.ones(n), np.zeros(n), positions)
    assert stats["msd_exponent"] == pytest.approx(2.0, abs=1e-6)


def test_frozen_position_has_no_exponent():
    n = 300
    stats = ms.episode_stats(np.zeros(n), np.zeros(n), np.zeros((n, 2)))
    assert stats["msd_exponent"] is None
    assert stats["mean_move_bout_s"] is None
    assert stats["bouts_per_min"] == pytest.approx(0.0)


def test_short_episode_has_no_exponent():
    n = 100  # < 2 * (5.0 / dt)
    positions = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    stats = ms.episode_stats(np.ones(n), np.zeros(n), positions)
    assert stats["msd_exponent"] is None


@pytest.mark.parametrize(
    "speeds, headings, positions",
    [
        ([], [], []),
        ([0.0], [0.0], [[0.0, 0.0]]),
        ([1.0, 1.0], [0.0, 0.0], [[0.0, 0.0], [1.0, 0.0]]),
    ],
)
def test_degenerate_inputs_return_no_nan(speeds, headings, positions):
    stats = ms.episode_stats(speeds, headings, positions)
    assert stats  # all keys present
    for value in stats.values():
        assert value is None or np.isfinite(value)


def test_constant_heading_has_no_saccades():
    n = 60
    stats = ms.episode_stats(np.ones(n), np.zeros(n), np.zeros((n, 2)))
    assert stats["saccade_rate_hz"] == pytest.approx(0.0)
    assert stats["mean_saccade_amplitude_rad"] is None


def test_every_fly_reference_carries_a_citation():
    for key, ref in ms.FLY_REFERENCE.items():
        assert ref["citation"] in ms.CITATIONS, key
        assert "units" in ref and "condition" in ref


def test_scored_reference_windows_are_ordered():
    for key in ("saccade_rate_hz", "mean_saccade_amplitude_rad", "msd_exponent"):
        ref = ms.FLY_REFERENCE[key]
        assert ref["low"] < ref["high"]


def test_in_window_handles_missing_and_present_values():
    assert ms.in_window("msd_exponent", None) == (None, (1.2, 1.95))
    passed, window = ms.in_window("msd_exponent", 1.8)
    assert passed is True and window == (1.2, 1.95)
    passed, _ = ms.in_window("msd_exponent", 2.0)
    assert passed is False
