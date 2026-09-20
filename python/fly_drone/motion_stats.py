"""Motion-structure statistics for the fly-anchored liveness bar (P4, workstream A).

The existing liveness criteria (L1 mobility, L2 exploration) only see aggregates and cannot
see the *structure* of motion: a brick drifting at 0.06 m/s on a constant heading passes both.
P4 adds L6-L8, which measure intermittency, saccadic turning and exploration structure, and
anchor their thresholds on published *Drosophila* free-flight statistics rather than on the RL
teacher.

This module is pure: functions over 1-D arrays, no simulation and no imports from ``env``.
Every function tolerates degenerate input (all-stationary, one frame, constant heading) and
returns ``None`` rather than raising or emitting ``nan``. ``None`` propagates to "criterion not
evaluable", never to "passed".

See ``docs/references.md`` for the sources behind every ``FLY_REFERENCE`` window.
"""

import numpy as np

# Frame rate of the rollout loop: one ``env.step`` = 0.04 s (``env.FRAME_SECONDS``).
FRAME_SECONDS = 0.04
# A frame counts as moving at the same threshold the ``slow_fraction`` metric uses, so bouts
# and L1 agree by construction (``distill.py`` stuck test, ``env.py`` speed).
MOVING_THRESHOLD = 0.05

# Declared saccade detector. The fly's own saccades exceed 2000 deg/s (~35 rad/s, Fry et al.
# 2005), but the rollout samples at 25 Hz, so a 90 deg turn is smeared over two-three frames;
# the threshold is set well below the fly peak and above smooth optomotor steering (a few
# rad/s). Refractory matches the ~50 ms saccade duration (Reynolds & Frye 2007).
SACCADE_THRESHOLD = 3.0
SACCADE_REFRACTORY = 0.1

# Mean-squared-displacement fit window, seconds.
MSD_MIN_TAU = 0.2
MSD_MAX_TAU = 5.0

# Publication keys used by ``FLY_REFERENCE``. Full citations live in ``docs/references.md``.
CITATIONS = {
    "tammero2002": (
        "Tammero & Dickinson 2002, J Exp Biol 205:327-343 (free flight, 1 m arena): body "
        "saccades turn the heading ~90 deg over ~50 ms."
    ),
    "schnell2017": (
        "Schnell, Ros & Dickinson 2017, Curr Biol 27:1200-1205: free-flight chamber analysis "
        "reports spontaneous saccades at ~0.5 Hz."
    ),
    "reynolds2007": (
        "Reynolds & Frye 2007, PLoS ONE 2(4):e354 (free flight, 1 m arena): inter-saccade "
        "segments are scale-free with structure-function exponent alpha ~ 0.9 "
        "(second moment ~ tau^1.8); saccade duration ~50 ms."
    ),
    "david1982": (
        "David 1982, as reported in Fry, Sayaman & Dickinson 2005, J Exp Biol 208:2303-2315: "
        "free flight preferred ground speed ~0.10 m/s; inter-saccade initial speed "
        "0.285-0.356 m/s (Reynolds & Frye 2007)."
    ),
    "fry2005": (
        "Fry, Sayaman & Dickinson 2005, J Exp Biol 208:2303-2315: body angular velocity during "
        "free-flight saccades can exceed 2000 deg/s (~35 rad/s)."
    ),
    "martin1999": (
        "Martin et al. 1999, as reported in Wolf et al. 2002, J Neurosci 22:11035-11044: adult "
        "*walking* occurs in intermittent bouts of ~1-3 min initially and ~1 s after "
        "acclimation, separated by pauses. Used here only as a broad order-of-magnitude proxy; "
        "no equivalent free-flight stop-start bout distribution is published (flies hover "
        "rather than stop), so the pause window is reported as context and not scored."
    ),
}


# Fly-anchored reference values and the windows L6-L8 score against. Each entry carries the
# published value (where one exists), the species/condition, a citation key and the
# order-of-magnitude window used by ``liveness``. A window is only as tight as the published
# number supports; every threshold traces to a line in ``docs/references.md``.
FLY_REFERENCE = {
    "saccade_rate_hz": {
        "value": 0.5,
        "low": 0.2,
        "high": 2.0,
        "units": "Hz",
        "condition": "free flight, spontaneous body saccades",
        "citation": "schnell2017",
    },
    "mean_saccade_amplitude_rad": {
        "value": 1.5708,
        "low": 0.8,
        "high": 2.1,
        "units": "rad",
        "condition": "free flight, ~90 deg turn over ~50 ms",
        "citation": "tammero2002",
    },
    "msd_exponent": {
        "value": 1.8,
        "low": 1.2,
        "high": 1.95,
        "units": "-",
        "condition": "free flight, tau <= 1 s; super-diffusive, below ballistic",
        "citation": "reynolds2007",
    },
    "median_speed_ms": {
        "value": 0.1,
        "low": 0.04,
        "high": 0.35,
        "units": "m/s",
        "condition": "free flight preferred ground speed; context, not a criterion",
        "citation": "david1982",
    },
    "mean_move_bout_s": {
        "value": None,
        "low": 0.3,
        "high": 10.0,
        "units": "s",
        "condition": "adult walking proxy; no free-flight stop-start bout distribution",
        "citation": "martin1999",
    },
    "mean_pause_bout_s": {
        "value": None,
        "low": 0.15,
        "high": 10.0,
        "units": "s",
        "condition": "adult walking proxy; context only, not scored",
        "citation": "martin1999",
    },
    "bouts_per_min": {
        "value": None,
        "low": 1.0,
        "high": 120.0,
        "units": "1/min",
        "condition": "adult walking proxy; catches the brick and the endless cruise",
        "citation": "martin1999",
    },
    "saccade_threshold_rad_s": {
        "value": 3.0,
        "units": "rad/s",
        "condition": "declared detector threshold; below the fly peak, above smooth steering",
        "citation": "fry2005",
    },
    "saccade_refractory_s": {
        "value": 0.1,
        "units": "s",
        "condition": "declared detector minimum separation; saccade duration ~50 ms",
        "citation": "reynolds2007",
    },
}


def _runs(mask):
    """Contiguous runs of ``True`` in a boolean array as ``(start, stop)`` pairs."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    edges = np.flatnonzero(np.diff(mask.astype(np.int8)))
    starts = np.concatenate([[0], edges + 1])
    stops = np.concatenate([edges + 1, [mask.size]])
    return [(int(s), int(e)) for s, e in zip(starts, stops, strict=True) if mask[s]]


def _bout_stats(speeds, dt):
    """Move/pause bout durations from a speed series; ``None`` where undefined."""
    moving = speeds >= MOVING_THRESHOLD
    move = [(e - s) * dt for s, e in _runs(moving)]
    pause = [(e - s) * dt for s, e in _runs(~moving)]
    out = {
        "mean_move_bout_s": float(np.mean(move)) if move else None,
        "median_move_bout_s": float(np.median(move)) if move else None,
        "mean_pause_bout_s": float(np.mean(pause)) if pause else None,
        "bouts_per_min": float(len(move) / (len(speeds) * dt / 60.0)),
        "bout_length_cv": None,
    }
    if len(move) >= 2:
        mean = float(np.mean(move))
        out["bout_length_cv"] = float(np.std(move) / mean) if mean > 0 else None
    return out


def _saccade_stats(headings, dt):
    """Saccade rate, amplitude and inter-saccade intervals from a heading series."""
    n = len(headings)
    out = {
        "saccade_rate_hz": 0.0,
        "mean_saccade_amplitude_rad": None,
        "median_isi_s": None,
        "isi_cv": None,
    }
    if n < 3:
        return out
    omega = np.diff(np.unwrap(headings)) / dt
    spans = _runs(np.abs(omega) >= SACCADE_THRESHOLD)
    peaks, amplitudes = [], []
    last = -np.inf
    for start, stop in spans:
        peak = start + int(np.argmax(np.abs(omega[start:stop])))
        if peak * dt - last < SACCADE_REFRACTORY:
            continue
        peaks.append(peak)
        amplitudes.append(float(np.sum(omega[start:stop]) * dt))
        last = peak * dt
    out["saccade_rate_hz"] = float(len(peaks) / (n * dt))
    if amplitudes:
        out["mean_saccade_amplitude_rad"] = float(
            np.mean(np.abs(np.asarray(amplitudes)))
        )
    if len(peaks) >= 2:
        isi = np.diff(peaks) * dt
        out["median_isi_s"] = float(np.median(isi))
        mean = float(np.mean(isi))
        out["isi_cv"] = float(np.std(isi) / mean) if mean > 0 else None
    return out


def _msd_exponent(positions, dt):
    """Least-squares slope of log MSD against log tau, or ``None`` if too short."""
    n = len(positions)
    # The spec's gate: fewer than 2*(MSD_MAX_TAU/dt) frames cannot support the fit.
    if n < 2 * int(MSD_MAX_TAU / dt):
        return None
    lags = np.arange(
        max(1, int(round(MSD_MIN_TAU / dt))),
        int(round(MSD_MAX_TAU / dt)) + 1,
    )
    taus, msd = [], []
    for lag in lags:
        delta = positions[lag:] - positions[:-lag]
        value = float(np.mean(np.sum(delta * delta, axis=1)))
        if value > 0 and np.isfinite(value):
            taus.append(lag * dt)
            msd.append(value)
    if len(taus) < 2:
        return None
    slope, _ = np.polyfit(np.log(taus), np.log(msd), 1)
    return float(slope) if np.isfinite(slope) else None


def episode_stats(speeds, headings, positions, dt=FRAME_SECONDS):
    """Per-episode motion-structure scalars; all keys always present, ``None`` where undefined.

    ``speeds`` (m/s), ``headings`` (rad, body yaw) and ``positions`` (2-D, metres) are the
    per-frame series kept local by the rollout loop. Callers reduce here rather than storing
    the raw series.
    """
    speeds = np.asarray(speeds, dtype=float).ravel()
    headings = np.asarray(headings, dtype=float).ravel()
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] < 2:
        positions = np.zeros((len(speeds), 2), dtype=float)
    out = {}
    if len(speeds) == 0:
        out.update(
            {
                "mean_move_bout_s": None,
                "median_move_bout_s": None,
                "mean_pause_bout_s": None,
                "bouts_per_min": None,
                "bout_length_cv": None,
                "saccade_rate_hz": None,
                "mean_saccade_amplitude_rad": None,
                "median_isi_s": None,
                "isi_cv": None,
                "median_speed": None,
                "p90_speed": None,
                "speed_cv": None,
                "msd_exponent": None,
            }
        )
        return out
    out.update(_bout_stats(speeds, dt))
    out.update(_saccade_stats(headings, dt))
    out["median_speed"] = float(np.median(speeds))
    out["p90_speed"] = float(np.percentile(speeds, 90))
    mean = float(np.mean(speeds))
    out["speed_cv"] = float(np.std(speeds) / mean) if mean > 0 else None
    out["msd_exponent"] = _msd_exponent(positions[:, :2], dt)
    return out


def in_window(key, value):
    """``(passed, window)`` for a statistic against its cited ``FLY_REFERENCE`` window."""
    ref = FLY_REFERENCE[key]
    window = (ref["low"], ref["high"])
    if value is None or not np.isfinite(value):
        return None, window
    return bool(window[0] <= value <= window[1]), window
