# L7 saccadic turning is unpassable on the quadrotor body

**Status:** recorded 2026-09-20. Free audit (`scripts/yaw_audit.py`), 3 × 60 s seeds on codec v2,
level 3, plus a 120 s teacher / scripted reference. **No threshold was changed.** L7 was already
failing in the P4 2×2; this explains why, and it is not the connectome's fault.

Artifacts: [`yaw-audit.json`](yaw-audit.json), [`../../specs/2026-09-20-ongoing-state-and-faithful-readout.md`](../../superpowers/specs/2026-09-20-ongoing-state-and-faithful-readout.md).

## 1. The mismatch

L7 scores the free-flight saccade rate (~0.5 Hz, Schnell et al. 2017) with a declared detector that
fires when the **heading rate exceeds 3 rad/s** — chosen "well below the fly's own peak (~35 rad/s,
Fry et al. 2005)". But the simulated body's yaw-rate command limit is **0.8 rad/s**
(`plant.LIMITS[3]`; the 16 × 16 m arena limits are `[0.7, 0.5, 0.3, 0.8]`). A controlled turn
therefore commands at most 0.8 rad/s, so the detector can only fire on a *discontinuity*, never on
flight.

## 2. Measurement (codec v2, `yaw-audit.json`)

| seed | all-frame max \|ω\| | clean-frame max \|ω\| | detections ≥ 3 rad/s | of those at a respawn | teleport frames |
| ---- | ------------------ | -------------------- | -------------------- | --------------------- | --------------- |
| 0    | 7.910              | 0.490                | 3                    | 3                     | 30              |
| 1    | 2.821              | 1.116                | 0                    | 0                     | 30              |
| 2    | 3.961              | 0.869                | 1                    | 1                     | 25              |

- The **clean-frame ceiling is 1.116 rad/s** (`threshold_reachable = false`): every controlled turn
  stays below the 3 rad/s detector even where the plant transiently overshoots its 0.8 rad/s command
  limit.
- **All 4 detections are crash-respawn teleports** (position jumps up to 1.33 m in one 0.04 s frame),
  which the heading derivative reads as a 4–8 rad/s "saccade". The v2 2×2's `saccade_rate_hz =
  0.013` was therefore a respawn artifact, not behaviour.

## 3. This is a body gap, not a brain gap

Re-running the detector relative to the body's own envelope (120 s, clean frames) shows the ceiling
is in the *body*, and that no controller on this body produces fly-like pulsatile reorientations:

| controller            | clean max \|ω\| | p99 \|ω\| | events ≥ 0.4 rad/s | events ≥ 0.6 rad/s |
| --------------------- | --------------- | --------- | ------------------ | ------------------ |
| codec v1              | 0.862           | 0.705     | 0.042 Hz           | 0.042 Hz           |
| codec v2              | 3.666           | 0.446     | 0.086 Hz           | 0.017 Hz           |
| RL teacher (skill)    | 1.079           | 0.925     | 0.167 Hz           | 0.092 Hz           |
| scripted light-seeker | 1.342           | 1.335     | 1.439 Hz           | 1.347 Hz           |

Even the accepted RL skill stays below the 0.2 Hz floor of L7's window when the threshold is scaled
to the body; the scripted seeker only exceeds it because it saturates the yaw command continuously
(a sustained turn, the opposite of a saccade). So L7's 0.2–2 Hz at 3 rad/s encodes a **fly-body
property** — a fast, low-inertia body that can saccade at ~35 rad/s — that a 0.8 rad/s quadrotor does
not have. It is the same kind of structural mismatch the P4 spec found in L1/L2 (which were
teacher-anchored); L7 is **fly-body-anchored**.

## 4. Why the free-roam arena is not the excuse

Level 3 has pillars, a beacon and threats, so there is structure to saccade to — as the scripted
seeker shows by turning at limit. The deficit is neither richness of the arena nor the readout: the
body cannot turn fast enough to register, and no controller we have (including the teacher) produces
discrete high-rate reorientations.

## 5. Decision required (bar / body — user sign-off)

L7 is a pre-registered criterion. Nothing below is taken without the user's choice.

- **A — Fly-like body first.** Raise the plant's yaw authority/rate toward fly values so saccades
  are physically possible, then L7 stands as written. This is the roadmap's stated destination, but
  it changes the body and invalidates/needs re-validation of every accepted actor and A1–A7.
- **B — Re-express L7 on the current body (interim).** Keep L7's *intent* (discrete reorientations,
  not smooth cruising) but anchor the detector on the body's own yaw envelope, and record the fly's
  35 rad/s / 0.5 Hz as the unimplemented fidelity target for the fly-like-body milestone.
- **C — Defer L7.** Mark it not-evaluable on a non-fly body and pursue the remaining unmet
  criterion, L2 coverage (0.104 vs 0.15), with a declared search process.

The other P4 gap, **L2 coverage**, is independent of all this and remains open.
