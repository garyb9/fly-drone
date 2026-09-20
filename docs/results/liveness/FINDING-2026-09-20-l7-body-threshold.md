# L7 saccadic turning is unpassable on the quadrotor body

**Status:** recorded 2026-09-20. Free audit (`scripts/yaw_audit.py`), 3 × 60 s seeds on codec v2,
level 3, plus a 120 s teacher / scripted reference. **No threshold was changed; L7 is deferred, not
relaxed** (user-approved, §5). L7 was already failing in the P4 2×2; this explains why, and it is
not the connectome's fault.

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

## 5. Decision (adopted): defer L7 on the drone body

The user chose to **defer L7** until the fly-like-body milestone, keeping the fly's own value as the
target. Implemented as a declared, automatic rule, not a hand-tuned exception:

- `liveness.liveness(..., yaw_limit=...)` takes the body's yaw authority (default
  `plant.LIMITS[3]`). When it is below `motion_stats.SACCADE_THRESHOLD`, **L7 is marked `deferred`**
  — reported as `passed: null` with a reason and the fly target (`0.5 Hz`, `3 rad/s`) — and is
  **excluded from the aggregate** `passed`. It is not a silent pass: `deferred` and `reason` are in
  the report, and the criterion is distinct from `not_evaluable` (an undefined statistic, which
  still fails).
- The moment a fly-like body raises the yaw limit above the detector, L7 becomes evaluable again
  with no code change; `tests/test_liveness.py` covers both directions.

Recomputed on the P4 smoke artifacts (the raw per-seed summaries are unchanged):

| cell        | passed | deferred | not-evaluable | failing |
| ----------- | ------ | -------- | ------------- | ------- |
| v1 canonical| no     | L7       | —             | L1, L2, L6 |
| **v2 canonical** | no | L7      | —             | **L2**  |
| v1 tonic    | no     | L7       | —             | L1, L2, L6 |
| v2 tonic    | no     | L7       | —             | L2, L3, L8 |

So the codec-v2 cell is now **one criterion from alive**, and the sole remaining liveness gap is
**L2 coverage** (0.104 vs 0.15). That is the next workstream. `ACCEPTANCE` A1–A7 were untouched.

Options **A (fly-like body first)** and **B (re-express L7 body-relative)** were considered and not
taken; B was rejected as muddy because even the accepted RL teacher sits below the 0.2 Hz floor when
scaled to the body.
