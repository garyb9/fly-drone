# Free roam: one connectome decoder flies an open arena

Code: `python/fly_drone/{arena,plant,env,feasibility,teacher,distill,attribution,roam_eval,server}.py`.
Plan: [`superpowers/plans`](superpowers/plans/) (free-roam plan, 2026-09-13).

## 1. What "the brain controls it" means here

| Contract                       | How it is enforced                                                                                                              |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| Only neurons reach the decoder | The observation is the 2,022 descending/motor traces, as in every other task. No task id, pose, beacon position or last action. |
| One decoder, no mode switching | The live server loads one free-roam actor and never swaps it based on scenario or state.                                        |
| Frozen brain                   | Wiring, weights, neuron parameters, tonic bias and encoder v4 are unchanged. Only the MLP decoder is learned.                   |
| Realisable teacher             | Labels use simulator geometry only for objects the eyes can see: in the field, unoccluded, within measured cue range (§4).      |
| Legacy results reproducible    | The accepted visual/looming room is byte-identical (`test_legacy_room_mjcf_unchanged`); per-seed replays match.                 |
| Pre-registered acceptance      | `roam_eval.ACCEPTANCE` was committed before any free-roam decoder was trained (§6).                                             |

## 2. Arena

`DronePlant(arena=ArenaSpec())` builds a separate model; `arena=None` builds the legacy room.

| Element   | Value                                                                                      | Why                                                      |
| --------- | ------------------------------------------------------------------------------------------ | -------------------------------------------------------- |
| Room      | 16 × 16 m interior, walls at ±8 m, 3 m tall, grey (0.35)                                   | Larger than the 8 m trial room                           |
| Wall band | near-black, z 0.8–1.2 m on every wall                                                      | Approaching a wall expands a dark region: a loom cue     |
| Pillars   | up to 16 mocap cylinders, r 0.3 m, grey with a 0.3 m near-black ring centred at z 1 m      | Moved mocap bodies register contact; see §3 for the ring |
| Beacon    | emissive sphere, r 0.3 m, one at a time, collected within 0.5 m                            | Light cue readable to 12 m                               |
| Threat    | dark sphere, r 0.25 m, thrown every 8–20 s from 3–4 m ahead at 0.8–1.4 m/s                 | The looming pathway in flight                            |
| Floor     | uniform grey                                                                               | The upstream checker sits at the dark threshold          |
| Lighting  | headlight ambient 0.6, diffuse 0.3                                                         | §3                                                       |
| Limits    | `[0.7, 0.5, 0.3, 0.8]` m/s, m/s, m/s, rad/s (stored in each actor; mismatches are refused) | §3                                                       |

Layouts are seeded: pillars keep ≥ 2.5 m of gap, stay ≥ 1 m from walls and ≥ 2 m from the spawn,
and a 0.25 m occupancy grid must stay connected. Half of all beacons spawn outside the field of
view, so search is required. Levels: L0 beacons only, L1 + 6 pillars, L2 + 16 pillars, L3 + threats.
A crash in continuous mode respawns the drone at the nearest clear point without resetting the
brain; only the encoder's previous frame is cleared, so the teleport is not read as looming.

## 3. Sensory tuning (encoder v4 unchanged)

The loom cue is dark-area growth between frames (`sensory-model.md` §2). Its firing threshold is
about 5 changed pixels per 3,072, so anything that changes dark-pixel counts while the drone turns
reads as a threat. Every row below is a measured kinematic scan (`fly-drone roam-feasibility`).

| Problem found                                                     | Measurement                                                                        | Fix                                                                          |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Eyes blind within 1.8 m                                           | camera near clip = `znear × extent`; parked bodies made extent 176 m (clip 1.76 m) | pin arena `statistic extent=17` (clip 0.17 m, as legacy)                     |
| Grey surfaces at grazing angles rendered below the dark threshold | turning with **nothing dark** in the room: loom in 59% of frames                   | flat lighting: 0%                                                            |
| Fully dark pillars sweep through the view while turning           | 64–70% of turning frames                                                           | grey pillar with a 0.3 m dark ring: 0.3% / 5% / 19% at 0.4 / 0.8 / 1.2 rad/s |
| A 1.2 m wall band changes apparent height near corners            | 55–57% near a corner                                                               | 0.4 m band: 17% / 35% at 0.8 / 1.2 rad/s near a corner; 0% at the centre     |

Resulting cue ranges (surface distance at which loom fires on 3 consecutive frames):

| Approach      | 0.5 m/s | 0.7 m/s | 1.0 m/s |
| ------------- | ------- | ------- | ------- |
| Ringed pillar | 0.90 m  | 1.14 m  | 1.18 m  |
| Wall band     | 1.56 m  | —       | 2.12 m  |

Forward limit 0.7 m/s keeps about 1.6 s of pillar warning; yaw limit 0.8 rad/s keeps turning false
loom near 5%. Forward flight over the floor produced no false loom. Beacon light cue: 0.97 at 8 m,
0.35 at 12 m (firing threshold 0.22), with the left/right sign correct at ±0.3 rad.

## 4. Teacher (labels only)

`teacher.teacher_action(env)` blends four drives with sigmoid weights:

| Drive           | Gate                                                             | Label (normalised)                 |
| --------------- | ---------------------------------------------------------------- | ---------------------------------- |
| Evade threat    | threat in view, unoccluded (`mj_ray`), within 2 m                | full lateral away from its side    |
| Avoid           | pillar within 1.2 m or wall within 2.0 m, inside ±35° of heading | full yaw away, forward ∝ clearance |
| Approach beacon | beacon in view, unoccluded, within 12 m                          | yaw 1.5 β, forward when facing     |
| Explore         | none of the above                                                | forward 0.8, constant gentle yaw   |

It never commands altitude: none of the four cues carries height. Tests check that the explore
label is identical for a hidden beacon on either side, and that an occluding pillar or a ghosted
threat removes its drive (`tests/test_teacher.py`).

## 5. Distillation into one decoder

`roam-collect` flies the teacher (with action noise) or, with `--beta < 1`, the current student,
and records features every other frame with the teacher's label and drive. `roam-fit` clones the
teacher onto a 2022 → 64 → 64 → 4 tanh decoder with equal weight per drive, a held-out set of whole
flights, and per-drive MSE and R². It exports a parity-checked actor with the arena limits.
`roam-screen` compares policies, the teacher, the cue script and random under any brain condition.

## 6. Evaluation and pre-registered acceptance

`fly-drone evaluate --task free_roam --policy actor.json` runs held-out seeds 1000–1049 for 120 s at
L3, with respawn, under seven brain conditions (intact, zeroed, shuffled, all vision silenced, light
silenced, loom silenced, ghost objects) plus teacher, cue-script and random baselines.

| ID  | Criterion                                                                                                                                                         |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A1  | beacons/min ≥ 0.6 × teacher and ≥ 2 × the best of zeroed, shuffled, vision-silenced, light-silenced, random; paired bootstrap interval above 0                    |
| A2  | collisions/min ≤ 0.5 × min(loom-silenced, ghost) and ≤ 0.5                                                                                                        |
| A3  | threat dodge rate ≥ 0.8 and ≥ 0.8 on each side; ghost-threat dodge rate ≤ 0.3                                                                                     |
| A4  | light silencing cuts beacons by ≥ 50% with the upper bound of the collision increase ≤ 0.25/min; loom silencing doubles collisions while keeping ≥ 50% of beacons |
| A5  | in-arena skill probes (separate report)                                                                                                                           |
| A6  | coverage ≥ 0.4, slow fraction ≤ 0.1, mean yaw bias ≤ 0.25, no tilt/bounds/altitude losses                                                                         |
| A7  | live server ≥ 1× real time                                                                                                                                        |

## 7. Live probes

`serve --task-policy free_roam=runs/<run>/actor.json`, then choose Free roam. In free roam the
socket accepts `place_beacon`, `launch_threat`, `pathway` (silence light or loom live), `ghost`, and
`reset` with a `level`. Frames carry `policy_status` (`none` means the drone deliberately holds
still), `attribution` (top cell types driving each command, gradient × input), and `free_roam` stats
with recent events.

## 8. Status

See [`validation.md`](validation.md) for measured results. The feasibility gate (§3 controller
comparison) and threat analysis are recorded in `docs/results/roam-feasibility.json` once accepted.
