# At the pre-registered 120 s, codec v2 passes L1–L8 except L4 (anti-luck)

**Status:** recorded 2026-09-20. Definitive run: 15 seeds × **120 s** × level 3, `seed_base 2000`,
6 workers, codec v2 on the canonical bundle. Artifact:
[`liveness-check-v2-120s.json`](liveness-check-v2-120s.json).

**Correction.** The P4 2×2 smoke used 60 s. L2 coverage accumulates with time, and the pre-registered
liveness duration is **120 s** (`roam_eval.liveness_check` default; the spec §10 example), so the
smoke understated coverage. At 120 s the v2 cell's failing set is not `{L2}` — L2 passes — it is
`{L4}`.

## 1. Result (v2 canonical, 15 seeds, 120 s)

| criterion | value | passed |
| --------- | ----- | ------ |
| L1 mobility | `slow_fraction` 0.0256 ≤ 0.25 | yes |
| L2 exploration | coverage **0.1888** ≥ 0.15 | **yes** |
| L3 sense causality | intact 0.026 vs sensory-silenced 0.051, CI (−0.029, −0.022) | yes |
| L4 anti-luck | **ghost is also "alive"** | **no** |
| L5 non-degenerate | yaw bias 0.052, no loss of control | yes |
| L6 intermittency | within window | yes |
| L7 saccadic turning | deferred (body yaw authority 0.8 < detector 3.0) | — |
| L8 exploration structure | within window | yes |

`passed = false`, `deferred = ["L7"]`, failing `["L4"]`.

## 2. Why L4 fails: the blind drone is nearly as alive

`L4` marks a control "alive" if it moves (`slow_fraction ≤ 0.25`) **and** explores (coverage ≥ 0.15).

| condition (120 s, 15 seeds) | slow_fraction | coverage | counted alive |
| --------------------------- | ------------- | -------- | ------------- |
| v2 intact                   | 0.026         | 0.189    | —             |
| **v2 ghost (objects hidden)** | **0.029**   | **0.177** | **yes**       |
| random                      | 0.070         | 0.043    | no            |
| cue_script                  | 0.037         | 0.096    | no            |
| teacher (context)           | 0.010         | 0.256    | —             |

Hiding every arena object barely changes the trajectory: coverage 0.189 → 0.177. The anti-luck rule
("a blind condition cannot pass by chance") therefore fails.

## 3. Cause: locomotion is intrinsic, not vision-carried

The v2 forward drive reads the flight-power motoneurons (`power_l`/`power_r`), which carry the
connectome's own tonic drive (`b = 0.85` on DLMn/DVMn). That drive cruises the drone regardless of
vision; vision only modulates *steering*. A 10 s probe (seed 0) of the codec's own inputs:

| ablation | mean power | steering-command std | mean speed |
| -------- | ---------- | -------------------- | ---------- |
| none     | 0.542      | 0.0132               | 0.339      |
| ghost    | 0.517      | 0.0145               | 0.330      |
| sensory  | 0.503      | **0.0000**           | 0.261      |

- `ghost` hides objects but leaves the visual pathway and its input noise: the steering fluctuation
  is unchanged (0.0132 → 0.0145), so it is intrinsic, not vision-driven.
- `sensory` (`brain.silence_sensors`) removes *all* sensory input: the steering command goes
  constant (std 0.0) and speed drops (0.339 → 0.261), which is why L3 passes — silencing the senses
  does make it more stationary.

So mobility/coverage is a property of the brain's ongoing state (the point of P4's C3), and vision's
causal contribution is steering, which L3 tests directly. L4's `move ∧ cover` control instead
demands that a blind body be *immobile*, which an intrinsically-driven body is not.

## 4. What this means

- **L2 is not the remaining gap** — at the correct duration the codec clears it (0.189 vs 0.15).
- **L4 is.** By the project's own rule ("a blind condition cannot pass by chance") the current
  liveness is not attributable: a blind connectome-driven body still covers 17.7 %. Whether that
  invalidates "alive" or means L4 tests the wrong thing is a **bar question** (below).

## 5. Decision (adopted): leave L4 failing as a recorded near-miss

The user chose to **leave the bar exactly as pre-registered**. No rule or threshold was changed:

- The v2/canonical cell is **alive by L1–L8 with L7 deferred, except L4**, which fails because a
  blind connectome-driven body still covers 17.7 %. `passed` remains `false`.
- This is reported as a **recorded near-miss**: the drone moves because of its neurons (its own
  tonic drive), but the liveness is **intrinsic, not vision-attributable**, and the project's
  anti-luck rule ("a blind condition cannot pass by chance") therefore does not clear.
- The two non-connectome controls do fail (`random` 0.043, `cue_script` 0.096), and L3 shows vision
  is causally used for steering, so the picture is coherent — it is the strict ghost rule that
  stands.

Options A (move ghost into the causal test) and B (gate the forward drive on vision) were
considered and **not** taken. P4 closes here: the idling is fixed and measured, and the remaining
gap is an honest, recorded attribution caveat rather than an open workstream.

The 60 s "fails only L2" statement in
[`FINDING-2026-09-20-c3-and-codec-v2.md`](FINDING-2026-09-20-c3-and-codec-v2.md) is superseded by
this run.
