# Encoder v5 runs: decisions log (2026-09-15)

Decisions taken autonomously while executing plan 05
(`docs/superpowers/plans/2026-09-14-fly-drone-05-encoder-v5-sac.md`) after the user asked for autonomous
execution. Guiding light: `AGENTS.md` (the brain decides, frozen connectome, causal proof, honest labels,
pre-registered thresholds never relaxed). Each entry: what was decided, why, and what it costs if wrong.

## Task 11: Stage 1 DAgger (L2, encoder v4)

Screen on validation seeds 9000–9009, 60 s, level 2, intact:

| policy         | beacons/min | collisions/min | visited cells |
| -------------- | ----------- | -------------- | ------------- |
| teacher        | 2.3         | 0.0            | 41.0          |
| random         | 0.0         | 0.1            | 7.8           |
| it0 warm-actor | **2.2**     | 1.0            | 35.0          |
| it1            | 1.5         | 2.1            | 29.8          |
| it2            | 2.0         | 0.9            | 33.6          |
| it3            | 2.0         | 1.0            | 29.4          |

- **Choice: iteration 0** (`runs/v5/dagger/choice.json`), by the plan's rule (highest beacons/min, ties by fewer
  collisions/min). DAgger iterations 1–3 did not improve on the teacher-only fit on this 10-seed screen. The
  it0 fit's held-out R² was 0.38 (avoid), 0.25 (beacon), −1.26 (explore). Cost if wrong: the round-0 decoder
  in Task 12 is fitted on all DAgger data anyway, so the choice only sets the sanity-gate reference (1.76).

## Scope of autonomy

- **User decision (2026-09-15):** run autonomously through Task 15, without asking before Tasks 13–15.
- **How the plan's stop rules are applied:**
  - **Task 12 sanity gate fails:** stop and report. SAC would start from a broken system.
  - **Task 13, a round does not improve near-dodge:** do not start the next round. Pick the best round so far and continue.
  - **Task 14, the bypass beats the full system:** still run the Task 15 measurements. The result is recorded as a failed E3, and no
    behaviour is attributed to neurons.
  - **The smoke test fails:** debug and fix it, with review, before Task 13.
- **Cost if wrong:** some evaluation compute spent after a failed E3.

## Critic-only warm-up review (8dda027)

- Re-run review (opus) **approved** it. The actor and entropy coefficient stay frozen for the first 50k frames:
  there is no Adam state for the actor, the actor and critic have separate feature extractors, and Polyak averaging updates the critic only.
- **Four minors are parked:**
  - A crash resume repeats the warm-up. This is documented, and alternating rounds want it anyway.
  - The unit round test never steps the actor for decoder or bypass rounds.
  - The actor losses are still computed during the warm-up. This only wastes compute.
  - A negative `--actor-warmup` is treated as 0.
- The missing test coverage is folded into the pre-Task-13 smoke test: the actor parameters must change after the warm-up
  on the real 6-worker path. Cost if wrong: a regression is caught by the smoke test instead of the unit suite.

## Task 12: Stage 2a

- **Clone collection started while the warm-up commit (8dda027) was still under review.** `encoder-collect`
  does not use the SAC/warm-up code, and workers import at spawn, so a later fix cannot change a running
  collection. Cost if wrong: none measurable; the rest of Task 12 waits for the review.
- **Task 12 runs from one driver script.** It chains clone → round-0 decoder → sanity screen → validation → v4 E1 baseline, and
  it stops automatically if the sanity gate fails (beacons/min < 0.8 × 2.2 = 1.76). The gate uses the Task 11 choice (it0) as its reference, as the plan
  specifies, not the teacher. Cost if wrong: none; the gate is an engineering check, not acceptance.
- **Clone data check:** 48,000 frames from 32 distinct flights of 1,500 frames each (60 s at 25 Hz). No flight
  was cut short, which settles the parked Task 4 concern about the reported flight count for this dataset.

- **Clone fit** (`learned-v5:b3f4c7b4cbb99f28`, 20k steps, 3 held-out flights):

  | channel | mi1 L / R     | tm3 L / R     | lc4 L / R     | lplc2 L / R   |
  | ------- | ------------- | ------------- | ------------- | ------------- |
  | r       | 0.989 / 0.988 | 0.989 / 0.988 | 0.952 / 0.953 | 0.951 / 0.953 |
  | mse     | 0.012 / 0.013 | 0.012 / 0.013 | 0.010 / 0.011 | 0.010 / 0.011 |

  The loom channels (lc4, lplc2) are the hardest to clone. That fits their sparse, event-driven signal.

- **Round-0 decoder:** export max error 8.3e-7 (limit 1e-4). The encoder version matches the clone.

### Sanity gate FAILED: stopped before SAC

Screen: level 2, seeds 9000–9009, 60 s. Values per minute.

| system                                            | beacons | collisions | visited cells |
| ------------------------------------------------- | ------- | ---------- | ------------- |
| it0 warm actor (v4, linear head), reference       | 2.2     | 1.0        | 35.0          |
| **round 0: clone encoder + tanh decoder, all DAgger data** | **0.9** | 2.0        | 24.1          |
| diag A: clone encoder + tanh decoder, it0 data only        | 1.1     | 2.8        | 25.6          |
| diag B: v4 encoder + tanh decoder, it0 data only           | 1.8     | 1.8        | 32.1          |

Gate: 0.8 × 2.2 = 1.76. Almost all the extra collisions are wall hits. As instructed, I stopped and report.

Diagnostics (a few minutes of compute each, no training runs):

- **Not the fit quality.** Diag A's held-out drive errors match the v4 fit (avoid 0.113 vs 0.106).
- **Not the wiring.** v4 and v5 drive exactly the same cells (light 1903/1924, loom 165/146). With identical
  currents, the v5 runtime reproduces v4 bit for bit (open-loop replay of held-out flight 625, 500 frames).
- **Mostly the clone, amplified by the brain.** The clone's currents are close to v4's: r 0.95–0.99, and loom
  peaks reach 91–94% of v4's. In the replay, though, the 2,022 descending/motor features the decoder reads have a
  median correlation of only 0.90 with the v4 run, and half the active features fall below 0.9. The round-0 decoder
  was fitted on activity recorded under v4, which it never sees under the clone. This costs 1.8 → 1.1 beacons/min.
- **Partly the tanh head.** It costs 2.2 → 1.8 with more collisions. A third of the avoid labels have |yaw| ≥ 0.97,
  where tanh saturates.

The fix is the user's call (see the report). The Tasks 13–15 pipeline was never launched.

### User decision: refit under the clone and loosen the tanh fit (Task 12b, new)

- **`roam-collect --encoder`:** the round-0 decoder is fitted on activity recorded while the clone drives the
  brain. The teacher labels are unchanged (simulator geometry gated by visibility).
- **Ruling: data loading enforces the encoder version.** `_load` refuses files recorded under a different
  encoder than the decoder will run with. This is the check whose absence let the mismatch through. Cost if wrong:
  the v4 DAgger files can no longer seed a learned decoder, which is the point.
- **Ruling: the warm start fits the pre-tanh output** against `atanh(clip(y, ±0.97))`, so saturated avoid turns keep
  their gradient. Held-out errors are still reported in action space. Cost if wrong: explore/beacon labels
  could fit slightly worse. The screen will show it.
- **Ruling: teacher-only collection first** (128 flights, level 2, the same seeds 200–327 as v4 it0), with no DAgger
  iterations under the clone. Under v4, iterations 1–3 did not beat it0. A DAgger iteration under the clone is
  added only if the gate fails again. Cost if wrong: one more collection round (~20 min).
- **Task 12b** landed as e285dd2 (124 tests). The review approved it with 5 minors, which are parked.
- **Ruling: keep the pre-tanh warm start.** The implementer questioned it: it beat the old loss only in short fits on
  toy data. So it was checked on real data before the collection. Screen: level 2, seeds 9000–9009. Values per minute.

  | decoder on v4, it0 data   | held-out avoid mse (action space) | beacons | collisions | visited cells |
  | ------------------------- | --------------------------------- | ------- | ---------- | ------------- |
  | linear head (DAgger it0)  | 0.106                             | 2.2     | 1.0        | 35.0          |
  | tanh head, old loss       | 0.113                             | 1.8     | 1.8        | 32.1          |
  | tanh head, pre-tanh loss  | 0.124                             | **2.2** | **0.5**    | 35.0          |

  The offline error is slightly higher, but in flight the linear head's foraging comes back with fewer collisions.
  Cost if wrong: none seen at level 2.

### Sanity gate FAILED again (re-run with clone-recorded data)

The round-0 decoder was refit on 128 teacher flights recorded with the clone driving the brain. The export error was
1.1e-5, and held-out avoid mse fell to 0.098, the best of all fits. Screen: level 2, seeds 9000–9009. Values per minute.

| system                                   | beacons | collisions | visited cells |
| ---------------------------------------- | ------- | ---------- | ------------- |
| diag B′: v4 encoder, same warm start     | 2.2     | 0.5        | 35.0          |
| round 0 v2: clone encoder, clone data    | **1.0** | 0.6        | 34.9          |

Obstacle avoidance and exploration are back. Only beacon seeking is lost.

**Cause: the clone blurs the left–right light difference.** On the clone's held-out frames:

| pathway     | per-channel r | left−right difference r | difference gain (clone / v4) |
| ----------- | ------------- | ----------------------- | ---------------------------- |
| light (Mi1) | 0.99          | 0.85                    | 0.85                         |
| light (Tm3) | 0.99          | 0.85                    | 0.85                         |
| loom (LC4)  | 0.95          | 0.94                    | 0.92                         |

In v4 the light difference is exactly 0 in 67% of frames. The clone adds side noise there and shrinks the real
side signal. Steering toward a beacon depends on that side signal.

- **Ruling: skip the announced DAgger iteration under the clone.** DAgger fixes decoder covariate shift, but this is the encoder's
  fidelity. Cost if wrong: one ~25 min iteration postponed.
- **Ruling: run a cheap test (clone refit with 60k steps instead of 20k), then stop and report**, as the Task 12 gate instruction
  says. Any encoder fix changes the encoder version, so the round-0 data must be re-collected. The user chooses.

Test result: the same clone data, refit for 60k steps instead of 20k. Held-out frames.

| metric                                        | 20k steps | 60k steps |
| --------------------------------------------- | --------- | --------- |
| light left−right difference r                  | 0.85      | 0.88      |
| light difference gain (clone / v4)            | 0.86      | 1.00      |
| light difference rmse (v4 difference std 0.28) | 0.149     | 0.141     |
| clone difference where v4's is exactly 0      | 0.021     | 0.018     |
| loom left−right difference r                   | 0.94      | 0.96      |

More training removes the shrinkage but leaves most of the error. That error sits in the frames where the beacon is
off to one side, not in the frames with no light difference. Training longer alone is unlikely to recover beacon seeking.

### User decision: difference-aware clone (Task 12c, new)

- The clone loss also penalises errors in each left−right difference, and batches oversample frames where the beacon is
  off to one side. The clone trains for 60k steps. Then the round-0 data is re-collected under the new clone, the decoder
  is refit, and the gate is re-checked.
- **Ruling: the difference term has weight 1.0.** It is added to the per-channel MSE over the 4 pairs. Batches are
  split 1/3 uniform, 1/3 loom-active (the existing > 0.05 rule), and 1/3 light-side frames (|light L−R| > 0.05). Equal weight keeps
  channel fidelity primary, and the oversampling targets the third of frames that carry the side signal. Cost if wrong:
  another clone refit (~8 min).
- **Ruling: the earlier artifacts are kept** as `clone-v1`, `round0-v2-clonedata` and `round0-data-v1`, as the record of both failed
  gates. This costs about 1.3 GB of disk.
- **The v4 teacher labels, the connectome and the thresholds are unchanged.** The clone target is still v4's own currents, so this is
  still an honest copy of v4, only one weighted toward the signal the brain steers with.
- **Task 12c** landed as 6ff5c85 (127 tests). The review approved it, with 4 minor test-strength items, which are parked. The two gaps
  that could have hidden a live bug are the left/right pairing and the absolute value in the side-frame pool. Both were checked
  directly in the committed code and are correct.
- **Ruling: the re-run started alongside the review**, to save about 30 min. It would have been stopped if the review had
  found a serious problem in the clone fit. Cost if wrong: only the compute already spent.
- **Difference-aware clone** (`learned-v5:c273d625abe368d1`, 60k steps). Held-out frames:

  | metric                        | plain 20k | plain 60k | difference-aware 60k |
  | ----------------------------- | --------- | --------- | -------------------- |
  | light left−right difference r  | 0.851     | 0.878     | **0.890**            |
  | light difference rmse          | 0.149     | 0.141     | **0.136**            |
  | light difference gain          | 0.86      | 1.00      | 1.04                 |
  | loom left−right difference r   | 0.940     | 0.959     | 0.955                |
  | per-channel r (light / loom)  | 0.99/0.95 | 0.99/0.97 | 0.99/0.96            |

  This is a modest gain over longer training alone. The sanity gate on the re-collected round 0 decides whether it is enough.

## Tasks 13–15: pipeline

- **Tasks 13–15 run as one chained script** once the smoke test passes. If this session dies overnight, the run
  still finishes.
- **Round 0 (clone encoder + round-0 decoder) is a candidate for the final pair.** Task 13 step 6 says "the round
  with the best validation near-dodge rate", and the stop rule compares against "the best earlier round". Round 0 has
  a validation file and belongs in the comparison. Cost if wrong: if round 0 wins, the "final" v5 has had no SAC,
  and the results will say so.
- **A near-dodge rate of `null` (no near threats) counts as −1 for comparisons**, so a round with no measurable dodging
  can never count as an improvement.
