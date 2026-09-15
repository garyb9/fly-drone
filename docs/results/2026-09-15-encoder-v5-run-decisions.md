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

| system                                                     | beacons | collisions | visited cells |
| ---------------------------------------------------------- | ------- | ---------- | ------------- |
| it0 warm actor (v4, linear head), reference                | 2.2     | 1.0        | 35.0          |
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

  | decoder on v4, it0 data  | held-out avoid mse (action space) | beacons | collisions | visited cells |
  | ------------------------ | --------------------------------- | ------- | ---------- | ------------- |
  | linear head (DAgger it0) | 0.106                             | 2.2     | 1.0        | 35.0          |
  | tanh head, old loss      | 0.113                             | 1.8     | 1.8        | 32.1          |
  | tanh head, pre-tanh loss | 0.124                             | **2.2** | **0.5**    | 35.0          |

  The offline error is slightly higher, but in flight the linear head's foraging comes back with fewer collisions.
  Cost if wrong: none seen at level 2.

### Sanity gate FAILED again (re-run with clone-recorded data)

The round-0 decoder was refit on 128 teacher flights recorded with the clone driving the brain. The export error was
1.1e-5, and held-out avoid mse fell to 0.098, the best of all fits. Screen: level 2, seeds 9000–9009. Values per minute.

| system                                | beacons | collisions | visited cells |
| ------------------------------------- | ------- | ---------- | ------------- |
| diag B′: v4 encoder, same warm start  | 2.2     | 0.5        | 35.0          |
| round 0 v2: clone encoder, clone data | **1.0** | 0.6        | 34.9          |

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

| metric                                         | 20k steps | 60k steps |
| ---------------------------------------------- | --------- | --------- |
| light left−right difference r                  | 0.85      | 0.88      |
| light difference gain (clone / v4)             | 0.86      | 1.00      |
| light difference rmse (v4 difference std 0.28) | 0.149     | 0.141     |
| clone difference where v4's is exactly 0       | 0.021     | 0.018     |
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
  | light left−right difference r | 0.851     | 0.878     | **0.890**            |
  | light difference rmse         | 0.149     | 0.141     | **0.136**            |
  | light difference gain         | 0.86      | 1.00      | 1.04                 |
  | loom left−right difference r  | 0.940     | 0.959     | 0.955                |
  | per-channel r (light / loom)  | 0.99/0.95 | 0.99/0.97 | 0.99/0.96            |

  This is a modest gain over longer training alone. The sanity gate on the re-collected round 0 decides whether it is enough.

### Sanity gate FAILED a third time, narrowly

Round 0 was refit on 128 flights recorded under the difference-aware clone. Export error 3.2e-6; held-out avoid mse 0.148.
Screen: level 2, seeds 9000–9009. SEM = standard error of the mean over the 10 seeds.

| system                                  | beacons/min (SEM)   | collisions/min | visited cells |
| --------------------------------------- | ------------------- | -------------- | ------------- |
| it0 warm actor (v4, linear), reference  | 2.2 (±0.29)         | 1.0            | 35.0          |
| diag B′: v4, pre-tanh warm start        | 2.2 (±0.25)         | 0.5            | 35.0          |
| round 0 v2: clone v1, clone data        | 1.0 (±0.33)         | 0.6            | 34.9          |
| **round 0 v3: difference-aware clone**  | **1.6 (±0.34)**     | 0.6            | 35.7          |

- **Gate: 1.76. Failed.** As instructed, I stopped and report. The threshold was not relaxed.
- The difference-aware clone recovered most of the lost foraging (1.0 → 1.6). Avoidance and exploration match v4.
- On 10 seeds the screen's standard error is about ±0.3 beacons/min. So the 0.16 shortfall below the gate, and the gap
  to v4, are both within noise. This 10-seed screen cannot separate this system from one that passes.

### User decision: re-screen the gate on 30 seeds

- Both the v4 it0 reference and round 0 are screened on seeds 9000–9029. The same rule applies to the 30-seed means:
  round 0 must reach at least 0.8 × the reference. Only the sample size changes; the ratio and the pre-registered `ACCEPTANCE` stay as they were.
- **Ruling: the reference is re-measured on the same 30 seeds** rather than reusing the 10-seed 2.2, so the ratio
  compares like with like. Cost if wrong: none.
- If it passes, execution continues autonomously: validation, the v4 E1 baseline, the smoke test, then Tasks 13–15. If it fails, I stop and report.

**Result: FAILED, clearly.** Level 2, seeds 9000–9029, 60 s. The ± value is the standard error over the 30 seeds.

| system                            | beacons/min      | collisions/min | visited cells | seeds with 0 beacons | mean yaw bias |
| --------------------------------- | ---------------- | -------------- | ------------- | -------------------- | ------------- |
| v4 it0 (reference)                | 1.93 ± 0.20      | 1.07           | 33.8          | 3                    | 0.077         |
| round 0, difference-aware clone   | **1.27 ± 0.19**  | 0.57           | 34.6          | 8                    | 0.106         |

- **Gate: 0.8 × 1.93 = 1.55.** Round 0 reaches 66% of the reference.
- **Paired difference (round 0 − v4): −0.67 ± 0.19 beacons/min**, about 3.5 standard errors. The 10-seed 1.6 came from the
  easier seeds: seeds 9010–9029 average 1.10.
- **Avoidance is intact** (fewer collisions than v4) and so is exploration. Beacon seeking is still impaired.
- **The larger yaw bias** (0.106 vs 0.077) suggests the clone copies the two eyes asymmetrically.
- Stopped and reported, as instructed.

### User decision: mirror-symmetric clone and double the clone data (Task 12d, new)

- **Ruling: the augmentation.** With probability 0.5 per training sample, the two eyes are swapped and flipped
  horizontally, and every left/right target pair is swapped. A mirrored world must give mirrored currents, which is the fly's bilateral
  symmetry. This is only valid if v4 itself is mirror-symmetric. The implementer must first check that on real v4 renders
  and stop if it is not. Cost if wrong: a clone trained toward a symmetry v4 lacks.
- **Ruling: 32 more clone flights**, seeds 632–663, disjoint from 600–631 and from every evaluation and validation seed. They are
  collected now, alongside the code work, because the collector is unchanged. Cost: ~6 min, +0.9 GB of disk.
- **Ruling: the 30-seed gate reuses the measured v4 reference** (1.933 on seeds 9000–9029). v4 is unchanged, and this saves 16 min.
  Cost if wrong: none.
- The clone is fit on both data files at 60k steps. Round 0 is then re-collected and refit, and the 30-seed gate is re-run.
  If it passes, execution continues autonomously.
- **v4 is mirror-symmetric, bit for bit.** On 180 real free-roam frames, the mirrored eye images gave exactly the swapped
  cues (maximum error 0.0 on all four). The augmentation is therefore an honest copy of v4's own symmetry.
- **Task 12d** landed as 4097de5. It changes only its own 5 files; another session's uncommitted viewer edits were left out.
- **The extra clone data** is 32 flights, seeds 632–663, 1,500 frames each.
- **Ruling: the re-run started alongside the 12d review**, as for 12c. It would have been stopped if the review had found a serious
  problem in the mirror or clone fit. Cost if wrong: the compute already spent.
- **The 12d review approved it**, with 4 minors parked. I read the training loop myself to confirm that each step mirrors a random half
  of the batch, because no test covers it.
- **Mirror-symmetric clone** (`learned-v5:1681bff17b4eda85`, 64 flights, 60k steps). Held-out frames. Mirror consistency
  compares the left−right difference on a frame with the negated difference on its mirror image; a perfect score is r = 1, rmse = 0.

  | metric                                  | difference-aware clone (32 flights) | mirror-symmetric clone (64 flights) |
  | --------------------------------------- | ----------------------------------- | ----------------------------------- |
  | light left−right difference vs v4, r     | 0.890                               | **0.978**                           |
  | light left−right difference vs v4, rmse  | 0.136                               | **0.085**                           |
  | light mirror consistency, r             | 0.886                               | **0.997**                           |
  | light mirror consistency, rmse          | 0.140                               | **0.029**                           |
  | loom left−right difference vs v4, r      | 0.955                               | 0.962                               |
  | loom mirror consistency, r              | 0.968                               | 0.994                               |

  The previous clone's asymmetry was as large as its error against v4, and it depended on the scene. Mirror training
  removes almost all of it and also brings the clone much closer to v4 on the steering signal. Caveat: the
  held-out split now covers 6 flights across both data files, not 3, so the two columns use different frames. The gate decides.

### Gate FAILED with the mirror-symmetric clone (30 seeds)

Round 0 was refit on 128 flights recorded under the mirror-symmetric clone: export error 5.9e-6, held-out avoid mse 0.118. Screen:
level 2, seeds 9000–9029, 60 s. Gate: 0.8 × 1.93 = 1.55.

| system                        | beacons/min (SEM) | collisions/min | visited cells | zero-beacon seeds | mean yaw bias | paired vs v4    |
| ----------------------------- | ----------------- | -------------- | ------------- | ----------------- | ------------- | --------------- |
| v4 it0 (reference)            | 1.93 ± 0.20       | 1.07           | 33.8          | 3                 | 0.077         | —               |
| difference-aware clone        | 1.27 ± 0.19       | 0.57           | 34.6          | 8                 | 0.106         | −0.67 ± 0.19    |
| **mirror-symmetric clone**    | **1.20 ± 0.22**   | 0.80           | 33.5          | 10                | **0.084**     | −0.73 ± 0.24    |

- **The steering bias is fixed** (0.084, close to v4's 0.077), and the clone copies v4's steering signal much more faithfully. Foraging still did not recover.
  So the clone is **no longer the likely bottleneck**.
- **Ruling: one more diagnostic before reporting** (~16 min). The pre-tanh decoder fitted on v4 data, flown with v4 (diag B′, 2.2 on 10 seeds), is
  screened on the same 30 seeds. If it also misses 1.55, the remaining loss is in the SAC decoder head or its warm start, not the encoder.
  That points to a different fix.

### Diagnosis: the clone strategy is exhausted

**The decoder is cleared.** Diag B′ (pre-tanh warm start, v4 it0 data, v4 encoder) on seeds 9000–9029 scores **1.90 ± 0.18** beacons/min
(0.67 collisions/min, 2 zero-beacon seeds). That matches the linear it0 actor: paired −0.03 ± 0.18. It passes 1.55.
The same decoder method under the mirror clone loses **0.70 ± 0.21** beacons/min, paired. The whole loss comes from running under the clone.

**The brain amplifies tiny input differences.** Open-loop replay of held-out flight 600 (500 frames) through the frozen
connectome. The features are the 2,022 descending/motor traces the decoder reads.

| currents fed to the brain            | currents vs v4, r | features vs v4, median r | features with r < 0.9 |
| ------------------------------------ | ----------------- | ------------------------ | --------------------- |
| clone v1                             | 0.94–0.99         | 0.858                    | 66%                   |
| mirror-symmetric clone               | 0.96–1.00         | 0.880                    | 58%                   |
| exact v4 + noise σ = 0.01            | ≈ 1.000           | 0.902                    | 49%                   |
| exact v4 + noise σ = 0.03            | 0.995–1.000       | 0.894                    | 53%                   |

**Correction:** these are the numbers at the environment's real timing, 8 brain ticks (40 ms) per camera frame. My first replay
stepped 40 ticks (200 ms) per frame and reported 0.835 / 0.866 / 0.921 / 0.919. The conclusion is unchanged and slightly stronger.

Even imperceptible noise on the exact v4 currents decorrelates almost half of the fine traces. An encoder that is not
bit-identical to v4 will never reproduce v4's brain activity. A better clone therefore cannot close the gap, and after the first
fixes the attempts stopped improving (30 seeds: 1.27 → 1.20). What remains is decoder robustness in closed loop under the encoder it
actually runs with. That is DAgger's and SAC's job, not the clone's. Stopped and reported. The user chooses the strategy.

### User decision: DAgger under the clone, then SAC

- **The clone is frozen** as the mirror-symmetric `learned-v5:1681bff17b4eda85`. No more clone work.
- **Ruling: the DAgger iterations follow the v4 Stage-1 scheme.**
  - Iteration 1: the student is the current round-0 decoder, flying half the time (beta 0.5), on seeds 1200–1327.
  - Iteration 2, only if iteration 1 fails the gate: the student is the iteration-1 decoder, beta 0.25, on seeds 2200–2327.
  - Each refit uses all flights recorded under the clone so far. The teacher labels are unchanged.
  - Cost if wrong: ~30 min per iteration.
- **Ruling: pick the best round 0.** It is the highest 30-seed beacons/min among teacher-only (1.20), DAgger 1 and DAgger 2, with ties going to fewer collisions.
  It is installed as `runs/v5/round0`, with `gate.json` recording whether it passed or is an override.
- **If no candidate passes after 2 iterations, SAC (Tasks 13–15) starts from the best round 0 anyway**, as the user decided. This is recorded as a
  gate override in the results. Round 0 avoids obstacles and forages at about 62% of v4, and SAC trains in closed loop under the real
  encoder, which is what is missing. Cost if wrong: hours of SAC compute from a weaker start, reported honestly.

### Gate PASSED after one DAgger iteration under the clone

DAgger iteration 1: the teacher-only round-0 decoder flew half the time (beta 0.5) under the mirror-symmetric clone, on 128 flights, seeds 1200–1327.
The decoder was then refit on both clone-recorded files: export error 3.5e-6, held-out avoid mse 0.181 (student-flown states make harder labels).
Screen: level 2, seeds 9000–9029, 60 s. Gate: 1.55.

| system                              | beacons/min (SEM) | collisions/min | visited cells | zero-beacon seeds | paired vs v4    |
| ----------------------------------- | ----------------- | -------------- | ------------- | ----------------- | --------------- |
| v4 it0 (reference)                  | 1.93 ± 0.20       | 1.07           | 33.8          | 3                 | —               |
| round 0, teacher-only, mirror clone | 1.20 ± 0.22       | 0.80           | 33.5          | 10                | −0.73 ± 0.24    |
| **round 0, DAgger it1, mirror clone** | **2.13 ± 0.27** | **0.53**       | **36.5**      | 5                 | **+0.20 ± 0.25** |

- **Passed without an override.** DAgger iteration 2 was not needed.
- The installed `runs/v5/round0` is `round0-dagger1`, recorded in `gate.json`.
- The v5 system (learned encoder, frozen connectome, SAC-style decoder) now forages as well as v4, statistically, with half the collisions.
- **Lesson:** the gap was the decoder never having flown under the encoder it runs with. Clone fidelity was not the missing piece. Training in closed loop under the
  real encoder closed the gap in one iteration.
- Next, automatically: validation, the v4 E1 baseline, the 6-worker smoke test, then Tasks 13–15.

**Round-0 validation baseline** (`sac-validate`, level 3 with threats, seeds 9000–9009, 60 s). This is the starting point for SAC, not a gate:

| near-dodge | balanced | ghost near-dodge | beacons/min | collisions/min | E1 loom AUC (≥ 0.8) | E2 margins (light ≥ 0.05, loom ≥ 0.1) |
| ---------- | -------- | ---------------- | ----------- | -------------- | ------------------- | --------------------------------------- |
| 0.27       | 0.18     | 0.30             | 1.9         | 2.7            | 0.69, fail          | 0.50 / 0.26, pass                       |

Round 0 was trained only on threat-free level 2. It forages on level 3 but does not dodge causally yet: intact near-dodge 0.27 against 0.30
for the ghost condition, and threat hits inflate collisions. Raising near-dodge, and the loom selectivity that E1 measures, is what the Task 13 SAC rounds are for.

**v4 E1 baseline** (`runs/v5/e1-v4-baseline.json`, 50 evaluation seeds, reported alongside E1): loom AUC **0.687** (fails the 0.8 bar),
E2 passes (light margin 0.49, loom margin 0.27). The hand-built v4 encoder does not meet E1 either; round-0 v5 is at the same 0.686. E1 therefore
asks SAC for loom selectivity that v4 never had.

**Task 12 complete** (18:39).
- **Encoder:** the mirror-symmetric clone `learned-v5:1681bff17b4eda85`.
- **Round 0:** DAgger iteration 1 under that clone (`runs/v5/round0`, gate passed).
- **Also written:** validation and the v4 E1 baseline.
- **Code added along the way:** Tasks 12b–12d (collection under a learned encoder, pre-tanh warm start, difference-aware and mirror-symmetric clone fit), each reviewed.
- **Next:** the 6-worker smoke test, then Tasks 13–15.

### Smoke test and launch of Tasks 13–15

- **Smoke encoder round** (9,000 frames, warm-up 6,000, 6 workers): exited cleanly in 117 s at **81 fps**. `round.json` is consistent.
  Loom and light metabolic costs are both logged (0.0091 / 0.0019).
- **Time estimate from that throughput:**
  - Encoder round, 350k frames: ≈ 72 min. Decoder round, 150k frames: ≈ 31 min. Validation: ~15 min. That is **~2 h per round**, ~6 h for 3 rounds.
  - Bypass (450k frames): ≈ 95 min, plus the E3 screen. Final evaluation: ~3 h.
  - **Total ~11–12 h.**
- **Ruling: the smoke check's warm-up log test only requires the flag to be logged and to end at 0.** SB3 writes its log at episode ends, and a smoke
  run is a single 1,500-frame episode per worker, so a 1→0 flip cannot be seen. The freeze itself is covered by the reviewed unit tests, and
  learning after the warm-up is proven by comparing actor weights before and after. Cost if wrong: a warm-up regression on the 6-worker path would only show up in round 1's logs.
- **Ruling: a gatekeeper script launches Tasks 13–15 automatically**, and only if every smoke check passes:
  - encoder version matches `round.json`;
  - export parity;
  - encoder and decoder actor weights moved after the warm-up;
  - warm-up and metabolic-cost logging present;
  - RAM headroom.

  Otherwise it stops and logs why. Cost if wrong: none. A failed check blocks the multi-hour run.
- **Smoke decoder round** (9,000 frames, 6 workers): exited cleanly in 103 s. Peak RAM used was 10.8 GB, with at least 13.3 GB still available.
- **Smoke check: all 12 passed (18:43).**
  - The encoder version matches `round.json`.
  - The encoder actor moved after the warm-up (max |Δw| 0.082 against the clone), and so did the decoder actor (0.074 against round 0).
  - The decoder is pinned to the new encoder, and export parity is 1.3e-5.
  - The warm-up flag and the metabolic cost are logged. The decoder's cost is 0 by design, because only encoder rounds pay the
    metabolic cost (`sac.py` step).
  - There are no tracebacks, and RAM headroom is fine.
- **RAM caveat:** the replay buffer fills lazily, so the smoke run held 9k transitions. A full round holds 100k (~5.3 GB), which should leave about 8 GB free.
  A memory watch runs during round 1.
- **Tasks 13–15 launched automatically at 18:43** (`runs/v5/pipeline13-15.log`), starting with the round 1 encoder SAC.

## Tasks 13–15: pipeline

- **Tasks 13–15 run as one chained script** once the smoke test passes. If this session dies overnight, the run
  still finishes.
- **Round 0 (clone encoder + round-0 decoder) is a candidate for the final pair.** Task 13 step 6 says "the round
  with the best validation near-dodge rate", and the stop rule compares against "the best earlier round". Round 0 has
  a validation file and belongs in the comparison. Cost if wrong: if round 0 wins, the "final" v5 has had no SAC,
  and the results will say so.
- **A near-dodge rate of `null` (no near threats) counts as −1 for comparisons**, so a round with no measurable dodging
  can never count as an improvement.
