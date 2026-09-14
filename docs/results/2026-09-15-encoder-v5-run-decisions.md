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

## Tasks 13–15: pipeline

- **Tasks 13–15 run as one chained script** once the smoke test passes. If this session dies overnight, the run
  still finishes.
- **Round 0 (clone encoder + round-0 decoder) is a candidate for the final pair.** Task 13 step 6 says "the round
  with the best validation near-dodge rate", and the stop rule compares against "the best earlier round". Round 0 has
  a validation file and belongs in the comparison. Cost if wrong: if round 0 wins, the "final" v5 has had no SAC,
  and the results will say so.
- **A near-dodge rate of `null` (no near threats) counts as −1 for comparisons**, so a round with no measurable dodging
  can never count as an improvement.
