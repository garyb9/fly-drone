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

## Task 12: Stage 2a

- **Clone collection started while the warm-up commit (8dda027) was still under review.** `encoder-collect`
  does not use the SAC/warm-up code, and workers import at spawn, so a later fix cannot change a running
  collection. Cost if wrong: none measurable; the rest of Task 12 waits for the review.
