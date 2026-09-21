# v2 is alive but not capable: the A1–A7 skill gate fails on the new default

**Status:** recorded 2026-09-20. Smoke: `adapter-check --adapter v2`, 15 seeds × 120 s × level 3,
seeds 1000–1014 (the committed v1 gate's base). Artifact:
[`adapter-check-v2-smoke.json`](adapter-check-v2-smoke.json). **No threshold changed.**

P4 fixed *liveness* (does the brain move the body?). This measures the next question: *capability*
(does it do the task?). The committed skill gate is `ACCEPTANCE` A1–A7, reused unchanged.

## 1. Result: every capability criterion fails

| criterion | v2 (15 seeds) | threshold | passed |
| --------- | ------------- | --------- | ------ |
| A1 beacon rate | **0.033 /min** (teacher 0.733; best rival `shuffle` 0.133) | ≥ 0.6× teacher **and** ≥ 2× rival | no |
| A2 collisions | **6.67 /min** (loom 6.23, ghost 7.37) | ≤ 0.5 /min and ≤ 0.5× min(loom, ghost) | no |
| A3 dodge | 0.636 (balanced 0.25, ghost 0.441) | ≥ 0.8 / ≥ 0.8 / ≤ 0.3 | no |
| A4 loom causality | loom collision ratio **0.935** | ≥ 2.0 | no |
| A5 skill probes | steer 0.0, approach 0.6, dodge 0.8, wall 1.0 | balanced ≥ 0.8 | no |
| A6 coverage / slow | coverage **0.185** / slow 0.028 | ≥ 0.4 / ≤ 0.1 | no (coverage) |
| A7 real-time | live | — | — |

`passed = false`.

## 2. What v2 did and did not fix

Against the committed v1 gate (canonical, 50 seeds, same base/level):

| metric | v1 | v2 | verdict |
| ------ | -- | -- | ------- |
| A6 `slow_fraction` | 0.622 | **0.028** | **fixed** (the idling) |
| A6 coverage | 0.032 | **0.185** | 5.8× better, still < 0.4 |
| A2 collisions/min | 1.25 | **6.67** | **worse** — it now moves fast into things |
| A1 beacon rate | 0.030 | 0.033 | unchanged: essentially no beacon seeking |
| A3 ghost dodge | 0.042 | **0.441** | **causality lost**: it "dodges" blind |
| A4 loom ratio | 2.82 (pass) | **0.935 (fail)** | loom silencing no longer raises crashes |
| A5 steer probe | 0.7 | **0.0** | stopped steering toward the target |

So v2 trades idling for collisions: it moves, covers more ground and intermittently, but the motion
is not *directed* by what it sees. The loops that matter — beacon seeking (A1), loom avoidance (A4,
A5-steer), causal dodging (A3) — are absent or lost.

## 3. Reading

- The idling was a readout problem and v2 fixed it. What remains is a **behaviour** problem: the
  declared bridge reads only the 2,022 descending/VNC motor traces, and at rest those carry a tonic
  forward drive and weak steering, not a target-seeking or threat-avoidance signal.
- Tuning the codec further cannot create a signal that is not in the readout; more gain would mostly
  add collisions (A2 is already 13× the bar). This is the P4 spec's own falsifier: after the readout
  is faithful, the residual is the brain's ongoing dynamics and its sensory drive.
- The candidate fixes are the deferred P1/P3 machinery and new contract work: **richer senses and
  body feedback into the connectome** (vision/optic flow/feedback), which is a C4+ declared,
  versioned change needing sign-off. The fly's own seeking and saccadic avoidance are closed-loop
  behaviours; the open-loop tonic bridge is not expected to reproduce them.

## 4. Caveats and next gate

- 15-seed smoke, not the 50-seed gate; the direction is unambiguous (A1 is 4.5 % of the teacher, not
  a marginal miss), so the full gate was not spent.
- v1's numbers are the committed 50-seed run; v2's are 15 seeds from the same seed base. The
  comparison is directional, not paired.

**Recommendation:** do not ship v2 as a *capable* default (it is correctly shipped as the *alive*
default). Next workstream: a C4 spec for closed-loop senses/feedback into the connectome, aimed at
A1/A3/A4, with the A1–A7 gate re-run after each declared addition.
