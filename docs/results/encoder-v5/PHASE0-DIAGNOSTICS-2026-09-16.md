# Phase 0 diagnostics (2026-09-16): what broke encoder v5 round 1

Recovery session after the run was stopped at 01:30 on 2026-09-16
(`HANDOFF-2026-09-16.md`). Nothing here is training; every number is a validation of a saved
artifact (seeds 9000–9009, 60 s, level 3, 6 workers), offline inference, or an existing run log.

## 0. Summary

The round-1 **encoder** broke first, and it broke in the first ~13 k frames after its actor
unfroze: its deterministic mean output collapsed to a **constant 1.0 current on all 8 channels**,
so the deployed encoder is blind. The decoder round inherited that blind encoder and never had a
chance. The collapse happened while the entropy coefficient was still at its SB3 default of 1.0.

Three independent lines of evidence agree:

1. **D1 (behaviour):** round-1 encoder + the *good* round-0 decoder forages **0.00 beacons/min**
   (the same decoder under the clone forages 2.13).
2. **D4 (offline):** the exported encoder's mean current is bit-identical to the clone at 50 k,
   then 0.977 ± 0.026 at 100 k, 0.999 at 150 k, and 1.000 ± 0.005 (constant) at 350 k.
3. **The run logs (in-run):** `loom_cost = 0.01 · mean(loom currents)` jumps from 0.0017 (clone)
   to a flat **0.010** between 45 k and 63 k — i.e. every loom channel pinned at exactly 1.0 —
   and stays there for the remaining 280 k frames.

This refines the handoff's §2.2 hypothesis from "the entropy bonus swamps the small reward" to the
specific failure it produces: an entropy-driven collapse of the policy's **deterministic mean**
(the deployed artifact) to the max-entropy centre, not sigma inflation/tanh saturation.

## 1. D1 — which learner broke round 1? (encoder)

Repinned round-0's decoder JSON onto the final round-1 encoder and validated
(`sac-export --repin-decoder`, `runs/v5/diag/d1/`).

| pair                                   | near_dodge | balanced | ghost | beacons/min | collisions/min | E1 loom AUC | E2  |
| -------------------------------------- | ---------- | -------- | ----- | ----------- | -------------- | ----------- | --- |
| clone + round0 (baseline)              | 0.269      | 0.182    | 0.296 | 1.9         | 2.7            | 0.686       | fail |
| **round1 encoder + round0 decoder**    | **0.111**  | **0.000** | 0.087 | **0.00**    | **5.40**       | **0.441**   | fail |

The only changed part is the encoder, and foraging goes to zero. **Conclusion: the encoder round
was fatal on its own; the decoder round merely inherited the wreckage** (the handoff's §2.3 reading
is confirmed). Near-dodge at 0.111 with balanced 0.0 and ghost 0.087 is not a dodge skill.

## 2. D4 — offline current drift (the collapse timeline)

Deployed currents are the deterministic mean `tanh(mu(eyes)) + 1` (`encoder.py:138-142`); D4 runs
the exported `.pt` over the same held-out clone frames as `lr_fidelity.py`
(`runs/v5/diag/current_drift.py`, log `runs/v5/diag/d4-current-drift.log`).

| checkpoint            | overall mean | std   | fraction sat > 0.9 | mean abs diff vs clone |
| --------------------- | ------------ | ----- | ------------------ | ---------------------- |
| clone (round-0 init)  | 0.337        | 0.587 | 0.675              | 0.000                  |
| round1 50 k           | 0.337        | 0.587 | 0.675              | 0.000                  |
| **round1 100 k**      | **0.977**    | 0.026 | 0.000              | 0.836                  |
| round1 150 k          | 0.999        | 0.012 | 0.000              | 0.850                  |
| round1 200 k          | 0.992        | 0.008 | 0.000              | 0.845                  |
| round1 350 k (final)  | 1.000        | 0.005 | 0.000              | 0.851                  |

The clone produces strongly-varying, mostly-saturated currents (loom channels mean ≈ 0.19, 67.5 %
of all values at the 0/2 extremes). The round-1 encoder produces a nearly constant 1.0 everywhere:
it is a **dead encoder**, carrying no visual information. The 50 k checkpoint is bit-identical to
the clone (same `learned-v5:1681bff17b4eda85` version hash) — the warm-up freeze worked perfectly.

## 3. In-run log corroboration

The round-1 encoder log already contained the signal
(`docs/results/encoder-v5/run-records/round1/round1-encoder.log`).
`metabolic_cost = λ_loom·mean(loom) + λ_light·mean(light)`, λ = 0.01 / 0.002, so a constant current
of 1.0 on every channel yields exactly 0.012 and loom_cost exactly 0.010:

| steps | actor_frozen | ep_rew_mean | loom_cost | metabolic_cost | ent_coef |
| ----- | ------------ | ----------- | --------- | -------------- | -------- |
| 45 k  | 1            | −251        | 0.0017    | 0.0030         | 1        |
| 54 k  | 0            | −285        | 0.0058    | 0.0073         | 1        |
| 63 k  | 0            | −271        | **0.0100**| **0.0120**     | 0.675    |
| 108 k | 0            | −233        | 0.0100    | 0.0120         | 0.0075   |
| 342 k | 0            | −151        | 0.0100    | 0.0120         | 5e-13    |

The collapse lands in the **13 k frames after unfreeze at 50 k**, while α ≈ 1.0 → 0.675. Once μ is
at the centre it never recovers, even as α decays to ~0 over the next 280 k frames — consistent
with a weak Q gradient through a saturated/max-entropy mean.

## 4. D2/D3 — behavioural timeline

D2 pins round-0's decoder to round-1 encoder checkpoints (isolates the encoder); D3 validates
round-1 decoder checkpoints under the round-1 encoder (isolates the decoder).

| diagnostic | pair                                   | near_dodge | balanced | ghost | beacons/min | collisions/min |
| ---------- | -------------------------------------- | ---------- | -------- | ----- | ----------- | -------------- |
| D2 100 k   | r1 encoder@100k + round0 decoder       | 0.000      | 0.000    | 0.074 | 0.10        | 5.0            |
| D2 150 k   | r1 encoder@150k + round0 decoder       | 0.083      | 0.062    | 0.091 | 0.10        | 5.9            |
| D3 50 k    | r1 decoder@50k + r1 encoder            | 0.111      | 0.000    | 0.087 | 0.00        | 5.4            |
| D3 100 k   | r1 decoder@100k + r1 encoder           | 0.933      | 0.857    | 0.875 | 0.00        | 13.1           |

D2 at 100 k is already degenerate (beacons 0.10 vs round-0's 1.9): the 50 k → 100 k collapse is
visible behaviourally, matching D4 and §3.

D3 100 k is the decoder's own SAC flailing under the blind encoder: near-dodge 0.933 looks like a
skill, but ghost is 0.875, beacons are 0.00 and collisions nearly triple (13.1/min). This is the
exact signature the round-1 validation saw (near 0.926 / ghost 0.815) and the metric the
near-dodge-only stop rule rewarded.

**D3 50 k is identical to D1** (0.111 / 0.000 / 0.087 / 0.00 / 5.4), exactly as it must be: the
decoder's actor is also frozen for the first 50 k frames, so at 50 k it is still the round-0
decoder — the same pair D1 measured. This shows the decoder round was **already fully degenerate
at its first frame, before its own actor moved at all**: zero beacons, no dodging, from the blind
encoder. Decoder SAC never had a working system to train on.

This separates the failure into two stages:

1. **Encoder collapse** (round-1 frames ~50 k–63 k): foraging to zero, vision gone. Encoder round
   alone, at α ≈ 1.
2. **Decoder flailing** (round-1 frames 50 k–150 k): the decoder's own SAC, under a blind encoder,
   degrades ghost dodge 0.087 → 0.815 and near-dodge → 0.926 — the erratic flight the near-dodge
   stop rule then counted as progress.

The entropy fix addresses stage 1; the selection guard catches stage 2.


## 5. Conclusion against the pre-registered interpretation matrix

- **D1 degenerate** → the encoder round broke the pair; the α/log_std fix belongs on encoder
  rounds first, and no decoder anchor is indicated (matches the user's decision to drop Pack 3 D1).
- **D2 broken at 100 k, not 50 k** → the break is *after* unfreeze, in the high-α window; the
  entropy-transient theory is **confirmed**, not refuted (the 50 k checkpoint is the untouched
  clone). No need to re-diagnose a critic/random-action-phase cause.
- The mechanism is specifically **mean collapse**, which sharpens the fix set.

## 6. Implications for Phase 1 (one addition to the approved scope)

The approved fixes (`ent_coef="auto_0.01"`, matched target entropy, log_std clamp, selection guard)
target this mechanism directly and are confirmed as the right first move. One refinement:

- **Log `train/log_std_mean` alone is not sufficient** — σ was not what collapsed; μ was. The
  logging item must also record **per-channel deterministic mean-current drift** on a fixed probe
  observation batch (cheap, CPU). The existing `metabolic_cost` log caught it by accident; make it
  deliberate.
- The selection guard (ghost ≤ 0.3, beacons ≥ ½ round-0, E2 pass) would have stopped the run at
  round 1 instead of running two more degenerate rounds (~7 h saved).

## 7. Artifacts

- `runs/v5/diag/d1/` — repinned decoder, validation, checks (D1)
- `runs/v5/diag/d2/` — exported encoder checkpoints + repinned decoders; `d2-100k/`, `d2-150k/` (D2)
- `runs/v5/diag/d3/` — exported decoder checkpoints; `d3-50k/`, `d3-100k/` (D3)
- `runs/v5/diag/current_drift.py`, `d4-current-drift.log` (D4)
- `runs/v5/diag/run_d2_d3.sh`, `d2-d3-chain.log` (chain driver)
- Unchanged: `roam_eval.ACCEPTANCE`, the plan's Task 13 rules, `runs/v5/round0` (best valid pair).
