# vy/vz training signal for the decoder/encoder — design

> **Superseded (2026-09-16).** Absorbed into three places and no longer maintained as a standalone
> design: its instrumentation (M1–M3) landed as Pack 1 R2–R4 of
> [`2026-09-15-training-improvements.md`](2026-09-15-training-improvements.md); its lateral-lag
> diagnosis now motivates [`2026-09-16-wing-level-action-design.md`](2026-09-16-wing-level-action-design.md)
> §1; and its strategy option (c), "increase evade exposure", was approved as the
> threats-without-pillars curriculum in
> [`2026-09-16-retinotopic-sensing-v6-design.md`](2026-09-16-retinotopic-sensing-v6-design.md) §5.
> Retained for provenance: §2 (the vy/vz diagnosis) and §6 (the axis asymmetry) are still cited.

**Status:** absorbed 2026-09-16; retained as a historical record. Originally proposed 2026-09-15,
awaiting user decision — documentation only, no code, no runs. Read
[`docs/free-roam.md`](../../free-roam.md) and the encoder v5 spec
([`2026-09-14-learned-encoder-sac-design.md`](2026-09-14-learned-encoder-sac-design.md)) first;
this doc cross-references both and reopens neither.

## 1. Why

The user noticed free-roam training isn't producing much vertical movement, and on inspection
lateral is in the same position. `roam_eval.ACCEPTANCE` (A1–A7,
[`roam_eval.py:14-30`](../../../python/fly_drone/roam_eval.py)) can be satisfied by a policy
that barely uses `vy` or `vz` at all: A3 (`A3_min_dodge_rate`, `A3_min_balanced_dodge_rate`) is
a binary hit/dodge outcome keyed by which **side** the threat came from, not by which action
axis produced the dodge; A6 checks only yaw bias and loss-of-control crash categories. Nothing
today would flag it if a decoder learned to "dodge" mostly through forward braking and lucky
positioning rather than actually commanding lateral/vertical velocity. This doc diagnoses why
that gap can exist, proposes instrumentation to see it, and lays out options — it does not
propose a fix without first measuring whether one is needed.

| Drive (`teacher.py:96-163`) | `vx` | `vy` | `vz` | `yaw` |
| ---------------------------- | ---- | ---- | ---- | ----- |
| Explore (`teacher.py:30-42`) | ✅ (`EXPLORE_FORWARD`) | 0.0 | 0.0 | ✅ (yaw cast) |
| Approach beacon (`teacher.py:114-116`) | ✅ | 0.0 | 0.0 | ✅ |
| Avoid pillar/wall (`teacher.py:131`) | ✅ | 0.0 | 0.0 | ✅ |
| Evade threat (`teacher.py:159`) | fixed brake (-0.5) | ✅ (`evade_dir`) | ✅ (`evade_vertical`) | 0.0 |

## 2. Diagnosis

### 2.1 Where `vy`/`vz` come from

`teacher_action` (`teacher.py:96-163`) blends whichever drives are active that frame as
`weight * drive_action + (1 - weight) * action` (e.g. `teacher.py:132`, `teacher.py:160`).
Explore, approach-beacon, and avoid all construct their action arrays with literal `0.0` in the
`vy`/`vz` slots (`teacher.py:42`, `teacher.py:114-116`, `teacher.py:131`). Only the evade-threat
drive's action, `np.array([-0.5, threat["evade_dir"], threat["evade_vertical"], 0.0])`
(`teacher.py:159`), has nonzero `vy`/`vz`. Since every other term contributes exactly zero to
those two components, **the blended label's `vy`/`vz` is nonzero only while the evade-threat
drive has nonzero weight** — i.e., only during an active, in-range, unoccluded threat encounter.

### 2.2 Exposure is sparse

Threats exist only at arena level 3 (`arena.py:63`, `LEVELS = {0: (0, False), 1: (6, False),
2: (16, False), 3: (16, True)}` — the tuple is `(pillar count, threats enabled)`). Level 3 always
bundles 16 pillars; there is no level with threats but no/fewer pillars. Within a threat-enabled
episode, the first throw is scheduled `uniform(3.0, 8.0)` s after reset (`env.py:246`), and every
subsequent one `uniform(8.0, 20.0)` s after the previous one resolves (`env.py:352`, inside
`_finish_threat`). A free-roam episode is 1500 frames at 25 Hz (`HORIZON_FRAMES`, `FRAME_SECONDS`
— `env.py:10,28`) = 60 s. So an episode sees on the order of 3–7 threat encounters, each lasting
only as long as the drone is within `THREAT_LOOM_RANGE + 0.5` m and the threat is visible
(`teacher.py:141`) — a small fraction of total frames, against explore/beacon/avoid being active
essentially every frame of every episode at every level.

### 2.3 Stage 1 contributes zero `vy`/`vz` examples, by design

The encoder v5 spec's training-stages table (§4) puts Stage 1 (DAgger decoder warm start) on
**L2, no threats**, explicitly because "v4 loom cues on L3 would teach the decoder to escape
from walls." This is a deliberate, already-approved design choice — this doc does not propose
revisiting it. It does mean the round-0 decoder that seeds Stage 2b/3 starts having seen exactly
zero non-zero `vy`/`vz` labels. Threat/`vy`/`vz` exposure is intended to arrive during Stage 2b/3
(3 alternating encoder/decoder SAC rounds, L3, 350k + 150k frames/round per the spec's §4 table)
— but §2.2 shows throws stay sparse there too, so `vy`/`vz`-labeled experience remains a small
fraction of frames even once L3 is in play.

### 2.4 Isotropic exploration init

SAC's actor `log_std` — the per-action-dimension exploration noise — is set identically at both
initialization points: in the DAgger-clone fit (`sac.py:436-438`,
`actor.log_std.weight.zero_(); actor.log_std.bias.fill_(-2.5)`) and again whenever a round starts
from a `.pt` v4-clone init (`sac.py:667-669`, same two lines). Both zero the weight matrix and
fill the bias with one scalar applied uniformly across all four action dimensions
(`ACTOR_WARMUP_FRAMES = 50_000`, `sac.py:41`, freezes the actor for the first 50k frames of every
round so this initial noise level persists unchanged through warm-up). There is currently no
mechanism biasing initial exploration toward `vy`/`vz` specifically.

### 2.5 No instrumentation sees this today

- `roam_eval.ACCEPTANCE` (`roam_eval.py:14-30`) has no `vy`/`vz`-specific bar. `dodge_rates`
  (`roam_eval.py:47-57`) computes a binary `dodged` rate from `threat_log` entries, split by
  `t["side"]` (which side the threat came from) — not by which action axis the drone actually
  used to evade.
- The DAgger-clone fit's per-drive error report (`sac.py:439-457`) computes
  `err = (tanh(pre_tanh) - target).square().mean(1)` (`sac.py:443-447`) — `.mean(1)` averages
  across all four action dimensions **before** grouping by drive, so even the "threat" drive's
  own `held_out_mse` number hides `vy`/`vz` error inside a combined figure alongside `vx`/`yaw`.
- `_finish_threat` (`env.py:335-353`) records `side`, `min_distance`, `hit`, `dodged`
  (`env.py:338-344`) — no record of the peak `vy`/`vz` actually commanded during the encounter.
- No `round0*/decoder.json` or `validation.json` under `runs/v5/` contains any per-axis field
  (confirmed by inspection; these files carry only `version, encoder_version, dataset_hash,
  feature_ids, mean, scale, layers, action_limits, output`).

Net effect: a policy can pass every pre-registered acceptance bar (A1–A7) while barely using
`vy`/`vz`, or using them in a degenerate/imbalanced way, and nothing in the current pipeline
would surface it.

### 2.6 `vy` and `vz` are not one problem

`docs/results/roam-step-response.json`: from a 0.56 m/s forward cruise, a commanded vertical step
reaches 80% of target in **0.685 s**; a commanded lateral step takes **1.48 s** — vertical is
direct-thrust, lateral requires the airframe to tilt first. `teacher.py:148-155`'s own comment
already reflects this: the evade label always commands both axes at full authority ("lateral and
vertical are independent axes... commanding both at full authority costs nothing", `teacher.py:
156-158`), but the code's design note explains vertical was chosen as "the primary escape axis"
specifically *because* lateral alone was judged "too slow to matter" against a threat typically
~1.9 m away. So even with identical training exposure, lateral has a harder physical ceiling to
reach — a fix for one axis should not be assumed to transfer to the other, and any future
measurement or acceptance criterion should keep them as separate columns, not a combined
"vy/vz" figure.

## 3. What already exists and is not reopened by this doc

- Stage 1 stays on L2 (encoder v5 spec §4) — not revisited here.
- Task 13's stop rule (`docs/superpowers/plans/2026-09-14-fly-drone-05-encoder-v5-sac.md`,
  Task 13 Step 5) stays keyed on `near_dodge_rate` alone.
- Task 14 (E3 brain-bypass check) and Task 15 (final evaluation) are unaffected.
- `roam_eval.ACCEPTANCE` is unchanged; no threshold in it is proposed for modification here.

## 4. Proposed instrumentation (prerequisite to any strategy decision)

All additive: new dict keys / new functions, reusing existing machinery rather than building a
parallel system. None of these are implemented by this doc — see §8.

| ID | What | Where | Reuses |
| -- | ---- | ----- | ------ |
| M1 | Per-dimension DAgger/clone error: `threat_vy_mse`, `threat_vz_mse` alongside the existing per-drive `{split}_mse` | `sac.py:439-457` (the `err` array already has per-dimension columns before `.mean(1)`; index columns 1/2 for the `threat` drive's rows instead of averaging them away) | existing `report["drives"]` structure |
| M2 | Per-axis dodge decomposition: peak `\|vy\|`, `\|vz\|` commanded during each evade window, conditioned on dodged vs. hit | new fields on the `outcome` dict in `_finish_threat` (`env.py:338-344`); a new function beside `dodge_rates` (`roam_eval.py:47-57`) that reads them | existing `threat_log`/`dodge_rates` filtering logic |
| M3 | Rollout-level action stats during SAC training: `rollout/action_vy_absmean`, `rollout/action_vz_absmean` | a callback sibling to `MetabolicLogger` (`sac.py:572-578`), added to the same `CallbackList` in `train_round` (`sac.py:673-`) | existing `logger.record_mean` pattern |
| M4 | Round-by-round summary table combining M1+M2 | small script assembling the table Task 13 Step 5 already requires ("a table of rounds 0..k: near-dodge, balanced, ghost near-dodge, beacons/min, collisions/min, E1 AUC, mean metabolic cost") | augments that table with two columns, does not replace it |

**Explicitly not proposed:** any change to `roam_eval.ACCEPTANCE` — a pre-registered threshold
change needs the user's separate agreement (per `roam_eval.py:1-5`'s own docstring and
AGENTS.md), and is out of scope for this doc.

## 5. Strategy options

Gated on §4's data existing first — none of these should be chosen blind.

| Option | What changes | Cost | Risk | Trigger |
| ------ | ------------ | ---- | ---- | ------- |
| (a) Measure first | Land M1–M2 (read-only reporting), read them at the existing Task 13 Step 5 round boundary | none beyond what Task 13 already runs | none | always do this first |
| (b) Bias initial exploration | Fill `log_std.bias` less negative for indices 1–2 (`vy`,`vz`) than 0,3 at both init sites (`sac.py:436-438`, `667-669`) | small, isolated, reversible code change; no new data collection | touches SAC init only, not reward/arena/data semantics | if (a) shows `threat_vy_mse`/`threat_vz_mse` not improving round-over-round, or dodges succeeding with near-zero peak `\|vy\|`/`\|vz\|` |
| (c) Increase evade exposure | A threats-without-pillars scenario, or reweighting DAgger/collection toward threat-drive frames (precedented by Task 12c's light-side-frame oversampling, `sac.py` clone-fit sampler) | collection/arena-semantics change | touches pre-registered `arena.LEVELS` and related constants — needs explicit user sign-off, separate from this doc | only if (b) alone doesn't close a measured gap |

**Recommendation:** (a) → (b) → (c), in that order, escalating only when the prior step's
measurement shows the next one is actually warranted. Option (c) is left as a named placeholder
here, not fully specified, to avoid speculative design on something that may never be needed.

## 6. `vz` vs. `vy`: differentiated treatment

| Axis | Actuation ceiling (`roam-step-response.json`) | Current teacher bias | What "fixed" looks like |
| ---- | ----------------------------------------------- | --------------------- | ------------------------ |
| `vz` (vertical) | 80% of step in 0.69 s (direct thrust) | primary escape axis by default | used whenever appropriate; a real ceiling is plausible to reach |
| `vy` (lateral) | 80% of step in 1.48 s (airframe must tilt first) | secondary, judged "too slow to matter" for a ~1.9 m threat | *used* when geometry favors it (e.g. more warning distance), not necessarily to the same magnitude as `vz` |

Any future evaluation of this gap must report these as two separate numbers. "Fixed" should mean
each axis is used when the situation calls for it, not that the two axes reach matched
magnitudes — the actuation asymmetry is a body/physics property this project isn't changing.

## 7. Relationship to the live pipeline

This doc does not pause, alter, or add a new gate to the currently-running
`stage2a-dagger.sh` → `pipeline13-15.sh` chain. The proposed instrumentation (§4) is read as
additional columns on the round-boundary report Task 13 Step 5 already produces, not a new stop
condition. The Task 13 stop rule remains `near_dodge_rate`-only, as approved.

## 8. Landing the code: now vs. after

Per `docs/results/2026-09-15-encoder-v5-run-decisions.md`, the user has already granted autonomy
through Task 15, which **supersedes Task 13 Step 1's per-round "ask the user to start round k"**
— so there is no natural per-round pause left to use as a safe landing window for source edits.
`env.py` and `sac.py`, the two files M1–M3 would touch, are imported fresh by every new
subprocess the running/about-to-run chain spawns (`roam-collect`, `sac-round`, etc.).

This is a distinct hazard from AGENTS.md's "ask before long data collection / DAgger / PPO runs"
rule (that rule is about *initiating* runs; this is about *editing source under* a run already
authorized and in motion). Recommendation: **hold all of M1–M3 until the user explicitly says
when to land them** — e.g. after Task 15 completes, or after a deliberate pause the user
requests — rather than landing them mid-chain against files the unattended pipeline is actively
importing.

## 9. Open questions

1. Confirm: hold all instrumentation code (M1–M3) until the pipeline reaches a natural stop
   (Task 15 done) or a deliberate pause, since the autonomy grant through Task 15 removes the
   per-round ask that would otherwise be the safe landing window (§8).
2. Should strategy option (c) be fully specified now, or left as the placeholder in §5 until
   §4's data shows it's actually needed? (Recommendation: placeholder.)
3. Any objection to this living as a standalone spec (this file) rather than as an amendment
   section appended to the encoder v5 spec?

## 10. Decisions (user, 2026-09-16)

Recorded by the recovery session after Phase 0 diagnostics
(`docs/results/encoder-v5/PHASE0-DIAGNOSTICS-2026-09-16.md`). The pipeline has stopped, so the
§8 landing hazard is moot.

1. **(a) Measure first — adopted.** M1–M3 land as Pack 1 R2–R4 of
   `2026-09-15-training-improvements.md` in Phase 1; M4's columns ride in the new round selection
   guard's report. No new stop condition, no `ACCEPTANCE` change.
2. **(b) Bias initial exploration — gated on measurement.** Only after the re-run's M1–M3 data
   says `threat_vy_mse`/`threat_vz_mse` are not improving, or dodges succeed with near-zero peak
   `|vy|`/`|vz|`. It touches the same two init sites as the Phase 1 log_std clamp, so it stays a
   small addition.
3. **(c) Increase evade exposure — placeholder.** Fully specified only if (b) proves insufficient;
   it touches the pre-registered `arena.LEVELS` and needs separate sign-off.
4. **Open question 2:** (c) stays a placeholder as recommended. **Question 3:** this file stays
   standalone. **Question 1 (hold M1–M3 until the pipeline stops):** resolved — the pipeline is
   stopped and the instrumentation lands in Phase 1.
