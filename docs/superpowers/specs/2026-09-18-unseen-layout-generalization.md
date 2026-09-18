# Unseen-layout generalization (M5) — design

**Status:** proposed, awaiting user decision (2026-09-18). Documentation only: no source under
`python/fly_drone/` changes, no runs, no threshold moved.

**Where this sits.** M5 of the retinotopic-sensing v6 design
([§6, §8](2026-09-16-retinotopic-sensing-v6-design.md)). It is the goal-level check that the
free-roam skills — foraging, obstacle avoidance, threat dodging — survive an arena the agent was
never trained on, not merely held-out seeds of the same arena family. It runs **after** the frozen
v6 free-roam actor is accepted
([plan 08](../plans/2026-09-18-fly-drone-08-frozen-v6-free-roam.md)), on that actor.

**Related:** [`roam_eval.py`](../../../python/fly_drone/roam_eval.py) (A1–A7, E1–E4),
[`arena.py`](../../../python/fly_drone/arena.py), [`env.py`](../../../python/fly_drone/env.py),
[`distill.py`](../../../python/fly_drone/distill.py), [`free-roam.md`](../../free-roam.md).

## 1. The gap

`evaluate --task free_roam` flies held-out **seeds** (default 1000+, 50 episodes) but builds every
arena from the same distribution: `ConnectomeEnv` takes `spec=None` and falls back to
`arena.ArenaSpec()` (half_size 8, 16 slots, radius 0.3, spacing 2.5, band 0.8–1.2), and
`generate_layout` draws 16 pillars for `level=3`. Training and DAgger used other seeds of the same
family. Held-out seeds therefore test **sample** generalization, not **distribution** generalization.

The v6 spec §6 names this directly: *"The current held-out seeds share the arena family."* A decoder
can score A1–A7 by reading the same cue statistics it saw in training while never meeting a layout
family it did not. The goal ("open, **unseen** arenas", AGENTS.md) needs a shifted distribution.

## 2. What "unseen" must mean here

Two independent axes, both drawn per episode:

- **Layout draw** — already random through `generate_layout(rng, spec, level)`; training and
  evaluation must use disjoint layout sets and we must record that they are disjoint.
- **Arena family** — the `ArenaSpec` / `LEVELS` combination, currently fixed. This is the axis M5
  adds: train on a *support* of families, evaluate on *shifts* outside it.

The drone contract does not move: `ArenaSpec.limits`, `altitude`, the PID, the connectome, the
encoder identity and the decoder interface are unchanged. Only obstacle geometry and level
(threat on/off) vary, and only within what the eyes can see.

## 3. Shift set (proposed)

A small, explicitly insufficient-for-training set, each with a difficulty metric so "shifted" is
measured, not asserted. Candidate shifts, to be fixed before any run:

| Shift | Change vs training family | Reads as |
| --- | --- | --- |
| S0 in-family | same `ArenaSpec`, disjoint layout seeds | the current A1–A7 baseline |
| S1 denser | `pillar_spacing` ↑ / more pillars per area | tighter corridors |
| S2 larger room | `half_size` 8 → 10, walls farther, beacon range scaled | long-range search |
| S3 threat regime | train level 2, evaluate level 3 (thrown threats) | the hardest, and the one A3 scores |
| S4 adversarial obstacle | a wall segment dividing the room (a forced detour) | structure never trained |

Difficulty metric per layout, computed from geometry only: free-area fraction, mean/percentile
clearance, connected-component count (must be 1), and nearest-pillar distance distribution. A shift
is only valid if its metric distribution is outside the training support; otherwise it is not a
test.

## 4. Training under domain randomization (DR)

If the accepted pair does not already generalize (§6), the remedy is DR, in this order:

1. **Collection/DAgger DR** — `roam-collect` draws a family per flight from the training support
   (pillar spacing, half_size within bounds, level mix). Labels use simulator geometry, as always,
   and only for visible objects.
2. **Decoder SAC under DR** — the frozen-v6 decoder round runs on the randomized families with the
   existing guard/revert. The connectome and encoder stay frozen; the v6 encoder sees only its eye
   stack, and the layout variation reaches it as pixels.
3. **No mode switch.** One decoder over the whole training support and every shift; the server and
   the actor never key on arena identity.

DR is **additive** to the accepted plan: the canonical arena and its A1–A7 evidence stay.

## 5. Evaluation and pre-registration

- **Canonical (unchanged):** `evaluate --task free_roam` on `ArenaSpec()`, seeds as today. A1–A7 and
  E1–E4 are the accepted numbers; nothing here changes them.
- **Generalization (new, additive):** the same 50-episode protocol on each shift S1–S4, with a
  paired-bootstrap CI over shared seeds. Report beacons/min, collisions/min, dodge rate, ghost dodge,
  coverage, and the difficulty metric per shift.
- **Proposed generalization criterion (needs sign-off, pre-registered before the run):** on every
  shift, `A1` beacons ≥ 0.6 × teacher **and** A3 ghost dodge ≤ 0.3, with the paired CI excluding a
  collapse; plus each shift's beacons ≥ a pre-registered fraction (propose 0.6) of the S0 number.
  This is a new criterion; `roam_eval.ACCEPTANCE` is untouched and never relaxed.
- E1–E4 are reported on every shift (E4 = `test_legacy_room_mjcf_unchanged` plus v4 actors).

## 6. Falsifiers

- **Already generalizes.** If S1–S4 are within the CI of S0, M5 is a no-op and DR is not needed;
  record it.
- **Collapse on one shift.** If a skill is present in-family and absent on a shift, report the shift
  and the missing skill, then scope a DR run against that family. Do not average it away.
- **Difficulty not actually shifted.** If the metric distribution overlaps training, the shift is
  invalid; fix the shift, do not report it as evidence.

## 7. Task sketch

| # | Task | Notes |
| --- | --- | --- |
| 1 | Layout-family sampler + shift registry in `arena.py` | deterministic, connectivity-checked, difficulty metric; tests only |
| 2 | Thread an arena spec/layout through `env.py`, `distill.screen`/`collect`, `sac` env factories, CLI | default `ArenaSpec()` must leave legacy/eval bit-identical |
| 3 | Measure the accepted pair on S0–S4 | no training; this decides whether DR is needed |
| 4 | (if needed) DR collection + one gated decoder round | ask the user; guard/revert as plan 08 |
| 5 | Report + decision | `docs/results/encoder-v6/GENERALIZATION-<date>.md` |

## 8. Open decisions (need the user)

1. **Shift set** — accept S1–S4, or trim to S3+S4 (the two that test a skill, not just density)?
2. **Generalization bar** — accept the proposed criterion (A1 ≥ 0.6 teacher, ghost ≤ 0.3, shift ≥ 0.6
   of S0) or state your own before the run.
3. **DR trigger** — run DR only if a shift collapses (recommended), or train under DR regardless?
4. **Scope** — evaluate the accepted frozen pair only, or also the teacher as an upper bound per
   shift (recommended: both)?
