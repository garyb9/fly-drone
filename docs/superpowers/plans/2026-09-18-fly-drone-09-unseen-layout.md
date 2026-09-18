# Unseen-layout generalization (M5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Status:** proposed, not approved (2026-09-18). Docs only; no code written.

**Goal:** show the accepted free-roam actor keeps its skills in arena families it was never trained
on, and if it does not, fix it with domain randomization rather than redefining "unseen" as
"another seed".

**Spec:** [`2026-09-18-unseen-layout-generalization.md`](../specs/2026-09-18-unseen-layout-generalization.md).
**Depends on:** the accepted frozen-v6 actor from
[plan 08](2026-09-18-fly-drone-08-frozen-v6-free-roam.md). **Mechanics:** plan 04
[free-roam handoff](2026-09-14-fly-drone-04-free-roam-handoff.md).

## Global constraints

- Connectome, encoder and decoder interfaces frozen as in plan 08; only obstacle geometry and level
  vary. `ArenaSpec.limits`/`altitude` and the PID never change.
- `roam_eval.ACCEPTANCE`, A1–A7 and E1–E4 are never relaxed. The generalization criterion is
  **additive** and pre-registered before any shifted run.
- The default `ArenaSpec()` path stays bit-identical, so canonical evaluations and
  `test_legacy_room_mjcf_unchanged` reproduce.
- Workers ≤ 6; `env -u PYTHONPATH .venv/bin/python …`; ask the user before every long run.
- No `fly_drone/*.py` edits while a `sac-round`/`roam-*` process is live. Commit and push per task.

## Design

```
training support D_train                  evaluation
  ArenaSpec family × layout seed            S0 in-family (canonical A1–A7)
        |                                    S1 denser   S2 larger
        v                                    S3 threats  S4 adversarial wall
  roam-collect / DAgger  ──>  decoder SAC (frozen v6 pair)  ──>  evaluate A1–A7 + E1–E4
```

- **Layout family** is `(level, ArenaSpec)`. `generate_layout` already randomizes within a family;
  M5 adds a sampler over families and an explicit evaluation shift registry.
- **DR is the remedy, not the premise.** Task 3 measures the accepted pair first; DR runs only if a
  shift collapses (decision 3), because those runs are long.

## File structure

| File | Responsibility | Change |
| --- | --- | --- |
| `python/fly_drone/arena.py` | layout-family sampler, shift registry, difficulty metric | modify (additive) |
| `python/fly_drone/env.py` | accept and use an explicit arena spec/layout | modify (default unchanged) |
| `python/fly_drone/distill.py` | thread the spec through `collect`/`screen` | modify |
| `python/fly_drone/sac.py` | thread the spec into the round env factories | modify |
| `python/fly_drone/cli.py` | `--arena`/`--shift` options on collect/screen/evaluate | modify |
| `tests/test_arena.py` | sampler determinism, connectivity, difficulty ordering | modify |
| `docs/results/encoder-v6/` | `GENERALIZATION-<date>.md` | create |

## Tasks

### Task 1: Layout-family sampler, shift registry, difficulty metric
- Add a deterministic `sample_family(rng)` over a bounded support and a `SHIFTS` registry
  (`S0`–`S4`), each a `(level, ArenaSpec)` plus a seed base.
- Add `difficulty(spec, pillars)` → `{free_fraction, clearance_p10, clearance_mean, components}`.
- Tests: same seed → same family; every sampled layout is connected; S1–S4 metric distributions are
  disjoint from `S0`; no change to `ArenaSpec()` defaults.

### Task 2: Thread the arena through the pipeline
- `ConnectomeEnv` already takes `spec`; thread `level`+`spec` through
  `distill._roam_env`/`collect`/`screen`, `sac` env factories, and `evaluate_free_roam`.
- CLI: `roam-collect`/`roam-screen`/`evaluate` gain `--arena <S0..S4>` (default: current behaviour).
- Tests: default path is bit-identical (reuse the legacy-room and screen regression tests); an
  explicit `S4` arena changes `plant.pillars` deterministically.

### Task 3: Measure the accepted pair on S0–S4 (no training)
- `evaluate --task free_roam --episodes 50 --arena S0..S4` per shift; also the teacher and random
  baselines per shift.
- Report beacons/min, collisions/min, dodge, ghost dodge, coverage, difficulty metric, paired CI.
- **Gate:** does any skill collapse on a shift? If none, M5 is a no-op (record it) and Tasks 4–5
  collapse to the report. If one does, continue.

### Task 4: DR collection + one gated decoder round (needs sign-off)
- `roam-collect` with `--family-support` (or repeated `--arena`) so the training mix spans the
  support; DAgger refit; screen each.
- One decoder SAC round with the plan-08 guard/revert and the decoder anchor (plan 08 Phase 4).
- Accept only on canonical G1/G2 **and** no shift collapse.

### Task 5: Report
- Write `docs/results/encoder-v6/GENERALIZATION-<date>.md`: per-shift A1–A7, E1–E4, difficulty
  metrics, CIs, the DR run if any, and the honest verdict.
- Update `docs/validation.md`, `docs/free-roam.md` and overview §9–§10.

## Open decisions (need the user)

1. **Shift set** — S1–S4, or trim to the skill-bearing shifts S3+S4?
2. **Generalization bar** — accept the proposed additive criterion before the run (see spec §5).
3. **DR trigger** — only on collapse (recommended) vs always.
4. **Upper bound** — include the teacher per shift (recommended) or actor only?

## Risks

| Risk | Mitigation |
| --- | --- |
| Shift is not actually harder | difficulty metric computed and reported; invalid shifts are not evidence |
| Threading a spec breaks the canonical eval | default `ArenaSpec()` and regression tests pin the legacy path |
| DR round collapses as encoder rounds did | decoder-only + anchor + guard/revert (plan 08) |
| Scope creep into training | Task 3 is diagnostic; Task 4 only on a measured collapse |
