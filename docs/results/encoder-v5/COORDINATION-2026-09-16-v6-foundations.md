# Coordination: v6 foundations (separate session) and the running Phase 2 pipeline

**Written 2026-09-16 by the v6 foundations session**, for the session driving
`pipeline13-15.sh` (Plan 05 / encoder v5, Phase 2). Read this before touching shared modules
or the ledger for the next iteration.

## 1. The live run (updated 2026-09-16 ~14:25)

- The Phase 2 `pipeline13-15.sh` (pid 332789) **exited at 14:13** right after launching the round-1
  decoder SAC. The Phase 2 session then committed `c5796d7` and `10b7a2b` (pin loaded rounds to the
  post-warm-up entropy regime) and `f246f66` (preload pre-START round paths), and relaunched the
  round-1 decoder by hand (`sac-round decoder ...`, pid 363990, started 14:21).
- Shared working tree, same filesystem, same branch (`main`). There are no separate clones, so any
  file write is immediately visible to both sessions; the risk is a concurrent write, not a merge.
- `git status` was clean at each of this session's commits; no staged or uncommitted work to clobber.

## 2. Hard rule for this session while the pipeline runs

Per `docs/superpowers/specs/2026-09-15-training-improvements.md` §8 and the vy-vz spec §8: every
`sac-round` / `roam-*` subprocess re-imports `python/fly_drone/*.py` at spawn, so **no edit to a
shared module lands while any such subprocess is active**. This session will only add **new files**
until the pipeline stops or a window is named. Specifically held back:

- `python/fly_drone/brain.py` — v6 runtime wiring (spatial input roles)
- `python/fly_drone/plant.py` — `advance_wrench` + reflex rate controller
- `python/fly_drone/env.py` — frame-transform / domain-randomisation hook
- any change to `sac.py`, `encoder.py`, `distill.py`, `teacher.py`, `roam_eval.py`, `cli.py`

## 3. What the v6 foundations session has already added (all additive, committed)

| Commit | Content |
| --- | --- |
| `6d59eed` | Specs `2026-09-16-retinotopic-sensing-v6-design.md` and `2026-09-16-wing-level-action-design.md`; five read-only probe scripts under `scripts/` |
| `1e6b7e3` | Marked the vy-vz spec superseded; annotated the training-improvements status |
| `a78b4f0` | `python/fly_drone/retinotopy.py` + `tests/test_retinotopy.py` |
| `ef34d90` | `python/fly_drone/spatial_encoder.py` + `tests/test_spatial_encoder.py` (the v6 conv net, `learned-v6:`) |
| `5946d77` | Coordination-note update |
| `e667a9d` | `python/fly_drone/predictive.py` + `tests/test_predictive.py` (auxiliary predictive objective) |
| `7b8b1b5` | Fixed the v6 light pathway: added `mi1`/`tm3` spatial channels (12 maps, 816 currents) |
| `16b1e6f` | `python/fly_drone/augment.py` + tests (domain randomisation) |
| `334b32f` | `python/fly_drone/wrench.py` + tests (wing-level mixer and rate loop) |
| `a77c18f` | `python/fly_drone/spatial_clone.py` + tests (v4-cue -> v6 target maps) |
| `1b06172` | `python/fly_drone/spatial_policy.py` + tests (SB3 actor/extractor adapter) |
| `1f705d2` | `scripts/spatial_maps.py` (map inspection) |

These are **new files only** (except edits to this session's own earlier modules). The pipeline does
not import `retinotopy`, `spatial_encoder`, `predictive`, `augment`, `spatial_clone` or
`spatial_policy`, and its T15 E4 pytest runs only `test_arena.py::test_legacy_room_mjcf_unchanged`
and `test_env.py -k "legacy or replay_is_bit_identical"`, so the new test files are not collected by
the run. Full suite 215 passed, `ruff` clean.

## 4. What v6 is, in one paragraph

This is **next-iteration groundwork, not a change to Plan 05**. The M1/M1b feasibility gate (run
2026-09-16) found that the current 8-scalar encoder cannot carry loom selectivity (E1 0.687 vs the
0.8 bar) and that the viable retinotopic injection targets are **Tm4** (2D sheet, drives LC4/escape,
lateralises, sub-region selective) and **T2** (wide-field), while Mi1/Tm3/T4/T5 do **not** propagate
to the loom circuit. v6 widens the sensory bandwidth (spatial Tm4/T2 maps + direct LC4/LPLC2) and
deepens the action output (body wrench over a reflex rate loop). Nothing in it changes v4/v5,
`roam_eval.ACCEPTANCE`, or the frozen connectome. Details: the two specs in §3.

## 5. Request to the Phase 2 session

- Continue as planned. Do not adjust anything for v6; Plan 05's result stands on its own.
- The pipeline exited at 14:13 and the decoder round is now running under your manual control. When
  the chain reaches a natural stop (T15) or you name a landing window, add a line to
  `docs/results/encoder-v5/ledger.md` saying so. The v6 session will then land the shared-module
  changes (brain/plant/env) in one window, between runs.
- If the Phase 2 run produces a usable v5 final pair, note it in the ledger; the v6 specs assume
  the v5 baseline (`runs/v5/round0` or the chosen final) stays reproducible.
- Keep the ledger's existing conventions (append, date, cost-if-wrong where relevant); this note
  is additive and does not change any ruling.
