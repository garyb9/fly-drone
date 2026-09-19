# Liveness bar + P3 visual relay — plan

**Status: Part A landed, Part B landed (infra + offline validation), 2026-09-19.** The liveness bar
(`liveness.py`, `liveness-check`) and the declared optic-flow relay (`relay.py`, `--visual relay`)
are in. The relay's yaw flow validates against simulator egomotion (r ≈ 0.97–0.99); the pitch channel
is declared but weak under vertical translation. No behaviour gate has been run yet — the 15-seed
smoke and 50-seed gate are the next steps and need sign-off.

**Documented deviations from the first draft.** The relay uses global phase correlation on the
rendered eye images (sub-pixel) rather than ommatidial sampling; the measured eye geometry is not
needed for a declared translation model and the ommatidial join is absent. The relay adapter is
`adapter-relay.json` with `visual="relay"` folded into the version (as planned); canonical
`adapter.json` is unchanged.

Part A and Part B are additive. `roam_eval.ACCEPTANCE`, `CONDITIONS` and A1–A7 are never touched.
The canonical bundle and accepted actors stay byte-identical. No teacher enters the behaviour path.

**Goal.** Define a measured, teacher-free "feels alive" bar, then let the frozen connectome drive
the drone through a declared optic-flow relay so the bar becomes reachable — without a teacher, a
learned decoder, or relaxing acceptance.

**Why.** P0 showed the declared adapter cannot forage (0.03 vs teacher 0.73 beacons/min) and the
drone is 62 % slow / 3.2 % coverage. P1 gave one causal effect (`slow_fraction` 0.572→0.623). P2's
declared prior was null. The current v4 eye gives the brain only two global image statistics per
eye (light, loom) and no direction-selective motion; the connectome therefore has little to act on.
P3 activates the motion pathway with a declared feature and anatomically chosen targets, and the
liveness bar measures whether that makes the drone move like it is being driven by what it sees.

## Invariants

- `ACCEPTANCE` / `CONDITIONS` / A1–A7 untouched; liveness is an additive diagnostic.
- No teacher in the behaviour path; the teacher is at most an unweighted context row.
- Canonical bundle and accepted actors byte-identical; `adapter.json` / `adapter-check.json`
  unchanged.
- One brain, one bridge; the relay is a global declared bridge version, never scenario-switched.
- ≤6 workers; ask before the 15-seed and 50-seed runs.

## Part A — liveness bar

**A1. `python/fly_drone/liveness.py`.** Pre-registered thresholds and `liveness(results, policy)`
returning per-criterion detail and `passed`. Teacher-free:

| Criterion | Metric | Bar |
| --- | --- | --- |
| L1 mobility | `slow_fraction` | ≤ 0.25 |
| L2 exploration | `coverage = mean_visited_cells / 256` | ≥ 0.15 |
| L3 sense-causality | paired-bootstrap CI of intact − `sensory` on `slow_fraction` | upper bound < 0 |
| L4 anti-luck | `ghost`, `random`, `cue_script` | each fails L1∧L2 |
| L5 non-degenerate | `mean_abs_yaw_bias` ≤ 0.25, no loss-of-control | — |

Baseline evidence: canonical 0.622 slow / 3.2 % coverage (fails L1, L2); random 0.070 / 4.3 %
(fails L2); cue-script 9.8 % (fails L2); teacher 0.009 / 26.2 % (passes). The teacher appears only
as context, never in the pass/fail.

**A2. `roam_eval.liveness_check(...)`** mirrors `feedback_check` (`roam_eval.py:434`). Combos =
`policy|none`, `|sensory`, `|ghost`, plus `teacher|cue_script|random` `none`. Writes
`docs/results/liveness/liveness-check.json`; reuses `paired_bootstrap` / `_per_seed`.

**A3. CLI.** `fly-drone liveness-check` subparser + dispatch; exit 1 when not passed (diagnostic,
like `adapter-check`).

**A4. Tests.** `tests/test_liveness.py`: synthetic-summary thresholds, CI reuse, CLI exit-code
monkeypatch, source assertion that the behaviour path contains no teacher.

**A5. Docs.** `docs/results/liveness/FINDING-<date>.md` after the first run; spec status note;
`docs/free-roam.md` §6 mention.

## Part B — P3 visual relay

**B1. `python/fly_drone/relay.py`.** Declared optic-flow front-end: sample stereo luma at measured
ommatidial directions (`eye_geometry.EyeMap.sample`), compute per-eye signed horizontal (yaw) and
vertical (pitch) flow between frames, clip to `[0, 2]`. Four new channels
`flow_yaw_l/r`, `flow_pitch_l/r`; existing `light_l/r`, `loom_l/r` retained. Deterministic, no
learning.

**B2. Declared target mapping.** Derive roles at runtime from `cells.json` (same pattern as loom,
`brain.py:121`): signed flow splits to `T4a/b`,`T5a/b` (horizontal) and `T4c/d`,`T5c/d` (vertical)
by side. Canonical bundle untouched. The subtype tuning is the declared prior; the direct drive is
a declared shortcut past the photoreceptor relay, supported by `external-prior-art.md:71`.

**B3. Brain/env plumbing.** `brain.py`: `relay_ids`, `inject_relay`, `silence_relay`, include relay
ids in `silence_sensors`. `env.py`: opt-in relay flag, per-frame flow observe, per-sub-tick
injection (mirrors feedback `env.py:424`), and a `relay` pathway ablation in `PATHWAYS`.

**B4. Bridge identity.** `relay-v1`; `docs/results/adapter/adapter-relay.json` with a `visual` field
folded into `_version` only when present (canonical `adapter.json` byte-identical). Calibrate via
`adapter-calibrate --visual relay`; `--visual {v4,relay}` opt-in, default v4 until it wins.

**B5. Offline validation.** `scripts/validate_relay_flow.py`: correlate the declared flow against
simulator camera egomotion (labels only, never injected) before any behaviour run. FlyView ground
truth is a later follow-on needing the dataset.

**B6. Tests.** `tests/test_relay.py`: flow determinism + sim-egomotion correlation, role derivation,
injection/silence, identity pinning, canonical-adapter-unchanged guard.

**B7. Docs.** `docs/sensory-model.md` §8; FINDING after the gate; spec §7 P3 tick.

## Phasing

1. Save this plan; land Part A (code + tests + docs) — no long run.
2. Land Part B feature/mapping/plumbing/tests + offline validation — no long run.
3. **Ask** → 15-seed causal smoke (~45 min, non-pre-registered).
4. **Ask** → full 50-seed pre-registered gate (~3 h) only if the smoke is promising.
5. Adopt relay and tick P3 only on a liveness pass; otherwise record the negative and keep v4.

## Verify each step

```bash
env -u PYTHONPATH .venv/bin/python -m pytest -q
.venv/bin/ruff format python tests && .venv/bin/ruff check python tests
```

## Risks

| Risk | Mitigation |
| --- | --- |
| Coarse flow estimator | validate vs sim egomotion before any behaviour run; a null result is reportable |
| T4/T5 direct drive is a shortcut | state it in the FINDING; subtype tuning is the declared prior |
| Liveness unreachable without more control channels | that is the next lever, not a reason to relax the bar |
| New identity touches canonical artifacts | guard with a canonical-adapter-unchanged test |
