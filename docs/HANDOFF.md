# HANDOFF — where the project is, and where to continue

**Updated 2026-09-22 after the C4a A3 gate was run (negative).** This is the operational companion
to the governing spec; read it before touching code.

## 0. TL;DR

- The **brain is the fly's frozen connectome; the body is a simulated quadrotor** that the brain
  drives through a **declared bridge** (see `AGENTS.md` for the non-negotiables).
- **P4 closed.** Codec **v2** fixed the idling (the drone now moves). But v2 is **alive, not
  capable**: it fails every A1–A7 skill criterion.
- **C4a A3 was run 2026-09-22 and FAILED.** Codec v3's lateral escape is mechanical, not capable: it
  raises near-threat `|vy|` 0.3 → **0.8** and passes the ghost causality test (0.441 → **0.205**),
  but A3 dodge **fell 0.64 → 0.34** and **liveness L6 regressed** (move bout 9.14 → 12.80 s). By the
  C4 spec's own falsifier, **C4a alone is rejected; the 50-seed gate is not spent.** Evidence:
  [`FINDING-2026-09-22-c4a-a3.md`](results/adapter/FINDING-2026-09-22-c4a-a3.md).
- **C4b (selective loom) is now the prerequisite**, not a follow-up: until the loom cue separates a
  closing threat from a turn sweep, the lateral gain fires constantly and hurts. A v3 variant that
  **keeps the v2 climb and adds** the strafe is a candidate (needs sign-off).
- **Open decision (needs user):** v3 is still the shipped default (`DEFAULT_CODEC = "v3"`) by earlier
  user direction, but it has now failed its gate. Recommend reverting the default to v2 until a
  cell passes; do not silently keep an ungated default.
- **Nothing is blocked.** Next free step is B0 (C4b selectivity feasibility, §5); no long run until
  the user decides the default and signs off on the C4b front-end.

## 1. Read first (in order)

1. `AGENTS.md` — the goal, the principles, the pre-registered bars, the commands.
2. [`docs/superpowers/specs/2026-09-19-body-agnostic-fidelity-cyborg-design.md`](superpowers/specs/2026-09-19-body-agnostic-fidelity-cyborg-design.md)
   — **the governing spec**: principle, contracts C1–C4, sequencing table, risks.
3. [`docs/superpowers/specs/2026-09-20-ongoing-state-and-faithful-readout.md`](superpowers/specs/2026-09-20-ongoing-state-and-faithful-readout.md)
   — P4 (closed).
4. [`docs/superpowers/specs/2026-09-20-c4-escape-and-loom-selectivity-design.md`](superpowers/specs/2026-09-20-c4-escape-and-loom-selectivity-design.md)
   — **C4, the current work** (A0–A2 done, A3/C4b pending).
5. [`docs/overview/README.md`](overview/README.md) — architecture, math, roadmap.
6. The findings under `docs/results/{liveness,adapter,feedback,dynamics}/` (indexed in §8).

## 2. The contract ledger (from the governing spec)

Each addition to the frozen brain is a **declared, additive, versioned, silenceable** contract
change. Wiring, weights, signs and neuron parameters never change; no teacher enters the behaviour
path; one bridge, no mode switching.

| Change       | What                                             | Outcome                                                  |
| ------------ | ------------------------------------------------ | -------------------------------------------------------- |
| **C1** (P1)  | Ascending/proprioceptive body feedback           | landed; **weak positive** (moved `slow_fraction` only)   |
| **C2** (P2)  | Connectome-constrained per-neuron dynamics       | landed; **negative** (only `noise_sigma` varied)         |
| **C3** (P4)  | Ongoing state (tonic drive)                      | **C3a rejected** by its silencing gate; **C3b dropped**  |
| **C4** (now) | Use the brain's escape response + selective loom | **C4a A3 failed 2026-09-22 (rejected); C4b not started** |

P0 (declared adapter + teacher-free gate) and P3 (liveness bar; relay closed negative) are
unchanged/landed. `roam_eval.ACCEPTANCE` (A1–A7) and `liveness` (L1–L8) are pre-registered and are
**never relaxed without the user**.

## 3. State of the two bars (measured)

**Liveness** (is it alive?): on the canonical bundle, codec v2 × 15 seeds × **120 s** passes L1–L8
**except L4**; L7 is deferred. Artifact `docs/results/liveness/liveness-check-v2-120s.json`.

- **L7 deferred automatically** when `plant.LIMITS[3]` (0.8 rad/s) < `motion_stats.SACCADE_THRESHOLD`
  (3 rad/s): the quadrotor cannot saccade; it re-enables on a fly-like body.
- **L4 near-miss** (user decision: leave the bar as pre-registered): the blind (ghost) body is nearly
  as alive (coverage 0.177 vs 0.189) because locomotion is the brain's intrinsic tonic drive.

**Acceptance** (is it capable?): v2 × 15 seeds × 120 s fails **A1–A6** (A7 live). Artifact
`docs/results/adapter/adapter-check-v2-smoke.json`. Headline: A1 beacon rate **0.033/min** vs teacher
0.733; A2 collisions **6.67/min** vs ≤0.5; A3 dodge 0.636 / balanced 0.25 / ghost **0.441**; A4 loom
ratio 0.935 vs ≥2; A5 steer probe 0.0; A6 coverage 0.185 vs 0.4 (slow 0.028 passes).

**Acceptance, v3 A3 smoke (2026-09-22):** `adapter-check-v3-smoke.json` (15 seeds 1000–1014, same
seeds as v2). Near-threat peak `|vy|` **0.8** (v2 0.08–0.5); A3 dodge **0.341** (v2 0.636),
balanced 0.154 (v2 0.25), **ghost 0.205 — passes ≤0.3** (v2 0.441); A2 collisions 4.30/min (v2 6.67;
walls 178→90 but threat hits 12→29); A4 loom ratio 1.45 (v2 0.935); A6 coverage 0.257 (v2 0.185),
slow 0.018. Still `passed = false`. **Liveness v3 regressed:** L6 `mean_move_bout_s` 12.80 vs the
10.0 window (v2 9.14; paired CI [2.38, 4.83]), `bouts_per_min` 6.6 → 4.8; L1/L2/L3/L5/L8 pass, L4
fails as before, L7 deferred. Full analysis: `FINDING-2026-09-22-c4a-a3.md`.

**Why** (`FINDING-2026-09-20-loom-escape-audit.md`): the brain's `escape` readout fires ~0.9 on a
loom (like the teacher), but the v2 bridge commands a **climb** and a `0.5·yaw` sidestep
(near-threat peak `|vy|` 0.24–0.39 vs the teacher's 1.0) and has **no wall avoidance** (9–16
collisions/120 s, nearly all walls). The v4 loom scalar is also not threat-selective (turn sweeps
saturate it; documented E1 ≤ 0.686).

## 4. C4a: what is done, what is not

- **A0 (free, done):** `escape` is the mean of the two `DNp01` giant-fibre cells, discarding the
  side. Kept separate, `L−R` = **+0.257** (`loom_l`), **−0.176** (`loom_r`), **0** (`loom_both`),
  rest noise floor 0. → a sided escape signal exists. `scripts/escape_side_probe.py`; artifact
  `docs/results/adapter/escape-side-probe.json`; `FINDING-2026-09-20-c4a-a0.md`.
- **A1 (done):** codec **v3** = `declared-v3:467faae753cef735` (`docs/results/adapter/adapter-v3.json`).
  Reads `DNp01` L/R separately; `vy = −0.8·loom·sign(esc_L−esc_R)`; climb reduced to a declared 0.25
  fraction. Constants `ESCAPE_SIDE_GAIN=0.8`, `CLIMB_FRACTION=0.25` are **declared, not fitted**.
  Additive and silenceable; the canonical artifacts are byte-identical. (v3 is now the default by user direction.)
- **A2 (done, free falsifier):** `scripts/escape_command_audit.py` → `loom_l vy −0.800`, `loom_r
vy +0.800`, `loom_both vy 0.000`, calm 0. `passed = true`; artifact
  `docs/results/adapter/escape-command-audit.json`; `FINDING-2026-09-20-c4a-a1-a2.md`.
- **A3 (run 2026-09-22, FAILED):** v3 dodge 0.341 / balanced 0.154 / ghost 0.205; A4 1.45; A2
  4.30/min; liveness L6 regressed. **C4a alone rejected; 50-seed gate not spent.**
- **C4b (NOT started, now the prerequisite):** a selective loom front-end (the documented v6
  spatial route, or the P3 relay extended to LC4/LPLC2), gated by the E1/E2 selectivity tests,
  needed for **wall avoidance** and for A4 to be causal. A head-on wall is a symmetric loom, so C4a
  alone will not fix A2. Until the loom is selective, the v3 strafe fires constantly (non-selective
  cue) and lengthens move bouts (the L6 miss).

## 5. Exact next action

**First, decide the default (user):** v3 failed its gate. Recommended: set `adapter.DEFAULT_CODEC`
back to `"v2"` (the gate-passing _alive_ default) and keep v3 selectable as an instrument, until a
cell passes. Do not silently keep an ungated default — this is a user call (it reverses an earlier
user override).

**Then (free, no seeds), B0 of C4b:** run the documented E1/E2 selectivity gates on the current v4
front-end and on both candidates — the **v6 spatial (retinotopic) front-end** and the **P3 relay
extended to LC4/LPLC2** — and report which cells carry the selective signal. E1 bar ≥ 0.8
(`sensory-model.md` §6). No cell clears E1/E2 → stop C4b (loom stays non-selective).

**Only after B0:** a 15-seed smoke of the chosen C4b front-end (C4b alone and C4a+C4b), then a
v3-variant probe that **keeps the v2 climb** and adds the lateral escape rather than trading climb
for strafe (declared-constant change — needs sign-off). The 50-seed stage-3 gate is spent only on a
cell that first clears a free falsifier and a 15-seed smoke. **Ask the user before any long run.**

## 6. Invariants that must not break

- **Pre-registered bars:** `roam_eval.ACCEPTANCE` (A1–A7) and `liveness` (L1–L8) are scored, never
  relaxed, without the user.
- **Canonical artifacts are byte-identical:** `docs/results/adapter/adapter.json` (v1),
  `docs/results/adapter/adapter-check.json`; canonical `bundle_hash`
  `edc5439e291e65233e673b5baaed069f570f5003c245e414de2aa1e41b14aff5`. `data/malecns` is never edited.
- **Bridge rules:** reads neural activity only (no pose, target, task id or pixels); one formula; no
  mode switching; no teacher in the behaviour path.
- **Default bridge is codec v3 pending user decision** (`adapter.DEFAULT_CODEC = "v3"`, C4a; made
  default by earlier user direction, **now failed its A3 gate — recommended revert to v2**); v2/v1
  are opt-in. The viewer cycle is v3/v2/v1.
- **L7 stays deferred** on this body (auto rule in `liveness.liveness(yaw_limit=…)`).
- Legacy room MJCF and accepted actors remain reproducible (`test_legacy_room_mjcf_unchanged`).

## 7. Environment, commands, gotchas

```bash
# tests (ROS pollutes PYTHONPATH; strip it)
env -u PYTHONPATH .venv/bin/python -m pytest -q
# lint (note: scripts are linted too)
.venv/bin/ruff format python tests scripts && .venv/bin/ruff check python tests scripts
# web (web/dist is git-ignored; rebuild after TS changes; needed by `serve`)
yarn typecheck && yarn lint && yarn test && yarn build
```

- Keep parallel workers **≤ 6** (RAM).
- Codec switch: `serve --codec v3|v2|v1` (or `--adapter v3|v2|v1|<path>`); viewer free-roam bar
  **bridge → Codec v3/v2/v1** sends the server `codec` op and resets the sim; `roam-screen adapter
--codec v3 …`.
- A dev server may already be running on `127.0.0.1:8000` (`fly-drone serve`); if the port is busy,
  an older instance is holding it.
- **Test count at handoff: 391 passing.** New work should keep it green and add tests for any codec
  or liveness change (see `tests/test_adapter_v3.py` for the pattern).

## 8. Artifact index

**Specs:** `docs/superpowers/specs/2026-09-19-body-agnostic-fidelity-cyborg-design.md` (governing),
`…/2026-09-20-ongoing-state-and-faithful-readout.md` (P4, closed),
`…/2026-09-20-c4-escape-and-loom-selectivity-design.md` (C4, current),
`…/2026-09-16-retinotopic-sensing-v6-design.md` (C4b candidate).

**Findings (this arc):**
`docs/results/liveness/FINDING-2026-09-20-idling-cause.md`,
`…-c3-and-codec-v2.md`, `…-c3b-dropped.md`, `…-l7-body-threshold.md`,
`…-l2-coverage-and-l4-attribution.md`;
`docs/results/adapter/FINDING-2026-09-20-v2-acceptance.md`,
`…-loom-escape-audit.md`, `…-c4a-a0.md`, `…-c4a-a1-a2.md`, `…-2026-09-22-c4a-a3.md`.
(Earlier: `docs/results/adapter/FINDING-2026-09-19.md`, `docs/results/feedback/FINDING-2026-09-19.md`,
`docs/results/dynamics/FINDING-2026-09-19.md`.)

**Codec artifacts:** `docs/results/adapter/adapter.json` (v1 canonical),
`adapter-v2.json` (v2), `adapter-v3.json` (v3), plus `adapter-check*.json`,
`adapter-check-v2-smoke.json`, `adapter-check-v3-smoke.json`, `escape-side-probe.json`,
`escape-command-audit.json`. Liveness: `docs/results/liveness/liveness-check-v2-120s.json`,
`…-v3-smoke.json`.

**Free probes (scripts):** `scripts/command_audit.py`, `scripts/readout_audit.py`,
`scripts/escape_side_probe.py`, `scripts/escape_command_audit.py`, `scripts/yaw_audit.py`,
`scripts/make_tonic_bundle.py`.

**Code touchpoints:** `python/fly_drone/adapter.py` (codecs + identities),
`liveness.py` (L1–L8, L7 deferral), `motion_stats.py` (fly references), `roam_eval.py`
(ACCEPTANCE, adapter_check/liveness_check/tonic_check), `distill.py` (screen/rollout),
`brain.py` (readout_ids incl. `escape` = DNp01), `server.py` + `web/src` (viewer, codec switch),
`cli.py` (all commands, `--codec`/`--adapter`).

## 9. Open decisions and risks

- **v3 failed its A3 gate (2026-09-22)** but is still the default from an earlier user override.
  **Decide: revert `DEFAULT_CODEC` to v2 until a cell passes**, or explicitly keep v3 with the
  failure recorded. Recommended: revert.
- **C4b is required** for A2/A4 (wall avoidance) and is now the prerequisite for any lateral use to
  help; it is a front-end change with the documented E1/E2 selectivity gates and needs sign-off.
- **A v3 variant that keeps the v2 climb** and adds (rather than trades) the lateral escape is the
  other candidate lever, given the threat-hit rise (12 → 29) tracked the climb cut.
- **A1 (beacon seeking) is untouched by C4a/C4b.** Nothing in this arc makes the drone seek a beacon
  (A1 = 4.5 % of teacher); that is likely the next workstream after avoidance, and may need the
  documented v6 spatial sensing / closed-loop search.
- **L4 near-miss** and **L7 deferral** are recorded, not fixed.
- **Never** re-use the rejected C3b knob as a saccade/behaviour generator; **never** add a teacher
  or a hidden mode switch to make the numbers move.
