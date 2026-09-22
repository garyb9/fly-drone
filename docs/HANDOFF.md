# HANDOFF — where the project is, and where to continue

**Active program: fidelity-first roadmap, approved 2026-09-22.** Start with
[`the agent-ready roadmap`](superpowers/specs/2026-09-22-fidelity-first-roadmap-and-agent-handoff.md).
It supersedes the old next-action ordering below: S0 run management/evidence → bounded S1 A1
audit → S2 force-source audit → canonical-brain biomechanics → **whole-body articulation first**.
Do not wait for all drone skills to pass before body work. C4b remains parked during this sequence.

**Execution authorization:** autonomous approved local experiments, ≤2 hours per run including
resumes, ≤6 experiment-hours per stage, ≤6 workers total; one heavy run at a time. Report progress
and support mid-run stop. New biological contracts require a concrete design and sign-off.
Keep codec v3 as default. Preserve the dirty sibling fly-playground worktree; its articulation is
a partly assisted proxy, not verified biomechanics or canonical-brain parity.

**Current package:** S0 implementation. Baseline verification, managed experiments,
report-completeness checks and deterministic S1 audit are in progress. No long experiment is
running. Next: finish S0 validation, then execute registered S1 trials through the manager.
Do not run C4a's rejected 50-seed gate or start v6 retraining.

**Historical C4 state follows.** This is the operational companion
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
- **C4b B0 was run 2026-09-22 and is BLOCKED on E1.** Neither the v4 front-end (E1 **0.738**) nor
  the frozen v6 spatial clone (E1 **0.720**, motion 0.657, union 0.710) clears the **E1 ≥ 0.8**
  selectivity bar, and the relay route is structurally infeasible (T4/T5 do not reach LC4/LPLC2; a
  translation model cannot represent looming). Evidence:
  [`FINDING-2026-09-22-c4b-b0.md`](results/adapter/FINDING-2026-09-22-c4b-b0.md). So **wall
  avoidance (A2/A4) is not reachable by the current front-ends.**
- **Decisions (2026-09-22):**
  1. **Default codec: keep v3** (`DEFAULT_CODEC = "v3"`, user decision) with its A3 failure recorded;
     v2/v1 remain selectable. It is the _latest rig_, not a _capable_ one.
  2. **C4b fork: open** — (a) train the v6 encoder to clear E1 (long run), (b) design a new declared
     selective loom front-end, or (c) defer avoidance (A1/body work instead).
- **Nothing is running.** No long run until the user decides; B0 was free and is recorded.

## 1. Read first (in order)

1. `AGENTS.md` — the goal, the principles, the pre-registered bars, the commands.
2. [`docs/superpowers/specs/2026-09-19-body-agnostic-fidelity-cyborg-design.md`](superpowers/specs/2026-09-19-body-agnostic-fidelity-cyborg-design.md)
   — **the governing spec**: principle, contracts C1–C4, sequencing table, risks.
3. [`docs/superpowers/specs/2026-09-20-ongoing-state-and-faithful-readout.md`](superpowers/specs/2026-09-20-ongoing-state-and-faithful-readout.md)
   — P4 (closed).
4. [`docs/superpowers/specs/2026-09-20-c4-escape-and-loom-selectivity-design.md`](superpowers/specs/2026-09-20-c4-escape-and-loom-selectivity-design.md)
   — C4 historical work (A0–A2 done, A3 failed, C4b B0 blocked).
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
| **C4** | Use the brain's escape response + selective loom | **C4a A3 failed; C4b B0 ran and is blocked on E1** |

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
- **C4b (B0 run 2026-09-22, BLOCKED on E1):** a selective loom front-end is needed for **wall
  avoidance** and for A4 to be causal. B0 measured the candidates: v4 E1 **0.738**, frozen v6
  spatial clone E1 **0.720** (motion 0.657, union 0.710) — **neither clears E1 ≥ 0.8**; the relay
  route is structurally infeasible (its T4/T5 targets do not reach LC4/LPLC2, and a
  global-translation model cannot represent looming). `FINDING-2026-09-22-c4b-b0.md`.

## 5. Exact next action

**Default decided (2026-09-22): keep codec v3** as shipped, with the A3 failure recorded (user
call). v2/v1 stay selectable.

**B0 is DONE (2026-09-22): no adoptable front-end, C4b blocked on E1** (§0, §4). The next action is
a **user decision**, not a run:

- **(a) Train the v6 encoder to clear E1** (the paused Option; long run, needs sign-off), then re-run
  B0; only if E1 ≥ 0.8 does C4b B1/B2 proceed.
- **(b) Design a new declared selective loom front-end** (spec first; must clear E1).
- **(c) Defer avoidance**, record C4b as blocked-on-E1, and move to A1 beacon seeking or body
  fidelity.

The **default codec decision is closed: keep v3**. No 50-seed gate is spent until a cell
clears a free falsifier and a 15-seed smoke. **Ask the user before any long run.**

## 6. Invariants that must not break

- **Pre-registered bars:** `roam_eval.ACCEPTANCE` (A1–A7) and `liveness` (L1–L8) are scored, never
  relaxed, without the user.
- **Canonical artifacts are byte-identical:** `docs/results/adapter/adapter.json` (v1),
  `docs/results/adapter/adapter-check.json`; canonical `bundle_hash`
  `edc5439e291e65233e673b5baaed069f570f5003c245e414de2aa1e41b14aff5`. `data/malecns` is never edited.
- **Bridge rules:** reads neural activity only (no pose, target, task id or pixels); one formula; no
  mode switching; no teacher in the behaviour path.
- **Default bridge is codec v3** (`adapter.DEFAULT_CODEC = "v3"`, C4a; made default by user
  direction and **kept after its A3 failure by user decision 2026-09-22**); v2/v1 are opt-in. The
  viewer cycle is v3/v2/v1.
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
`…-loom-escape-audit.md`, `…-c4a-a0.md`, `…-c4a-a1-a2.md`, `…-2026-09-22-c4a-a3.md`,
`…-2026-09-22-c4b-b0.md`.
(Earlier: `docs/results/adapter/FINDING-2026-09-19.md`, `docs/results/feedback/FINDING-2026-09-19.md`,
`docs/results/dynamics/FINDING-2026-09-19.md`.)

**Codec artifacts:** `docs/results/adapter/adapter.json` (v1 canonical),
`adapter-v2.json` (v2), `adapter-v3.json` (v3), plus `adapter-check*.json`,
`adapter-check-v2-smoke.json`, `adapter-check-v3-smoke.json`, `escape-side-probe.json`,
`escape-command-audit.json`, `b0-v4-e1.json`, `b0-v6-e1.json`. Liveness:
`docs/results/liveness/liveness-check-v2-120s.json`, `…-v3-smoke.json`.

**Free probes (scripts):** `scripts/command_audit.py`, `scripts/readout_audit.py`,
`scripts/escape_side_probe.py`, `scripts/escape_command_audit.py`, `scripts/yaw_audit.py`,
`scripts/make_tonic_bundle.py`.

**Code touchpoints:** `python/fly_drone/adapter.py` (codecs + identities),
`liveness.py` (L1–L8, L7 deferral), `motion_stats.py` (fly references), `roam_eval.py`
(ACCEPTANCE, adapter_check/liveness_check/tonic_check), `distill.py` (screen/rollout),
`brain.py` (readout_ids incl. `escape` = DNp01), `server.py` + `web/src` (viewer, codec switch),
`cli.py` (all commands, `--codec`/`--adapter`).

## 9. Open decisions and risks

- **v3 failed its A3 gate (2026-09-22)** but **stays the default** by user decision (2026-09-22);
  it is the latest rig, not a capable one. The failure is recorded, not hidden.
- **C4b is BLOCKED on E1** (B0, 2026-09-22): no front-end clears 0.8, so A2/A4 (wall avoidance) are
  not reachable by the current front-ends. Needs a user decision (train v6 / new front-end / defer).
- **A v3 variant that keeps the v2 climb** and adds (rather than trades) the lateral escape is the
  other candidate lever, given the threat-hit rise (12 → 29) tracked the climb cut — but it is only
  worth testing once a selective loom exists, since the non-selective cue fires the strafe always.
- **A1 (beacon seeking) is untouched by C4a/C4b.** Nothing in this arc makes the drone seek a beacon
  (A1 = 4.5 % of teacher); that is likely the next workstream after avoidance, and may need the
  documented v6 spatial sensing / closed-loop search.
- **L4 near-miss** and **L7 deferral** are recorded, not fixed.
- **Never** re-use the rejected C3b knob as a saccade/behaviour generator; **never** add a teacher
  or a hidden mode switch to make the numbers move.
