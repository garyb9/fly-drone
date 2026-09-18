# Frozen-v6 free roam (Option 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Status:** approved by the user 2026-09-18 (Option 3 from
[`HANDOFF-2026-09-18.md`](../../results/encoder-v6/HANDOFF-2026-09-18.md) §3). No encoder training.

**Goal:** ship a free-roam actor whose **only** learner is the decoder, on the frozen v6 pair
(encoder `learned-v6:01280e414169ff9c` + round-0 decoder), and prove causal skill with A1–A7 plus
the harvested controls (rewired graph, eye alignment, stimulus battery, fly.ai oracle).

**Why now.** Three encoder-training routes are clean negative results (alternating ×5, joint
single-α, joint per-head-α); `runs/v6/round0` passes the foraging gate; the E1 bar (0.8) is above
both v4 (0.732) and round 0 (0.720); and the goal is behaviour and causal attribution, not E1. The
external survey shows every strong project puts learning in the readout, not the visual front end.

**Spec:** [`2026-09-16-retinotopic-sensing-v6-design.md`](../specs/2026-09-16-retinotopic-sensing-v6-design.md)
§5–§8. **Mechanics:** plan 04 [`2026-09-14-fly-drone-04-free-roam-handoff.md`](2026-09-14-fly-drone-04-free-roam-handoff.md)
(now v4-era; re-based here on v6). **Diagnostics:** [`external-prior-art.md`](../../external-prior-art.md)
and the harvest spec.

## Scope exception (declare, do not hide)

The goal permits a **learned** encoder for free roam; it does not require one. Option 3 freezes the
visual front end at the proven neutral clone and declares **"encoder learning is paused, three
routes negative"**. The brain/decoder contract is unchanged: the decoder still reads only the 2,022
DN + VNC traces, and no pose/target/pixels reach it. Revisit Option 2 (decoupled critic) only as a
time-boxed run if A3 fails specifically on loom (see §Open decisions).

## Global constraints

- Connectome frozen and non-differentiable. Encoder frozen: `SpatialEncoder` weights, `learner-v6:`
  identity and the patch map never change in this plan.
- Decoder input is the 2,022 DN + VNC traces only. Simulator state is for labels/rewards/metrics.
- One decoder for free roam; no mode switch. `roam_eval.ACCEPTANCE` A1–A7 and E1/E2 bars never move.
- Legacy v4/v5 and every accepted actor reproduce bit-for-bit; `test_legacy_room_mjcf_unchanged`.
- Workers ≤ 6; `env -u PYTHONPATH .venv/bin/python …`; ask the user before every long run.
- No `fly_drone/*.py` edits while a `sac-round`/`roam-*` process is live. Commit and push per task.

## Design

```
eyes (6x48x64) ──> FROZEN encoder (learned-v6:01280e414169ff9c) ──> 816 patch currents
                                                                        │
                                              FROZEN CONNECTOME ────────┘
                                                        │
                                              2022 DN + VNC traces
                                                        │
                                   LEARNED decoder (DAgger clone -> SAC) ──> [vx,vy,vz,yaw] ──> PID
```

- **What moves:** the decoder only, via the existing SB3 SAC decoder path
  (`sac-round --learner decoder`), warm-started from the round-0 decoder and DAgger recovery.
- **What does not:** encoder, patch map, connectome, PID.
- **Why SAC, not PPO:** the v5/v6 pipeline already trains the decoder under SAC and exports the pair
  via `sac-export`; the legacy PPO `train` path still exports legacy `LIMITS` and needs fixing
  (plan 04 blocker note), so it is out of scope here.
- **E3 (bypass) is retained** as a hard anti-wire gate: a decoder that reads the 816 currents
  directly must not beat the full pair.

## Gates

| Gate | What | Source |
| --- | --- | --- |
| G0 realisability | held-out features predict each drive label with R² ≥ 0.5 | `roam-fit` per-drive R² |
| G1 round guard | beacons ≥ ½ round 0, ghost near-dodge ≤ 0.3, E2 pass | `roam_eval.round_eligible` |
| G2 anti-wire | a bypass decoder reading the 816 currents is **not** better than the full pair | `roam_eval.bypass_comparison` |
| G3 acceptance | A1–A7 on 50 held-out seeds | `evaluate --task free_roam` |
| G4 causal (harvest) | rewired-graph, eye-alignment, battery, oracle diagnostics reported alongside | harvest scripts |

E1/E2 are frozen with the encoder and are **reported, not gated** (round 0: E1 loom 0.720, E2 light
0.488 / loom 0.208). If G3 fails only on A3 (threat dodge), see the Option-2 trigger below.

## File structure

| File | Responsibility | Change |
| --- | --- | --- |
| `docs/results/encoder-v6/` | frozen-pair record, DAgger/round tables, final E1–E4 + A1–A7 | create/append |
| `docs/results/accepted-policies.json` | the accepted `free_roam` actor | modify at Task 7 |
| `python/fly_drone/evaluate`/`roam_*` plumbing | only if a smoke exposes a v6-pair gap | modify if needed (Task 0) |

No new module is expected; all commands already take `--encoder`.

## Tasks

### Task 0: Pin the frozen pair and smoke every command path
- [x] Pair recorded: encoder `learned-v6:01280e414169ff9c` (`runs/v6/clone/encoder.pt`) + decoder
      `runs/v6/round0/decoder.json` (`dataset_hash 60cb1821…`). Record:
      [`FROZEN-PAIR-2026-09-18.md`](../../results/encoder-v6/FROZEN-PAIR-2026-09-18.md).
- [x] Smoked `roam-collect/fit/screen`, `sac-validate`, `evaluate`, `sac-round bypass`,
      `sac-init-decoder → sac-round decoder`, `sac-export` against the pair
      (`runs/roam/v6-smoke/`).
- [x] Fixed what the smoke broke: `roam-fit --encoder` (was v4-only) and
      `training.export_actor` encoder identity (was hard-coded v4); both v4 paths unchanged.
- [x] Corrected Task 4's warm start to `sac-init-decoder` (the DAgger `warm-ppo.zip` space does not
      match a v6 decoder round).

### Task 1: Realisability probe (G0)
- Collect a held-out teacher set under the frozen clone: `roam-collect --output
  runs/roam/v6-realise.npz --flights 64 --seconds 60 --encoder runs/v6/clone/encoder.pt --levels 0 1 2`.
- `roam-fit runs/roam/v6-realise.npz --encoder runs/v6/clone/encoder.pt --output runs/roam/v6-realise-fit`;
  read held-out **per-drive R²** (features), and a cues-only control if `roam-fit` exposes one.
- Gate: each drive R² ≥ 0.5. If a drive is below, record it and let the user decide before DAgger.

### Task 2: E3 bypass control on the frozen pair (G2)
- Train the bypass decoder that reads the flat 816 currents
  (`sac-round --learner bypass --encoder runs/v6/clone/encoder.pt --frames <small>`), then run
  `roam_eval.bypass_comparison` against the round-0 pair.
- Gate: the bypass is **not** better. If it is, the brain is decorative on the v6 currents; stop and
  report rather than train further.

### Task 3: DAgger iterations 0–3
- Iter 0: `roam-collect --output runs/roam/v6-d0.npz --flights 128 --seconds 60 --encoder
  runs/v6/clone/encoder.pt --levels 0 1 2`; `roam-fit runs/roam/v6-d0.npz --encoder … --output
  runs/roam/v6-it0`; screen `roam-screen runs/roam/v6-it0/warm-actor.json teacher random --encoder …`.
- Iters 1–3: `roam-collect --student runs/roam/v6-itK/warm-actor.json --beta 0.5/0.25/0 --encoder …
  --seed-base <new>`; refit on all npz; screen each. Include recovery-focused data (threat frames).
- Gate each: screens improve or hold; E1/E2 stay at the frozen baseline.

### Task 4: Decoder SAC fine-tune (gated, one round at a time)
- Warm-start in the SAC spaces (a DAgger `warm-ppo.zip` cannot load: its observation is
  `Box(2022)`, a v6 decoder round's is `Dict(dn, geometry)`):
  `sac-init-decoder <it3-npz> --encoder runs/v6/clone/encoder.pt --output runs/roam/v6-it3/init`,
  then `sac-round decoder --encoder runs/v6/clone/encoder.pt --init runs/roam/v6-it3/init/decoder.zip
  --frames <agreed> --workers 6`.
- Add the **decoder behaviour-cloning anchor** first (Phase 4): `sac.anchor_penalty` is currently
  encoder-only, so a decoder round can still drift. Anchor to the DAgger actor with a small fixed α.
- `sac-validate` + `roam_eval.round_eligible`; accept the round only on G1 and G2, else revert to
  the DAgger actor and record. Budget/kill rule: see Open decisions — ask before starting.

### Task 5: Full evaluation (G3)
- `evaluate --task free_roam --policy <decoder> --encoder runs/v6/clone/encoder.pt --episodes 50
  --workers 6 --output runs/roam/v6-evaluation.json` (A1–A7 + teacher/cue/random + A5 probes).
- Record E1/E2 as frozen; do not relax `ACCEPTANCE`.

### Task 6: Accept the actor and run the harvest controls (G4)
- Add the accepted `free_roam` actor to `docs/results/accepted-policies.json`.
- Run `scripts/rewired_report.py`, `scripts/eye_alignment.py`, `scripts/stimulus_battery.py` and
  `scripts/check_fly_ai_oracle.py`; record outcomes next to the acceptance.

### Task 7: Report
- Write `docs/results/encoder-v6/FREE-ROAM-<date>.md`: G0–G4, all E1/E2 numbers, A1–A7, the
  harvest diagnostics, and the honest verdict. Update `docs/validation.md` and `docs/free-roam.md`.

## Open decisions (need the user)

1. **DAgger budget/threats.** Start at levels 0–2 (no threats) as in plan 04, or include level 3
   from iteration 0? Recommend 0–2 for iterations 0–1, then level 3 for 2–3.
2. **SAC decoder budget and kill rule.** Frames per round (recommend one 150k round first) and the
   exact revert condition (G1 or G2 fail).
3. **Option-2 trigger.** If G3 fails **only** on A3 (threat dodge) and loom is the cause, do we
   time-box one decoupled-critic encoder run, or ship free roam v1 without thrown threats?
4. **E1 framing.** Confirm E1/E2 are reported-only under the frozen encoder (recommended), so the
   0.8 bar is not silently lowered.

## Risks

| Risk | Mitigation |
| --- | --- |
| Frozen encoder's neutral Tm4/T2 leaves threat dodge loom-poor | Teacher gate already passed; A3 is the test; Option-2 trigger above |
| Decoder SAC collapses as in the encoder rounds | Decoder-only is the v5-proven path; per-round guard + revert; small round |
| Bypass decoder matches the pair (brain decorative) | Hard G2 before long training |
| Legacy PPO path confusion | Use the SAC decoder path only; PPO stays out of scope |
| Scope creep into encoder training | This plan trains the decoder only; encoder work needs a new plan |
