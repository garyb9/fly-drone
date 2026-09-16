# Encoder v5 recovery status (2026-09-16, end of session): Phase 0 done, Phase 1 partly landed

Written for a fresh session. Read [`PHASE0-DIAGNOSTICS-2026-09-16.md`](PHASE0-DIAGNOSTICS-2026-09-16.md)
first (the diagnosis); this file says what has been changed and what is left. The previous run's
handoff (`HANDOFF-2026-09-16.md`) is superseded.

## 1. What this session did (all committed and pushed)

Commits, oldest first:

| Commit    | What |
| --------- | ---- |
| `b1945ab` | Phase 0 diagnostics + spec decisions (report, ledger, `docs/superpowers/specs/*`) |
| `6cb5173` | **Phase 1a** round selection guard (`roam_eval.round_eligible` / `pick_best_round` / `round_gate_report`), pipeline rewired, plan Task 13 Steps 5–6 amended |
| `b471d06` | **Phase 1b** post-warm-up entropy regime (`auto_0.01`, matched target entropy, `ClampedActor` log_std ∈ [−4, −1]) |
| `0f5deee` | **Phase 1c** `ProbeLogger` (log_std, action level, action-state drift) |
| `a7d9475` | **Phase 1d** Pack 1: fixed-probe E1/E2, per-axis clone error, per-evade peaks, action-axis logging, worker clamp |

Full test suite is green (162 passed) and `ruff check python tests` is clean as of `a7d9475`.
`roam_eval.ACCEPTANCE` and every pre-registered bar are untouched. No training was run.

## 2. The diagnosis in one paragraph

The round-1 **encoder** broke the pair on its own: in the first ~13 k frames after its actor
unfroze at SB3's default α = 1.0, its deterministic mean output collapsed to a **constant 1.0
current on all 8 channels**, so the deployed encoder is blind (D1: 0.00 beacons/min with the good
round-0 decoder; D4: 0.337 → 0.977 → 1.000 across 50 k/100 k/350 k; run logs: `loom_cost` flat at
0.010 = λ·1.0 from 63 k on). The decoder round was degenerate from frame 0 under that encoder
(D3 50 k ≡ D1), then its own SAC flailed (ghost 0.087 → 0.875). Details and all numbers:
`PHASE0-DIAGNOSTICS-2026-09-16.md`.

## 3. What is landed (Phase 1a–d)

- **Guard** (`roam_eval.py`): eligible only if beacons ≥ ½ round 0, ghost ≤ 0.3, E2 passes; falls
  back to round 0. Verified against the recorded round 0/1/2 validations: rounds 1 and 2 are
  ineligible, pick = 0. `pipeline13-15.sh` uses it.
- **Entropy** (`sac.py`): `ent_coef="auto_0.01"`; `target_entropy = dim·(0.5·ln(2πe) − 2.5)`;
  `ClampedActor` clamps `log_std` to [−4, −1]; `WARM_START_LOG_STD = −2.5`.
- **Logging** (`sac.py`): `ProbeLogger` records `probe/log_std_mean`, `probe/action_mean`,
  `probe/action_state_std`; `ActionLogger` records `rollout/action_vy_absmean`,
  `rollout/action_vz_absmean`.
- **Pack 1**: `encoder_checks(controller="teacher")` default (fixed probe; `E1_policy`/`E2_policy`
  kept), CLI `--controller`; per-axis clone MSE (`{split}_{vx,vy,vz,yaw}_mse`); `peak_vy`/`peak_vz`
  on each threat outcome; worker defaults 16 → 6.

## 4. Phase 1e (Pack 2) — done

All four packed changes landed as opt-in flags whose defaults reproduce the pre-Pack-2 path
(pinned by `test_the_default_mode_reproduces_a_plain_dict_buffer`):

| ID | What | Flag / default |
| -- | ---- | -------------- |
| S4 | Critic-only warm-up skips the actor forward/backward, not just its optimiser step (`WarmupSAC._train_critic_only`) | always on during warm-up |
| S3 | n-step returns in the Bellman target only, stopping at done or timeout | `--n-step` (default 1) |
| S1 | Replay buffer stores obs once and derives `next_obs`, halving RAM | `--optimize-memory` (off) |
| S2 | One matched model+buffer snapshot every `resume_every` steps; `--resume` continues it, restoring `num_timesteps` so the warm-up is not repeated, and runs the remainder of the round | `--resume` (off) |

S1/S3 live in `python/fly_drone/replay.py` (`FlyDictReplayBuffer`). Disk: the S2 snapshot is one
buffer copy (~5.3 GB at 100k) and is deleted when the round completes on its own.

## 5. What is left before the re-run

1. **v4 fixed-probe E1 reference**: run `fly-drone encoder-checks --policy <v4 actor>` (teacher
   controller) once and store the number (`runs/v5/e1-v4-fixed.json`, back it up under
   `docs/results/encoder-v5/run-records/`). The pre-2026-09-16 v4 baseline was 0.687 policy-flown;
   the fixed-probe number will differ and must be reported alongside, not used to relax the 0.8 bar.
2. **Phase 1f**: full `pytest` + ruff are already green after Pack 2 (170 passed); re-run once
   more if anything above changes.
3. **Phase 1.5**: re-plan Task 13 in `pipeline13-15.sh` — shorter rounds with a mid-round
   checkpoint validation (tooling: `sac-export --learner … <ckpt.zip>` → repin/validate) and an
   explicit abort on degeneration; lengths informed by the D2/D3 timing (collapse was visible by
   the 100 k checkpoint).
4. **Phase 2**: re-run Task 13 from round 0 — **needs fresh user approval**. It will differ from
   rounds 1–2 collectively (α init, target entropy, log_std clamp, 200 k buffer, guard), so gains
   will be attributed to the bundle, not single fixes.

## 6. Rules that still bind

`env -u PYTHONPATH` on every python/fly-drone call; ≤ 6 workers; long runs via `setsid nohup`;
no `.py` edits while any `sac-round`/`roam-*` chain is active; stage by explicit path (never
`git add -A`); pre-registered thresholds are never relaxed without the user; commit and push after
each completed step (no agent trailer, per user 2026-09-16).

## 7. Artifact map for this recovery

- `docs/results/encoder-v5/PHASE0-DIAGNOSTICS-2026-09-16.md` — the diagnosis (read first)
- `docs/results/encoder-v5/RECOVERY-STATUS-2026-09-16.md` — this file
- `docs/results/encoder-v5/ledger.md` — running record; `.superpowers/sdd/…/progress.md` is the live copy
- `docs/results/encoder-v5/diag/current_drift.py`, `run_d2_d3.sh`, `d4-current-drift.log` — Phase 0 tools
- `runs/v5/diag/{d1,d2,d3,d2-100k,d2-150k,d3-50k,d3-100k}/` — Phase 0 validations (git-ignored)
- `runs/v5/round0/` — still the best valid pair (clone encoder + DAgger-it1 decoder, 2.13 bpm)
