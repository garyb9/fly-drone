# Training improvements for encoder v5 and the decoder — design

**Status:** Packs 1–2 **landed** in Phase 1 (2026-09-16, commits `b471d06`–`4f199ee`); Pack 3
deferred; Pack 4 (encoder representation) **superseded** by
[`2026-09-16-retinotopic-sensing-v6-design.md`](2026-09-16-retinotopic-sensing-v6-design.md), which
subsumes it as motion input, domain randomisation and channel differentiation. Documentation only —
no code.
Read [`2026-09-14-learned-encoder-sac-design.md`](2026-09-14-learned-encoder-sac-design.md) and
[`2026-09-15-vy-vz-training-signal.md`](2026-09-15-vy-vz-training-signal.md) first; this doc
absorbs the latter's M1–M3 instrumentation as a prerequisite and does not reopen either.

This spec collects, with file references, the improvements that look worth making to the way the
fly's encoder and decoder are trained. It is deliberately **not** an implementation plan: each
pack below needs the user's decision on scope first (see §6). It does not touch
`roam_eval.ACCEPTANCE`, the v4 encoder, the frozen connectome, or any pre-registered threshold.

**Landing hazard.** Tasks 13–15 are running under another agent
(`.superpowers/sdd/2026-09-14-fly-drone-05-encoder-v5-sac/pipeline13-15.sh`,
`sac-round ...`). Every `sac-round`/`roam-*` subprocess re-imports `python/fly_drone/*.py` at
spawn, so editing source under a live run can change the code a worker loads mid-round. Writing
this document is safe. **No `.py` edit lands until the pipeline stops or the user names a
window** (same rule as `2026-09-15-vy-vz-training-signal.md` §8).

## 1. Scope

| In scope                                                              | Out of scope                                            |
| --------------------------------------------------------------------- | ------------------------------------------------------- |
| Evaluation rigor for E1–E2 and the round stop rule                    | Editing `roam_eval.ACCEPTANCE`                          |
| Replay-buffer memory, crash-resume, sample reuse                       | Reopening v4 or the connectome                          |
| Keeping the decoder from forgetting the round-0 skill                  | Re-lowering the clone-fidelity work (declared exhausted) |
| Encoder input/representation and domain randomisation                  | Joint encoder+decoder training (still forbidden)        |

The four packs are independent enough to land one at a time; Pack 4 is the only one that changes
the learned encoder version and therefore invalidates all current `runs/v5/round0*` data.

## 2. Findings

### 2.1 E1/E2 measure the policy, not only the encoder

`roam_eval.encoder_checks` (`roam_eval.py:483`) flies each seed by asking the **policy** for
actions — `_checks_job` calls `brain.infer(obs)` (`roam_eval.py:473`) — and then scores the
currents the encoder applied. The set of frames that are threat-positive / beacon-positive, and
the number of positives, therefore depend on how that round's policy happened to fly. Comparing
E1 across rounds 1–3 mixes encoder semantics with behaviour, and the sample (`THREAT_POSITIVE_RANGE`,
`roam_eval.py:363`) can shrink to a handful of positives when a policy avoids threats well.

The pending vy-vz spec already flags that `dodge_rates` (`roam_eval.py:47`) keys on threat side,
not action axis, and that only M1–M3 would make the gap visible. E1/E2 have the same class of
problem one level up.

### 2.2 The E1 bar may be unattainable as written

`runs/v5/e1-v4-baseline.json` records v4 loom AUC **0.687** against the pre-registered
`E1_min_loom_auc = 0.8` (`roam_eval.py:365`); round-0 v5 is at 0.686. E1 asks SAC for a
selectivity the hand-built v4 never had, even before the encoder is at fault. This is reported
here as a risk to surface, **not** a proposal to relax a threshold — that needs the user.

### 2.3 Replay memory is the binding constraint on sample efficiency

The design and `docs/training.md` §9 note that SB3's `DictReplayBuffer` does not support
`optimize_memory_usage`, so a 100 k buffer stores `obs` **and** `next_obs` (≈ 5.3 GB). The encoder
observation is a 3-frame luma stack (`encoder.py`/`brain.FrameStack`), so consecutive transitions
share two of three frames; storing `next_obs` separately duplicates nearly all of it. The
100 k ceiling is what stops the buffer from covering the sparse-throw tail better.

### 2.4 A crash wastes a whole round

`train_round` (`sac.py:595`) always builds a fresh model/buffer and calls
`model.learn(..., reset_num_timesteps=True)` (`sac.py:683`). A crash mid-round cannot resume the
buffer or the step count; `docs/training.md` §9 tells the operator to pass the remaining frames
and that the warm-up (`ACTOR_WARMUP_FRAMES`, `sac.py:41`) repeats. For an ~11–12 h chain this is a
real exposure to lost compute.

### 2.5 Decoder rounds have no anchor to the cloned skill

Decoder rounds run SAC from the round-0 decoder at `learning_rate = 3e-4` (`sac.py:276`) for
150 k frames, with no KL/BC term against the previous decoder. The run-decisions log shows how
fragile that cloned skill is: the whole clone-fidelity effort moved foraging by ~0.1 beacons/min,
while simply flying the decoder under its own encoder moved it 1.0 → 2.13. Nothing in the decoder
round protects the 2.13 starting point while it learns to dodge.

### 2.6 The clone collapses the 8 channels onto 4

`brain.V4_TO_V5 = (0, 1, 0, 1, 2, 3, 2, 3)` (`brain.py:30`) means the clone targets
(`encoder.v4_targets`) make `mi1 == tm3` and `lc4 == lplc2` exactly. The 8-channel design's whole
point is distinctions the connectome already draws (spec §3.1); at round-0 they are absent, and
the metabolic cost (`λ_loom` averaged over LC4+LPLC2, `sac.py:32,156`) gives SAC no gradient to
separate them. E1 then asks SAC to discover the LC4/LPLC2 selectivity from task reward alone.

### 2.7 Smaller items

- **Worker defaults.** `evaluate_free_roam` (`roam_eval.py:319`) and `distill.screen`
  (`distill.py:411`) default to 16 workers while `AGENTS.md` caps at 6; every worker carries a
  brain + renderer. The documented commands pass 6, but the default is a RAM foot-gun.
- **Warm-up computes actor losses it discards.** `WarmupSAC.train` (`sac.py:255`) still runs the
  full `super().train()` and only skips the optimizer step, so the actor loss/gradients are
  computed for the first 50 k frames (parked minor, pure compute).
- **Band randomisation is the only domain randomisation.** `SacRoamEnv._randomise_bands`
  (`sac.py:138`) re-tints wall bands per episode; brightness/contrast remain fixed, so the encoder
  can still key on absolute luma.
- **Loom clone targets are spiky** and fit with plain MSE (`encoder.clone_loss`, `encoder.py:236`).

## 3. Pack 1 — Evaluation rigor

Reuses existing machinery; no threshold or v4 change.

| ID | What | Where | Why |
| -- | ---- | ----- | --- |
| R1 | Fixed-probe E1/E2: a `controller` field on the checks job. Default `teacher` flies `teacher_action` (geometry-driven, never reads neurons) while the loaded encoder's currents still drive the brain, so trajectories and threat/beacon labels are identical across encoders and only the currents differ. Keep the current policy-driven mode as a secondary number. | `roam_eval._checks_job` (`roam_eval.py:447`), `encoder_checks` (`:483`) | Removes the behaviour confound in §2.1; makes E1 comparable across rounds |
| R2 | M1: per-dimension clone error (index the `err` columns before `.mean(1)`), so `threat_vy_mse`/`threat_vz_mse` exist | `sac.warm_start_decoder` (`sac.py:439-457`) | vy-vz spec M1 |
| R3 | M2: peak `\|vy\|`/`\|vz\|` commanded per evade, on the `outcome` dict | `env._finish_threat` (`env.py:335`) | vy-vz spec M2 |
| R4 | M3: `rollout/action_vy_absmean`, `rollout/action_vz_absmean` | callback beside `MetabolicLogger` (`sac.py:572`) | vy-vz spec M3 |
| R5 | Round summary guard: report `beacons_per_min` and E1 alongside `near_dodge_rate`; keep the stop bar on near-dodge only | `sac.validate` (`sac.py:715`) | A dodge gain must not silently trade away foraging |
| R6 | Clamp `workers` to ≤ 6 in the two defaults | `roam_eval.py:319`, `distill.py:411` | §2.7 |

Tests: teacher-probe determinism (two encoders → identical labels, differing currents); a stored
v4 fixed-probe E1 reference; M1–M3 fields present in a short run.

## 4. Pack 2 — Sample efficiency and robustness

| ID | What | Where | Why |
| -- | ---- | ----- | --- |
| S1 | `DictReplayBuffer` variant that stores only `obs` and derives `next_obs` by index, respecting episode boundaries. Same RAM → 2× buffer. | `sac.build_sac` (`sac.py:276`), new buffer class | §2.3; directly increases reuse of rare threat transitions |
| S2 | Crash-resume: `--resume-buffer` + `--resume-steps`; `train_round` loads the buffer, restores `num_timesteps`, and calls `learn(reset_num_timesteps=False)`. Allowed only for the same learner + same frozen partner; alternating rounds keep fresh buffers. | `sac.train_round` (`sac.py:595`), `cli.py` | §2.4; protects the multi-hour chain |
| S3 | Optional n-step targets (`--n-step`, default 1) in the Bellman target only. | `sac.WarmupSAC`/`build_sac` | Sparse −20 collision signal propagates faster |
| S4 | Skip the actor loss during warm-up. | `sac.WarmupSAC.train` (`sac.py:255`) | §2.7 compute |

PER is named but **not** in this pack's first cut: it changes sampling semantics enough to deserve
its own measurement. Tests: memory-footprint assertion, save/load round-trip equivalence, resume
skips warm-up, `n-step=1` leaves training numerically unchanged.

## 5. Pack 3 — Decoder stabilization

| ID | What | Where | Why |
| -- | ---- | ----- | --- |
| D1 | Optional KL/BC anchor to a reference decoder actor (`--bc-anchor W`, default 0 = bit-identical). Reference = the previous round's decoder, loaded from `--init`. | `sac.WarmupSAC`/`build_sac` (`sac.py:236,276`) | §2.5; protects the cloned foraging while dodging is learned |
| D2 | Phase 2, gated on D1's measurement: blend visibility-gated teacher labels on rare in-range threat frames into decoder updates (a side buffer of labels, sampled aligned). | `sac.SacRoamEnv`/`train_round` | Attacks the vy/vz exposure gap at its source instead of via reward shaping |

Tests: `W=0` no-op (parameters bit-identical); `W>0` pulls a toy actor toward the reference.
No reward/`ACCEPTANCE` change is part of this pack. An optional threat-clearance shaping term is
named as a possible follow-up but deliberately **not** specified here.

## 6. Pack 4 — Encoder representation (needs sign-off, changes the version)

Every item here changes `EyesExtractor`/`LearnedEncoder` weights or behaviour and therefore the
`learned-v5:` version hash (`encoder.weights_hash`, `encoder.py:93`). Landing it invalidates
`runs/v5/round0*` and requires re-collecting round-0 data.

| ID | What | Where | Why |
| -- | ---- | ----- | --- |
| E1 | Frame-difference input channels (current − previous, or a small temporal conv), so motion/loom is directly representable. v4's cue is literally `ΔD`. | `encoder.EyeNet` (`encoder.py:56`) | Sample efficiency and loom selectivity |
| E2 | Image domain randomisation (brightness/contrast/noise) behind an optional `frame_transform` hook, default `None` so v4 hashes and legacy tests are untouched. | `env.ConnectomeEnv.push_frame` (`env.py:376`), `sac._randomise_bands` | §2.7; less keying on absolute luma |
| E3 | Visible-obstacle critic geometry (nearest visible pillar/wall), training-only. | `sac.visible_geometry` (`sac.py:44`), `CriticExtractor` (`sac.py:187`) | Better avoidance credit assignment; actor still never sees it |
| E4 | Channel differentiation. Lower-risk option: an auxiliary self-supervised term that shapes the 8 channels distinctly, rather than re-deriving hand-built per-type targets. | `encoder.clone_loss`/`fit_clone` | §2.6; the clone starts `mi1==tm3`, `lc4==lplc2` |

E4 is a design decision (auxiliary objective vs. synthetic per-type targets vs. accept and let SAC
learn it). The other three are mechanics.

## 7. Relationship to the vy-vz spec

- M1–M3 land here as Pack 1 R2–R4, exactly the instrumentation that spec says must exist before
  its strategy options (a)/(b)/(c) are chosen. This doc does not decide (b) or (c).
- Pack 3's D1/D2 are additional options that spec does not mention; if the user prefers, they can
  be sequenced after the vy-vz measurement rather than before it.
- Task 13's stop rule stays keyed on `near_dodge_rate` (vy-vz spec §3); R5 only adds reported
  columns.

## 8. Sequencing and safety

1. **Now:** this document only. No `.py` edits while `pipeline13-15.sh`/`sac-round` run.
2. **After the pipeline stops:** Pack 1 (read-only reporting + worker clamp), then Pack 2
   (defaults preserve semantics). Both leave v4, `ACCEPTANCE`, and the connectome untouched.
3. Verify: `env -u PYTHONPATH .venv/bin/python -m pytest -q`, `ruff format`/`ruff check`,
   `tests/test_arena.py::test_legacy_room_mjcf_unchanged`, the v4 replay golden hashes.
4. **Pack 3** default-off, measured before Pack 2's S3/S4 if the user wants the vy-vz data first.
5. **Pack 4 only on explicit go-ahead**, followed by a fresh clone/round-0 collection; no result
   from the current pipeline is comparable afterwards.
6. Commit and push after each completed step (`AGENTS.md`); ask before each long run.

## 9. Open decisions

1. **E1 definition/bar.** Do we keep the absolute 0.8 against a v4 baseline of 0.687, or make E1 a
   measured fixed-probe number with the bar revisited by the user? (Thresholds are never relaxed
   silently.)
2. **Pack 4 scope and timing.** E1–E3 change the version and force re-collection; do we run them
   after the current pipeline, or after the vy-vz measurement?
3. **n-step/PER.** Is S3 in the first cut, and is PER ever in scope?
4. **Pack 3 vs vy-vz order.** Anchor first, or measurement first?
5. **BC anchor target.** KL against the reference policy, or a pre-tanh MSE against its mean
   (matching the existing warm-start loss)?

## 10. Decisions (user, 2026-09-16)

Recorded by the recovery session after Phase 0 diagnostics
(`docs/results/encoder-v5/PHASE0-DIAGNOSTICS-2026-09-16.md`) and before re-running Task 13.

| Pack | Decision |
| ---- | -------- |
| **1 — evaluation rigor** | **Adopted.** R1 fixed-probe E1/E2 (teacher controller, policy-driven kept as a secondary number), R2–R4 (= vy-vz M1–M3), R5 (the round guard's report), R6 worker clamp. Bars unchanged: E1 stays ≥ 0.8 even though v4 measures 0.687 and fixed-probe E1 will differ; that is reported, not relaxed. |
| **2 — sample efficiency** | **Adopted in full before the re-run.** S1 next_obs-by-index buffer (dedicated tests: no cross-episode `next_obs`, halved footprint, save/load round-trip), S2 crash-resume (`--resume-buffer`/`--resume-steps`, same learner + frozen partner only, resume skips warm-up, single rotating buffer checkpoint), S3 `--n-step` default 1 (mechanism only; n = 1 leaves training numerically unchanged — the first re-run stays at 1 to keep reward semantics identical to rounds 1–2), S4 skip the actor loss during warm-up. PER stays out. |
| **3 — decoder stabilization** | **Deferred.** D1's BC anchor is not taken; Phase 0 shows the encoder — not the decoder — broke round 1, and D3 shows the decoder round was degenerate from frame 0 under a blind encoder. Revival trigger: D1/D3 implicate the decoder, or the re-run shows the cloned skill being forgotten despite the entropy fixes. If revived, use a decaying pre-tanh MSE anchor to the `--init` decoder. |
| **4 — encoder representation** | **Deferred, explicit sign-off required.** It changes the `learned-v5:` version and invalidates all `runs/v5/round0*` data. Reconsider only if the fixed setup still fails E1/E2. |

Sequencing note (§8): the "no `.py` edits under a live run" hazard is past — the pipeline stopped
on 2026-09-16 01:30. It still binds whenever runs restart: land all Phase 1 code between runs, and
never edit `fly_drone/*.py` while a `sac-round`/`roam-*` chain is active.

