# SDD ledger — plan: docs/superpowers/plans/2026-09-14-fly-drone-05-encoder-v5-sac.md

Spec: docs/superpowers/specs/2026-09-14-learned-encoder-sac-design.md (c9b3cf2). Scope of this run: code Tasks 1–10. Tasks 11–15 are runs, each gated on the user.
Start HEAD: 7ec6af8

## Pre-flight scan

| Tasks | Produces → consumes | Finding |
| --- | --- | --- |
| T1 ↔ T5 | v4 replay goldens → env refactor must keep them | consistent |
| T2 ↔ T3 | Rust `learned-v5:` prefix → `LEARNED_PREFIX`/`LEARNED_EXTERNAL` | same string |
| T2 ↔ T7 | Rust `output: tanh` → `export_decoder` payload `"output": "tanh"` | consistent |
| T3 ↔ T4 | brain constants/FrameStack → encoder.py; brain imports encoder lazily | no import cycle |
| T3 ↔ T5 | `learned`, `push_frame`, `encode_stack`, `set_currents` → env | consistent |
| T4 ↔ T6 | `EyesExtractor`, `FEATURES`, `eyes_space` → sac.py; `from_actor` needs pi=[] linear mu | build_sac encoder pi=[] ✓ |
| T6 ↔ T7 | `learner_spaces`, `build_sac`, `set_dn_stats` → SpacesOnlyEnv, warm start | consistent |
| T6 ↔ T8 | `LIGHT`/`LOOM`, test helper `zero_actor` → roam_eval, test_roam_eval | consistent |
| T7 ↔ T9 | `export_decoder`, `init_decoder` → train_round, CLI | consistent |
| T8 ↔ T9 | `screen(encoder=)`, `encoder_checks` positional order → validate | consistent |
| T4 ↔ T9 | `collect_clone`/`fit_clone` positional args → CLI | consistent |
| T5 ↔ T8 | both edit roam_eval.py (`_probe_setup` vs `_probe_job`) | distinct functions |
| T1 self | fill-me then paste digests | intended two-step |
| T2 self | tests vs code (clip [1,1,0,-1], tanh×limits) | agree |
| T3 self | role counts 886/887/1017/1037/71/55/94/91 match data; `encode_stack() is cues` | agree |
| T4 self | 0.2 s job → 5 frames; fit report keys | agree |
| T5 self | newest stack frames [2,5] vs `luma_u8(plant.images)` | agree |
| T6 self | band test needs `import mujoco` (added); launch_threat at level 0 | agree |
| T7 self | warm start uses graph.bin sha as dataset hash | agree |
| T8 self | synthetic non-selective loom fails E1 and E2 margin | agree |
| T9 self | resume from .zip with buffer kwargs; bypass screen | agree |
| T10 self | docs only | agree |

Scan clean apart from the environment issues below.

## Log

- User decision: work on main, commit + push each task (AGENTS.md norm). No worktree.
- User decision: stop `fly-drone serve --accepted` (pid 1935403) before Task 2's rebuild; restart it after Task 2 passes.
- Task 1: complete (commits 7ec6af8..8da2f70, review clean)
- Task 2: BASE 8da2f70; viewer server (pid 1935403) stopped before rebuild, restart after Task 2 passes
- Task 2: complete (commits 8da2f70..65f4405, review clean; minor: Rust accepts any learned-v5: suffix, exact version match is Python load_policy's job)
- Viewer server restarted after Task 2
- Task 3: BASE 65f4405
- Task 3: complete (commits 65f4405..eec2331, review clean)
- Task 4: BASE eec2331
- Task 4: implementer hit API rate limit after RED (uncommitted encoder.py/test_encoder.py); resumed same agent
- Task 4: implementer DONE 1a47c16; review dispatched
- Task 4: complete (commits eec2331..1a47c16, review clean; minor parked: _clone_job collision clears stack only, collect_clone reports requested flights — check at Task 11 run)
- Task 5: BASE 1a47c16
- Task 5: implementer DONE f243735; review dispatched
- Task 5: complete (commits 1a47c16..f243735, review clean; the 2 suite warnings were confirmed pre-existing in the Task 4 review)
- Task 6: BASE f243735
- Task 6: implementer DONE 3aee2fa; review dispatched
- Task 6: complete (commits f243735..3aee2fa, review clean, opus reviewer)
- Ruling: park yaw-only visibility (plan-mandated, inherited teacher.visible/beacon_visible) — critic-only, threats launch at drone altitude ±0.1 m, same function the teacher gate A3 used — if wrong, critic sees geometry for steeply above/below objects; revisit in final review / before Task 13.
- Parked minors T6: bypass learner with "external" encoder gives unclear error; wrong comment sac.py:152 on band alpha; untested: currents actor_key isolation, metabolic reward, band grey change, per-key critic use.
- Task 7: BASE 3aee2fa
- Task 7: implementer DONE cecd9f0 (reports new torch UserWarning float(loss) in warm_start_decoder); review dispatched
- Ruling: fix plan-mandated warm_start_decoder UserWarning directly (float(loss.detach())) instead of a fix round — one line, zero behaviour change, test output must be pristine — cost if wrong: none. Commit a15d541, 9 passed with -W error::UserWarning.
- Task 7: complete (commits 3aee2fa..a15d541, review clean)
- Task 8: BASE a15d541
- Task 8: implementer DONE a1f8c0e; review dispatched
- Task 8 review: approved; Important (plan-mandated) E1 positive-label path and bypass controller untested.
- Ruling: fix round 1 now (label test with hand-placed threat, bypass screen test, clear bypass-without-encoder error, doc comments) — E1 is a pre-registered gate and its positive branch never ran under test — cost if wrong: small extra tests.
- Parked T8 minors: v4 renders after _move_objects so v4 AUC is offset one 40 ms threat step vs v5 (note when comparing in Task 14); previous_gap not reset on respawn (negligible); bypass encodes currents twice; screen() routing untested.
- Task 8: fix round 1 dispatched (resumed implementer)
- Task 8: fix round 1 DONE aabfd94 (112 passed); scoped re-review dispatched
- Task 8: complete (commits a15d541..aabfd94, review + re-review clean)
- Task 9: BASE aabfd94
- Task 9: implementer DONE 32c0ead (113 passed); review dispatched
- Task 9 review: approved (SB3 .zip resume allocates fresh buffer; SubprocVecEnv spawn pickling OK). Minor: no guard for missing frozen partner (plan-mandated).
- Ruling: add train_round guard + test directly (controller) — a mistyped flag on a multi-hour run would fail inside a spawned worker or leave an empty output dir — cost if wrong: none.
- Parked T9: SubprocVecEnv workers>1 path untested (first exercised in Task 11–13 runs; do a short 6-worker smoke before long runs).
- Task 9: complete (commits aabfd94..e235130, review clean + controller guard)
- Task 10: BASE e235130
- Task 10: implementer DONE bc1db92 (concern: README §9 milestone table stale vs §7.3); review dispatched
- Full suite at bc1db92: 114 passed, 2 pre-existing fastapi/starlette warnings
- Task 10 review: needs fixes — architecture.html roadmap item 0 and README §9 row 0 still say v4 sufficiency is open, contradicting rewritten §7.3.
- Ruling: include README §9 row in fix round though outside brief's literal file sections — same-file contradiction of a decided fact — cost if wrong: one table row edit.
- Task 10: fix round 1 dispatched (resumed implementer)
- Task 10: fix round 1 DONE 348fc3a, 4426754 (accidental whole-file prettier on architecture.html reverted; net diff 2 lines html + README §9 row). Controller verified net diff bc1db92..4426754 directly (docs-only, 2 rows) instead of a re-review dispatch.
- Task 10: complete (commits e235130..4426754)
- Final whole-branch review: 7ec6af8..4426754 dispatched (opus)
- Final review (opus): With fixes. No Critical. Important: (1) --seed ignored on .zip resume; (2) no export of checkpoints / no way to validate an encoder round (decoder pins old version); (3) spec-level: untrained critic may wreck warm starts (critic-only warm-up).
- Ruling: one fix dispatch for Important 1, 2 + minors (sensory-model.md broken markdown, split loom/light cost logging, build decoder BrainRuntime after vec.close, document crash resume restarts budget, warn on ignored --encoder) — all local, behaviour-neutral for v4 — cost if wrong: small rework.
- Ruling: Important 3 (critic warm-up / ent_coef) NOT implemented — changes the approved training design (spec §4); goes to the user before Task 13. Tasks 11–12 unaffected.
- Ruling: time estimate for rounds is ~5–8 h + ~2 h bypass (not 3 h) — tell user before Task 13.
- Parked findings adjudicated acceptable by final review; SubprocVecEnv 6-worker smoke required before Task 13.
- User decision (2026-09-14): critic-only warm-up, actor frozen for first 50k frames of each SAC round; spec amended. Added to the final-fix dispatch as a separate commit.
- User decision (2026-09-14): start Tasks 11–12 once the final fixes land and pass re-review.
- CORRECTION: controller mis-described Tasks 11–12 to the user (said 11=clone, 12=decoder init). Plan: T11 = Stage 1 DAgger decoder on L2 with v4 (~2 h, 4 iterations, seeds 200+1000k, validation screens 9000); T12 = Stage 2a clone + round-0 decoder + sanity gate + v4 E1 baseline (~1.5 h, needs T11 data). Re-asking the user before any run.
- User decision (2026-09-14, after correction): run Task 11 now, then continue into Task 12 without asking; stop and report if Task 12's sanity gate fails.
- Task 11: started 2026-09-14 21:03:48 via setsid nohup scratchpad/stage1-dagger.sh, log runs/v5/dagger/stage1.log
- Final fixes landed: 541c13a (seed on resume, sac-export/repin-decoder), 8e0dfa8 (minors). 118 passed. Fix agent declined the warm-up (relayed approval); scoped re-review of 4426754..8e0dfa8 dispatched (sonnet).
- Ruling: warm-up dispatched as its own implementer (opus, brief warmup-brief.md) citing the recorded user decision (line 89) — it is the user's direct choice, not a relayed claim — if wrong, one extra commit to revert.
- Ruling: Task 12 waits for re-review approval AND warm-up commit + review (Task 12 doesn't use train_round, but avoid running against half-edited sac.py).
- Final re-review (sonnet): Approved; all findings addressed. New Minor: sac-export repin path doesn't check --encoder before torch.load(None).
- Ruling: park the repin --encoder Minor — docs always pass it and the traceback is only opaque, not wrong — if wrong, a confusing crash during manual export.
- Warm-up implemented: 8dda027 (121 passed). Concerns: crash resume re-applies warm-up (docs: --actor-warmup 0); plain-SAC zips default 50k; ent_coef frozen during warm-up too; 6-worker path unexercised. Task review of 8e0dfa8..8dda027 dispatched (sonnet).
- Task 11 runs finished 2026-09-14 22:41:29 (no errors). Screens bpm/cpm: teacher 2.3/0.0, random 0.0/0.1, it0 2.2/1.0, it1 1.5/2.1, it2 2.0/0.9, it3 2.0/1.0 → choice it0. choice.json + commit + "Task 11: complete" still to do.
- Warm-up task review (sonnet) died on session limit before writing warmup-review.md — re-dispatch.
- Session handoff written: HANDOFF.md (this folder).
- New session 2026-09-15: user says run autonomously, log decisions in a reviewable doc → docs/results/2026-09-15-encoder-v5-run-decisions.md.
- Warm-up task review re-dispatched (opus) → warmup-review.md
- Task 11: choice.json written (it0, 2.2/1.0, teacher 2.3)
- Task 11: complete (runs only; runs/ git-ignored; decisions doc committed)
- Ruling: start Task 12 encoder-collect before warm-up review returns — collect doesn't touch sac.py, workers import at spawn — cost if wrong: none. Started 00:56:15, log runs/v5/clone/collect.log
- Warm-up review (opus): Approved, spec compliant (resume re-applies warm-up, disclosed/documented). Actor/alpha frozen verified vs SB3 internals (no Adam state, separate extractors, polyak critic-only). No Critical/Important.
- Warm-up: complete (commits 8e0dfa8..8dda027, review clean)
- Parked warm-up minors: (1) crash resume repeats warm-up (documented; want for alternating rounds); (2) round smoke test never steps actor for dec1/byp with 40-frame rounds; (3) actor/alpha losses still computed in warm-up (waste only); (4) negative --actor-warmup accepted as 0.
- Ruling: park warm-up minor (2) and cover it in the pre-Task-13 6-worker smoke (check actor params change after warm-up on the real path) — cost if wrong: an actor-update regression on decoder/bypass would only be caught by the smoke.
- Task 12: driver .superpowers/sdd/2026-09-14-fly-drone-05-encoder-v5-sac/stage2a.sh launched 00:58:32 (waits on collect pid 3449911; gate exits 2), log runs/v5/stage2a.log
- Background waiter bug: pgrep -f matched its own shell; killed, driver waits on PID instead.
- Ruling: "run autonomously" covers decisions inside approved steps (warm-up review, Task 11 close, Task 12, pre-T13 smoke). Tasks 13, 14, 15 each still need explicit user approval, as the user's same message states — cost if wrong: a pause of a few hours waiting for approval.
- User decision (2026-09-15): run autonomously up to and including Task 15 (supersedes per-task approval for 13–15).
- Ruling: plan stop rules still bind as written: T12 gate fail → stop & report; T13 "no near-dodge improvement" → don't start the next round, go to Step 6 with the best round; T14 bypass_better → still run Task 15 evaluation (measurements, no conclusions change), but record the E3 failure prominently and do not claim attribution to neurons; smoke failure → debug/fix via SDD before T13 — cost if wrong: extra evaluation compute after a failed E3.
- Task 12: encoder-collect done 01:02:10 — 48000 frames (= 32 × 60 s × 25 Hz), stacks (48000,6,48,64); encoder-clone started. Parked T4 minor checked: 32 distinct flights, 1500 frames each (no collision truncation) — resolved for this run.
- Task 12: clone done 01:04:56, version learned-v5:b3f4c7b4cbb99f28, 3 held-out flights: r mi1 0.989/0.988, tm3 0.989/0.988, lc4 0.952/0.953, lplc2 0.951/0.953; mse 0.010–0.013.
- Task 12: round0 decoder done 01:05:18 — export_max_error 8.3e-7 (≤1e-4 ✓), encoder_version matches clone, 384k samples, held-out mse avoid 0.194, beacon 0.038, explore 0.029 (threat null: L2 has no threats). Sanity screen running.
- Task 12: SANITY GATE FAIL 01:07:11 — round0 (clone enc + round0 dec) L2 9000–9009: 0.9 bpm, 2.0 cpm (walls), 24.1 cells vs it0 v4 2.2/1.0/35.0. Gate 1.76. Driver stopped before validation; smoke skipped. Per user instruction: stop and report.
- Ruling: run one cheap diagnostic (~3 min, not a long run) before reporting — refit round-0 decoder on it0.npz only with clone encoder and screen it — separates "all-DAgger refit" from "clone/tanh" as the cause — cost if wrong: 3 min compute.
- Observation: warm_start_decoder fits on x recorded under v4 in the DAgger npz (not re-simulated with the clone). Round0 held-out avoid mse 0.194 vs v4 roam-fit it0 0.106 on same-kind data → SAC tanh-head fit is a suspect independent of the clone.
- Ruling: diagnostic B (chained after A) — init_decoder on it0.npz with encoder=None (v4) and screen with v4 — isolates "SAC-head fit" from "clone encoder" — cost if wrong: ~3 min compute.
- Observation: roam-fit (v4 warm-actor) and warm_start_decoder are identical (4000 steps, Adam 1e-3, batch 256, class weights, same holdout seed) except the head: linear vs tanh with labels clipped to ±0.97. it0 avoid labels: 33% have |yaw| ≥ 0.97 (mean |yaw| 0.75); beacon 21% |fwd| ≥ 0.97. Hypothesis: tanh saturation underfits hard avoid turns → wall collisions. Awaiting diag A/B.
- Diag A (tanh head, it0.npz only, clone encoder): fit held-out mse avoid 0.113 / beacon 0.031 / explore 0.026 (≈ v4 roam-fit 0.106/0.029/0.025), export 5e-6; L2 screen 1.1 bpm, 2.8 cpm, 25.6 cells. → fit quality and all-DAgger mixing are NOT the cause; suspects: clone encoder in closed loop, or tanh head in closed loop. Diag B decides.
- Diag B (tanh head, it0.npz, v4 encoder): L2 1.8 bpm, 1.8 cpm, 32.1 cells (linear it0: 2.2/1.0/35.0). Tanh head costs some (collisions).
- Diag wiring: v4 input cells == v5 pathway cells exactly (light 1903/1924, loom 165/146). Replay of held-out flight 625 (500 frames): v4 vs v5 runtime with identical cues → max|diff| 0 (bit-identical). v5 with clone currents vs v4 cues → median DN/motor feature r 0.899, 50% of active features r<0.9, max|diff| 0.83.
- Diagnosis: clone current errors (r≈0.95, loom peaks 91–94%) amplify through the connectome; the round-0 decoder is fitted on v4-recorded activity it never sees under the clone (1.8 → 1.1 bpm). Tanh head adds a smaller loss (2.2 → 1.8). Ledger: STOPPED at T12 gate per user instruction; reporting.
- User decision (2026-09-15, after gate-fail report): "Refit under clone + loosen the tanh fit" — add --encoder to roam-collect, re-collect L2 teacher flights with the clone driving the brain, refit round 0 with a non-saturating tanh warm start, re-run the gate; if it passes continue autonomously through T15.
- Task 12b (new code task, not in plan): BASE 68c94ac (decisions-doc commit pushed just before the implementer started)
- Ruling: _load enforces encoder_version (param, default v4); warm_start_decoder requires it — the missing check let v4 data seed a learned decoder — cost if wrong: v4 DAgger npz can't init learned decoders (intended).
- Ruling: warm start loss on pre-tanh z vs atanh(clip(y,±0.97)); held-out mse stays in action space — saturated avoid labels keep gradient — cost if wrong: slightly worse beacon/explore fit, visible in screen.
- Ruling: teacher-only collection under clone first (128 flights, L2, seeds 200–327); add one DAgger iteration under clone only if the gate fails again — v4 DAgger iterations didn't beat it0 — cost if wrong: one extra ~20 min collection.
- Task 12b: implementer dispatched (opus), brief task-12b-brief.md, report task-12b-report.md
- Task 12b: implementer DONE_WITH_CONCERNS e285dd2 (124 passed). Concerns: pre-tanh loss beats old only at short training on toy data (300 steps old 0.099 vs new 0.128); may push outputs toward ±1.
- Ruling: before the 45-min collection, run diag B′ (new warm start, it0.npz, v4, L2 screen) vs diag B (old loss: 1.8/1.8) — decides keep vs revert pre-tanh on real data — cost if wrong: ~3 min.
- Task 12b: review dispatched (opus), package review-68c94ac..e285dd2.diff → task-12b-review.md; diag B′ running (log runs/v5/diag/diagb2.log)
- User request (2026-09-15): alert the user if the project data grows over 20 GB (project was 9.4 GB: .venv 5.6, runs 2.9, target 0.45). Size monitor running (du every 5 min; re-arm on expiry); also check size at each milestone.
- Task 12b review (opus): Spec ✅, Approved, no Critical/Important. ⚠️ items resolved by controller: e285dd2 on origin/main, ruff format/check clean, report shows 124 passed + 2 known warnings.
- Parked T12b minors: student pinned to another encoder fails only inside a worker (no early check/test); duplicated test fixture code; round-0 collection reuses DAgger it0 seeds 200–327 (brief-mandated, same scenes as v4 it0 — intended for comparability); new saturated-label test uses one seed / 50 steps (wide margin).
- Task 12b: complete (commits 68c94ac..e285dd2, review clean)
- Diag B′ fit (pre-tanh loss, it0.npz, v4): export 2.3e-5; held-out action-space mse avoid 0.124 / beacon 0.038 / explore 0.033 vs old-loss diag B 0.113 / 0.031 / 0.026 (slightly worse offline). Screen pending — closed loop decides keep vs revert.
- Diag B′ screen (pre-tanh loss, v4, it0): L2 2.2 bpm, 0.5 cpm, 35.0 cells — vs old tanh loss 1.8/1.8/32.1 and linear it0 2.2/1.0/35.0.
- Ruling: keep the pre-tanh warm start — closed loop recovers the linear head's foraging with fewer collisions despite slightly higher offline action-space mse — cost if wrong: none seen at L2.
- Task 12 re-run: stage2a-v2.sh launched 07:4x, log runs/v5/stage2a-v2.log; smoke13.sh chained on its PID (log runs/v5/smoke13.log).
- Task 12 re-run: clone-driven collect done 08:07:15 (~23 min); round0 refit + gate running.
- Task 12 re-run: round0-data/it0.npz verified (encoder_version learned-v5:b3f4c7b4cbb99f28, 128 flights, 96000×2022, 373 MB). Round0 refit 08:07:27: export_max_error 1.1e-5 (≤1e-4 ✓), held-out action-space mse avoid 0.098 / beacon 0.040 / explore 0.034 (old v4-data round0: 0.194/0.038/0.029). Gate screen running.
- Task 12 re-run: SANITY GATE FAIL #2 08:09:25 — round0 (clone-recorded data, pre-tanh) 1.0 bpm (gate 1.76). Diagnosing closed-loop v5 path before the pre-announced clone DAgger iteration. Breakdown: collisions 0.6 (B′ v4 0.5), cells 34.9 (35.0) → avoidance/exploration RECOVERED; only beacons lost (1.0 vs 2.2). env.step timing checked: v5 encodes the frame pushed at end of previous step (same static L2 scene as v4 render-now). Suspect: clone L−R light difference fidelity.
- Diag L−R (clone held-out, 4500 frames): mi1/tm3 L−R diff r 0.85, gain 0.85, rmse 0.15; v4 diff exactly 0 in 67% of frames; lc4/lplc2 L−R r 0.94. Per-channel r 0.99 hides a blurred beacon-side signal.
- Ruling: do NOT run the pre-announced clone DAgger iteration — evidence points at encoder L−R fidelity, not decoder covariate shift; DAgger can't fix the encoder — cost if wrong: one ~25 min iteration postponed.
- Ruling: cheap test only (clone refit 60k steps → runs/v5/diag/clone60k, same L−R metric), then stop and report to user per T12 gate instruction — a fidelity fix changes the encoder version and forces re-collection (~30 min cycle), user should choose — cost if wrong: a pause.
- clone60k running (started ~08:13); harness killed the waiter for low memory (transient; 16.9 GB available after). lr_fidelity.py fixed to load stacks once (npz key access re-reads 885 MB per batch). Waiter re-armed.
- clone60k result (held-out): light L−R r 0.851→0.878, gain 0.86→1.00, rmse 0.149→0.141 (diff std 0.28); |pred diff| where v4 diff==0 0.021→0.018; loom L−R r 0.940→0.959, rmse 0.145→0.120. More steps help a little; the light side-signal error stays ~half its std and sits in beacon-to-one-side frames. STOPPED; reporting to user with options.
- User decision (2026-09-15): "Difference-aware clone" — clone loss also penalises L−R difference errors, batches oversample beacon-to-one-side frames, 60k steps; re-collect round-0 data under the new clone, refit, gate; if it passes continue autonomously through T15.
- Ruling: DIFF_WEIGHT = 1.0 on MSE of the 4 L−R pair differences (added to channel MSE); batch = 1/3 uniform, 1/3 loom-active (existing rule >0.05), 1/3 light-side frames (|light L−R| > 0.05); clone.json gains per-pathway held-out L−R r/rmse/gain — equal weight keeps channel fidelity primary, oversampling targets the 33% of frames carrying the side signal — cost if wrong: another clone refit (~8 min).
- Ruling: keep previous artifacts: runs/v5/clone → clone-v1, round0 → round0-v2-clonedata, round0-data → round0-data-v1 — record of both failed gates — cost: ~1.3 GB disk (project ~10 GB).
- Task 12c (new code task, not in plan): brief task-12c-brief.md; BASE 12bc680; implementer dispatched (opus), report task-12c-report.md
- Task 12c: implementer DONE 6ff5c85 (127 passed). Notes: sampler draw order changed (thirds); pools computed once (clone_pools); docs/training.md §9.2 also set to --steps 60000.
- Ruling: launch stage2a-v3.sh in parallel with the 12c task review — saves ~30 min if clean; kill it if the review returns Critical/Important findings on fit_clone — cost if wrong: the elapsed compute.
- Task 12c: review dispatched (opus), package review-12bc680..6ff5c85.diff → task-12c-review.md. stage2a-v3.sh running (pid 4089200, log runs/v5/stage2a-v3.log), smoke13.sh chained.
- Task 12c review (opus): Spec ✅, Approved, no Critical/Important; ran the 4 new clone tests (pass, no warnings). ⚠️ resolved by controller: report has RED (4 failed) / GREEN / full 127 passed; controller read the committed code for the two untested minors — PAIRS derived from V5_CHANNELS _l/_r names (encoder.py:34), light-side pool uses np.abs (encoder.py:217) — so the live 60k fit is not affected.
- Parked T12c minors: loss test would pass with wrong L/R pairing; sampler test uses left-only beacons (abs untested); fallback test lacks one-empty-one-nonempty case; 0.05 threshold duplicated in docs/docstring.
- Task 12c: complete (commits 12bc680..6ff5c85, review clean)
- v3: difference-aware clone done 08:35:19, learned-v5:c273d625abe368d1, held-out L−R r mi1/tm3 0.89, lc4/lplc2 0.955 (plain 20k 0.851/0.940; plain 60k 0.878/0.959). Collection under new clone running. Collect done 09:01:16 (npz learned-v5:c273d625abe368d1, 128 flights, 96000×2022). Round0 refit 09:01:43: export 3.2e-6 ✓, version matches, held-out mse avoid 0.148 / beacon 0.038 / explore 0.030 (v2: 0.098/0.040/0.034). Gate screen running.
- Task 12 re-run #3: SANITY GATE FAIL #3 09:06:04 — round0 (diff-aware clone) 1.6 bpm (gate 1.76; v1 clone 1.0, v4-data 0.9). Stopped per user instruction; reporting with options. Gate threshold NOT relaxed. Per-seed SEM ≈ 0.3 on 10 seeds (round0 1.6±0.34, v4 it0 2.2±0.29).
- User decision (2026-09-15): re-screen the gate on 30 seeds (9000–9029) for BOTH v4 it0 and round0, same 0.8× ratio on the 30-seed means; pass → continue autonomously (validation, E1 baseline, smoke, T13–15); fail → stop and report.
- Ruling: reference is re-measured on the same 30 seeds (not the 10-seed 2.2) — the ratio rule compares like with like — cost if wrong: none.
- stage2a-gate30.sh written.
- Gate30: v4 it0 reference on seeds 9000–9029 = 1.933 bpm (10-seed was 2.2) → threshold 0.8× = 1.547. Screen took ~16 min. Round0 30-seed screen running.
- Gate30 FAIL 09:30:06 — round0 (diff-aware clone) 1.267 bpm on 30 seeds vs threshold 1.547 (ratio 0.66). Stopped per user instruction; reporting.
- Gate30 detail: v4 1.93±0.20 cpm 1.07 zero-seeds 3 yaw_bias 0.077; round0 1.27±0.19 cpm 0.57 zero-seeds 8 yaw_bias 0.106; paired −0.67±0.19; seeds 9010–9029 round0 1.10.
- User decision (2026-09-15): "Symmetric clone" — L/R mirror augmentation in the clone fit + double clone data (64 flights); re-collect round 0, refit, gate on 30 seeds (same 0.8× rule, reference 1.93 re-used from gate30-v4-it0.json); pass → continue autonomously.
- Ruling: augmentation = with p=0.5 per sample, new_left = hflip(right stack), new_right = hflip(left stack), targets swap every _l/_r pair — a mirrored world must give mirrored currents (bilateral symmetry) — requires v4 itself to be mirror-symmetric; implementer must verify on real v4 encode and report NEEDS_CONTEXT if not — cost if wrong: clone trained toward a symmetry v4 lacks.
- Ruling: extra 32 clone flights seeds 632–663 (disjoint from 600–631 and all eval/validation seeds) collected now in parallel with code — encoder-collect unchanged — cost if wrong: ~6 min compute, +0.9 GB.
- Ruling: 30-seed gate reuses the measured v4 reference 1.933 (gate30-v4-it0.json, same seeds) instead of re-screening v4 — v4 is unchanged, saves 16 min — cost if wrong: none.
- Task 12d (new code task): brief task-12d-brief.md; BASE 1ed4808 (controller docs commit landed before any implementer commit); implementer dispatched (opus), report task-12d-report.md
- Concurrent session (fullscreen-simulation-hud-overlay) editing web/src/main.ts, web/src/style.css and a one-line additive plant.py state() angular_velocity key. Told it: no yarn py:build/prep/ci or venv reinstall while fly-drone processes run; plant.py change additive only, keep test_env/test_arena goldens green; pull --rebase, never force-push. Watch for plant.py changes before SAC rounds (spawned workers import at start).
- 09:36: peer's uncommitted edits in the shared tree (plant.py, web/src/scene/theme.ts, world.ts, style.css). Told the 12d implementer to stage only its own files by path and to check `git diff --cached --stat`; the 12d review must confirm no foreign files in the diff.
- Extra clone collection done 09:38:52: runs/v5/clone/data2.npz 48000 frames / 32 flights (885 MB); verified 32 distinct flights 632–663, 1500 frames each. Project 11 GB.
- Task 12d: implementer DONE_WITH_CONCERNS 4097de5 (147 passed — suite grew from other commits; test_encoder 11 passed; ruff clean). v4 mirror symmetry bit-exact on 180 free-roam frames (max err 0.0, test tol 1e-5). Concerns: mirror mask changes RNG sequence (12c fit not exactly reproducible with new code); head can still be asymmetric → watch held_out_mirror. Staged only its 5 files.
- Ruling: launch stage2a-v4.sh in parallel with the 12d review (same as 12c) — kill if review finds Critical/Important in fit_clone/mirror — cost if wrong: elapsed compute.
- Task 12d: review dispatched (opus), package review-1ed4808..4097de5.diff (1 commit, 5 files, no peer files) → task-12d-review.md
- Task 12d review (opus): Spec ✅, Approved, no Critical/Important. Named risks checked: flip axis = width matching EyesExtractor dims=[3]; symmetry test feeds swapped flipped RGB to a fresh BrainRuntime; commit has only the 5 task files.
- Controller verified the untested training path: fit_clone mirrors a Bernoulli(0.5) half of every batch (encoder.py:291-292) via mirror_batch (flip width, swap eye halves, swap targets, :244-252).
- Parked T12d minors: pooled held_out_mirror r dominated by common mode (controller measures mirror consistency of the L−R difference with runs/v5/diag/lr_mirror.py after the clone fit); no test that fit_clone applies the mirror (verified by reading); symmetry test asserts loom non-zero but not L≠R; held-out mirror pass swaps targets it discards.
- Task 12d: complete (commits 1ed4808..4097de5, review clean)
- Baseline mirror consistency (lr_mirror.py, clone-v2-diffaware on its own data.npz split, 4500 frames): light L−R vs v4 r 0.890 rmse 0.136; mirror diff r(dp,−dm) 0.886 rmse 0.140; constant bias −0.006; loom L−R r 0.955, mirror r 0.968. → the clone's self-asymmetry is as large as its v4 error, and it is scene-dependent, not a constant offset.
- 09:45 peer session reports its web UI work done: plant.py state() gains angular_velocity (additive), test_env + test_arena 31 passed, uncommitted in tree. Controller verified `git diff python/`: only plant.py +3 lines (one "angular_velocity" key + comment in state(); no physics/advance/camera change) — safe for running and future simulations.
- v4 run: mirror-symmetric clone done 09:57:16, learned-v5:1681bff17b4eda85; clone.json held-out L−R r mi1/tm3 0.978, lc4/lplc2 0.962; pooled mirror r 1.0/0.996. Collection under it started.
- lr_mirror.py on the new clone (fit's own 2-file split, 6 held-out flights, 9000 frames): light L−R vs v4 r 0.978 rmse 0.085 (prev 0.890/0.136); mirror diff r 0.997 rmse 0.029 (prev 0.886/0.140); bias ≈0; loom L−R r 0.962 rmse 0.103, mirror r 0.994. Caveat: different held-out flights than the 3-flight baseline split.
- v4 run: collect done 10:22:49, refit 10:23:04, GATE30 FAIL 10:28:42 — round0 (mirror-symmetric clone) 1.2 bpm vs 1.547. Clone fidelity no longer explains the gap (L−R r 0.978, mirror 0.997, but foraging no better than the 1.27 diff-aware clone).
- Ruling: before reporting, run diag B′ on 30 seeds (pre-tanh decoder fit on v4 it0 data, v4 encoder; 10-seed 2.2) — if it also falls below 1.547, the bottleneck is the SAC tanh decoder/warm start, not the clone, which changes the recommended fix — cost: ~16 min. Launched 12:30:36 (log runs/v5/diag/diagb30.log).
- Gate30 #2 breakdown (mirror clone): 1.20±0.22 bpm, cpm 0.80, cells 33.5, zero-beacon seeds 10, yaw_bias 0.084 (diff-aware 0.106, v4 0.077), paired vs v4 −0.73±0.24; refit export 5.9e-6, held-out mse avoid 0.118 / beacon 0.043 / explore 0.033. Steering bias fixed, foraging not recovered. Smoke skipped as designed.
- Diag B′ on 30 seeds (pre-tanh decoder, v4 it0 data, v4 encoder): 1.90±0.18 bpm, cpm 0.67, cells 34.2, zero 2, yaw 0.085; paired vs linear it0 −0.03±0.18 → passes 1.55. Decoder head/warm start CLEARED. Mirror-clone round0 vs B′ paired −0.70±0.21 → the whole loss is from running under the clone.
- Ruling: one more cheap diagnostic before reporting — open-loop brain replay of a held-out flight (mirror clone's own split) with v4 vs mirror-clone currents (v1 clone gave median DN feature r 0.90) — separates brain amplification (clone can't fix) from closed-loop drift (DAgger under clone fixes) — cost: ~5 min, one process.
- Replay2 (held-out flight 600 of the 2-file split, 500 frames, v5 runtime): clone-v1 currents r 0.941–0.994 → DN/motor features median r 0.835 (74% <0.9); mirror clone currents r 0.958–0.996 → features median r 0.866 (64% <0.9). Near-perfect currents barely raise brain-level agreement: the brain amplifies small current differences.
- Noise sensitivity (flight 600, 500 frames): exact v4 cues + N(0, 0.01) (currents r ≈1.000) → features median r 0.921, 44% <0.9; σ 0.03 → 0.919. The connectome decorrelates fine DN traces under imperceptible input noise → trace-level cloning is unattainable; clone-fidelity strategy exhausted. STOPPED; asking user to choose strategy (DAgger under clone / SAC from round 0 with gate re-basing or override / stop).
- User decision (2026-09-15): "DAgger under clone, then SAC" — 1–2 DAgger iterations with the student flying under the mirror clone, refit, 30-seed gate; pass → continue autonomously through T15; still failing after 2 → start SAC from the best round 0 with the gate override recorded.
- Ruling: iteration k=1 student = current mirror-clone round0 decoder, beta 0.5, seeds 1200–1327; k=2 student = dagger1 decoder, beta 0.25, seeds 2200–2327 (v4 Stage-1 scheme 200+1000k, betas 0.5/0.25); refit on all clone-recorded files so far; teacher labels unchanged — cost if wrong: ~30 min per iteration.
- Ruling: best round 0 = highest 30-seed beacons/min among teacher-only (1.20), dagger1, dagger2 (ties → fewer collisions); installed as runs/v5/round0 with gate.json {passed, override, candidates}; the teacher-only mirror round0 archived as runs/v5/round0-mirror-it0 — cost if wrong: SAC could start from a slightly weaker round 0.
- Ruling: validation, v4 E1 baseline, smoke and T13–15 run after either a pass or the recorded override, per the user's choice — cost if wrong: hours of SAC compute from a round 0 at ~62% of v4 foraging, reported honestly.
- DAgger it1 (beta 0.5, seeds 1200–1327, student = mirror-clone teacher-only round0) collect done 18:08:34 (~25 min); it1.npz verified (learned-v5:1681bff17b4eda85, 128 flights, 96000×2022, student round0-mirror-it0, beta 0.5). Refit 18:08:50 on it0+it1: export 3.5e-6, held-out mse avoid 0.181 / beacon 0.037 / explore 0.033 (student-state avoid labels are harder). Gate30 screen running.
- Task 12: GATE PASS 18:14:12 — dagger it1 round0 2.133 bpm / 0.533 cpm on 30 seeds (gate 1.547; v4 it0 1.933). Installed runs/v5/round0 from round0-dagger1 (gate.json passed=true, override=false). No it2. Validation + v4 E1 baseline running, smoke chained.
- Gate pass breakdown: dagger1 2.13±0.27 bpm, cpm 0.53, cells 36.5, zero 5, yaw 0.100, paired vs v4 +0.20±0.25 (teacher-only mirror −0.73±0.24). Decisions doc pushed a4947a1; memory project_v5_clone_limit updated with the outcome.
- CORRECTED noise sensitivity at 8 ticks/frame (flight 600, 500 frames): σ 0.01 → features median r 0.902, 49% <0.9; σ 0.03 → 0.894, 53% <0.9 (40-tick run had said 0.921/0.919). Conclusion stronger, unchanged. Clone replays at 8 ticks: clone-v1 median r 0.858 (66% <0.9), mirror clone 0.880 (58% <0.9). Decisions doc table and memory corrected.
- Task 12: round0 validation done 18:20:34 (L3, seeds 9000–9009): near_dodge 0.269, balanced 0.182, ghost near_dodge 0.296, beacons 1.9, collisions 2.7, E1 loom AUC 0.686 (fail, ≥0.8), E2 pass (light margin 0.50, loom margin 0.26). Baseline for T13 (round0 trained on threat-free L2 → no causal dodging yet). v4 E1 baseline running.
- v4 E1 baseline (runs/v5/e1-v4-baseline.json, 50 eval seeds): E1 loom AUC 0.687 (fail ≥0.8); E2 pass (light margin 0.49, loom margin 0.27). v4 itself does not meet E1; round0 v5 0.686.
- Task 12: complete 18:39:08 (runs; code Tasks 12b–12d 900a946..4097de5; clone learned-v5:1681bff17b4eda85 mirror-symmetric on 64 flights; round0 = DAgger it1 under clone, 30-seed gate 2.13 vs 1.55 passed; validation + v4 E1 baseline written). Smoke test starting.
- Smoke test started 18:39 (sac-round encoder running, 13.5 GB available). Pipeline13-15 inputs verified: clone encoder.pt, round0 decoder.json/.zip/validation.json/gate.json present; round0 decoder pinned to learned-v5:1681bff17b4eda85 = clone version; bash -n OK.
- Smoke: encoder round exit 0, 117 s wall for 9000 frames (6 workers); decoder round running. round.json ok (actor_warmup 6000, init clone, version learned-v5:485aa5d2fecc36fe); fps 81; metabolic 0.011 (loom 0.0091, light 0.0019); no errors.
- Ruling: smoke_check warm-up log check relaxed to "actor_frozen logged, final 0" — SB3 logs at episode ends and the smoke is a single 1500-frame episode per worker, so a 1→0 flip is not observable; freeze covered by reviewed WarmupSAC unit tests, post-warm-up learning by the weight comparison — cost if wrong: a warm-up regression on the 6-worker path would surface in round 1's logs (many episodes) instead.
- Smoke: decoder round exit 0, 103 s; smoke done 18:42:54; RAM peak used 10.8 GB, min available 13.3 GB. Caveat: SB3 replay buffer fills lazily (smoke stored 9k transitions; full rounds 100k ≈ 5.3 GB) → expect ~8 GB available in round 1; watch round 1 memory.
- SMOKE_CHECK OK 18:43 (12/12): hash==round.json; encoder actor Δw 8.2e-2 vs clone; decoder pinned to smoke encoder; export parity 1.3e-5; decoder actor Δw 7.4e-2 vs round0; actor_frozen final 0 both; metabolic logged (encoder 0.011, decoder 0 — by design: sac.py:152-158 charges λ_loom·mean(loom)+λ_light·mean(light) only when learner == "encoder"); no tracebacks; RAM ok.
- Task 13: pipeline13-15.sh launched by gatekeeper 18:43:09, pid 788490, log runs/v5/pipeline13-15.log.
- 19:12 round1 encoder: 99k/350k frames, fps 60 (smoke 81 before gradient steps dominate), ep_rew_mean −239, metabolic 0.012, checkpoints 50k/100k, no errors, MemAvailable 8.6 GB (buffer full, stable), project 12 GB. Revised estimate: encoder ~97 min, decoder ~42 min, validation ~15 → ~2.6 h/round, 3 rounds ~7.8 h, bypass+E3 ~2.5 h, eval ~3 h → done ~08:00–09:00 on 2026-09-16.
- Warm-up verified on the real 6-worker path (round1-encoder.log): actor_frozen=1 at dumps 18k–45k, flips to 0 between 45k and 54k (boundary 50k), stays 0 to 99k. Resolves parked warm-up minor "6-worker path not exercised" and the relaxed smoke log check.
- T13–15 time estimate from smoke fps 81: encoder round 350k ≈ 72 min, decoder 150k ≈ 31 min, validation ~15 min → ~2 h/round; 3 rounds ~6 h; bypass 450k ≈ 95 min + E3 screen; evaluation ~3 h → total ~11–12 h.
- Prepared runs/v5/diag/smoke_check.py (hash vs round.json, export parity, encoder + decoder actor weights moved after warm-up, actor_frozen 1→0 and metabolic_cost in logs, RAM headroom); decoder.json layout (layers[{weights,bias}]) verified compatible. Run it when smoke13 logs "smoke done", then launch pipeline13-15.sh if SMOKE_CHECK OK.
- Note for options: Rust policy.rs supports Output::Clip (PPO default) and Output::Tanh (SAC) — a clip/linear-head decoder runs without Rust changes, but SAC's actor needs tanh squashing to train.
- Prepared (launched with 12d review): stage2a-v4.sh — archives clone-v2-diffaware / round0-v3 / round0-data-v2, reads v4 30-seed ref from round0-v3/gate30-v4-it0.json, clone on data.npz+data2.npz 60k, collect under clone, refit, gate30 (0.8× ref), validate, v4 E1. Launch after 12d review + data2 done: stage2a-v4.sh → smoke13.sh <pid> runs/v5/stage2a-v4.log.
- Prepared (launched with review): stage2a-v3.sh (archives clone-v1/round0-v2-clonedata/round0-data-v1, clone 60k, collect under new clone, refit, gate, validate, v4 E1; syntax OK). Launch after 12c review: stage2a-v3.sh → smoke13.sh <pid> runs/v5/stage2a-v3.log → inspect → pipeline13-15.sh.
- Prepared (not launched): stage2a-v2.sh (moves failed round0 → round0-v4data, collect under clone, refit, gate, validate, v4 E1; syntax OK); smoke13.sh now takes `<driver pid> <driver log>`. Launch order after 12b review passes: stage2a-v2.sh → smoke13.sh <pid> runs/v5/stage2a-v2.log → inspect → pipeline13-15.sh.
- Tasks 13–15 compute pipeline written: pipeline13-15.sh (syntax OK; START_ROUND env to resume). Launch only after smoke passes.
- Ruling: round 0 (clone encoder + round0 decoder) is a candidate in T13 stop rule and Step 6 — plan's table covers rounds 0..k and has round0/validation.json — cost if wrong: final pair could be un-SAC'd v5 (reported honestly).
- Ruling: null near_dodge_rate → −1 in comparisons — no measurable dodge never counts as improvement — cost if wrong: none.
- Pre-T13 smoke queued: smoke13.sh (waits on driver, runs only if "stage 2a done"): encoder+decoder sac-round, 9000 frames, warm-up 6000, 6 workers, RAM sampled; log runs/v5/smoke13.log, outputs runs/v5/smoke/.
- Gate30 launched 09:19:08, pid 4182088, log runs/v5/stage2a-gate30.log; smoke13 chained.
- Extra clone collection started 09:32:15 pid 8950: 32 flights seeds 632–663 → runs/v5/clone/data2.npz (log collect2.log)
- stage2a-v4 launched 09:40:33 pid 25495, log runs/v5/stage2a-v4.log; smoke13 chained
- stage2a-dagger.sh launched 17:43:43 pid 695240, log runs/v5/stage2a-dagger.log; smoke13 chained
- CORRECTION: replay2.py / noise_sens.py stepped 40 brain ticks (200 ms) per frame; env uses 8 ticks (40 ms). Trace-correlation numbers (0.835/0.866/0.921) were at wrong timing; behavioural evidence (B′ 1.90 vs clone 1.20; DAgger pass 2.13) unaffected. Re-running at 8 ticks → runs/v5/diag/replay8.log; correct the decisions doc and memory with the new numbers.
- Gatekeeper gate-smoke-then-pipeline.sh launched 18:42:06: waits for smoke done → smoke_check.py → launches pipeline13-15.sh only on SMOKE_CHECK OK (log runs/v5/gate-smoke.log)
- Round 1 done: encoder SAC 20:22 (learned-v5:b9722d66acced4f1, ep_rew −151), decoder SAC 20:56 (ep_rew flat ≈ −690 after unfreeze), validation 21:05: near_dodge 0.93, balanced 0.91, ghost 0.81, beacons 0.0, collisions 6.1, E1 0.48 fail, E2 fail. DEGENERATE (erratic flight scores as dodging; not causal). Stop rule (near-dodge only) accepted it; round 2 encoder SAC started 21:05 from round 1 (ep_rew ≈ −2480 at 252k).
- Attempted kill -STOP of pipeline group was denied by the permission classifier; pipeline still running. Reported to user with recommendation: stop, keep round 0, add selection guard (beacons ≥ ½ round 0, ghost ≤ 0.3), investigate decoder SAC collapse. User replied "isn't spastic how a fly flies?"; answered (jerky OK, non-causal + no foraging not). PENDING USER DECISION: stop vs let round 2 finish (validation ≈ 23:40).
- Backup 22:50: ledger, handoff, run scripts, diag scripts and small run JSONs copied to docs/results/encoder-v5/ (subfolder run-records/, since .gitignore `runs/` matches any dir named runs); decisions doc gained "Round 1: result and where we stand". Untracked docs/superpowers/specs/2026-09-15-training-improvements.md belongs to another session — not staged.
- Checkpoint 23:20 (user asked, usage may run out): round 2 encoder SAC 315k/350k, ep_rew ≈ −2500; pipeline pid 788490 still running; user decision (stop vs finish) still pending. Resume doc: docs/results/encoder-v5/CHECKPOINT-2026-09-15-2320.md. Project 12 G (6.3 G w/o .venv), RAM 9 G free.
- Round 2 validation 00:52: near_dodge 1.0, balanced 1.0, ghost 1.0, beacons 0.0, collisions 10.3/min, E1 0.554 fail, E2 fail (light +0.016, loom +0.037). MORE degenerate than round 1 (blind condition dodges 100%, collisions up). Near-dodge-only stop rule accepted it (1.0 > 0.926); round 3 encoder SAC started 00:52 (validation ≈ 04:30). User decision (stop vs finish) still pending.
- User decision (2026-09-16 01:30): "stop everything here, prepare a thorough handoff". Pipeline group 788490 killed (SIGTERM) during round 3 encoder SAC (~117k/350k). Tasks 14–15 not run; runs/v5/final not created. Lead found: build_sac leaves SB3 auto-α at init 1.0 (frozen through warm-up) → entropy bonus swamps ~0.05/step reward after unfreeze; α decays to 1e-13 (series in handoff §2.2). Handoff: docs/results/encoder-v5/HANDOFF-2026-09-16.md.
- Recovery session 2026-09-16 08:30: handoff re-verified against code/logs/plan — all load-bearing claims confirmed; D1 command corrected to `sac-export --repin-decoder <json> --encoder <pt> --output <json>` (no `--learner`, no checkpoint; cli.py:220-224). Plan approved: Phase 0 diagnostics; Phase 1 = selection guard + `ent_coef="auto_0.01"` + matched target entropy + log_std clamp [−4,−1] + logging + Pack 1 (R1–R6) + all Pack 2 (S1–S4) before any re-run; Pack 3 D1/BC anchor dropped; Pack 4 deferred; dispositions recorded in all three specs. Deleted runs/v5/round3 (19 MB).
- Phase 0 D4 (runs/v5/diag/current_drift.py, d4-current-drift.log): exported round-1 encoder mean current 0.337±0.587 @50k (bit-identical to clone, version 1681bff17b4eda85) → 0.977±0.026 @100k → 0.999 @150k → 1.000±0.005 @350k. Deployed path is deterministic `tanh(mu)+1` (encoder.py:138-142), so the encoder is a constant 1.0 on all 8 channels = blind (clone: 0.337/0.587/67.5% saturated).
- Phase 0 in-run log (run-records/round1/round1-encoder.log): metabolic_cost 0.0030 (clone, warm-up) → 0.0073 @54k → flat 0.012 from 63k to 342k; loom_cost flat 0.010 = λ_loom·1.0; ent_coef 1.0 @54k → 0.675 @63k. Collapse in the first ~13k frames after the 50k unfreeze, never recovers.
- Phase 0 D1 (runs/v5/diag/d1/): round-1 encoder + round-0 decoder = near 0.111, balanced 0.000, ghost 0.087, beacons 0.00, collisions 5.40/min, E1 0.441, E2 fail. Same decoder under clone = 0.269 / 0.182 / 0.296 / 1.9 / 2.7 / 0.686. Encoder round alone destroys foraging → encoder broke first (handoff §2.3 confirmed); no decoder anchor indicated.
- Phase 0 D2/D3 (runs/v5/diag/run_d2_d3.sh, d2-d3-chain.log): D2 enc@100k 0.000/0.10/5.0 E1 0.457, enc@150k 0.083/0.10/5.9 E1 0.517 — both degenerate; D3 dec@50k = D1 exactly (0.111/0.000/0.087/0.00/5.4) because the decoder actor is still frozen at round-0 weights → decoder round degenerate at frame 0; D3 dec@100k flails (0.933/0.857/ghost 0.875/0.00/13.1 E1 0.530) = stage-2 decoder collapse.
- Ruling: two-stage failure model — (1) encoder mean collapses to the max-entropy centre while α≈1 in frames 50k–63k; (2) decoder SAC flails under the blind encoder and inflates near-dodge/ghost. Phase 1 entropy fixes target (1), the guard catches (2) — cost if wrong: re-run degenerates and the guard aborts early.
- Ruling: Phase 1 logging must record per-channel deterministic mean-current drift, not only train/log_std_mean — μ collapsed, not σ; the existing metabolic_cost log caught it by accident — cost if wrong: none (additive).
- Report: docs/results/encoder-v5/PHASE0-DIAGNOSTICS-2026-09-16.md.
- Phase 1a (selection guard): `roam_eval.round_eligible` / `pick_best_round` / `round_gate_report` + `ROUND_GATE` (beacons ≥ ½ round 0, ghost ≤ `A3_max_ghost_dodge_rate` 0.3, E2 pass; round 0 always eligible). 6 unit tests. Run against the recorded round 0/1/2 validations: round 0 eligible, rounds 1 and 2 ineligible, `pick_best_round` = 0 — the stopped run would have halted at round 1. `pipeline13-15.sh` stop rule and Step 6 rewired to call the guard; plan Task 13 Step 5/6 amended. Full suite 153 passed, ruff clean.
