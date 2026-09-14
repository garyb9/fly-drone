# Handoff: Free Roam, one connectome decoder in an open arena

Written 2026-09-14 on `main` (local main is ahead of `origin/main`; see "Git state").
Read first: the approved plan `~/.claude/plans/when-youre-back-lets-dynamic-tome.md` (copy of its
intent is below), then [`docs/free-roam.md`](../../free-roam.md), then `docs/training.md` and
`docs/validation.md` for the accepted steering/looming work this builds on.

## The ask (user's words, condensed)

A **free roam mode**: the drone flies around an open arena and **every action comes from the fly
connectome**, "as if the fly controls it — not hand-waving". RL/fine-tuning is fine. Thorough and
evidence-based. User decisions already made:

- Keep **encoder v4 frozen** (upgrade to v5 only if a feasibility gate proves the 4 cues are
  insufficient, and only with user sign-off; v5 invalidates every accepted actor).
- **Foraging arena + live probes** (silence light/loom pathways live, launch threats, place
  beacons, "who is flying" neuron attribution panel).
- **Larger room** (16 × 16 m).
- The user said **pause training** at the end of this session. Do not start collection or
  training without checking with the user.

## Non-negotiable contracts

1. Decoder input is only the 2,022 descending/VNC motor traces. No task id, pose, beacon position,
   last action. Simulator state is for labels, rewards, respawn and metrics only.
2. One decoder for free roam; the server never swaps actors by scenario/state.
3. Frozen brain: wiring, weights, neuron params, tonic bias, encoder v4 unchanged.
4. Teacher labels must be **observationally realisable** (only react to what is in view,
   unoccluded, within measured cue range).
5. Legacy room reproducible: `tests/test_arena.py::test_legacy_room_mjcf_unchanged` pins the MJCF
   hash; accepted visual/looming per-seed replays were verified identical after the arena work.
6. Acceptance `python/fly_drone/roam_eval.py::ACCEPTANCE` (A1–A6, A5 probes, A7 live RTF) is
   **pre-registered**. Do not relax thresholds without the user.

## What exists and works (all committed, tests pass, `yarn ci` green at be42e69)

| Piece                                                                                                                                                                                                               | File(s)                                        | Status                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------- |
| Arena plant variant (4 walls ±8 m, thin dark band z 0.8–1.2, 16 mocap pillars grey with 0.3 m dark ring, emissive beacon, mocap threat, flat lighting, pinned model extent)                                         | `plant.py`, `arena.py`                         | done                                                    |
| Seeded layout, reachability, beacon (50 % out of view), threat planner (intercept course), respawn, clearance                                                                                                       | `arena.py`                                     | done                                                    |
| `task="free_roam"` env: beacon collection, threat scheduler, respawn without brain reset (only encoder history cleared), contact kinds, occlusion-aware beacon visibility, `light`/`loom`/`ghost` ablations, reward | `env.py`                                       | done                                                    |
| Rust binding `clear_vision_history`; `BrainRuntime.silence_inputs`                                                                                                                                                  | `crates/brain-python/src/lib.rs`, `brain.py`   | done                                                    |
| Sensor scan + cue-only probe + random baseline + gate report                                                                                                                                                        | `feasibility.py`, `fly-drone roam-feasibility` | done                                                    |
| Composite visibility-gated teacher (threat / avoid / beacon / explore)                                                                                                                                              | `teacher.py`                                   | done, **threat drive ineffective (see blockers)**       |
| DAgger tools: `roam-collect` (`--levels`, `--student --beta`), `roam-fit` (class-balanced BC, held-out flights, per-drive R², parity export with arena limits), `roam-screen`                                       | `distill.py`, `cli.py`                         | done, never run at scale                                |
| Exact gradient×input attribution by cell type                                                                                                                                                                       | `attribution.py`                               | done                                                    |
| Evaluation + pre-registered acceptance A1–A6, A5 skill probes (steer/approach/dodge/wall)                                                                                                                           | `roam_eval.py`, `evaluate --task free_roam`    | done, never run on a policy                             |
| Live server: free roam continuous, `room` message, probes (`place_beacon`, `launch_threat`, `pathway`, `ghost`), `policy_status`, attribution, free-roam stats/events                                               | `server.py`                                    | done                                                    |
| Viewer redesign (room-driven walls/pillars, true-scale drone, free roam in task list) + "DRONE HOLDING — no decoder" message                                                                                        | `web/src/main.ts`, `web/src/scene/*`           | done; **no probe panel / attribution panel yet**        |
| `yarn dev` → `serve --accepted` (loads accepted steering/looming actors from `docs/results/accepted-policies.json`)                                                                                                 | `package.json`, `cli.py`, `server.py`          | done; fixes the "drone stuck" report for visual/looming |
| Docs                                                                                                                                                                                                                | `docs/free-roam.md`                            | done; threat section needs the final numbers            |

## Measured facts you should not re-derive

Sensors (kinematic scans, arena, encoder v4 — details in `docs/free-roam.md` §3):

- Eyes were blind inside 1.76 m until the arena `statistic extent` was pinned (parked bodies
  inflated the extent). Now near clip 0.17 m like legacy.
- Grey surfaces at grazing angles rendered below the 0.18 dark threshold: 59 % false loom while
  turning with nothing dark in view → flat lighting (ambient 0.6 / diffuse 0.3) → 0 %.
- Fully dark pillars: 64–70 % false loom when yawing → grey pillar + 0.3 m dark ring: 0.3 % /
  5 % / 19 % at 0.4 / 0.8 / 1.2 rad/s. Wall band 1.2 m → 0.4 m: corner false loom 55 % → 17 %.
- Loom warning distance: ringed pillar 0.90 / 1.14 / 1.18 m at 0.5 / 0.7 / 1.0 m/s; wall 1.56 /
  2.12 m at 0.5 / 1.0 m/s. Beacon light cue 0.97 at 8 m, 0.35 at 12 m, side sign correct.
- Arena limits chosen from this: `[0.7, 0.5, 0.3, 0.8]` (m/s, m/s, m/s, rad/s).

Feasibility gate (`runs/roam/feasibility.json`, 20 seeds × 120 s, L3, old threats):

- Cue-only script 1.38 beacons/min vs random 0; **0 pillar/wall collisions** in 40 min.
  Collisions were all thrown threats (47 % dodged vs random 33 %) → gate **FAILED** only on the
  collision ratio (0.76 vs required < 0.5). Foraging + static avoidance are feasible on v4 cues.
- Ghosting pillars raised the cue script's pillar hits 2 → 11–21: pillar avoidance is causally
  loom-driven.

## Blockers (why training was not started)

### 1. Thrown-threat dodging is not achievable even by the privileged teacher

Screens (`roam-screen`, 10 seeds × 60 s, L3; reports in `runs/roam/screen-*.json`):

| Threat design                                                                        | Teacher dodge (visible) | Teacher dodge (ghost) | Random dodge                    |
| ------------------------------------------------------------------------------------ | ----------------------- | --------------------- | ------------------------------- |
| 3–4 m, 0.8–1.4 m/s, aimed at launch position (original)                              | 71 % (1.1 coll/min)     | 46 %                  | 39 %                            |
| 5–6 m, 0.6–1.0 m/s, intercept lead                                                   | 55 %                    | 33 %                  | 76 % (drift escapes slow shots) |
| 3–4 m, 0.9–1.3 m/s, intercept lead, teacher commits side + brakes (current, 8b87cc1) | **24 %**                | 21 %                  | 68 %                            |

Frame trace of the current teacher (reproduce with the inline script pattern in this session's
history, or write `runs/roam/trace_threat.py`): the teacher enters the threat drive at gap ≈ 1.9 m
and commands full lateral (±1 → 0.5 m/s) + brake, but **the drone reaches only 0.2–0.3 m/s lateral
~1.2 s later**, and the bearing to the threat stays constant (collision course). Minimum distance
ends at 0.30 m (contact at 0.31 m). Also seen: drive flickers back to `beacon` for several frames
mid-dodge (visibility or weight dropping) — check `teacher.visible()` for the threat.

Root cause candidates, in the order to check:

1. **Stabiliser lateral response.** `plant.advance` integrates the velocity command into a position
   hold (`K_P=0.4`, `K_D=0.9`); step lateral response is slow. Measure a 0 → 0.5 m/s lateral step
   while cruising 0.56 m/s forward. If it takes > 0.8 s to reach 0.4 m/s, the threat is
   physically undodgeable at 1.9 m. Options: raise the arena's lateral limit, add velocity
   feed-forward, or give more warning (the loom cue fires ≈ 2 m; the teacher could trigger at the
   first loom frame instead of gap < 2 m).
2. **Intercept lead makes dodging harder**: the shot leads the drone's current velocity; braking
   - sidestepping still leaves it near the lead point because the threat's relative bearing is
     constant. Try aiming at the launch position again (the original design had the best teacher
     rate, 71 %) and instead fix chance escapes by shortening flights/raising threat radius.
3. The `dodged` metric only counts threats that came within 2 m; verify with ghost controls.

Decision rule: **the teacher (upper bound) must reach ≥ 0.8 balanced dodge with ghost ≤ 0.3**
before threats are used in training/evaluation. If it cannot after checking the stabiliser, ask
the user whether free roam v1 should ship without thrown threats (levels 0–2) and keep threats as
a later milestone. Do **not** lower A2/A3.

### 2. Cue-only probe trade-off (affects only the gate report, not the decoder)

Yaw-escape: good foraging (1.38/min), poor dodging. Sidestep-escape: 77 % dodging, foraging 0.1/min.
The current probe (f516f1f) turns for slowly rising loom and sidesteps for saturating loom; on the
latest threats it foraged 0.4/min. The gate criterion compares against a random baseline that
barely moves (7–11 cells visited), which makes the collision ratio a weak comparator — raise this
with the user rather than silently changing the gate.

## Next steps (in order, after the user says to resume)

1. Resolve blocker 1 (stabiliser step test → threat design) with the teacher screen:
   `fly-drone roam-screen teacher random --ablations none ghost --seeds 10 --seconds 60 --level 3 --workers 16 --output runs/roam/screen-X.json`
2. Re-run the gate: `fly-drone roam-feasibility --episodes 20 --seconds 120 --workers 16 --output runs/roam/feasibility.json`; copy accepted report to `docs/results/roam-feasibility.json`.
3. Realisability probe (plan Phase 3): fit features→teacher labels and cues→labels on held-out
   flights; R² ≥ 0.5 per drive (`roam-fit` already reports held-out per-drive R²).
4. DAgger iteration 0: `fly-drone roam-collect --output runs/roam/dagger-0.npz --flights 128 --seconds 60 --workers 16` (add `--levels 0 1 2` if threats are deferred), then `fly-drone roam-fit runs/roam/dagger-0.npz --output runs/roam/it0`, then `roam-screen runs/roam/it0/warm-actor.json teacher random`.
   Iterations 1–3: `roam-collect --student runs/roam/itK/warm-actor.json --beta 0.5/0.25/0 --seed-base <new>`, fit on all npz files.
5. PPO fine-tune on the free-roam reward with curriculum (plan Phase 5): `train` needs `--task free_roam`, `--resume runs/roam/itK/warm-ppo.zip --learning-rate 1e-4 --log-std -1.5`, net arch 64-64, and **export with arena limits** (`export_actor(..., limits=ArenaSpec().limits)` — `training.train` still exports legacy `LIMITS`; fix before training, and add `--curriculum` level scheduling as planned).
6. Evaluate: `fly-drone evaluate --task free_roam --policy <actor> --episodes 50 --workers 12 --output runs/roam/evaluation.json` (runs 7 brain conditions + teacher/cue/random + A5 probes). Copy to `docs/results/`, fill `docs/validation.md`.
7. Add the accepted actor to `docs/results/accepted-policies.json` as `"free_roam"`; `serve --accepted` then loads it (limits match the arena, so `policy_status` becomes `loaded`).
8. Viewer (plan Phase 9, the user wants it): probe panel (silence light/loom, launch threat, ghost, level, click-to-place beacon), "who is flying" attribution panel (frames already carry `attribution`), trail, event ticker. Put it in `web/src/app/roam.ts` with a small hook in `main.ts`; extend `scripts/browser-check.mjs`.
9. Attribution report + causal DN hold check (plan Phase 6) using `attribution.py`.

## Lessons that will bite

- **Never run `yarn prep`/`yarn ci`/`py:build` while a simulation/training/evaluation process
  runs** — rebuilding the native extension can crash it.
- `pytest ... | tail` masks failures: use `set -o pipefail`.
- `pkill -f "<pattern>"` inside a Bash tool command matches its own shell (exit 144). Find PIDs
  with `pgrep -f "python.*..."` and kill children (`pgrep -P`) + parent.
- Free-roam threats park at z = −20; code that re-validates object positions (e.g. plant reset)
  must allow that (`park_obstacle=True`).
- The live viewer is one shared session: resets from any client (scripts, other browsers) change
  what the user sees.
- Legacy actors have limits `[0.4, 0.4, 0.2, 0.8]`; the server refuses them in the arena
  (`policy_status: "limits mismatch"`), which the viewer now reports as "DRONE HOLDING".
- Frames/events: env clears `roam["events"]` after each free-roam step; probes issued between
  steps are reported in the next frame.
- Loom gain 150 + 5-pixel threshold makes the cue extremely sensitive to anything dark changing
  size in view; any scene change must be re-scanned with `roam-feasibility`'s sensor scan.

## Environment and running processes

- A viewer server started by this session may still be running: `fly-drone serve --accepted` on
  :8000 (the user is using it). Leave it unless the user asks.
- Runs (git-ignored): `runs/roam/` holds feasibility/screen reports, the aborted
  `dagger-0-L012.npz` attempt (incomplete — delete or re-collect), `verify-*.png` screenshots,
  and the diagnostic scripts `verify-accepted.mjs`, `diagnose-live.mjs`.
- Machine: 28 threads; each brain + renderer process ≈ 0.5 GB; 16 workers is a safe default.

## Git state

Everything above is committed on `main`. `be42e69` (viewer redesign) was pushed; later commits in
this session (the holding message, `--levels`, this handoff) are local only unless pushed after
this document was written — check `git status -sb`.
