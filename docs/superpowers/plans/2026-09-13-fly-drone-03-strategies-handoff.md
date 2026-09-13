# Handoff: add four new flight strategies (rough)

For the next agent. Written 2026-09-13 at commit `1290df8` on `main` (pushed). Read
`docs/validation.md` and `docs/training.md` first; they cover the accepted steering and looming
work and every lesson below in detail.

## The ask

The user wants four more strategies next to the existing **steer to target** (`visual`) and
**dodge obstacle** (`looming`):

1. **approach**: fly forward and end within ~0.5–0.6 m of the bright target, not just face it.
2. **track**: the target orbits sideways and reverses; keep it centred for 30 s.
3. **steer_dodge**: turn toward the target while an obstacle launches mid-episode; dodge, then
   keep steering.
4. **escape**: an obstacle from ahead triggers a vertical climb (the giant-fiber escape idea).

Decisions already made by the user: **one policy per task** (not a generalist), and all four
tasks. The UI part of that request (trials panel moved to the top) is done.

## State of the working tree (uncommitted, untested)

Five files carry a partial implementation. The code has not been formatted, linted or tested.
Review it before building on it; `git diff` shows everything.

- `python/fly_drone/env.py`, written in full:
  - `TASKS`, `HORIZON_FRAMES`, `EVAL_SECONDS` and the success constants.
  - Per-task reset placement, including the orbiting target (`_orbit_position`).
  - The looming, escape and steer_dodge obstacle launch.
  - Per-task reward, and new `info` fields: `target_distance`, `side`, `climb`.
  - A new `EpisodeTracker` class holding every task's success rule, meant to be shared by
    evaluation and the live server. This replaces duplicated logic.
  - **The seed draw order for `visual` and `looming` must stay the same**, so accepted
    evaluations still replay identically by seed. Check that it did.
- `python/fly_drone/training.py`: `_rollout_chunk` uses `EpisodeTracker`. `evaluate` takes
  `seconds=None` (the per-task default) and writes generic acceptance keys `success_rate`,
  `balanced_success`, `passed`, `ablation_passed`, plus `hover_passed` for visual only.
  `TASK_NOTES` describes each rule. The `THREAT_RANGE` and `MAX_DISPLACEMENT_AT_THREAT`
  constants moved to `env.py`.
- `python/fly_drone/calibration.py`: `teacher_action(task, info)` for all six tasks, and
  `collect_closed_loop` generalised with `TEACHER_AXES` exploration noise. Visual labels are
  still unscaled, with `warm_start` applying `--teacher-scale`. **New tasks store the exact
  teacher action, so train them with `--teacher-scale 1.0`.**
- `python/fly_drone/cli.py`: task choices come from `TASKS`, `evaluate --seconds` defaults per
  task, and `serve --task-policy TASK=PATH` can be repeated.
- `python/fly_drone/server.py`: tasks come from `env.TASKS`, there are per-task policies, and
  the live result comes from `EpisodeTracker`. The frame outcome adds `target_distance` and
  `climb`. **This was a string-surgery edit; read it carefully.**

## Known breakage and TODO

- `tests/test_env.py:139` still asserts `"avoidance_passed"`; change it to `"passed"`.
  `tests/test_server.py` asserts `metadata["tasks"] == ["visual", "looming"]`; update it for
  six tasks.
- Add tests: an env smoke test per new task (`vision=False`: reset, a few steps, the new `info`
  fields, the track target actually moves), and `EpisodeTracker` rules on fabricated `info`
  dicts.
- **UI (`web/src/main.ts`) is not updated.** The `#task` select still has only two options.
  Add the four new ones with readable names, per-task outcome text (target distance, tracking
  error, climb), and the trial-header label map. Keep `scripts/browser-check.mjs` passing.
- **Before training anything**, check feasibility with the scripted teachers
  (`teacher_action`, noise 0) on about 20 seeds per task. This saved hours twice:
  - Escape was **infeasible as first designed**. The vertical limit is only 0.2 m/s: a climb
    triggered at 2.0 m survived 70%, at 1.5 m 0%. The WIP aims the obstacle 0.12 m _below_
    the drone to fix this. Verify it; if it still fails, ask the user before changing
    `LIMITS`, since that changes the command interface for all tasks.
  - `approach` and `track` are unverified. Track relies on the ±2.6° binocular overlap for sign.
- Then, per task: `fly-drone calibrate --closed-loop --task T --trials 64 --frames <horizon>`,
  `train --task T --calibration ... --teacher-scale 1.0 --envs 8`, and a 50-seed `evaluate`
  (run tasks two at a time). Accept at ≥ 80% raw **and** ≥ 0.8 balanced, beating every ablation.
  Record results in `docs/validation.md` and `docs/results/`, update `training.md` with each
  task's math, and add checklist items.

## Lessons that will bite

- **Blind policies pass naive metrics.** Always use balanced success (by side) and, for any
  dodge, threat-specificity (displacement at threat onset < 0.25 m). The shuffled-feature
  ablation survived 90% of looming episodes by drifting away.
- **Calibrate in closed loop.** With loom gain 150, turning fires looming cells in 9–15% of
  frames. Decoders fitted on static frames collapsed (8% steering).
- **A warm start leaves the model at lr 1e-5 and log_std −2.5.** PPO resumed from it barely
  learns. Pass `--learning-rate 1e-4 --log-std -1.5` when fine-tuning (this took looming from
  84% to 96%).
- **Never run `yarn dev` / `yarn ci` / `py:build` while training or evaluation runs.** Rebuilding
  the native extension can crash processes that have it loaded. Use
  `scripts/venv.sh fly-drone serve ...` to view.
- The encoder is `ENCODER_VERSION = bright-contrast-400-splay075-noaa-loom150-v4`. Any camera,
  render or cue change must bump it (in `policy.rs` and `brain.py`), which invalidates every
  actor and calibration.
- Accepted policies: steering `runs/v4-closed-s04/actor.json`, looming
  `runs/v4-loom-ft/actor.json`. `runs/` is git-ignored; the reports are in `docs/results/`.

## Environment notes

- Workflow: `yarn prep` before committing, then `yarn ci` (the same command CI runs). Commit and
  push to `main` is allowed.
- A viewer server from this session may still be running on :8000 with both accepted policies.
  Stop it with `pkill -f "fly-drone serve"`.
- The machine has 28 threads and ~19 GB free. Each brain plus renderer process uses ~0.5 GB.
  Evaluation at 12 workers takes ~2–6 min; training at 8 envs runs ~110 steps/s.
