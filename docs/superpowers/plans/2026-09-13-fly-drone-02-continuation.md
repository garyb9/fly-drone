# fly-drone: finish initial setup, add yarn prep/dev/ci, reach milestone-1 acceptance

## Context

The previous agent built most of milestone 1 (Rust brain + PyO3, MuJoCo plant, env/PPO/calibration/evaluate, server, web viewer, passing `scripts/check.sh`) but hit its session limit before committing anything (repo has zero commits), before running held-out evaluation/ablations, and before writing validation docs. Quick 6-seed check of `runs/conservative/actor.json`: turns toward target 6/6, but only 4/6 meet the "halve bearing error" rule → below the 80% target. The user also wants a fly-playground-style `prep` / `dev` / `ci` workflow (chosen: **root package.json with yarn scripts**, see Phase 1), a fly-playground-like `docs/` folder, README documentation, and commits pushed to `main`.

Data decision: mirror fly-playground, which commits `public/data/malecns/**` (incl. full graph.bin) directly — no LFS. Commit `data/malecns/` (84 MB) directly.

Note: a second Claude process (session e9c34caa, fork of the old run) is still alive in this repo. Before editing, check `ps` / file mtimes; if it's actively writing, stop and tell the user.

## Phase 0 — Hygiene and first commit (push to main)

1. `.gitignore`: add `.ruff_cache/`, `pipeline/out/` stays tracked (fixtures used by Rust tests — verify with `cargo test` referencing it), `node_modules/`, `*.npz` outside `runs/` not needed (runs/ ignored). Confirm `python/fly_drone/_brain.abi3.so` ignored (`*.so` ✓).
2. Check `vendor/mujoco-drones` submodule is pinned to a commit and `docs/source-provenance.json` names it.
3. `git add` specific paths; review `git status` for secrets/large junk; commit "Initial import: connectome drone laboratory (milestone 1 in progress)"; `git push -u origin main` (graph.bin 61 MB → GitHub warning only, under 100 MB hard limit).

## Phase 1 — Root yarn task runner (mirror fly-playground `yarn prep`/`yarn ci`)

User changed choice: a **root `package.json` managed by yarn** (`packageManager: yarn@1.22.22`, `.nvmrc` = 22) orchestrates Rust, Python and TS, same as fly-playground. It also gets real TS tooling now (prettier + eslint over `web/src`) so browser-side TS work can grow later.

- Fold `web/` into the root: move `web/package.json` deps/devDeps to root, keep Vite root at `web/` (`vite --config web/vite.config.ts`), delete `web/package-lock.json`, generate `yarn.lock`. Add `prettier`, `eslint`, `@eslint/js`, `typescript-eslint`, copy `.prettierrc.json`, `.prettierignore`, `eslint.config.js` from fly-playground (adapt paths). Add `rust-toolchain.toml` (stable, clippy+rustfmt).
- `ruff` added to pyproject `dev` extras + `[tool.ruff]` (py311, line 88).
- Python tools invoked via `.venv/bin/…` so yarn works without activating the venv.

| script                                               | runs                                                                                                                                    |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `format` / `format:check`                            | prettier `web/src/**/*.ts`, `*.{json,md}`, `docs/**/*.md`                                                                               |
| `lint` / `lint:fix`                                  | eslint `web/src`                                                                                                                        |
| `typecheck`                                          | `tsc --noEmit -p web`                                                                                                                   |
| `test`                                               | vitest (web)                                                                                                                            |
| `build`                                              | vite build (web)                                                                                                                        |
| `rs:fmt` / `rs:fmt:check`                            | `cargo fmt --all [--check]`                                                                                                             |
| `rs:lint`                                            | `cargo clippy --workspace --all-targets -- -D warnings`                                                                                 |
| `rs:test`                                            | `cargo test -p brain-core --locked`                                                                                                     |
| `rs:bench`                                           | `cargo run --release --bin brain-bench`                                                                                                 |
| `py:fmt` / `py:fmt:check`, `py:lint` / `py:lint:fix` | `ruff format [--check]`, `ruff check [--fix]` on `python tests scripts`                                                                 |
| `py:build`                                           | `maturin develop --release --extras dev,train`                                                                                          |
| `py:test`                                            | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q`                                                                                            |
| `smoke`                                              | `fly-drone baseline --seconds 2`                                                                                                        |
| `setup`                                              | `./scripts/setup.sh` (submodule, venv, maturin, `yarn install`)                                                                         |
| `predev` / `dev`                                     | `py:build` then `fly-drone serve` (README: `yarn web:dev` in 2nd terminal for Vite HMR)                                                 |
| `dev:policy`                                         | serve with `--policy runs/<best>/actor.json`                                                                                            |
| `assay`, `calibrate`, `train`, `evaluate`            | CLI wrappers                                                                                                                            |
| `prep`                                               | format → lint:fix → rs:fmt → py:fmt → py:lint:fix → typecheck → rs:lint → py:build → build                                              |
| `ci`                                                 | format:check → lint → rs:fmt:check → py:fmt:check → py:lint → typecheck → rs:lint → rs:test → py:build → py:test → test → build → smoke |

- Fix clippy/ruff/eslint findings (no behavior changes).
- `scripts/check.sh` → `exec yarn ci`; setup.sh uses `corepack enable && yarn install --frozen-lockfile` instead of `npm ci`.
- CI workflow: add clippy component, node yarn cache, final step `yarn ci`.
- Code-split three.js in vite config to remove the 655 kB warning.
- Commit + push.

## Phase 2 — Evaluation correctness & speed

In `python/fly_drone/training.py::evaluate` and `env.py::info`:

- **Bug:** `info()["bearing"]` is not wrapped to [-π, π] while `final` is wrapped → initial/final compared inconsistently. Add a single `wrapped_bearing()` helper in env.py used by `step`, `info`, calibration & evaluate.
- Add hover acceptance: 30 s policy hover run reporting altitude RMS after settling (<0.15 m) into the same report (plan requirement; `baseline` only covers PID).
- Speed: run the 4 ablation modes in parallel processes (`concurrent.futures.ProcessPoolExecutor`, one BrainRuntime per process, ~0.5 GB RSS each); add `--workers` CLI flag. Record wall time, tick latency, missed deadlines in report.
- Add a tiny pytest for `wrapped_bearing` and for evaluate on 1 episode × 2 s (marked fast enough for CI, or `@pytest.mark.slow` skipped by default).
- Run `yarn evaluate --policy runs/conservative/actor.json --episodes 50` → baseline numbers. Commit.

## Phase 3 — Improve visual steering to ≥80% with ablation degradation

1. Training throughput: use `SubprocVecEnv` with N envs (`--envs`, default 4) in `train()`; adjust `n_steps` so rollout = 1024 total. Keep seed determinism per env.
2. Reward shaping in `env.py` (visual): replace constant `2*cos(bearing)` with progress term `k*(|b_prev|-|b_now|)` + small `cos(bearing)`; keep altitude + action penalties.
3. Curriculum run: warm start from `runs/calibration.npz` (existing `calibration.warm_start`), then PPO ~50k steps (`--envs 4`), checkpoint every 5k; evaluate every checkpoint on 10 seeds via script; pick best, then full 50-seed evaluation with ablations.
4. If success <80% after that: increase calibration trials (64→256) and teacher scale sweep (0.4/0.7/1.0), retrain. Stop and report to user if still failing after two iterations (don't silently lower the bar).
5. Copy accepted report to `docs/results/evaluation-visual.json`. Commit.

## Phase 4 — Looming task

- Train `--task looming --resume <best visual ppo.zip>` ~30k steps; extend `evaluate` with `--task looming` metric (collision rate vs obstacle placed ahead, compare to ablations).
- Store `docs/results/evaluation-looming.json`. Commit.

## Phase 5 — Viewer verification

- `yarn dev`, then `node scripts/browser-check.mjs`; extend it to exercise pause/resume, reset (episode increments, tick resets), reconnect (close socket → frames resume), neuron silence → activity drops, target left/right → command sign. Record in `docs/manual-checklist.md` (ticked = automated, unticked = needs human eye, same format as fly-playground).
- Commit.

## Phase 6 — docs/ folder (fly-playground style, drone-relevant)

- Move result JSONs into `docs/results/` (`baseline.json`, `native-benchmark.json`, `sensory-assay.json`, evaluations); update README links.
- Keep `architecture.md`, `THIRD_PARTY.md`, `source-provenance.json`, `reference-motor-registry.ts`.
- New: `validation.md` (acceptance table: hover RMS, steering %, ablations, parity, latency/RTF, missed deadlines — with numbers and dates), `neuron-model.md` (LIF, graph format, determinism — adapted from fly-playground's), `sensory-model.md` (cameras, encoder, Mi1/Tm3 & LC4/LPLC2 mapping, limitations), `control-and-physics.md` (frames, motor order/signs, lag, PID cascade, clock ratios), `training.md` (calibration → PPO → evaluate workflow, reward, run layout), `data-pipeline.md` (data source, manifest hash, how copied from fly-playground), `manual-checklist.md`, `references.md`, `superpowers/plans/2026-09-13-fly-drone-01-foundations.md` (the original plan) and this continuation plan.
- Slim architecture.md where it duplicates the new files.

## Phase 7 — README

Replace run section with a scripts table:

- **Bootstrap once:** `yarn setup` (or `./scripts/setup.sh`)
- **Before committing:** `yarn prep`
- **Run:** `yarn dev` (+ optional `yarn web:dev`), `yarn dev:policy`
- **CI (local == GitHub):** `yarn ci`
- Train/evaluate: `yarn calibrate`, `yarn train`, `yarn evaluate`.
  Link docs index. Final `yarn ci`, commit, push.

## Critical files

`package.json` (new, root), `yarn.lock`, `eslint.config.js`, `.prettierrc.json`, `rust-toolchain.toml`, `pyproject.toml`, `.gitignore`, `.github/workflows/ci.yml`, `scripts/{check,setup}.sh`, `python/fly_drone/{env,training,calibration,cli}.py`, `scripts/browser-check.mjs`, `web/vite.config.ts`, `README.md`, `docs/**`.

## Verification

- `yarn ci` green locally; GitHub Actions green after push (`gh run watch`).
- `yarn evaluate` report: `acceptance.steering_passed` and `ablation_passed` true; hover RMS < 0.15 m.
- Export parity < 1e-4 (existing check in `export_actor`).
- `yarn rs:bench` p50/p95 recorded; missed deadlines reported, graph never reduced.
- Browser check passes against live `yarn dev`.
