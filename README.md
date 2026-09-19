# Fly / Drone

A full MaleCNS fly connectome inside a simulated quadrotor. Camera pixels enter a frozen Rust
spiking network of 166,700 neurons, a learned decoder turns descending/motor neural activity into
motion commands, and a stabilising controller drives four MuJoCo rotors.

The browser shows the drone, its two eyes, a live anatomical activity graph, and an illustrative
fly receiving the same neural outputs. The drone is the primary body; the fly has an independent
trajectory.

## Goal

**The fly's connectome is the brain; the body is an embodiment, not the intelligence.** Today the
body is a simulated quadrotor — the cyborg milestone. The fidelity reference is the fly's own
biomechanical body (`flybody`/NeuroMechFly); the deployment target is a physical drone. Only
descending/motor (and, once added, ascending) neurons reach the body adapter; the wiring stays
frozen; the body supplies reflexes (stabilisation), never decisions. Every skill must be causal — it
disappears when the pathway carrying it is silenced. Start with
[`docs/overview/README.md`](docs/overview/README.md) and the diagrams in
[`docs/overview/architecture.html`](docs/overview/architecture.html); agents also read
[`AGENTS.md`](AGENTS.md).

## Requirements

Rust (stable, via `rust-toolchain.toml`), Python 3.11+, Node 22 (`.nvmrc`) with Yarn 1, and
OpenGL/EGL (Mesa is fine). The first setup downloads PyTorch and the RL dependencies, several GB.

```bash
git clone --recurse-submodules https://github.com/garyb9/fly-drone.git
cd fly-drone
yarn setup          # submodule, .venv, native Rust extension (maturin), yarn install, web build
```

Python tools run through `scripts/venv.sh`, so no `source .venv/bin/activate` is needed for
`yarn` scripts. Activate the venv only to call `fly-drone` directly.

## Scripts

Like fly-playground, one root `package.json` drives every language:

| When                                 | Command                                 | What it does                                                                                                                            |
| ------------------------------------ | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| once                                 | `yarn setup`                            | bootstrap everything (`scripts/setup.sh`)                                                                                               |
| before committing                    | `yarn prep`                             | **auto-fix**: prettier, eslint `--fix`, `cargo fmt`, `ruff format` + `ruff --fix`; then typecheck, clippy, rebuild extension, build web |
| run the lab                          | `yarn dev`                              | rebuild extension + web, serve on **http://127.0.0.1:8000** (PID hover baseline, brain observing)                                       |
| run with a trained brain             | `yarn dev:policy runs/<run>/actor.json` | the neural decoder commands the drone                                                                                                   |
| frontend work                        | `yarn web:dev`                          | Vite with HMR, proxies `/ws` to a running `yarn dev`                                                                                    |
| CI (identical locally and on GitHub) | `yarn ci`                               | format/lint checks (TS, Rust, Python), typecheck, clippy, Rust tests, extension build, pytest, vitest, web build, 2 s flight smoke test |
| after `yarn dev` is up               | `yarn browser:check`                    | Playwright: telemetry, pause/resume, reset, reconnect, interventions, mobile layout                                                     |

Individual steps are also scripts: `format`, `lint`, `typecheck`, `test`, `build`, `rs:fmt`,
`rs:lint`, `rs:test`, `rs:bench`, `py:fmt`, `py:lint`, `py:build`, `py:test`, `smoke`.
`./scripts/check.sh` is an alias for `yarn ci`.

## Train and evaluate

```bash
yarn assay                                     # causal gate: vision reaches the policy features
yarn calibrate --trials 256 --output runs/calibration.npz
yarn train --task visual --steps 50000 --envs 4 \
     --calibration runs/calibration.npz --teacher-scale 0.7 --output runs/visual
yarn evaluate --policy runs/visual/actor.json --episodes 50 --workers 12 \
     --output runs/visual/evaluation.json
yarn train --task looming --steps 30000 --resume runs/visual/ppo.zip --output runs/looming
scripts/venv.sh fly-drone export runs/visual/checkpoints/rl_model_20000_steps.zip \
     --output runs/visual/actor-20k.json            # any checkpoint → parity-checked Rust actor
```

The actor sees only the activity of 2,022 descending/VNC motor neurons: never pixels, cues,
pose or target position. Neither PPO nor the warm start changes the connectome or neuron
parameters. Training refuses to start if the sensory causal gate fails. Evaluation compares the
trained brain against zeroed features, silenced visual inputs and shuffled features, plus a
30 s hover check. `runs/` is git-ignored. Accepted results go to `docs/results/`.

## Interact

### Trials and evaluation replay

Panel **05 · Trials & Replay** re-runs any evaluation episode exactly. Seeds fully determine
target placement, obstacle launch, brain noise and rendering on one machine.

1. Start the service with the policies you want to watch (this skips the extension rebuild):
   ```bash
   scripts/venv.sh fly-drone serve --policy runs/<steering-run>/actor.json \
       --looming-policy runs/<looming-run>/actor.json
   ```
2. **Run a trial:** choose the task (steer to target / dodge obstacle), the brain condition
   (intact, zeroed features, vision silenced, shuffled features) and a seed, then press
   **Run trial**.
3. **Replay an evaluation:** pick any `evaluation*.json` under `runs/` or `docs/results/`. Its seeds
   appear as pass/fail buttons for the selected condition. Clicking one loads that report's policy
   and replays the episode. The outcome line shows live bearing or obstacle distance, then
   PASSED/FAILED using the same rules as `fly-drone evaluate`.

Only `.json` actors under `runs/` or `docs/results/` can be loaded from the browser.

### Scene controls

- Orbit/zoom the main scene, or follow the drone.
- Pause or reset all clocks and bodies together.
- Move the visual target left/centre/right, and place an obstacle ahead.
- Select a displayed neuron and pulse, hold, silence or restore it.
- Inspect actual/commanded motor RPM, camera currents, neural readouts and motion commands.

The inspector shows measured somata and selected real connections, coloured by source activity.
It is not a measurement of transmission along axons. The fly's dynamics are illustrative. The
drone mesh is drawn 6× larger than its physical size.

## Documentation

| Doc                                                                                | Contents                                                                                                          |
| ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| [`docs/overview/`](docs/overview/)                                                 | **start here**: the cyborg goal, architecture diagrams (HTML), math, training, evidence, roadmap                  |
| [`docs/architecture.md`](docs/architecture.md)                                     | system overview, process boundaries, contracts                                                                    |
| [`docs/neuron-model.md`](docs/neuron-model.md)                                     | LIF equations, weights and signs, tonic fixed point, activity trace, CSR format, determinism, Rust actor          |
| [`docs/sensory-model.md`](docs/sensory-model.md)                                   | camera geometry, encoder equations, looming math, splay dead-zone experiment, causal assay                        |
| [`docs/control-and-physics.md`](docs/control-and-physics.md)                       | frames, CF2X constants, rotor wrench, drag, motor lag, cascaded PID and mixer, clock identities                   |
| [`docs/training.md`](docs/training.md)                                             | MDP, reward shaping, PPO/GAE, warm-start turn gain, export parity, evaluation statistics                          |
| [`docs/data-pipeline.md`](docs/data-pipeline.md)                                   | MaleCNS provenance, selection and transforms, dataset hash                                                        |
| [`docs/validation.md`](docs/validation.md)                                         | acceptance table with measured results and performance                                                            |
| [`docs/free-roam.md`](docs/free-roam.md)                                           | free-roam arena, sensory tuning evidence, composite teacher, distillation, pre-registered acceptance, live probes |
| [`docs/manual-checklist.md`](docs/manual-checklist.md)                             | automated and human verification items                                                                            |
| [`docs/connectome-navigation-findings.md`](docs/connectome-navigation-findings.md) | external navigation-connectome findings, relevance filter, model gaps, future path-integration milestone          |
| [`docs/references.md`](docs/references.md)                                         | papers, datasets and upstream code                                                                                |
| [`docs/superpowers/plans/`](docs/superpowers/plans/)                               | milestone plans                                                                                                   |

## Status and limits

This is a research simulator, not flight software. The stabiliser uses ideal simulated state,
and the visual adapter is engineered (two image statistics per eye). Onboard operation keeps the
full connectome and may need a larger airframe to carry compute. Hardware selection, state
estimation, motor identification, firmware validation and flight tests are later milestones.
The frontend can be hosted statically; simulation and training need a persistent service.

Project code: MIT. MaleCNS data: CC-BY 4.0. Fly asset: Apache-2.0. See
[`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md) and the included notices.
