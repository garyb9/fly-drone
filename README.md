# Fly / Drone

A full MaleCNS fly connectome inside a simulated quadrotor: camera pixels enter a frozen Rust spiking network, a learned decoder turns neural activity into motion commands, and a stabilizing controller drives four MuJoCo rotors.

The browser shows the drone, its two eyes, a live anatomical activity graph, and an illustrative fly receiving the same neural outputs. The drone is the primary body; the fly has an independent trajectory.

## Run locally

Requires Rust, Python 3.11+, Node 22, and a working OpenGL/EGL implementation. The Rust extension is native; no Python reimplementation of the brain is used. First setup downloads Python/RL dependencies, which can take several GB with GPU-enabled PyTorch wheels.

```bash
git clone --recurse-submodules https://github.com/garyb9/fly-drone.git
cd fly-drone
./scripts/setup.sh
source .venv/bin/activate
fly-drone serve
```

Open **http://127.0.0.1:8000**. Without `--policy`, the scene explicitly runs a PID hover baseline while the brain observes the cameras and drives the fly panel. To give the neural decoder authority over drone movement:

```bash
fly-drone serve --policy runs/calibrated/actor.json
```

The service binds localhost. For frontend development, keep it running and execute `npm run dev --prefix web`. The Vite server proxies simulation traffic. MuJoCo's native viewer is also available with `MUJOCO_GL=glfw fly-drone baseline --viewer` on a desktop with a display.

## Train and evaluate

```bash
# Physics baseline and synthetic + actual-camera causal checks
fly-drone baseline --seconds 30
fly-drone assay

# Efficient first decoder: supervised neural-only warm start, then PPO
fly-drone calibrate --output runs/calibration.npz
fly-drone train --steps 1024 --calibration runs/calibration.npz --output runs/calibrated

# Pure PPO and continuation are also supported
fly-drone train --task visual --steps 20000 --output runs/visual
fly-drone train --task looming --steps 20000 --resume runs/calibrated/ppo.zip --output runs/looming

# Held-out seeds and all three ablation controls
fly-drone evaluate --policy runs/calibrated/actor.json --episodes 50
```

Tasks are `hover`, `visual`, and `looming`. The current visual reward encourages target orientation and approach; the warm-start teacher teaches yaw toward the target while preserving hover. Calibration uses simulator bearing only as an offline label. The actor sees 2,022 descending/motor neural activities, never raw image features, target coordinates, or simulator pose. PPO does not change the connectome or neuron parameters.

Training refuses to start if matched sensory stimulation/silencing fails. Checkpoints, exported Rust actor JSON, export-parity metrics, and evaluation metrics are stored under ignored `runs/`. A saved policy validates the graph hash, encoder version, neuron identities and layer dimensions before loading. Long training and evaluation are separate from CI.

## Interact

- Orbit/zoom the main scene; follow the drone.
- Pause or reset all clocks and bodies together.
- Move the visual target left/center/right and place an obstacle ahead.
- Select a displayed neuron and pulse, hold, silence or restore it.
- Inspect actual/commanded motor RPM, camera currents, neural readouts and motion commands.

The neural inspector displays measured somata and selected real connections, colored by source activity. It does not depict measured electrical transmission along axons. The fly's independent dynamics are adapted from `fly-playground`, with no movement assistance or noise; its wing animation is slowed for visibility. The drone mesh is enlarged 6x in the main view while physics stays at Crazyflie scale.

## Verification and architecture

```bash
./scripts/check.sh
# With the local server running and Chrome installed:
node scripts/browser-check.mjs
```

See [architecture](docs/architecture.md), [validation results](docs/validation.md), [source provenance](docs/source-provenance.json), and [attribution](docs/THIRD_PARTY.md). `requirements-tested.txt` records the direct package versions tested in this checkout; Cargo and npm have lockfiles. The drone simulator is pinned as a git submodule.

The current plant uses ideal state feedback for stabilization and an engineered visual adapter. It is a research simulator, not validated real-flight software. The onboard goal preserves the full connectome and allows a larger airframe carrying compute. Hardware selection, sensor estimation, identified motors, firmware validation and flight tests remain subsequent milestones. The frontend is suitable for later static hosting; the persistent simulation service remains separate.

Project code: MIT. MaleCNS data: CC-BY 4.0. Fly asset: Apache-2.0. See the included notices.
