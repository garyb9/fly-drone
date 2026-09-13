# Architecture and scientific contract

The default dataset contains all 166,700 MaleCNS neurons and 10,520,431 retained directed edges.
No runtime performance fallback reduces the graph. Smaller synthetic assets only support unit
tests.

## Components

```
            ┌─────────────────────────── Python process (one simulation thread) ───────────────────────────┐
 MuJoCo ──► │ DronePlant: cameras (64×48 ×2) ─► BrainRuntime.sense ─► brain-core (Rust, PyO3) ─► features │
 physics    │      ▲                                                     │ LIF 166,700 neurons             │
 1 kHz      │      │ rotors ◄─ mixer ◄─ cascaded PID ◄─ motion intent ◄─ actor (Rust MLP) ◄──────────────┘  │
            │      └── FlyMirror (illustrative, shares readouts, no feedback)                                │
            └──────────────── newest telemetry frame ─► FastAPI WebSocket ─► Three.js viewer ──────────────┘
```

| Layer                                              | Owner                                                       | Details                                                                      |
| -------------------------------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Neural simulation, vision encoder, actor inference | `crates/brain-core` (no Python/MuJoCo/browser deps)         | [`neuron-model.md`](neuron-model.md), [`sensory-model.md`](sensory-model.md) |
| Python binding                                     | `crates/brain-python` (PyO3, abi3)                          | `BrainRuntime` in `python/fly_drone/brain.py`                                |
| Rigid body, rotors, stabiliser, clocks             | `python/fly_drone/plant.py` + pinned `vendor/mujoco-drones` | [`control-and-physics.md`](control-and-physics.md)                           |
| Gym env, PPO, calibration, evaluation              | `python/fly_drone/{env,training,calibration}.py`            | [`training.md`](training.md)                                                 |
| Service                                            | `python/fly_drone/server.py`                                | below                                                                        |
| Viewer                                             | `web/src` (Three.js, Vite)                                  | below                                                                        |

## Contracts

- **Only neurons reach the actor.** The policy input is the activity of 2,022 descending/VNC
  motor cells. Simulator state and image features cannot bypass the graph. Simulator state is
  used only by the stabiliser, rewards, calibration labels and evaluation.
- **Frozen brain.** Wiring, weights, signs, neuron parameters and tonic bias never change during
  learning. Only the decoder MLP is trained.
- **Identity binding.** Actors and calibrations store the dataset SHA-256, feature ids and
  `ENCODER_VERSION` (camera geometry, rendering, cue equations). Loading rejects any mismatch.
  Export checks Rust/PyTorch parity to 1e−4.
- **Causal gate.** Training refuses to start unless visual stimuli change the features and
  silencing removes that change.
- **Privileged stabilisation.** The PID sees ideal state. This is declared, not hidden.
- **Command interface.** Normalised action `[−1,1]⁴` → `[v_x, v_y, v_z, ψ̇]` limited to
  `[0.4, 0.4, 0.2, 0.8]` m/s, rad/s. Velocity integrates into a bounded position hold, and the PID
  owns attitude and motor allocation.
- **Clocks.** Physics 1 kHz, brain + PID 200 Hz, camera + policy 25 Hz;
  `brain_tick = physics_tick/5 + 40`. Time advances by ticks, independent of wall clock and
  viewer frame rate.
- **Units and frames.** SI units, right-handed `+X` forward / `+Y` left / `+Z` up, quaternions
  `[w,x,y,z]`, rotor speeds in RPM.

## Service, replay and interventions

One simulation thread owns the brain, renderer, controller and fly. Network tasks never touch
simulation objects. Commands go through a bounded queue, and each client receives only the
newest complete telemetry frame, never a backlog. Browser disconnects do not stop the
simulation. Exceptions stop the session and are reported to clients. Slow clients are dropped
after a send timeout. The WebSocket rejects unrelated origins. This is a localhost service, not a
hosted multi-user design.

Pause freezes all state. Reset clears neural state, sensory history, controller integrators,
motor state, interventions and fly state, increments the episode, and repeats the supplied seed.
Seeds and tick ids make runs repeatable on one machine. Pixel-identical rendering across
GPUs/drivers is not promised.

Brain geometry shows measured somata. Up to 1,000 displayed edges are real connections between
displayed cells, and their brightness follows source activity. The fly is a port of
fly-playground's force/integration model with noise and movement assistance removed. It shares
the brain's readouts, moves independently, and provides no sensory input. Wing animation is
slowed for legibility.

## Hardware and hosting

The frontend can be hosted statically. Long-running simulation and training need a persistent
service. Onboard deployment must profile `brain-core` on candidate compute, add camera drivers
and state estimation, identify motor dynamics, and validate command timeouts and an independent
emergency stop. Full-graph fidelity takes priority over Crazyflie size. Desktop RSS includes
Python metadata, so it is not an onboard power or payload measurement. See
[`control-and-physics.md`](control-and-physics.md) §7.
