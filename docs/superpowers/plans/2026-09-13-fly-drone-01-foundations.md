# fly-drone 01: a connectome-controlled drone laboratory

Original milestone plan (recorded verbatim in substance, 2026-09-13). Status of
each item is tracked in [`../../validation.md`](../../validation.md) and the
continuation plan [`2026-09-13-fly-drone-02-continuation.md`](2026-09-13-fly-drone-02-continuation.md).

## Summary

Create `/home/gb/projects/fly-drone` and private GitHub repository `garyb9/fly-drone`.

Build locally first: Rust brain and policy inference, Python MuJoCo simulation and RL, browser
visualization. The first milestone is one Crazyflie-sized drone hovering and steering toward
visual stimuli, with a docked brain inspector and a fly receiving the same neural outputs.

The eventual goal is autonomous onboard operation. Preserve full-connectome fidelity and permit
a larger airframe if onboard compute requires it.

## Control loop and implementation

- Copy the reusable Rust simulation core and necessary visualization assets from the current
  fly-playground working tree. Record source commit, copied-file hashes, local modifications, and
  attribution; leave the original project untouched.
- Use the full MaleCNS bundle: 166,700 neurons and 10,520,431 retained edges. Keep wiring,
  weights, and neuron parameters frozen during RL. Smaller fixtures serve development tests only.
- Expose the Rust core to Python through PyO3/maturin: load, reset, inject sensory currents,
  step, read neural features, and inspect selected activity. Use the same Rust runtime during
  training and playback.
- Pin MuJoCo-drones-gym to a reviewed commit behind a project-owned environment adapter. Start
  with CF2X physics and its cascaded controller; verify motor order, coordinate conventions,
  thrust units, and torque signs before integration.
- Run physics at 1,000 Hz, neural simulation and stabilization at 200 Hz, and camera/policy
  updates at 25 Hz. Advance by simulation ticks, independently of viewer frame rate.
- Simulate two body-mounted cameras, aimed left/right of forward, initially 64×48 pixels each.
  Rust extracts bounded luminance, temporal contrast, and image-expansion cues. Map these through
  the existing annotated visual and looming cell sets, explicitly documenting the engineered
  sensory model.
- Train a small PPO actor on descending/motor neural activity. Its four outputs are body-frame
  velocity targets and yaw rate. The actor receives no direct camera features or simulator
  state; simulator state remains available to stabilization, rewards, and evaluation.
- Export actor weights, normalization, bounds, and neuron IDs into a versioned artifact for Rust
  inference. Keep the critic and training machinery in Python.
- Convert motion targets through stabilization and motor allocation into four nonnegative rotor
  speeds. Add configurable motor lag, saturation, and thrust/drag parameters. Show commanded and
  actual rotor speeds and fixed rotation directions.

Public interfaces comprise `BrainRuntime`, `MotionCommand`, `MotorState`, and timestamped
`TelemetryFrame`; document SI units, quaternion ordering, and body/world frames.

## Viewer and training workflow

- First establish headless stepping and MuJoCo viewer playback; then build the
  TypeScript/Three.js interface using the existing rendering approach. WebGPU acceleration is a
  later optimization.
- Make the drone the main view, with orbit/follow controls, rotor visualization, camera
  thumbnails, and target/obstacle placement.
- Dock a brain inspector showing measured activity, selected anatomical connections,
  stimulation/silencing controls, and the sensory → neural → command → rotor trace.
- Feed identical neural readouts into the existing fly body model in a separate docked scene.
  The fly has its own trajectory and contributes no sensory feedback to the drone. Label it as
  an illustrative body model.
- Stream telemetry over a local WebSocket with bounded queues. Viewer disconnects do not
  interrupt physics; pause/reset explicitly synchronize brain, controller, cameras, motors, and
  fly.
- Train through hover-preserving commands, visual target orientation/approach, then
  looming-obstacle response. Save checkpoints and reproducible evaluation episodes.
- Provide documented commands for setup, baseline flight, training, evaluation, and interactive
  playback. Keep training output and caches outside version control.
- Host the frontend later on Vercel-like infrastructure, with simulation/training on a
  persistent service. Remote provisioning and authentication are outside the first milestone.

## Validation and acceptance

- Physics: equal motor thrust supports hover; individual motor perturbations produce expected
  roll/pitch/yaw; motor limits and lag behave correctly.
- Runtime: preserve Rust golden traces; verify Python bindings, reset isolation, inference
  export parity, and frame-rate-independent stepping.
- Sensory causality: left/right visual stimuli produce distinguishable downstream neural
  responses. Silencing the relevant pathways changes those responses. If this fails, report the
  failed mapping experiment before spending on RL.
- Flight: target 30-second hover with altitude RMS error below 0.15 m after settling, and
  correct target-directed steering in at least 80% of 50 held-out seeded trials.
- Brain contribution: compare trained runs against zeroed readouts, silenced sensory pathways,
  and shuffled neural features. Hover alone does not demonstrate connectome control; require
  measurable visual-steering degradation under ablation.
- Viewer: all panels share episode/tick identifiers; verify pause, reset, reconnect,
  intervention controls, and displayed motor telemetry.
- Performance: measure full-graph memory, neural tick latency, camera cost, and complete-loop
  real-time factor. Report missed deadlines explicitly; never silently reduce the brain.
- Include CI for Rust tests, Python integration tests, frontend checks, and a short simulation
  smoke test. Long training evaluations run separately.

## Onboard path and assumptions

- The first milestone demonstrates a frozen-connectome controller with learned decoding and
  explicit stabilization; it does not establish biological equivalence.
- Initial state feedback to stabilization is idealized. Before hardware, introduce sensor
  estimation, delay, noise, motor identification, and firmware-in-the-loop validation.
- Package Rust sensory processing, brain stepping, and actor inference independently of Python
  so they can move onto onboard compute.
- Select onboard hardware and the final airframe only after measuring memory, latency, power,
  and camera requirements. Full-connectome fidelity takes priority over Crazyflie size.
- Hardware flight is a subsequent milestone requiring onboard deadline handling, command timeout
  behavior, and an independent emergency stop.
