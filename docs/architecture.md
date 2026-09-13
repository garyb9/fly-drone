# Runtime and scientific contract

The default dataset contains all 166,700 MaleCNS neurons and 10,520,431 retained directed edges. No runtime performance fallback reduces the graph. Smaller synthetic assets only support unit tests. Project data retains its CC-BY-4.0 attribution.

`brain-core` is a native Rust library with no Python, browser, or MuJoCo dependency. It contains the copied deterministic LIF simulator, camera preprocessing, and exported MLP inference. `brain-python` exposes it through PyO3. The Python `BrainRuntime` registers anatomical sensory and motor identities. The live policy input is the activity of 2,022 descending/VNC motor cells; simulator state and image features cannot bypass the graph into the actor.

Python owns MuJoCo, camera rendering, a cascaded PID, motor lag, PPO, and the local HTTP/WebSocket service. The controller uses ideal simulated state; this is explicitly privileged stabilization, not onboard state estimation. Rust performs deployed actor inference. The Gym action is normalized to [-1,1]; the runtime command is [vx,vy,vz,yaw_rate], limited to [0.4,0.4,0.2,0.8] in m/s and rad/s. Velocity targets integrate into position targets with room bounds. The policy controls targets; PID controls attitude and allocates motor effort.

## Clocks and coordinates

Physics steps every 1 ms. Stabilization and the brain step every 5 ms. Camera and policy updates occur every 40 ms. Neural reset includes 40 settling ticks with a stationary body; consequently brain tick = physics_tick/5 + 40. Each subsequent camera period advances eight neural and controller steps and forty physics steps. A cue is held and injected once per neural tick. Training can run faster/slower than wall time without changing these ratios.

Drone coordinates are right-handed +X forward, +Y left, +Z up. Quaternions are [w,x,y,z]. Display conversion is [x,z,-y]. Motor positions in order are (+x,+y), (-x,+y), (-x,-y), (+x,-y). Positive rotation phase is counterclockwise viewed from above; rotor directions are [+,-,+,-], reaction yaw signs [-,+,-,+]. Thrust is `kf * RPM²`, torque `km * RPM²`; upstream coefficient comments suggesting rad/s are inconsistent with its implementation. The model uses the implementation convention throughout. Motor lag is a first-order 25 ms response integrated at 1 kHz, with nonnegative saturated speeds. Single-drone drag is enabled; downwash is not relevant to this milestone.

The viewer enlarges the drone 6x for legibility; world positions, target geometry and camera physics remain at simulation scale. The fly panel uses Y-up illustrative units rather than drone meters.

## Vision and learning

Two 64×48 cameras have 75° vertical FOV and ±0.45 rad horizontal splay. The Rust encoder emphasizes luminance above 0.55 with gain 400, followed by bounded sustained ON current and positive temporal change. The threshold and gain are engineering calibration for the demonstrator's camera range. Dark-area expansion drives annotated left/right LC4/LPLC2 looming populations. This is not retinal reconstruction or optical-flow estimation; brightness changes and rotation can confound these proxies.

Visual currents enter the original annotated Mi1/Tm3 sets. The neural graph, synaptic weights, neuron equations and tonic power-cell bias remain fixed. PPO learns a small tanh MLP decoder with optional fixed feature normalization. An optional supervised warm start labels rendered camera trials with desired yaw, using simulator bearing only to construct training labels. Rust actor inference accepts only neural activity. `calibrate`, `train --calibration`, and `evaluate` distinguish calibration fit, policy optimization and held-out flight tests.

The causal gate covers synthetic images and actual rendered targets with matched seeds and pathway silencing. Training refuses to start if this gate fails. Passing the gate proves software signal propagation, not successful behavior or biological validity. Policy export checks dataset hash, feature IDs, dimensions and finite parameters, then checks Rust/PyTorch output parity.

## Viewer, replay, and interventions

One simulation thread owns brain, MuJoCo renderer, controller and fly. Network tasks never touch simulation objects directly. Commands use a bounded queue; each client receives the newest complete telemetry frame rather than an unbounded backlog. Browser disconnects do not stop simulation. Exceptions stop the session and are shown to clients. Slow clients are disconnected after a send timeout. This is a localhost service, not a hosted multi-user security design.

Pause freezes all simulation state. Reset clears neural state, sensory history, controller integrators, motor state, interventions and fly state, increments episode, and repeats the supplied seed. The neural/body seed and tick IDs make evaluation rerunnable. Policy runs are saved with seed-specific metrics; exact cross-platform camera pixels and floating-point traces are not promised.

Brain geometry samples measured somata. Up to 1,000 displayed edges are real connections between displayed cells; their brightness follows source activity and is not a measurement of transmission along an axon. The fly is a port of the source project's force/integration model with noise and movement assistance omitted, sharing the brain's readouts but generating independent motion. Its dynamics are illustrative; it cannot supply drone sensory input. Wing animation is slowed for legibility and is not a measured wingbeat rate.

## Hardware and hosting

The frontend can later be statically hosted; a persistent service is needed for long-running simulation and training. Onboard deployment must profile the Rust library on candidate compute, add camera drivers and state estimation, identify motor dynamics, and validate firmware/command timeouts and emergency stop. Full-graph fidelity takes priority over Crazyflie size. Current desktop RSS includes Python metadata; it is not an onboard power/payload measurement.
