# Hardware estimate: flying the connectome on a real drone

Status: **estimate, not validated** (2026-09-14). Only the connectome size and tick cost are
measured, and only on a desktop (`results/native-benchmark.json`, `neuron-model.md` §5). Hardware
weights, power draw and ARM speed ratios are rough ballparks from general knowledge and must be
measured before any purchase or design decision. The existing gap list is
[`control-and-physics.md`](control-and-physics.md) §7.

## 1. Verdict

It is feasible. **The brain's compute is not the hardest part:** the connectome needs about 90 MB
of RAM and 3.9 ms per tick on one desktop core. The hard parts are:

- cameras behaving differently from the simulator (sim-to-real perception);
- estimating the drone's own state without the simulator's exact position and velocity;
- latency and missed real-time deadlines.

## 2. Component sizes

| Part                                       | Size                       | Compute per second                                                                                          | Onboard verdict                                                 |
| ------------------------------------------ | -------------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| Connectome (166,700 neurons, 10.5 M edges) | 64 MB file, **~90 MB RSS** | 200 ticks; **3.9 ms p50 / 5.1 ms p95** per tick, single thread, i7-14700K, against a 5 ms budget (measured) | **Bottleneck.** p95 is already over budget on a desktop core    |
| Encoder v4 (4 image statistics)            | a few lines of Rust        | ~150 k pixel operations                                                                                     | trivial                                                         |
| Encoder v5 CNN (as specced)                | ~115 k parameters, ~0.5 MB | ~2.3 M multiply-adds per eye per frame, ~120 M/s at 25 Hz                                                   | trivial on any small board's CPU                                |
| Decoder MLP (2022 → 64 → 64 → 4)           | ~134 k parameters, ~0.5 MB | ~134 k multiply-adds per frame at 25 Hz                                                                     | trivial; a microcontroller could run it                         |
| Stabiliser (cascaded PID + mixer)          | —                          | 200 Hz                                                                                                      | off-the-shelf flight controllers (PX4, ArduPilot) already do it |

Encoder v5 parameter and compute counts assume "same" padding on the three stride-2 convolutions
in the v5 design spec (48×64 → 24×32 → 12×16 → 6×8).

### 2.1 Why the brain is the bottleneck

- `brain-core` is single-threaded, with no parallel or GPU code (its only dependencies are
  `serde` and `serde_json`).
- The largest per-tick cost is drawing 166,700 Gaussian noise samples (`neuron-model.md` §5).
- Embedded ARM cores (Raspberry Pi 5, Jetson Orin CPU) are plausibly 2–3× slower per core than
  the benchmark machine. That would give roughly **8–12 ms per tick, 0.4–0.6× real time** on one
  core. _Estimate; not measured._

Ways to reach real time onboard, none of which reduce the graph:

| Approach                | Effect                                            | Cost                                                                                                                           |
| ----------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Multithread the tick    | Neuron integration is embarrassingly parallel     | Spike scatter needs per-thread buffers; changed summation order breaks the bit-identical golden trace unless handled carefully |
| Cheaper noise generator | Removes the dominant Θ(N) cost                    | Same maths, different random stream: golden trace and seeded replays must be regenerated                                       |
| GPU (e.g. Jetson)       | 10.5 M synapses at 200 Hz is small work for a GPU | New backend and a parity test against the CPU reference                                                                        |

## 3. Build options

| Option                            | Hardware (rough)                                                                                                                                                 | All-up size                                                 | Difficulty         | Notes                                                                                                             |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ------------------ | ----------------------------------------------------------------------------------------------------------------- |
| **A. Brain on the ground**        | ~250 mm quadrotor with 2 cameras streaming video; a desktop runs brain + decoder and sends velocity commands back (e.g. PX4 offboard velocity mode over MAVLink) | ~0.4–0.7 kg                                                 | **Moderate**       | Still a cyborg: the brain decides, just not onboard. The radio link adds ~30–80 ms of delay. Best first real demo |
| **B. Onboard companion computer** | Jetson Orin Nano/NX-class (~7–25 W) or Raspberry Pi 5 (~5–12 W), plus a flight controller and 2 global-shutter cameras                                           | 5–7" or ~450 mm frame, ~0.8–1.5 kg, perhaps 8–15 min flight | **Hard**           | Needs the brain port (§2.1), a real-time scheduler, and thermal and power budgeting                               |
| **C. Tiny and efficient**         | Neuromorphic chip or FPGA                                                                                                                                        | could be small                                              | **Research-grade** | Hardware is hard to obtain and toolchains are immature. Not a near-term path                                      |

A Crazyflie-sized airframe is out: full-graph fidelity takes priority over size
([`architecture.md`](architecture.md), Hardware and hosting).

### 3.1 Camera requirements

Derived from the simulated eyes (`sensory-model.md`), not from a sensor study:

| Property         | Requirement                                                     | Why                                                                                        |
| ---------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Count and layout | 2 cameras, splayed ±0.75 rad, together ~177° with small overlap | Matches the left/right input roles                                                         |
| Shutter          | Global shutter                                                  | Rolling-shutter skew changes dark area while turning and reads as looming                  |
| Exposure         | Locked (no auto-exposure, no auto-gain)                         | v4 uses absolute brightness thresholds (0.55, 0.18); exposure changes would fire both cues |
| Frame rate       | ≥ 25 Hz, stable                                                 | Decoder and encoder run at 25 Hz                                                           |
| Resolution       | Any; downsampled to 64 × 48 per eye                             | The encoder input size is fixed                                                            |

## 4. Challenges beyond compute

| Challenge               | Why it bites this project specifically                                                                                                                                                                                                                               | Mitigation                                                                                                                                                                                                                       |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Real-world vision**   | v4 uses fixed brightness thresholds. Auto-exposure, 50/60 Hz light flicker, motion blur and rolling shutter all change dark area and trigger false loom. Loom already misfires from self-motion in the clean simulator. v5 would be trained on flat MuJoCo rendering | Global-shutter cameras with locked exposure. Build a **physical replica arena**: grey walls, dark bands, bright LED beacons, dark foam balls. Randomise noise, blur, exposure and latency in simulation before touching hardware |
| **State estimation**    | The simulated PID uses exact position and velocity, and velocity intent is integrated into a position hold. Real drones estimate state from IMU plus optical flow, rangefinder, visual odometry or motion capture, with noise, drift and delay                       | PX4 EKF with downward optical flow and a rangefinder, or motion capture for a first demo. Model the estimator in simulation and re-run the acceptance checks                                                                     |
| **Latency**             | The simulator has no camera exposure or processing delay, and one decision already takes 40 ms                                                                                                                                                                       | Measure end-to-end delay on the rig, put it into the simulator, and confirm the actor still passes                                                                                                                               |
| **Deadline misses**     | p95 tick time already exceeds 5 ms on a desktop; the live server misses 14% of 40 ms frame deadlines (`validation.md`)                                                                                                                                               | Real-time thread priority; command timeout → hover or land                                                                                                                                                                       |
| **Motors and airframe** | The simulated drone uses upstream motor and propeller constants                                                                                                                                                                                                      | Identify the real motor and propeller curves and refit the simulated plant                                                                                                                                                       |
| **Safety**              | A learned controller will do unexpected things                                                                                                                                                                                                                       | Netted room, prop guards, independent kill switch, geofence. A geofence is a body reflex and fits the principles, but it must be declared                                                                                        |

Favourable for hardware: the free-roam limits are slow (0.7 m/s forward, 0.8 rad/s yaw) and
thrown threats are slow (≤ 1.3 m/s), so real balls can be launched safely.

## 5. Difficulty by goal

| Goal                                                                                               | Difficulty                                                           |
| -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Brain on the ground flying a real drone in a replica arena                                         | **Moderate:** weeks to a few months for someone experienced with PX4 |
| Fully onboard, replica arena                                                                       | **Hard:** adds the brain port and real-time, thermal and power work  |
| Onboard in arbitrary real rooms, with foraging and dodging still passing silencing and ghost tests | **Very hard / research:** mostly sim-to-real perception, not compute |

The causal bar carries over to hardware: a real-world skill counts only if silencing its pathway
removes it on the real drone, and a ghost-like control (for example, an obstacle the cameras
cannot see) must not pass by chance.

## 6. First steps, in order

1. **Benchmark the brain on the target board.** Cross-compile `brain-core` and run the native
   benchmark on a Raspberry Pi 5 and a Jetson Orin. About a day; settles whether option B is
   realistic and replaces the ARM estimate in §2.1 with a measurement.
2. **Make the simulator more realistic.** Camera noise, exposure changes, motion blur, latency and
   an estimated-state stabiliser; re-run the pre-registered acceptance on the accepted actor.
3. **Build option A** in a physical replica arena, and repeat the silencing and ghost tests on the
   real drone.
