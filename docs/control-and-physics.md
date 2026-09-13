# Control and physics: command → stabiliser → rotors → rigid body

Code: `python/fly_drone/plant.py` (`DronePlant`), the pinned upstream simulator
`vendor/mujoco-drones` (`multi_drone_mujoco/envs/base_aviary.py`, `control/pid_control.py`), and
`python/fly_drone/env.py` (the clock schedule). The upstream is a git submodule pinned in
`.gitmodules`; its commit is recorded by `git submodule status`.

## 1. Frames, units and conventions

| Item        | Convention                                                                      |
| ----------- | ------------------------------------------------------------------------------- |
| World       | right-handed, `+X` forward (towards targets), `+Y` left, `+Z` up; metres        |
| Body        | same axes attached to the drone; yaw `ψ` about `+Z`, counter-clockwise positive |
| Quaternion  | `[w, x, y, z]` (MuJoCo order)                                                   |
| Euler       | roll `φ`, pitch `θ`, yaw `ψ` from `_quatToRPY` (ZYX)                            |
| Rotor speed | RPM throughout (see §3 note)                                                    |
| Viewer      | Three.js is Y-up: display `= [x, z, −y]`; the drone mesh is drawn 6× larger     |

## 2. Airframe constants (Crazyflie 2.x, X configuration)

| Symbol        | Name                          | Value                                 |
| ------------- | ----------------------------- | ------------------------------------- |
| `m`           | mass                          | 0.027 kg                              |
| `L`           | motor-to-centre arm           | 0.0397 m                              |
| `J`           | inertia `diag(Ixx, Iyy, Izz)` | `diag(1.4e−5, 1.4e−5, 2.17e−5)` kg·m² |
| `k_f`         | thrust coefficient            | 3.16e−10 N/RPM²                       |
| `k_m`         | reaction-torque coefficient   | 7.94e−12 N·m/RPM²                     |
| `T/W`         | thrust-to-weight ratio        | 2.25                                  |
| `c_xy`, `c_z` | rotor drag coefficients       | 9.1785e−7, 1.0311e−6                  |
| `r_p`         | propeller radius              | 0.02325 m                             |

Derived:

```
hover RPM   ω_h   = √(m g / 4 k_f) = √(0.26487 / 1.264e−9) ≈ 14,476 RPM
max RPM     ω_max = √(T/W · m g / 4 k_f) = 1.5 ω_h ≈ 21,714 RPM
max thrust  T_max = 4 k_f ω_max² = 2.25 m g ≈ 0.596 N
yaw arm     k_m / k_f ≈ 0.0251 m
```

The PID baseline in [`results/baseline.json`](results/baseline.json) holds all four motors at
14,475.8 RPM, matching `ω_h`.

## 3. Rotor forces and body wrench

Motor `i` sits at `(±L/√2, ±L/√2, 0)` in order `(+x,+y), (−x,+y), (−x,−y), (+x,−y)` and produces:

```
f_i = k_f ω_i²        (thrust along body +Z)
q_i = k_m ω_i²        (reaction torque magnitude)
```

Upstream `_physics` combines these into a body wrench:

```
T   = Σ f_i
τ_x = (L/√2)(f₀ + f₁ − f₂ − f₃)           roll
τ_y = (L/√2)(−f₀ + f₁ + f₂ − f₃)          pitch
τ_z = −q₀ + q₁ − q₂ + q₃                  yaw
```

Rotor spin directions are `[+,−,+,−]` (counter-clockwise when positive), so the reaction yaw
signs are `[−,+,−,+]`. The wrench is rotated to world coordinates with the body rotation matrix
`R` and written into MuJoCo `xfrc_applied` every physics step:
`F_world = R [0, 0, T]ᵀ` and `τ_world = R τ_body`.

> **Units note.** Upstream comments label `k_f`, `k_m` in `(rad/s)²`, but the implementation
> squares RPM directly, and the hover and max speeds above are only self-consistent in RPM. This
> project uses the implementation convention everywhere.

### Drag

A rotor-induced linear drag (Forster 2015) is enabled:

```
F_drag = −R · ( diag(c_xy, c_xy, c_z) · (2π/60) Σ ω_i ) · (Rᵀ v)
```

That is velocity-proportional drag in the body frame, scaled by the total rotor speed in rad/s.
Ground effect and downwash exist upstream but are not applied (single drone, hovering at 1 m).

## 4. Motor lag and saturation

The stabiliser outputs commanded speeds `ω_c ∈ [0, ω_max]`. Each motor follows a first-order lag
integrated exactly at the 1 kHz physics step `h = 1 ms` with `τ_mot = 25 ms`:

```
τ_mot ω̇ = ω_c − ω
ω(t+h) = ω(t) + α (ω_c − ω(t)),     α = 1 − e^(−h/τ_mot) = 1 − e^(−0.04) ≈ 0.0392
```

A step command reaches 63% in 25 ms and 95% in ≈ 75 ms. Speeds stay non-negative and saturated.
`motor_tau = 0` gives `α = 1` (no lag). Rotor phase for the viewer integrates
`φ_i += ω_i · 2π/60 · h · dir_i`.

## 5. Cascaded stabiliser (upstream `PIDControl`, 200 Hz)

The policy never commands attitude or rotors. It commands a **motion intent**:

```
u = [v_x, v_y, v_z, ψ̇]  (body frame),   |u| ≤ [0.4, 0.4, 0.2, 0.8]  (m/s, m/s, m/s, rad/s)
```

### 5.1 Intent → setpoints (`DronePlant.advance`, every 5 ms)

```
v_world  = Rz(ψ) [v_x, v_y, 0]ᵀ + [0, 0, v_z]ᵀ
p_hold  ← clip(p_hold + v_world Δt, [−3, −3, 0.4], [3, 3, 2.5])
ψ_target ← ψ_target + ψ̇ Δt                       Δt = 5 ms
```

Integrating velocity into a position hold means that a zero command holds position rather than
drifting. The room bounds keep the hold inside the arena.

### 5.2 Position loop → desired acceleration and thrust

```
e_p = p_hold − p,   e_v = v_world − v,   I_p ← clip(I_p + e_p Δt, −2, 2)
a_des = K_P e_p + K_I I_p + K_D e_v + [0, 0, g]
T_des = m ‖a_des‖                  (a_des,z clamped ≥ 0: no downward thrust)
K_P = [0.4, 0.4, 1.0],  K_I = [0.01, 0.01, 0.01],  K_D = [0.9, 0.9, 2.0]
```

### 5.3 Desired attitude (differential flatness, small angle)

```
z_b = a_des / ‖a_des‖
x_c = [cos ψ_target, sin ψ_target, 0]
y_b = (z_b × x_c)/‖z_b × x_c‖,   x_b = y_b × z_b
φ_des = asin(y_b,z),   θ_des = atan2(−x_b,z, z_b,z)
```

### 5.4 Attitude loop → torques

```
e_R = [φ_des − φ, θ_des − θ, wrap(ψ_target − ψ)],   I_R ← clip(I_R + e_R Δt, −0.5, 0.5)
τ_des = K_P,R e_R + K_I,R I_R + K_D,R (e_R − e_R,prev)/Δt
K_P,R = [2e−3, 2e−3, 1e−3],  K_I,R = [0, 0, 1e−4],  K_D,R = [5e−4, 5e−4, 2e−4]
```

The attitude derivative is taken on the error, not on measured body rate.

### 5.5 Allocation (mixer)

Solve for per-motor thrusts `f = [f₀..f₃]`:

```
⎡ T  ⎤   ⎡   1      1      1      1   ⎤ ⎡f₀⎤
⎢ τ_x⎥ = ⎢  L/√2   L/√2  −L/√2  −L/√2 ⎥ ⎢f₁⎥
⎢ τ_y⎥   ⎢ −L/√2   L/√2   L/√2  −L/√2 ⎥ ⎢f₂⎥
⎣ τ_z⎦   ⎣ −k_m/k_f k_m/k_f −k_m/k_f k_m/k_f ⎦ ⎣f₃⎦
```

Before solving, torques are clipped to 30% of the achievable envelope:
`|τ_xy| ≤ 0.3 (L/√2) k_f ω_max²` and `|τ_z| ≤ 0.3 (k_m/k_f) k_f ω_max²`. Then
`f ← clip(A⁻¹b, 0, k_f ω_max²)` and `ω_c = √(f/k_f)`.

This controller receives **ideal state** (exact `p, q, v, ω` from MuJoCo). That is privileged
stabilisation. A real vehicle needs state estimation (§7).

## 6. Clock schedule and tick identities

```
physics  1000 Hz   h  = 1 ms      MuJoCo mj_step, motor lag, wrench, drag
brain     200 Hz   Δt = 5 ms      1 neural tick + 1 stabiliser update per 5 physics steps
camera     25 Hz   40 ms          render both eyes, encode cues, policy decision
```

One `ConnectomeEnv.step(action)` = 1 camera frame = 8 neural ticks = 40 physics steps. The
reset performs 40 neural settling ticks with the body frozen. So, at any time:

```
brain_tick = physics_tick / 5 + 40,      sim_time = physics_tick · 1 ms
```

Checked by `tests/test_env.py::test_clocks_reset_and_seed_replay`: after one step,
`tick = 48`, `physics_tick = 40`, `time = 0.04 s`. Episodes truncate at 750 frames (30 s). They
terminate on collision, `z ∉ [0.15, 3]` m, `|x|,|y| > 3.5` m, or `|φ|,|θ| > 1.2` rad.
Simulation time advances only by these ticks, never by wall-clock or viewer frame rate.

## 7. Gap to hardware

| Idealisation here                | Needed before flight                                              |
| -------------------------------- | ----------------------------------------------------------------- |
| exact state to PID               | IMU + optical-flow / mocap estimator with delay and noise         |
| upstream `k_f, k_m`, lag `τ_mot` | identified motor/propeller curves per airframe                    |
| no sensor delay                  | camera exposure + processing latency in the loop                  |
| no deadline misses               | onboard scheduler, command timeout → safe hover/land              |
| Python server                    | Rust brain + actor on onboard compute, independent emergency stop |
