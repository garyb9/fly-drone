# Wing-level action (decoder v6) — design

**Status:** proposed, awaiting user decision (2026-09-16). Documentation only — no code, no training,
no `ACCEPTANCE` change.

**Related:** the sensing side is
[`2026-09-16-retinotopic-sensing-v6-design.md`](2026-09-16-retinotopic-sensing-v6-design.md). This
spec takes the spatial encoder as its input contract and shares its training/evaluation changes.

## 1. Why

The goal says the drone **acts the way the fly's own motor system acted**. Today the decoder emits a
**motion intent** `u = [vx, vy, vz, ψ̇]` handed to a cascaded PID (`control-and-physics.md` §5); the
connectome never commands a wing. But the connectome contains the fly's flight motor neurons — the
power muscles (DLMn, DVMn) and the steering muscles (b1, b2, b3) — and this project's whole claim is
that the drone moves **because of what the brain's descending and motor neurons do**.

Two concrete costs of the velocity abstraction:

1. **Lateral action is artificially sluggish.** Measured 80% step response is **0.69 s vertical but
   1.48 s lateral** (`docs/results/roam-step-response.json`): the intent → PID → tilt → translate
   path inserts an airframe-tilt lag. That lag is exactly why the teacher falls back to vertical as
   the primary escape axis, and why `vy` stays near zero (vy/vz spec §2.6). Commanding roll/pitch
   directly makes lateral first-order, like the fly's differential wing steering.
2. **The motor interface is not the fly's.** The decoded quantity (velocity) is a control
   abstraction the fly does not have; the fly's descending/motor neurons command muscles.

This spec makes the decoder speak a lower-level body command — a **body wrench** — over a retained
reflex rate controller that plays the role the fly's VNC/halteres play. It does not remove
stabilisation; it deepens the learned command.

## 2. Findings (existing evidence)

| Item | Value / source |
| --- | --- |
| Power flight MNs | DLMn 10 (`a,b` + `c–f`), DVMn 14 (`1a–c`, `2a,b`, `3a,b`) |
| Steering MNs | b1 2, b2 2, b3 2; i1 2, i2 2, iii1 2, iii3 2, hg1–4 2 each |
| Group roles (`groups.json`) | `wing_l` 30, `wing_r` 30, `thrust` 16, `escape` 2 |
| Decoder input | 2,022 DN + VNC motor traces (`brain.py` `feature_ids`) — already includes every MN |
| Existing low-level hook | `plant.advance_rpm(rpm)` applies four rotor RPMs directly, bypassing the PID |
| Step response | vertical 0.69 s, lateral 1.48 s to 80% |
| Legacy step-response tool | `stabiliser.py` |

The motor neurons are already in the decoder's input; what is missing is a decoder **output** at the
wing/wrench level and a plant path that accepts it.

## 3. Principles kept

- Only neural activity reaches the decoder (unchanged 2,022 traces). No pose enters the actor.
- The body may execute **reflex-level stabilisation** — the role the fly's VNC and halteres play.
  A rate controller is that reflex, not a planner.
- One decoder, no mode switching.
- Legacy actors and the velocity-command acceptance stay reproducible.
- The new bars are additive; no existing threshold moves.

## 4. Architecture

```
2,022 DN + VNC motor traces
        |
   [Decoder]  ->  body wrench  w = [roll, pitch, yaw, thrust]   (bounded, tanh)
        |                                  (or, variant B: 4 rotor RPMs)
   [reflex rate controller, fixed]  ->  torques + collective  ->  rotor RPMs
        |
   plant.advance_rpm(rpm)  ->  rigid body
```

- **Output:** `[φ̇_ref, θ̇_ref, ψ̇_ref, T]` (desired body rates + collective), or directly
  `[τ_roll, τ_pitch, τ_yaw, T]`. Same 4-D interface as today, different semantics.
- **Reflex loop (fixed, not learned):** a fast rate/attitude controller maps the desired rates to
  rotor RPMs via the existing mixer (`control-and-physics.md` §5.5) and `advance_rpm`. This is the
  haltere/VNC analogue and is what keeps an unstable airframe flyable without vestibular input into
  the connectome.
- **Why not raw RPM (variant B):** without proprioceptive feedback into the brain, open-loop RPMs
  cannot stabilise the airframe — that requires ascending/gyro feedback into the connectome, a later
  milestone (`overview/README.md` §10). The reflex loop is the 80/20.
- **Training-only privileged critic** (already present) still gets visible geometry only.

### 4.1 Plant changes (outline)

- Add `advance_wrench(thrust, torques)` or a rate-controller entry point beside `advance` /
  `advance_rpm`.
- Keep `advance` (velocity) as the default so all legacy tasks, accepted actors and A1–A7 remain
  bit-identical.
- New action-space descriptor in the exported actor so a v6 actor is never loaded into a
  velocity-mode environment.

## 5. Training

- **Imitation warm start:** the same DAgger teacher labels become wrench targets via the fixed
  mapping "velocity intent → equivalent wrench" (a deterministic, honest transformation using
  simulator state only at label time, as today). This preserves the sight-gated teacher and the
  honest-labels rule.
- **RL:** SAC as today, on the same free-roam reward.
- **Action chunking + temporal ensembling** (L7 s59–60, hw3 H=16) becomes attractive here: wrench
  commands are high-frequency and smoothing helps. Keep temporal ensembling so the drone is not
  blind during a chunk (L7 s60).
- **Multimodal head** (mixture / flow-matching) only if the vy/vz instrumentation shows
  mode-averaging — the sensing spec §5 has the same trigger.

## 6. Evaluation

- **Intermediate gate (recommended):** reach the velocity-command A1–A7 first with the v6 encoder,
  then switch the action semantics. This isolates sensing gains from action gains.
- **New wing-level bars (proposed, additive):**
  - **Stability:** no tilt/bounds/altitude loss beyond the A6 limits under intact conditions.
  - **Lateral improvement:** 80% lateral step response strictly below the 1.48 s velocity baseline
    (the point of the change), measured with `stabiliser.measure_step_response`.
  - **Causality:** the spatial-silence gate and A3 ghost clause still hold.
  - **Realisability:** the wrench commands respect the actuator envelope (rate/torque clip).
- A1–A6, E1–E4, and `ACCEPTANCE` values are unchanged and reported alongside.

## 7. Risks

| Risk | Mitigation |
| --- | --- |
| Unstable/exploding flight without a velocity setpoint | Retained reflex rate loop; clip wrench to actuator envelope; keep velocity mode as fallback |
| Wrench targets from velocity labels are ill-defined | Deterministic label mapping at train time only; verify by replaying teacher flights |
| Breaks comparability with accepted velocity actors | Velocity mode stays default; v6 actor tagged; A1–A7 kept as the intermediate gate |
| Harder exploration for SAC | Clone init from the velocity teacher; action chunking; curriculum from the sensing spec |

## 8. Sequencing

| Step | What | Gate |
| --- | --- | --- |
| W0 | Plant `advance_wrench` + rate controller + step-response test | Lateral 80% step < 1.48 s |
| W1 | Wrench DAgger labels + clone | Teacher flights reproduce in wrench mode |
| W2 | SAC fine-tune | Stability + A1–A3 on wing-level bars |
| W3 | Full evaluation, new bars + preserved A1–A7/E1–E4 | Acceptance as decided by the user |

Wing-level action is sequenced **after** the sensing milestone's intermediate gate (sensing spec §8,
M3), so sensing and action gains are attributable separately.

## 9. Open decisions

1. **Command level:** desired body rates + thrust (recommended) vs body torque + thrust vs raw RPMs.
2. **Reflex loop:** keep the existing cascaded PID's attitude/mixer stages (recommended, least new
   code) vs a new dedicated rate controller.
3. **Timing:** after the sensing intermediate gate (recommended) vs in parallel.
4. **Action chunking:** include in W2 (recommended) vs after.
