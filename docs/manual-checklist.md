# Verification checklist

Ticked items are covered by an automated check: `yarn ci` (Rust + Python + web tests),
`yarn browser:check` (Playwright against a live `yarn dev` server), or `yarn evaluate`. Unticked
items need a person watching a real display, or are still open. Headless checks cannot judge
motion quality, colour, or whether behaviour _looks_ right.

Last full pass: 2026-09-13, commit `f272977` (Playwright 1.63 / Chromium, PID baseline server).

## Runtime and determinism

- [x] Full graph loads with all roles: 166,700 neurons, 2,022 features
      (`tests/test_runtime.py::test_full_graph_and_roles`).
- [x] Golden trace is stable and reproducible within a run
      (`crates/brain-core/tests/golden_trace.rs`).
- [x] Reset replays bit-identically and clears interventions
      (`test_runtime.py::test_reset_replays_and_interventions_clear`, `test_env.py::test_clocks_reset_and_seed_replay`).
- [x] Binding rejects invalid neuron ids, and separate runtime instances do not share state
      (`test_binding_rejects_invalid_ids`, `test_fixture_instances_are_independent`).
- [x] Rust actor export matches PyTorch to ≤ 1e−4 and rejects mismatched manifests or encoders
      (`test_runtime.py::test_export_inference_and_manifest_validation`; checked on every `train`).
- [x] Clock identities: after one frame, `tick = 48`, `physics_tick = 40`, `t = 0.04 s`
      (`test_env.py::test_clocks_reset_and_seed_replay`).

## Physics and control

- [x] Equal RPM at hover supports the weight (`test_physics.py::test_equal_rpm_supports_weight`).
- [x] Each single-motor perturbation produces the expected roll/pitch/yaw sign
      (`test_each_motor_torque_sign`).
- [x] Motor lag, saturation and invalid-input rejection (`test_motor_lag_saturation_and_bad_inputs`).
- [x] A relocated obstacle registers contact, including after reset (mocap body)
      (`test_physics.py::test_moved_obstacle_registers_contact`).
- [x] Hover recovers from a vertical perturbation (`test_hover_recovers_vertical_perturbation`).
- [x] Velocity/yaw commands move the body in the expected direction
      (`test_commands_move_in_expected_direction`).
- [ ] Rotor spin direction and phase animation _look_ right in the viewer (cyan/orange rotors
      alternate `[+,−,+,−]`). Needs a human eye.

## Sensory pathway

- [x] Rendered camera stimuli reach the frozen graph, and silencing removes the effect
      (`test_assay.py::test_rendered_camera_stimulus_reaches_frozen_graph`; `yarn assay`).
- [x] Light-cue sign follows the target side for bearings `±0.1…±0.6` rad
      (`test_assay.py::test_light_cue_sign_follows_target_side`). This guards the 0.45 rad
      splay dead zone described in [`sensory-model.md`](sensory-model.md).
- [ ] Eye thumbnails show the target in the eye on its side while turning. The first screenshot
      showed the target right → right eye, `light R = 1.70`. Recheck with a trained policy.

## Learning and evaluation

- [x] Evaluation runs every ablation plus the hover check and writes a report
      (`test_env.py::test_evaluate_smoke_with_zero_policy`).
- [x] Bearing is wrapped to `[−π, π]` everywhere (`test_env.py::test_target_bearing_wraps_to_pi`).
- [x] ≥ 80% steering on 50 held-out seeds, balanced left/right, degrading under zero/sensory/shuffle
      ablations: 100% / balanced 1.00 vs 0.00 / 0.00 / 0.12
      ([`results/evaluation-visual-v3.json`](results/evaluation-visual-v3.json)).
- [x] A blind fixed-direction turner cannot pass: balanced success is required (silenced vision scores
      48% raw, 0.00 balanced).
- [x] Looming episode: an aimed obstacle launches and hits a drone commanded to hold still, and
      evaluation writes looming metrics
      (`test_env.py::test_looming_obstacle_launches_and_hits_a_stationary_drone`, `test_evaluate_looming_smoke`).
- [ ] Looming-obstacle response. See [`validation.md`](validation.md).

## Service and viewer (`yarn dev`, then `yarn browser:check`)

- [x] Connects and renders all panels with no page errors.
- [x] Motor telemetry shows 4 rows of actual/commanded RPM near hover (10k–22k).
- [x] Pause freezes the tick counter; Resume advances it.
- [x] Reset trial increments the episode and restarts the tick count.
- [x] Reload/reconnect rejoins the same running episode (the simulation is not interrupted by a
      disconnect). The service also has a server-side test,
      `test_server.py::test_service_pause_reset_and_disconnect`.
- [x] Target left/centre/right, obstacle place/clear, and Pulse/Hold/Silence/Restore are accepted
      without errors.
- [x] The service rejects unrelated WebSocket origins (`test_server.py::test_service_rejects_unrelated_origin`).
- [x] 390 px mobile layout has no horizontal scroll.
- [ ] Silencing a displayed neuron visibly dims it in the Living Graph. Needs a human eye; the
      browser check only asserts the command is accepted.
- [ ] With `yarn dev:policy <actor.json>`, moving the target Left/Right turns the drone towards
      it, and the motion readout's yaw sign matches.
- [ ] Follow-drone camera and orbit controls feel smooth on a desktop GPU.
- [ ] Live loop reaches real time. The first pass showed **0.71× real time and 446 missed frame
      deadlines** on the PID baseline. After native-size eye rendering, the offline loop measures
      1.24×; the live server re-check is pending. See [`validation.md`](validation.md) → Performance.
