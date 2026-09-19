# Body-agnostic, fidelity-first cyborg — design

**Status:** proposed, authorized 2026-09-19. Documentation and goal reframing first; no source
under `python/fly_drone/` or `crates/` changes until this spec and the reframed goal land.

**P0 landed 2026-09-19 (negative).** The declared adapter is the default free-roam bridge and the
teacher-free gate (`adapter-check`) is implemented. The gate **failed**: the connectome plus a
declared readout adapter cannot fly the drone to `ACCEPTANCE` (0.03 vs teacher 0.73 beacons/min;
A1/A2/A3/A6 fail). Evidence and numbers:
[`../../results/adapter/FINDING-2026-09-19.md`](../../results/adapter/FINDING-2026-09-19.md).
`ACCEPTANCE` and the accepted actors are unchanged. P1/P2 proceed as the proposed next contract
changes (each needs user sign-off).

**P1 landed 2026-09-19 (weak positive).** Ascending/proprioceptive feedback is an opt-in, additive
alternate bundle (`data/malecns-feedback`, manifest marker `feedback`; generated, git-ignored), with
a new `bundle_hash` and a `feedback-check` silencing gate. The gate passed causally on
`slow_fraction` only (0.572 intact vs 0.623 silenced, 95% CI excludes 0); it did not move foraging
or collisions. Evidence:
[`../../results/feedback/FINDING-2026-09-19.md`](../../results/feedback/FINDING-2026-09-19.md).

**P2 infrastructure landed 2026-09-19 (prior negative; not fitted).** `dynamics.bin` carries
per-neuron leak/threshold and the core integrates them; a declared Shiu-2024 LIF prior is emitted as
an additive alternate bundle (`data/malecns-dynamics`, marker `dynamics`). The prior maps onto the
canonical constants (identical leak/threshold; only `noise_sigma` differs), and the causal gate
**failed to beat uniform LIF**, so the canonical bundle is kept per the §6 falsifier. No recorded
activity is in-repo, so no fit has been run; the recording-fitted array is the remaining P2 step.
Evidence: [`../../results/dynamics/FINDING-2026-09-19.md`](../../results/dynamics/FINDING-2026-09-19.md).

**Related:** the goal contract is [`../../overview/README.md`](../../overview/README.md) §1;
the current free-roam method is [`../../free-roam.md`](../../free-roam.md); the closed negative
that motivated P0/P2 is [`../../results/encoder-v6/CONCLUSION-2026-09-18.md`](../../results/encoder-v6/CONCLUSION-2026-09-18.md);
the external survey is [`../../external-prior-art.md`](../../external-prior-art.md); the fly-body
roadmap this spec defers to Scope A is
[`fly-playground/docs/2026-09-13-connectome-articulation-plan.md`](https://github.com/garyb9/fly-playground/blob/main/docs/2026-09-13-connectome-articulation-plan.md).

**Scope.** Change the _method_, not the fidelity of the stated goal: keep the frozen connectome as
the brain, shrink the learned bridge so no engineered teacher produces the behaviour, close the
loop with body-derived feedback, and anchor the neuron/synapse dynamics to recorded activity. The
drone stays the simulated body in v1; a fly-like body (`flybody`/NeuroMechFly) is the fidelity
reference and the physical drone is the deployment target.

## 1. Why: the drift this fixes

Three structural choices currently let the learned decoder, not the connectome, be the brain:

1. **An engineered teacher injects the behaviour.** `teacher.py` blends four geometry-driven drives
   and DAgger/PPO/SAC fit a decoder to imitate it. The decoder learns the teacher's policy; the
   connectome is a feature source.
2. **The connectome is open-loop about its own body.** There is no ascending/proprioceptive/haltere
   feedback (the fly's own VNC closure), so the decoder must absorb every body mismatch and model
   error.
3. **The dynamics are the coarsest available model.** Uniform LIF signs, silent modulators, no
   per-neuron tuning. Lappalainen/Shiu (_Nature_ 2024) show per-neuron/per-synapse parameters can be
   _fitted_ to recordings while the wiring stays frozen — a principled middle path never considered
   here.

The symptom is already recorded: v5 and both v6 decoder rounds collapse to blind flailing
(`../../results/encoder-v6/SAC-ROUND2-2026-09-19.md`), and `external-prior-art.md` §1 states that
nobody has solved the goal and that every strong survey project learns only the readout. This spec
takes that seriously.

**Alignment.** The documented goal already says the body must never decide. P0/P1/P2 remove the
ways the current build quietly violates that. The goal text is broadened only in the body clause
(drone → embodiment), not in the contract clauses.

## 2. Reframed goal (lands before code)

Broadened statement, to replace the drone-only wording in `README.md`, `AGENTS.md`,
`docs/overview/README.md` §1 and `docs/architecture.md`:

> **The connectome is the brain; the body is an embodiment, not the intelligence.** Today the body is
> a simulated quadrotor (the cyborg milestone); the fidelity reference is the fly's own biomechanical
> body (`flybody`/NeuroMechFly); the deployment target is a physical drone. Only descending/motor
> (and, once added, ascending) neurons touch the body adapter; the wiring stays frozen; any fitted
> dynamics or added feedback is a declared, versioned contract change; the body supplies reflexes,
> never decisions; every skill must be causal.

Every existing contract clause in `overview/README.md` §1 is preserved; only the body identity is
generalised.

## 3. Principles kept

- **Frozen wiring.** No workstream changes the adjacency, signs, or connectivity of the canonical
  `data/malecns` bundle. Fitted dynamics (P2) and feedback (P1) are separate, additive identities.
- **Acceptance untouched.** `roam_eval.ACCEPTANCE` and `CONDITIONS` are never relaxed. New gates
  (teacher-free, feedback silencing) are added, not substituted.
- **Accepted actors reproducible.** v4 canonical encoder and the accepted room actors stay
  byte-loadable; `test_legacy_room_mjcf_unchanged` and the accepted `.json` artifacts are preserved
  before any cleanup.
- **Learned paths retained.** The RL/SAC/DAgger encoder+decoder code stays in the tree as a
  deprecated, optional path (see §8), so parts remain available.
- **Additive identities.** Every new bundle gets a distinct `bundle_hash`; a policy can only run
  against the bundle it was trained/fitted on (the `sign-v2` pattern).
- **No long run without sign-off.** Fits and training runs ask the user first (AGENTS.md).

## 4. P0 — Declared body adapter (core de-overfit move)

Replace the single learned bridge with two declared pieces:

| Piece            | What                                                                                                                                                                                     | Where                                                                                      |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| **Action codec** | A fixed/calibrated map from a small set of named descending/motor populations to body commands. Calibrated on a stimulus battery (FlyDrones-style ridge), not trained against a teacher. | new `python/fly_drone/adapter.py`, reusing `calibration.py`, `scripts/stimulus_battery.py` |
| **Body adapter** | The quadrotor setpoint integrator + cascaded PID + mixer. Declared non-learnable behaviour.                                                                                              | `python/fly_drone/plant.py` (unchanged contract)                                           |

- The **declared-adapter policy is the new default**. The learned policy path
  (`training.py`, `sac.py`, `distill.py`, `teacher.py`, `crates/brain-core/src/policy.rs`) is kept
  behind an explicit flag and marked deprecated.
- A new **teacher-free causal gate**: the declared policy must pass A1–A7 with no teacher anywhere
  in the loop. Evaluation reports both declared and learned policies.
- **Falsifier.** If the declared adapter yields no useful behaviour, that is a real finding about
  connectome/model sufficiency. Report it; do not retreat to the teacher to pass a gate.

## 5. P1 — Closed loop: ascending/proprioceptive feedback (contract change C1)

Give the connectome the body signals a real fly has. New annotated input roles, a new `bundle_hash`,
and an explicit clause in `overview/README.md` §1:

| Feedback                | Source                     | Candidate cells                                                     |
| ----------------------- | -------------------------- | ------------------------------------------------------------------- |
| Angular rate / haltere  | body gyro, `plant.state()` | haltere / ascending rate channels                                   |
| Optic flow              | rendered camera motion     | T4/T5 (validated against FlyView)                                   |
| Wing/leg proprioception | actuator/joint state       | VNC proprioceptive afferents (`body-mappings.json` `joint`, `load`) |

- Reuse fly-playground `src/sim/body-mappings.json` input roles and the ascending cells named in
  [`../../connectome-navigation-findings.md`](../../connectome-navigation-findings.md) §2
  (`AN07B037`, `AN06B009`, …) as candidates.
- **Causal test:** injecting and then silencing the feedback must change behaviour, and removing the
  feedback-dependent skill must not be recoverable by chance (ghost control).
- **Contract:** this adds inputs beyond pixels. It is declared, versioned, and opt-in. It does not
  relax the "body never decides" rule; it only tells the brain what the body is doing.

## 6. P2 — Connectome-constrained dynamics (contract change C2)

Fit per-neuron/per-synapse free parameters to recorded activity, with **wiring frozen**, and emit an
**additive** bundle (new `model` marker + `bundle_hash`; canonical uniform-LIF bundle untouched).

- **Sources:** Shiu et al. _Nature_ 2024 activation/silencing dataset and Lappalainen et al.
  _Nature_ 2024 connectome-constrained recordings.
- **Method:** keep the graph fixed; fit leak/threshold per neuron and a weight scale per synapse (or
  a declared low-dimensional parameterisation) against recordings; guard against saturation.
- **Identity:** additive, exactly like `sign-v2`. Accepted actors never load it. It is a **new
  frozen identity**, not a trained policy; the rule and the fitted constants are fixed offline.
- **Falsifier.** If fitted dynamics do not beat uniform LIF on the causal gates, do not adopt:
  report and keep the canonical bundle.
- **Compute:** the fit is a long run; ask the user before starting.

## 7. P3–P6 — relay, transfer, sensing, evaluation

- **P3 Visual relay.** Validate/activate photoreceptor→lamina rather than only pooled v4 cues
  (fly.ai finding #2: the histaminergic relay). Validate the motion path against FlyView optic-flow
  ground truth.
- **P4 Transfer.** Domain randomization (arena, lighting, textures, camera/body noise), the
  pre-specified unseen-layout generalisation, quadrotor system-ID, latency and actuator noise — the
  sim-to-real package.
- **P5 Evaluation.** Keep A1–A7, ghost, rewired-graph, and E3-bypass untouched. Add the teacher-free
  gate (P0) and real-fly baselines: DeepFly3D leg kinematics, the 2025 multi-strain looming/escape
  dataset.
- **P6 Scope A (separate plan).** Adopt `flybody`/NeuroMechFly through fly-playground's articulation
  plan; the P0 adapter abstraction lets the same neural readout drive drone or fly body. Do this in
  fly-playground, never by editing the hash-pinned `web/src/fly/*` reference copies.

## 8. Deprecation and cleanup

| Action                | Target                                                                                                                                                                                                                               |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Preserve**          | the accepted room actors (`runs/v4-closed-s04/actor.json`, `runs/v4-loom-ft/actor.json`) and the current v6 pair (`runs/v6/clone/encoder.pt`, `runs/v6/round0/decoder.json`, gate report) into a committed location, before deletion |
| **Delete**            | the rest of `runs/` (old v3/v4/v5 sweeps, calibrations, logs)                                                                                                                                                                        |
| **Keep (deprecated)** | v6 encoder/decoder code + the RL/SAC/DAgger/teacher modules, behind an explicit flag; `docs/results/encoder-v6/DEPRECATION.md` records why it did not work                                                                           |
| **Keep**              | canonical v4 encoder, alternate-bundle scripts, legacy tests, `web/src/fly/*` reference copies, `vendor/mujoco-drones` (v1 is still the drone)                                                                                       |
| **Add**               | `docs/results/encoder-v6/DEPRECATION.md`; prior-art additions (Eon Systems Mar 2026, `rndlabsoy/fly-brain-full`, `snedea/flybrain`, Lappalainen/Shiu 2024) to `references.md` and `external-prior-art.md`                            |

## 9. Sequencing (v1)

| Step | What                                                            | Contract          | Compute              |
| ---- | --------------------------------------------------------------- | ----------------- | -------------------- |
| 0    | Goal reframing + prior-art additions + this spec                | —                 | none                 |
| 0b   | Cleanup (preserve actors, delete `runs/`, deprecation note)     | —                 | none                 |
| 1    | P0 declared adapter + teacher-free gate (landed; gate negative) | none              | calibration          |
| 2    | P1 ascending feedback (landed; weak positive)                   | C1 (new identity) | probes               |
| 3    | P2 connectome-constrained fit (infra landed; prior negative)    | C2 (new identity) | fit — ask first      |
| 4    | P4 transfer/domain randomization                                | none              | training — ask first |
| 5    | P3 visual relay                                                 | none              | probes               |
| 6    | P5 evaluation additions                                         | none              | eval runs            |
| 7    | P6 Scope A fly body                                             | goal broadening   | separate plan        |

Each step: `env -u PYTHONPATH .venv/bin/python -m pytest -q`,
`.venv/bin/ruff format python tests && .venv/bin/ruff check python tests`, then commit and push.

## 10. Risks and honest failure modes

- **Teacher removal may leave the connectome unable to fly.** That is a finding, not a bug; report it.
- **P1 can destabilise the frozen dynamics.** Guard with the causal assay and a ghost control; make
  the feedback opt-in and versioned.
- **P2 may not beat uniform LIF.** Adopt only on causal-gate evidence; otherwise keep canonical LIF.
- **"Fitted dynamics" can be misread as training the brain.** Prevent with fixed constants, frozen
  wiring, an additive identity, and clear labels — the same guard as the path-integration milestone
  in `connectome-navigation-findings.md` §6.
- **Scope A duplicates fly-playground.** Coordinate via that repo's articulation plan; do not fork
  its body quietly.

## 11. Artifacts

- This spec; reframed `README.md`, `AGENTS.md`, `docs/overview/README.md`, `docs/architecture.md`.
- `docs/references.md`, `docs/external-prior-art.md` (new prior art).
- `docs/results/encoder-v6/DEPRECATION.md`; preserved actors under `docs/results/actors/` (or the
  path chosen in Step 0b).
- Later: `python/fly_drone/adapter.py`, feedback role tables, the fitted bundle, new tests and gates.

## 12. Open decisions

1. **Where preserved actors live** — `docs/results/actors/` (committed) vs a curated `runs/accepted/`
   that survives cleanup. Recommended: `docs/results/actors/`, and update `accepted-policies.json` /
   `current-policies.json` to point there.
2. **P1 input granularity** — population rate codecs first vs the labelled ascending cells directly.
   Recommended: labelled cells where a defensible mapping exists, populations otherwise, each with a
   confidence label.
3. **P2 fitting target** — leak/threshold only vs additionally a per-synapse scale. Recommended:
   leak/threshold first (smallest free-parameter set), escalate only on evidence.
